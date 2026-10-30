"""
Stanza resource caching and offline optimization for Ukrainian TTS.

Prevents redundant network calls to GitHub (resources.json) and repetitive
archive extraction when Stanza Ukrainian models are already cached on disk.
"""

import glob
import logging
import os
import zipfile
from typing import Optional

logger = logging.getLogger(__name__)

_PATCHED = False


def find_cached_stanza_dir(lang: str = "uk") -> Optional[str]:
    """
    Search for directory containing cached Stanza resources for the specified language.
    Checks environment variables, project model folders, and default OS cache locations.
    """
    candidates = []

    # 1. Explicit environment variable
    env_dir = os.environ.get("STANZA_RESOURCES_DIR")
    if env_dir:
        candidates.append(env_dir)

    # 2. Local project directories
    cwd = os.getcwd()
    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([
        os.path.join(cwd, "models", "stanza"),
        os.path.join(cwd, "models", "ua_tts", "stanza"),
        os.path.join(cwd, "models", "ua_tts"),
        os.path.join(module_dir, "..", "models", "stanza"),
        os.path.join(module_dir, "..", "models", "ua_tts", "stanza"),
        os.path.join(module_dir, "..", "models", "ua_tts"),
    ])

    # 3. Default Stanza model dir if stanza is installed
    try:
        import stanza.resources.common as rc
        if getattr(rc, "DEFAULT_MODEL_DIR", None):
            candidates.append(rc.DEFAULT_MODEL_DIR)
    except Exception:
        pass

    # 4. Standard OS cache locations on Windows and Unix
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        candidates.extend(glob.glob(os.path.join(local_app_data, "StanfordNLP", "stanza", "Cache", "*", "resources")))
        candidates.extend(glob.glob(os.path.join(local_app_data, "Packages", "PythonSoftwareFoundation.Python.*", "LocalCache", "Local", "StanfordNLP", "stanza", "Cache", "*", "resources")))

    home = os.path.expanduser("~")
    candidates.extend([
        os.path.join(home, "stanza_resources"),
        os.path.join(home, ".cache", "stanza"),
    ])

    for c in candidates:
        if not c or not os.path.isdir(c):
            continue
        c_abs = os.path.abspath(c)
        res_json = os.path.join(c_abs, "resources.json")
        lang_dir = os.path.join(c_abs, lang)
        # Check if resources.json and language folder or zip exists
        if os.path.exists(res_json) and (os.path.isdir(lang_dir) or os.path.exists(os.path.join(c_abs, f"{lang}.zip"))):
            return c_abs

    return None


def is_stanza_model_cached(model_dir: Optional[str] = None, lang: str = "uk") -> bool:
    """
    Check if required Stanza model files for language exist locally.
    """
    if not model_dir:
        model_dir = find_cached_stanza_dir(lang)
    if not model_dir or not os.path.isdir(model_dir):
        return False

    res_json = os.path.join(model_dir, "resources.json")
    if not os.path.isfile(res_json):
        return False

    # Check for unpacked processor model (e.g. tokenize/iu.pt)
    tok_pt = os.path.join(model_dir, lang, "tokenize", "iu.pt")
    if os.path.isfile(tok_pt):
        return True

    # Check if default.zip is present, and extract if needed
    default_zip = os.path.join(model_dir, lang, "default.zip")
    if os.path.isfile(default_zip):
        try:
            with zipfile.ZipFile(default_zip, "r") as z:
                z.extractall(os.path.join(model_dir, lang))
            if os.path.isfile(tok_pt):
                return True
        except Exception as e:
            logger.warning("Failed to unpack cached Stanza archive %s: %s", default_zip, e)

    return False


def patch_stanza_cache(preferred_dir: Optional[str] = None) -> bool:
    """
    Apply monkey-patches to Stanza so it reuses cached models without network checks.
    
    1. Replaces stanza.download with a cached check: skips download if models already exist.
    2. Overrides download_method in Pipeline.__init__ with DownloadMethod.NONE when models exist locally,
       preventing network updates and redundant archive re-extraction.
    """
    global _PATCHED
    if _PATCHED:
        return True

    try:
        import stanza
        import stanza.pipeline.core
        import stanza.resources.common as rc
    except ImportError:
        logger.debug("Stanza is not installed; skipping Stanza cache patch.")
        return False

    cached_dir = preferred_dir or find_cached_stanza_dir("uk")
    if cached_dir and os.path.isdir(cached_dir):
        if not os.environ.get("STANZA_RESOURCES_DIR"):
            os.environ["STANZA_RESOURCES_DIR"] = cached_dir
        rc.DEFAULT_MODEL_DIR = cached_dir

    _orig_download = stanza.download
    _orig_pipeline_init = stanza.pipeline.core.Pipeline.__init__

    def cached_download(lang="en", *args, **kwargs):
        target_dir = kwargs.get("dir") or os.environ.get("STANZA_RESOURCES_DIR") or rc.DEFAULT_MODEL_DIR
        if is_stanza_model_cached(target_dir, lang):
            logger.info("Using cached Stanza %s models from %s (skipping re-download)", lang, target_dir)
            return
        return _orig_download(lang, *args, **kwargs)

    def cached_pipeline_init(self, lang="en", *args, **kwargs):
        target_dir = (
            kwargs.get("dir")
            or kwargs.get("model_dir")
            or os.environ.get("STANZA_RESOURCES_DIR")
            or rc.DEFAULT_MODEL_DIR
        )
        if is_stanza_model_cached(target_dir, lang):
            # Set DownloadMethod.NONE so Stanza loads directly from local files
            # without checking GitHub for updates or re-extracting archives.
            kwargs["download_method"] = stanza.pipeline.core.DownloadMethod.NONE
        return _orig_pipeline_init(self, lang, *args, **kwargs)

    stanza.download = cached_download
    stanza.pipeline.core.Pipeline.__init__ = cached_pipeline_init
    stanza.Pipeline = stanza.pipeline.core.Pipeline

    _PATCHED = True
    logger.debug("Stanza cache patch applied successfully.")
    return True
