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

    logger.debug("Searching for cached Stanza '%s' resources across %d candidate locations...", lang, len(candidates))

    for c in candidates:
        if not c or not os.path.isdir(c):
            continue
        c_abs = os.path.abspath(c)
        res_json = os.path.join(c_abs, "resources.json")
        lang_dir = os.path.join(c_abs, lang)
        has_res = os.path.exists(res_json)
        has_model = os.path.isdir(lang_dir) or os.path.exists(os.path.join(c_abs, f"{lang}.zip"))
        logger.debug("  Checking Stanza directory candidate: %s (resources.json: %s, lang_data: %s)", c_abs, has_res, has_model)
        # Check if resources.json and language folder or zip exists
        if has_res and has_model:
            logger.debug("Matched cached Stanza '%s' directory: %s", lang, c_abs)
            return c_abs

    logger.debug("No cached Stanza '%s' directory found among candidate paths", lang)
    return None


def is_stanza_model_cached(model_dir: Optional[str] = None, lang: str = "uk") -> bool:
    """
    Check if required Stanza model files for language exist locally.
    """
    if not model_dir:
        model_dir = find_cached_stanza_dir(lang)
    if not model_dir or not os.path.isdir(model_dir):
        logger.debug("Stanza model directory is invalid or not found: %s", model_dir)
        return False

    res_json = os.path.join(model_dir, "resources.json")
    if not os.path.isfile(res_json):
        logger.debug("resources.json missing in %s", model_dir)
        return False

    # Check for unpacked processor model (e.g. tokenize/iu.pt)
    tok_pt = os.path.join(model_dir, lang, "tokenize", "iu.pt")
    if os.path.isfile(tok_pt):
        logger.debug("Found unpacked Stanza processor model: %s", tok_pt)
        return True

    # Check if default.zip is present, and extract if needed
    default_zip = os.path.join(model_dir, lang, "default.zip")
    if os.path.isfile(default_zip):
        logger.info("Extracting cached Stanza archive '%s' to '%s'...", default_zip, os.path.join(model_dir, lang))
        try:
            with zipfile.ZipFile(default_zip, "r") as z:
                z.extractall(os.path.join(model_dir, lang))
            if os.path.isfile(tok_pt):
                logger.debug("Successfully unpacked Stanza archive: %s", tok_pt)
                return True
        except Exception as e:
            logger.warning("Failed to unpack cached Stanza archive %s: %s", default_zip, e)

    logger.debug("Stanza '%s' model weights not found in %s", lang, model_dir)
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
        logger.info("Stanza offline cache active (resources dir: %s)", cached_dir)
    else:
        logger.debug("No local Stanza cache found at patch time (preferred_dir: %s)", preferred_dir)

    _orig_download = stanza.download
    _orig_pipeline_init = stanza.pipeline.core.Pipeline.__init__

    def cached_download(lang="en", *args, **kwargs):
        target_dir = kwargs.get("dir") or os.environ.get("STANZA_RESOURCES_DIR") or rc.DEFAULT_MODEL_DIR
        logger.debug("stanza.download invoked for lang='%s', target_dir='%s'", lang, target_dir)
        if is_stanza_model_cached(target_dir, lang):
            logger.info("Using cached Stanza %s models from %s (skipping re-download)", lang, target_dir)
            return
        logger.info("Downloading Stanza %s models to %s...", lang, target_dir)
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
            logger.debug("Pipeline.__init__ for '%s': overridden download_method to DownloadMethod.NONE (offline mode)", lang)
            kwargs["download_method"] = stanza.pipeline.core.DownloadMethod.NONE
        else:
            logger.debug("Pipeline.__init__ for '%s': local cache not ready, using standard download method", lang)
        return _orig_pipeline_init(self, lang, *args, **kwargs)

    stanza.download = cached_download
    stanza.pipeline.core.Pipeline.__init__ = cached_pipeline_init
    stanza.Pipeline = stanza.pipeline.core.Pipeline

    _PATCHED = True
    logger.debug("Stanza cache patch applied successfully.")
    return True
