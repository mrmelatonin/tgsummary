import os
import json
import logging
import tempfile
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

CACHE_FILE_NAME = "fetched_posts_cache.json"
RESPONSE_CACHE_FILE_NAME = "gemini_response_cache.json"


def save_cached_response(summary_text: str, cache_dir: str = "cache") -> str:
    """Atomically saves the latest Gemini response text for explicit replay."""
    cache_path = os.path.join(cache_dir, RESPONSE_CACHE_FILE_NAME)
    temporary_path = None
    try:
        if not isinstance(summary_text, str) or not summary_text.strip():
            raise ValueError("Cannot cache an empty summary.")
        os.makedirs(cache_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir,
                                         delete=False) as stream:
            temporary_path = stream.name
            json.dump({"summary_text": summary_text}, stream, ensure_ascii=False, indent=2)
        os.replace(temporary_path, cache_path)
        logger.info("Cached Gemini response to '%s'.", cache_path)
        return cache_path
    except (OSError, ValueError) as error:
        logger.warning("Failed to cache Gemini response: %s", error)
        return ""
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def load_cached_response(cache_dir: str = "cache") -> str:
    """Loads the latest nonempty Gemini response text, or returns an empty string."""
    cache_path = os.path.join(cache_dir, RESPONSE_CACHE_FILE_NAME)
    try:
        with open(cache_path, encoding="utf-8") as stream:
            data = json.load(stream)
        summary_text = data.get("summary_text") if isinstance(data, dict) else None
        if not isinstance(summary_text, str) or not summary_text.strip():
            raise ValueError("Cached response has no valid summary text.")
        logger.info("Loaded cached Gemini response from '%s'.", cache_path)
        return summary_text
    except (OSError, ValueError) as error:
        logger.warning("Cannot load cached Gemini response: %s", error)
        return ""

def get_cache_path(cache_dir: str = "cache") -> str:
    """Returns absolute or relative path to cache file."""
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, CACHE_FILE_NAME)

def save_cached_posts(posts: List[Dict[str, Any]], cache_dir: str = "cache") -> str:
    """Saves fetched posts to a local JSON cache file."""
    cache_path = get_cache_path(cache_dir)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)
        logger.info(f"Successfully cached {len(posts)} posts to '{cache_path}'.")
        return cache_path
    except Exception as e:
        logger.warning(f"Failed to save posts to local cache: {e}")
        return ""

def load_cached_posts(cache_dir: str = "cache") -> List[Dict[str, Any]]:
    """Loads cached posts from JSON file if it exists."""
    cache_path = get_cache_path(cache_dir)
    if not os.path.exists(cache_path):
        logger.warning(f"Cache file '{cache_path}' does not exist.")
        return []

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            posts = json.load(f)
        logger.info(f"Loaded {len(posts)} post(s) from local cache '{cache_path}'.")
        return posts
    except Exception as e:
        logger.error(f"Failed to read local cache '{cache_path}': {e}")
        return []

def has_cached_posts(cache_dir: str = "cache") -> bool:
    """Checks whether valid local cache exists."""
    return os.path.exists(get_cache_path(cache_dir))
