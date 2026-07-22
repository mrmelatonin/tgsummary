import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

CACHE_FILE_NAME = "fetched_posts_cache.json"

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
