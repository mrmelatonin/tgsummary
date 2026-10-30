"""
Ukrainian voice registry and validation module.
"""

from enum import Enum
from typing import Dict, List, Any

try:
    from ukrainian_tts.tts import Voices as _LibVoices, Stress as _LibStress
    Voices = _LibVoices
    Stress = _LibStress
except ImportError:
    class Voices(Enum):
        Tetiana = "tetiana"
        Dmytro = "dmytro"
        Mykyta = "mykyta"
        Lada = "lada"
        Oleksa = "oleksa"
        Olena = "olena"
        Olga = "olga"

    class Stress(Enum):
        Dictionary = "dictionary"
        Model = "model"


# Dictionary of available Ukrainian voices with metadata
UKRAINIAN_VOICES: Dict[str, Dict[str, str]] = {
    "tetiana": {
        "name": "Tetiana",
        "gender": "Female",
        "description": "Natural Ukrainian female voice (Tetiana / Тетяна)",
        "engine_voice": "uk_tetiana"
    },
    "dmytro": {
        "name": "Dmytro",
        "gender": "Male",
        "description": "Expressive Ukrainian male voice (Dmytro / Дмитро)",
        "engine_voice": "uk_dmytro"
    },
    "mykyta": {
        "name": "Mykyta",
        "gender": "Male",
        "description": "Clear Ukrainian male voice (Mykyta / Микита)",
        "engine_voice": "uk_mykyta"
    },
    "lada": {
        "name": "Lada",
        "gender": "Female",
        "description": "Melodic Ukrainian female voice (Lada / Лада)",
        "engine_voice": "uk_lada"
    },
    "oleksa": {
        "name": "Oleksa",
        "gender": "Male",
        "description": "Resonant Ukrainian male voice (Oleksa / Олекса)",
        "engine_voice": "uk_oleksa"
    },
    "olena": {
        "name": "Olena",
        "gender": "Female",
        "description": "Classic Ukrainian female voice (Olena / Олена)",
        "engine_voice": "uk_olena"
    },
    "olga": {
        "name": "Olga",
        "gender": "Female",
        "description": "Expressive Ukrainian female voice (Olga / Ольга)",
        "engine_voice": "uk_olga"
    }
}

DEFAULT_VOICE = "tetiana"


def is_valid_voice(voice_name: Any) -> bool:
    """Check if the given voice name or Voices enum is a valid supported Ukrainian voice."""
    if hasattr(voice_name, "value"):
        voice_name = voice_name.value
    v_str = str(voice_name).lower().strip()
    # Handle prefixes like uk_ or enum names
    for key in UKRAINIAN_VOICES:
        if v_str == key or v_str.endswith(f"_{key}") or v_str.endswith(f".{key}"):
            return True
    return False


def get_available_voices() -> List[Dict[str, str]]:
    """Return a list of dictionary metadata for all supported Ukrainian voices."""
    result = []
    for key, info in UKRAINIAN_VOICES.items():
        result.append({
            "key": key,
            "name": info["name"],
            "gender": info["gender"],
            "description": info["description"]
        })
    return result


def resolve_voice(voice_name: Any) -> str:
    """Validate and return normalized voice key."""
    if hasattr(voice_name, "value"):
        voice_name = voice_name.value
    v_str = str(voice_name).lower().strip()
    for key in UKRAINIAN_VOICES:
        if v_str == key or v_str.endswith(f"_{key}") or v_str.endswith(f".{key}"):
            return key
    valid_keys = ", ".join(UKRAINIAN_VOICES.keys())
    raise ValueError(
        f"Unsupported Ukrainian voice '{voice_name}'. Available Ukrainian voices: {valid_keys}"
    )

