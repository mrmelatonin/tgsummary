"""
ua_tts - Ukrainian Text-to-Speech application and library.
Synthesizes Ukrainian text into speech WAV (or MP3) audio using neural Ukrainian TTS models.
"""

from ua_tts.stanza_cache import patch_stanza_cache
from ua_tts.converter import UkrainianTTSConverter
from ua_tts.audio import save_as_wav, save_as_mp3, save_audio
from ua_tts.voices import Voices, Stress, UKRAINIAN_VOICES

__version__ = "0.1.0"
__author__ = "Antigravity Developer"

__all__ = [
    "UkrainianTTSConverter",
    "patch_stanza_cache",
    "save_as_wav",
    "save_as_mp3",
    "save_audio",
    "Voices",
    "Stress",
    "UKRAINIAN_VOICES",
    "__version__",
]
