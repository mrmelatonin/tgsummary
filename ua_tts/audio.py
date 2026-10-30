"""
Audio processing, WAV generation, and audio exporting module for Ukrainian TTS.
"""

import io
import os
from typing import Optional, Dict, Any

try:
    import numpy as np
except ImportError:
    np = None

try:
    import soundfile as sf
except ImportError:
    sf = None

DEFAULT_SAMPLE_RATE = 24000  # Standard speech sample rate (24kHz)
DEFAULT_BITRATE = "128k"     # Speech-optimized MP3 bitrate


def normalize_audio(audio_data) -> Any:
    """
    Ensure float32 1D numpy array normalized within [-1.0, 1.0].
    """
    if np is None:
        raise ImportError("numpy is required for audio processing. Install with: pip install numpy")
    audio = np.asarray(audio_data, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.flatten()
    if len(audio) == 0:
        return audio
    max_val = float(np.max(np.abs(audio)))
    if max_val > 1.0:
        audio = audio / max_val
    return audio


def audio_diagnostics(audio_data, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Dict[str, Any]:
    """
    Compute key diagnostic metrics for synthesized audio array.
    Useful for debugging silent, noisy, or clipped audio.
    """
    if np is None:
        return {"samples": 0, "duration_sec": 0.0, "is_silent": True}
    audio = np.asarray(audio_data, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.flatten()

    if len(audio) == 0:
        return {
            "samples": 0,
            "duration_sec": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "rms": 0.0,
            "has_nan_inf": False,
            "is_silent": True,
            "is_clipped": False,
        }

    has_nan_inf = bool(np.isnan(audio).any() or np.isinf(audio).any())
    clean_audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

    min_val = float(np.min(clean_audio))
    max_val = float(np.max(clean_audio))
    mean_val = float(np.mean(clean_audio))
    std_val = float(np.std(clean_audio))
    rms_val = float(np.sqrt(np.mean(clean_audio ** 2)))

    return {
        "samples": len(audio),
        "duration_sec": round(len(audio) / float(sample_rate), 3),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "mean": round(mean_val, 6),
        "std": round(std_val, 4),
        "rms": round(rms_val, 4),
        "has_nan_inf": has_nan_inf,
        "is_silent": rms_val < 1e-4,
        "is_clipped": abs(min_val) >= 0.999 or abs(max_val) >= 0.999,
    }


def save_as_wav(
    audio_data: np.ndarray,
    output_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE
) -> str:
    """
    Save 1D float32 audio NumPy array to a standard 16-bit PCM WAV file.

    Uses `soundfile` if available, and gracefully falls back to Python's standard
    library `wave` module.

    Parameters:
        audio_data: NumPy float32 audio sample array
        output_path: Destination path for WAV file
        sample_rate: Audio sampling rate in Hz (default: 24000)

    Returns:
        Absolute path to created WAV file.
    """
    audio = normalize_audio(audio_data)

    # Ensure output filename ends with .wav
    if not output_path.lower().endswith(".wav"):
        output_path = output_path + ".wav"

    # Ensure parent directory exists
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    saved = False
    if sf is not None:
        try:
            sf.write(output_path, audio, sample_rate, format="WAV", subtype="PCM_16")
            saved = True
        except Exception:
            saved = False

    if not saved:
        try:
            import wave
            int16_audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(output_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(int16_audio.tobytes())
            saved = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to export audio as WAV to '{output_path}': {e}"
            )

    return os.path.abspath(output_path)


def save_as_mp3(
    audio_data: np.ndarray,
    output_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    bitrate: str = DEFAULT_BITRATE
) -> str:
    """
    Save 1D float32 audio NumPy array to an MP3 file at specified sample_rate and bitrate.

    Uses `soundfile` directly if MP3 format is supported by the installed libsndfile backend,
    otherwise falls back to `pydub` (via wav BytesIO buffer).

    Parameters:
        audio_data: NumPy float32 audio sample array
        output_path: Path where output MP3 will be saved
        sample_rate: Audio sampling frequency in Hz (default: 24000)
        bitrate: Target MP3 encoding bitrate (default: "128k")

    Returns:
        Absolute path to created MP3 file.
    """
    audio = normalize_audio(audio_data)

    # Ensure output filename ends with .mp3
    if not output_path.lower().endswith(".mp3"):
        output_path = output_path + ".mp3"

    # Ensure parent directory exists
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    # 1. Try soundfile MP3 direct write first
    saved = False
    if sf is not None:
        try:
            sf.write(output_path, audio, sample_rate, format="MP3")
            saved = True
        except Exception:
            saved = False

    # 2. Try pydub fallback
    if not saved:
        try:
            from pydub import AudioSegment
            wav_buf = io.BytesIO()
            int16_audio = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            if sf is not None:
                sf.write(wav_buf, audio, sample_rate, format="WAV", subtype="PCM_16")
            else:
                import wave
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(int16_audio.tobytes())

            wav_buf.seek(0)
            sound = AudioSegment.from_wav(wav_buf)
            sound.export(output_path, format="mp3", bitrate=bitrate)
            saved = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to export audio as MP3 to '{output_path}': {e}\n"
                "Please ensure 'soundfile' (with MP3 support) or 'pydub' with ffmpeg is installed."
            )

    return os.path.abspath(output_path)


def save_audio(
    audio_data: np.ndarray,
    output_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    bitrate: str = DEFAULT_BITRATE
) -> str:
    """
    Save audio NumPy array to output file, automatically selecting WAV or MP3 format based on file extension.
    Defaults to WAV format.
    """
    if output_path.lower().endswith(".mp3"):
        return save_as_mp3(audio_data, output_path, sample_rate=sample_rate, bitrate=bitrate)
    return save_as_wav(audio_data, output_path, sample_rate=sample_rate)
