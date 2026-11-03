"""Speech narration module supporting local Ukrainian neural TTS (ua-tts), Google Gemini TTS, and Azure Speech REST API."""

import base64
import io
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Union, Any
from xml.sax.saxutils import escape, quoteattr

logger = logging.getLogger("src.speech")

UA_TTS_VALID_VOICES = {"tetiana", "dmytro", "mykyta", "lada", "oleksa", "olena", "olga"}
GEMINI_TTS_PREBUILT_VOICES = {"kore", "puck", "charon", "fenrir", "aoede"}


def prepare_speech_text(text: str) -> str:
    """Remove digest Markdown and links while preserving spoken content."""
    text = re.sub(r"!?\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"(?m)^\s*(?:#{1,6}\s+|[-*+]\s+|>\s*)", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]", "", text)
    # XML 1.0 does not permit these control characters.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def split_speech_text(text: str, limit: int = 2500) -> list[str]:
    """Bound requests, preferring paragraph/sentence/word boundaries."""
    if limit < 1:
        raise ValueError("Speech chunk limit must be positive.")
    chunks = []
    while len(text) > limit:
        end = max(text.rfind("\n", 0, limit + 1), text.rfind(". ", 0, limit))
        if end < limit // 2:
            end = text.rfind(" ", 0, limit + 1)
        if end <= 0:
            end = limit
        elif text[end:end + 2] == ". ":
            end += 1
        chunks.append(text[:end].strip())
        text = text[end:].strip()
    if text:
        chunks.append(text)
    return chunks


class AzureSpeechSynthesizer:
    """Streams Ukrainian neural speech from the Azure Cognitive Services Speech REST API."""

    def __init__(self, api_key: str, region: str, voice: str = "uk-UA-PolinaNeural"):
        if not api_key or not region:
            raise ValueError("Azure Speech API key and region are required.")
        self.api_key = api_key
        self.region = region
        self.voice = voice

    def synthesize(self, text: str, output_path: Path) -> Path:
        clean_text = prepare_speech_text(text)
        if not clean_text:
            raise ValueError("The digest contains no text to narrate.")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = split_speech_text(clean_text)
        try:
            with open(output_path, "wb") as output_file:
                for chunk in chunks:
                    output_file.write(self._synthesize_chunk(chunk))
            return output_path
        except Exception:
            output_path.unlink(missing_ok=True)
            raise

    def _synthesize_chunk(self, text: str) -> bytes:
        ssml = (
            f"<speak version='1.0' xml:lang='uk-UA'>"
            f"<voice name={quoteattr(self.voice)}>{escape(text)}</voice>"
            f"</speak>"
        )
        request = urllib.request.Request(
            f"https://{self.region}.tts.speech.microsoft.com/cognitiveservices/v1",
            data=ssml.encode("utf-8"),
            headers={
                "Ocp-Apim-Subscription-Key": self.api_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
                "User-Agent": "tgsummary",
            },
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    audio = response.read()
                    if not audio or not response.headers.get("Content-Type", "").startswith("audio/"):
                        raise RuntimeError("Azure Speech returned an empty or non-audio response.")
                    return audio
            except urllib.error.HTTPError as error:
                status = error.code
                error.close()
                if status not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise RuntimeError(f"Azure Speech failed (HTTP {status}). Check resource, voice and quota.") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise RuntimeError("Azure Speech request timed out or could not connect.") from None
            time.sleep(2 ** (attempt + 1))


class GeminiSpeechSynthesizer:
    """Text-to-Speech synthesizer powered by Google Gemini 3.8 Flash TTS via the Interactions API."""

    def __init__(
        self,
        api_key: str = "",
        voice: str = "Kore",
        model: str = "gemini-3.8-flash-tts",
        style: Optional[str] = None,
        max_retries: int = 3,
    ):
        resolved_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini speech synthesis.")
        self.api_key = resolved_key

        voice_clean = str(voice or "").strip()
        voice_lower = voice_clean.lower()
        if voice_lower in GEMINI_TTS_PREBUILT_VOICES:
            self.voice = voice_lower.capitalize()
        elif voice_clean:
            self.voice = voice_clean
        else:
            self.voice = "Kore"

        self.model = model or "gemini-3.8-flash-tts"
        self.style = style
        self.max_retries = max(1, int(max_retries))
        self._client = None

        logger.debug(
            "GeminiSpeechSynthesizer initialized: voice=%s, model=%s, style=%s, max_retries=%d",
            self.voice, self.model, self.style, self.max_retries
        )

    def _get_client(self) -> Any:
        if self._client is None:
            logger.debug("Initializing Google GenAI client for Gemini TTS...")
            from google import genai
            http_options = {
                "retry_options": {
                    "attempts": self.max_retries,
                    "max_delay": 10.0,
                }
            }
            self._client = genai.Client(api_key=self.api_key, http_options=http_options)
        return self._client

    def _synthesize_chunk_raw(self, chunk: str) -> bytes:
        """Call Gemini Interactions API for a single text chunk and return raw WAV bytes."""
        client = self._get_client()
        content_item: dict[str, Any] = {
            "type": "text",
            "text": chunk,
        }
        if self.style:
            content_item["annotations"] = [
                {
                    "type": "speech_metadata",
                    "style": self.style,
                }
            ]

        last_error = None
        for attempt in range(self.max_retries):
            try:
                interaction = client.interactions.create(
                    model=self.model,
                    input=[
                        {
                            "type": "user_input",
                            "content": [content_item],
                        }
                    ],
                    response_format={"type": "audio"},
                    generation_config={
                        "speech_config": [
                            {"voice": self.voice}
                        ]
                    },
                )
                output_audio = getattr(interaction, "output_audio", None)
                if not output_audio or not getattr(output_audio, "data", None):
                    raise RuntimeError("Gemini TTS returned an interaction without audio data.")
                return base64.b64decode(output_audio.data)
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_transient = any(code in err_str for code in ("503", "429", "unavailable", "resourceexhausted", "quota"))
                if attempt < self.max_retries - 1 and is_transient:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(
                        "Gemini TTS request attempt %d failed (%s); retrying in %ds...",
                        attempt + 1, e, wait_time
                    )
                    time.sleep(wait_time)
                else:
                    break

        raise RuntimeError(f"Gemini TTS synthesis failed after {self.max_retries} attempts: {last_error}") from last_error

    def synthesize(self, text: str, output_path: Path) -> Path:
        """Synthesize text into a speech audio file (MP3 or WAV) using Gemini 3.8 Flash TTS."""
        clean_text = prepare_speech_text(text)
        if not clean_text:
            raise ValueError("The digest contains no text to narrate.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting Gemini TTS synthesis (%d chars, voice: '%s', model: '%s')...",
            len(clean_text), self.voice, self.model
        )
        preview = clean_text[:120].replace("\n", " ") + ("..." if len(clean_text) > 120 else "")
        logger.debug("Prepared speech text preview: %s", preview)
        logger.debug("Destination audio path: %s", output_path)

        t_start = time.perf_counter()
        chunks = split_speech_text(clean_text, limit=2000)
        logger.debug("Split text into %d chunk(s) for Gemini TTS", len(chunks))

        audio_arrays = []
        sample_rate = 24000

        try:
            import soundfile as sf
            has_soundfile = True
        except ImportError:
            sf = None
            has_soundfile = False

        try:
            import numpy as np
        except ImportError:
            np = None

        try:
            for idx, chunk in enumerate(chunks, 1):
                logger.debug("Synthesizing chunk %d/%d (%d chars)...", idx, len(chunks), len(chunk))
                raw_wav_bytes = self._synthesize_chunk_raw(chunk)
                if has_soundfile and np is not None:
                    arr, sr = sf.read(io.BytesIO(raw_wav_bytes), dtype="float32")
                    sample_rate = sr
                    audio_arrays.append(arr)
                else:
                    import wave
                    with wave.open(io.BytesIO(raw_wav_bytes), "rb") as wf:
                        sample_rate = wf.getframerate()
                        frames = wf.readframes(wf.getnframes())
                        if np is not None:
                            int16_arr = np.frombuffer(frames, dtype=np.int16)
                            arr = (int16_arr / 32768.0).astype(np.float32)
                            audio_arrays.append(arr)
                        else:
                            audio_arrays.append(frames)

            from ua_tts.audio import save_audio
            if np is not None and len(audio_arrays) > 0 and isinstance(audio_arrays[0], np.ndarray):
                if len(audio_arrays) == 1:
                    full_audio = audio_arrays[0]
                else:
                    full_audio = np.concatenate(audio_arrays)
                save_audio(full_audio, str(output_path), sample_rate=sample_rate)
            else:
                with open(output_path, "wb") as f:
                    for chunk_data in audio_arrays:
                        f.write(chunk_data if isinstance(chunk_data, bytes) else chunk_data.tobytes())

            saved_path = Path(output_path)
            if not saved_path.is_file() or saved_path.stat().st_size == 0:
                raise RuntimeError("Gemini TTS generated an empty audio file.")

            t_elapsed = time.perf_counter() - t_start
            file_kb = saved_path.stat().st_size / 1024
            logger.info(
                "Gemini TTS audio generated successfully: %s (%.1f KB) in %.2fs",
                saved_path, file_kb, t_elapsed
            )
            return saved_path
        except Exception as e:
            logger.error("Gemini TTS synthesis failed: %s", e)
            output_path.unlink(missing_ok=True)
            raise


class UaTtsSynthesizer:
    """Local Ukrainian neural Text-to-Speech synthesizer powered by the ua_tts library."""

    def __init__(
        self,
        voice: str = "dmytro",
        speed: float = 1.0,
        backend: str = "auto",
        device: str = "cpu",
        stress: str = "dictionary",
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        v_str = str(voice).lower().strip()
        matched_voice = None
        for key in UA_TTS_VALID_VOICES:
            if v_str == key or v_str.endswith(f"_{key}") or v_str.endswith(f".{key}"):
                matched_voice = key
                break
        if not matched_voice:
            valid_list = ", ".join(sorted(UA_TTS_VALID_VOICES))
            raise ValueError(
                f"speech.voice must be a valid Ukrainian voice ({valid_list}), got '{voice}'."
            )

        self.voice = matched_voice
        self.speed = max(0.25, min(4.0, float(speed)))
        self.backend = backend
        self.device = device
        self.stress = stress
        self.model_path = model_path
        self.cache_dir = cache_dir
        self._converter = None

        logger.debug(
            "UaTtsSynthesizer initialized: voice=%s, speed=%.2f, backend=%s, device=%s, stress=%s, model_path=%s, cache_dir=%s",
            self.voice, self.speed, self.backend, self.device, self.stress, self.model_path, self.cache_dir
        )

    def _get_converter(self) -> Any:
        if self._converter is None:
            logger.debug("Instantiating UkrainianTTSConverter backend for voice '%s'...", self.voice)
            try:
                from ua_tts import UkrainianTTSConverter
            except ImportError:
                import sys
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                from ua_tts import UkrainianTTSConverter

            self._converter = UkrainianTTSConverter(
                voice=self.voice,
                speed=self.speed,
                backend=self.backend,
                model_path=self.model_path,
                cache_dir=self.cache_dir,
                device=self.device,
                stress=self.stress,
            )
        return self._converter

    def synthesize(self, text: str, output_path: Path) -> Path:
        """Synthesize Ukrainian text into a speech audio file (WAV or MP3)."""
        clean_text = prepare_speech_text(text)
        if not clean_text:
            raise ValueError("The digest contains no text to narrate.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting Ukrainian TTS synthesis (%d chars, voice: '%s', backend: '%s')...",
            len(clean_text), self.voice, self.backend
        )
        preview = clean_text[:120].replace("\n", " ") + ("..." if len(clean_text) > 120 else "")
        logger.debug("Prepared speech text preview: %s", preview)
        logger.debug("Destination audio path: %s", output_path)

        t_start = time.perf_counter()
        converter = self._get_converter()
        try:
            saved_path, _ = converter.convert_text(
                text=clean_text,
                output_file=str(output_path),
            )
            saved_path = Path(saved_path)
            if not saved_path.is_file() or saved_path.stat().st_size == 0:
                raise RuntimeError("ua-tts generated an empty audio file.")

            t_elapsed = time.perf_counter() - t_start
            file_kb = saved_path.stat().st_size / 1024
            logger.info("Ukrainian TTS audio generated successfully: %s (%.1f KB) in %.2fs", saved_path, file_kb, t_elapsed)

            stats = getattr(converter, "last_conversion_stats", {})
            if stats:
                diag = stats.get("diagnostics", {})
                logger.debug(
                    "Synthesis stats: engine=%s, fallback=%s, chunks=%s, sample_rate=%s, audio_duration=%.2fs",
                    stats.get("engine"), stats.get("is_fallback"), stats.get("chunks_count"), stats.get("sample_rate"),
                    diag.get("duration_sec", stats.get("duration_sec", 0.0))
                )

            return saved_path
        except Exception as e:
            logger.error("Ukrainian TTS synthesis failed: %s", e)
            output_path.unlink(missing_ok=True)
            raise


def create_speech_synthesizer(
    config: dict,
    api_key: str = "",
    region: str = "",
    gemini_api_key: str = "",
) -> Union[AzureSpeechSynthesizer, UaTtsSynthesizer, GeminiSpeechSynthesizer]:
    """Factory creating appropriate synthesizer according to config and environment."""
    provider = str(config.get("provider", "")).lower().strip()
    voice = str(config.get("voice", "")).strip()
    is_ua_voice = voice.lower() in UA_TTS_VALID_VOICES
    is_gemini_voice = voice.lower() in GEMINI_TTS_PREBUILT_VOICES or voice.startswith("voice_")

    logger.debug(
        "create_speech_synthesizer: provider='%s', voice='%s', is_ua_voice=%s, is_gemini_voice=%s",
        provider, voice, is_ua_voice, is_gemini_voice
    )

    if provider in ("gemini", "gemini-tts", "google") or is_gemini_voice:
        return GeminiSpeechSynthesizer(
            api_key=gemini_api_key or os.getenv("GEMINI_API_KEY", ""),
            voice=voice or "Kore",
            model=config.get("model", "gemini-3.8-flash-tts"),
            style=config.get("style"),
        )

    if provider in ("ua-tts", "ua_tts", "local") or is_ua_voice:
        return UaTtsSynthesizer(
            voice=voice or "dmytro",
            speed=float(config.get("speed", 1.0)),
            backend=config.get("backend", "auto"),
            device=config.get("device", "cpu"),
            stress=config.get("stress", "dictionary"),
            model_path=config.get("model_path"),
            cache_dir=config.get("cache_dir"),
        )

    # Defaults to Azure if explicitly requested, if voice is Azure-style, or if provider is unset
    return AzureSpeechSynthesizer(
        api_key=api_key,
        region=region,
        voice=voice or "uk-UA-PolinaNeural",
    )
