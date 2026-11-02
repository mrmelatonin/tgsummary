"""Speech narration module supporting local Ukrainian neural TTS (ua-tts) and Azure Speech REST API."""

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
) -> Union[AzureSpeechSynthesizer, UaTtsSynthesizer]:
    """Factory creating appropriate synthesizer according to config and environment."""
    provider = str(config.get("provider", "")).lower().strip()
    voice = config.get("voice", "")
    is_ua_voice = str(voice).lower() in UA_TTS_VALID_VOICES

    logger.debug("create_speech_synthesizer: provider='%s', voice='%s', is_ua_voice=%s", provider, voice, is_ua_voice)

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
