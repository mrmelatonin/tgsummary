"""Optional Azure Speech narration using the official REST API."""

import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr


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
    def __init__(self, api_key: str, region: str, voice: str = "uk-UA-PolinaNeural"):
        if not api_key or not region:
            raise ValueError("AZURE_SPEECH_KEY and AZURE_SPEECH_REGION are required when speech is enabled.")
        if not re.fullmatch(r"[a-z0-9]+", region):
            raise ValueError("AZURE_SPEECH_REGION must be a region name, such as westeurope.")
        if not re.fullmatch(r"[a-z]{2,3}-[A-Z]{2}-.+", voice):
            raise ValueError("speech.voice must be an Azure voice name, such as uk-UA-PolinaNeural.")
        self.api_key = api_key
        self.region = region
        self.voice = voice

    def synthesize(self, text: str, output_path: Path) -> Path:
        """Write MP3 frames from bounded requests into a single audio file."""
        chunks = split_speech_text(prepare_speech_text(text))
        if not chunks:
            raise ValueError("The digest contains no text to narrate.")
        output_path = Path(output_path)
        try:
            with output_path.open("wb") as audio:
                for chunk in chunks:
                    audio.write(self._synthesize_chunk(chunk))
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
        return output_path

    def _synthesize_chunk(self, text: str) -> bytes:
        locale = "-".join(self.voice.split("-")[:2])
        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang={quoteattr(locale)}>'
            f'<voice name={quoteattr(self.voice)}>{escape(text)}</voice></speak>'
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
