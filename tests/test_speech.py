import tempfile
import io
import unittest
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import base64
import os
from src.speech import (
    AzureSpeechSynthesizer,
    UaTtsSynthesizer,
    GeminiSpeechSynthesizer,
    create_speech_synthesizer,
    prepare_speech_text,
    split_speech_text,
)


class SpeechTests(unittest.TestCase):
    def test_cleanup_preserves_ukrainian(self):
        self.assertEqual(prepare_speech_text("📌 **Новини**\n- [Текст](https://example.com) & факти"), "Новини\nТекст & факти")

    def test_long_input_preserved_and_bounded(self):
        text = "Новини. " * 1200 + "я" * 6000
        chunks = split_speech_text(text)
        self.assertTrue(all(0 < len(chunk) <= 2500 for chunk in chunks))
        self.assertEqual("".join("".join(chunks).split()), "".join(text.split()))

    @patch("src.speech.urllib.request.urlopen")
    def test_request_escapes_xml_and_writes_audio(self, open_url):
        response = MagicMock()
        response.read.return_value = b"mp3-frames"
        response.headers = {"Content-Type": "audio/mpeg"}
        open_url.return_value.__enter__.return_value = response
        speech = AzureSpeechSynthesizer("test-key", "westeurope")
        with tempfile.TemporaryDirectory() as directory:
            path = speech.synthesize("Новини & <факти>", Path(directory) / "digest.mp3")
            self.assertEqual(path.read_bytes(), b"mp3-frames")
        request = open_url.call_args.args[0]
        root = ET.fromstring(request.data)
        self.assertEqual(list(root)[0].text, "Новини & <факти>")
        self.assertEqual(request.get_header("Ocp-apim-subscription-key"), "test-key")

    @patch("src.speech.time.sleep")
    @patch("src.speech.urllib.request.urlopen")
    def test_quota_failure_is_bounded_and_removes_partial_file(self, open_url, sleep):
        open_url.side_effect = [
            urllib.error.HTTPError("https://example.com", 429, "quota", {}, io.BytesIO())
            for _ in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digest.mp3"
            with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
                AzureSpeechSynthesizer("key", "westeurope").synthesize("Text", path)
            self.assertFalse(path.exists())
        self.assertEqual(open_url.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_missing_credentials(self):
        with self.assertRaises(ValueError):
            AzureSpeechSynthesizer("", "")


class UaTtsSynthesizerTests(unittest.TestCase):
    def test_valid_voice_initialization(self):
        synth = UaTtsSynthesizer(voice="dmytro", speed=1.2)
        self.assertEqual(synth.voice, "dmytro")
        self.assertEqual(synth.speed, 1.2)

        synth_tetiana = UaTtsSynthesizer(voice="tetiana")
        self.assertEqual(synth_tetiana.voice, "tetiana")

    def test_speed_clamping(self):
        synth_slow = UaTtsSynthesizer(voice="dmytro", speed=0.1)
        self.assertEqual(synth_slow.speed, 0.25)

        synth_fast = UaTtsSynthesizer(voice="dmytro", speed=10.0)
        self.assertEqual(synth_fast.speed, 4.0)

    def test_invalid_voice_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "speech.voice must be a valid Ukrainian voice"):
            UaTtsSynthesizer(voice="unsupported_voice")

    def test_empty_text_raises_value_error(self):
        synth = UaTtsSynthesizer(voice="dmytro")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "empty.mp3"
            with self.assertRaisesRegex(ValueError, "The digest contains no text to narrate"):
                synth.synthesize("   ", out_path)
            with self.assertRaisesRegex(ValueError, "The digest contains no text to narrate"):
                synth.synthesize("https://example.com 📌", out_path)

    def test_successful_synthesis_with_mocked_converter(self):
        synth = UaTtsSynthesizer(voice="dmytro")
        mock_converter = MagicMock()
        def fake_convert(text, output_file):
            p = Path(output_file)
            p.write_bytes(b"mocked-audio-bytes")
            return str(p), text
        mock_converter.convert_text.side_effect = fake_convert
        synth._converter = mock_converter

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "output.mp3"
            res = synth.synthesize("📌 **Головне**: [Новина](https://site.com) про Україну", out_file)
            self.assertTrue(res.is_file())
            self.assertEqual(res.read_bytes(), b"mocked-audio-bytes")
            mock_converter.convert_text.assert_called_once_with(
                text="Головне: Новина про Україну",
                output_file=str(out_file),
            )

    def test_converter_failure_cleans_up_file(self):
        synth = UaTtsSynthesizer(voice="dmytro")
        mock_converter = MagicMock()
        def failing_convert(text, output_file):
            Path(output_file).write_bytes(b"partial-data")
            raise RuntimeError("Synthesis failed")
        mock_converter.convert_text.side_effect = failing_convert
        synth._converter = mock_converter

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "failed.mp3"
            with self.assertRaisesRegex(RuntimeError, "Synthesis failed"):
                synth.synthesize("Тестовий текст", out_file)
            self.assertFalse(out_file.exists())



def make_mock_wav_b64(sample_rate: int = 24000, duration_sec: float = 0.1) -> str:
    import wave
    buf = io.BytesIO()
    samples = int(sample_rate * duration_sec)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * samples)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class GeminiSpeechSynthesizerTests(unittest.TestCase):
    def test_voice_normalization(self):
        synth = GeminiSpeechSynthesizer(api_key="fake-key", voice="kore")
        self.assertEqual(synth.voice, "Kore")

        synth_puck = GeminiSpeechSynthesizer(api_key="fake-key", voice="puck")
        self.assertEqual(synth_puck.voice, "Puck")

        synth_custom = GeminiSpeechSynthesizer(api_key="fake-key", voice="voice_custom_123")
        self.assertEqual(synth_custom.voice, "voice_custom_123")

        synth_default = GeminiSpeechSynthesizer(api_key="fake-key", voice="")
        self.assertEqual(synth_default.voice, "Kore")

    def test_missing_api_key_raises_value_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY is required"):
                GeminiSpeechSynthesizer(api_key="")

    def test_empty_text_raises_value_error(self):
        synth = GeminiSpeechSynthesizer(api_key="fake-key")
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "empty.mp3"
            with self.assertRaisesRegex(ValueError, "The digest contains no text to narrate"):
                synth.synthesize("    ", out_file)
            with self.assertRaisesRegex(ValueError, "The digest contains no text to narrate"):
                synth.synthesize("https://example.com 🚀", out_file)

    def test_successful_synthesis_with_mocked_interaction(self):
        synth = GeminiSpeechSynthesizer(api_key="fake-key", voice="Kore", style="news")
        mock_client = MagicMock()
        mock_interaction = MagicMock()
        mock_interaction.output_audio.data = make_mock_wav_b64()
        mock_client.interactions.create.return_value = mock_interaction
        synth._client = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "output.mp3"
            res = synth.synthesize("📌 **Головне**: Новина дня.", out_file)
            self.assertTrue(res.is_file())
            self.assertGreater(res.stat().st_size, 0)
            mock_client.interactions.create.assert_called_once()
            call_kwargs = mock_client.interactions.create.call_args.kwargs
            self.assertEqual(call_kwargs["model"], "gemini-3.8-flash-tts")
            self.assertEqual(call_kwargs["generation_config"]["speech_config"][0]["voice"], "Kore")

    def test_chunking_and_concatenation_with_multiple_chunks(self):
        synth = GeminiSpeechSynthesizer(api_key="fake-key", voice="Puck")
        mock_client = MagicMock()
        mock_interaction = MagicMock()
        mock_interaction.output_audio.data = make_mock_wav_b64()
        mock_client.interactions.create.return_value = mock_interaction
        synth._client = mock_client

        # Text with enough sentences to split into multiple chunks
        long_text = ". ".join([f"Речення {i} про важливі події" for i in range(150)])
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "long_digest.mp3"
            res = synth.synthesize(long_text, out_file)
            self.assertTrue(res.is_file())
            self.assertGreater(mock_client.interactions.create.call_count, 1)

    def test_api_failure_cleans_up_file(self):
        synth = GeminiSpeechSynthesizer(api_key="fake-key", max_retries=1)
        mock_client = MagicMock()
        mock_client.interactions.create.side_effect = RuntimeError("API service down")
        synth._client = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "failed.mp3"
            with self.assertRaisesRegex(RuntimeError, "Gemini TTS synthesis failed"):
                synth.synthesize("Текст для озвучення", out_file)
            self.assertFalse(out_file.exists())

    def test_http_429_quota_exhaustion_aborts_immediately(self):
        synth = GeminiSpeechSynthesizer(api_key="fake-key", max_retries=3)
        mock_client = MagicMock()
        err = RuntimeError("429 Resource has been exhausted (quota)")
        err.status_code = 429
        mock_resp = MagicMock()
        mock_resp.headers = {"retry-after": "85427"}
        err.response = mock_resp
        mock_client.interactions.create.side_effect = err
        synth._client = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "quota.mp3"
            with self.assertRaisesRegex(RuntimeError, "quota exceeded"):
                synth.synthesize("Текст для озвучення", out_file)
            # Must abort on the very first attempt without looping or sleeping
            self.assertEqual(mock_client.interactions.create.call_count, 1)

    def test_http_429_small_retry_after_retries_and_succeeds(self):
        synth = GeminiSpeechSynthesizer(api_key="fake-key", max_retries=3)
        mock_client = MagicMock()
        err = RuntimeError("429 rate limit")
        err.status_code = 429
        mock_resp = MagicMock()
        mock_resp.headers = {"retry-after": "0.01"}
        err.response = mock_resp

        mock_interaction = MagicMock()
        mock_interaction.output_audio.data = make_mock_wav_b64()
        mock_client.interactions.create.side_effect = [err, mock_interaction]
        synth._client = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "retry_ok.mp3"
            res = synth.synthesize("Текст для озвучення", out_file)
            self.assertTrue(res.is_file())
            self.assertEqual(mock_client.interactions.create.call_count, 2)

    def test_get_client_disables_interactions_retry_config(self):
        synth = GeminiSpeechSynthesizer(api_key="fake-key")
        with patch("google.genai.Client") as mock_genai_cls:
            mock_inst = MagicMock()
            mock_genai_cls.return_value = mock_inst
            client = synth._get_client()
            self.assertIs(client, mock_inst)
            self.assertIsNone(mock_inst.interactions.sdk_configuration.retry_config)

class SpeechFactoryTests(unittest.TestCase):
    def test_create_ua_tts_by_provider(self):
        synth = create_speech_synthesizer({"provider": "ua-tts", "voice": "dmytro"})
        self.assertIsInstance(synth, UaTtsSynthesizer)
        self.assertEqual(synth.voice, "dmytro")

    def test_create_ua_tts_by_voice_name(self):
        synth = create_speech_synthesizer({"voice": "tetiana"})
        self.assertIsInstance(synth, UaTtsSynthesizer)
        self.assertEqual(synth.voice, "tetiana")

    def test_create_azure_by_provider(self):
        synth = create_speech_synthesizer(
            {"provider": "azure", "voice": "uk-UA-OstapNeural"},
            api_key="key",
            region="westeurope",
        )
        self.assertIsInstance(synth, AzureSpeechSynthesizer)
        self.assertEqual(synth.voice, "uk-UA-OstapNeural")

    def test_create_azure_by_voice_name(self):
        synth = create_speech_synthesizer(
            {"voice": "uk-UA-PolinaNeural"},
            api_key="key",
            region="westeurope",
        )
        self.assertIsInstance(synth, AzureSpeechSynthesizer)

    def test_create_gemini_by_provider(self):
        synth = create_speech_synthesizer(
            {"provider": "gemini", "voice": "Kore"},
            gemini_api_key="fake-key",
        )
        self.assertIsInstance(synth, GeminiSpeechSynthesizer)
        self.assertEqual(synth.voice, "Kore")

    def test_create_gemini_by_voice_name(self):
        synth = create_speech_synthesizer(
            {"voice": "puck"},
            gemini_api_key="fake-key",
        )
        self.assertIsInstance(synth, GeminiSpeechSynthesizer)
        self.assertEqual(synth.voice, "Puck")



if __name__ == "__main__":
    unittest.main()
