import tempfile
import io
import unittest
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.speech import (
    AzureSpeechSynthesizer,
    UaTtsSynthesizer,
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


if __name__ == "__main__":
    unittest.main()
