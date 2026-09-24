import tempfile
import io
import unittest
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.speech import AzureSpeechSynthesizer, prepare_speech_text, split_speech_text


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


if __name__ == "__main__":
    unittest.main()
