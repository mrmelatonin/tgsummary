import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.publisher import TelegramPublisher


class AudioPublisherTests(unittest.IsolatedAsyncioTestCase):
    @patch("src.publisher.urllib.request.urlopen")
    async def test_bot_upload(self, open_url):
        response = MagicMock()
        response.read.return_value = b'{"ok":true}'
        open_url.return_value.__enter__.return_value = response
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digest.mp3"
            path.write_bytes(b"audio-bytes")
            self.assertTrue(await TelegramPublisher(bot_token="test-token").publish_audio("@destination", path))
        request = open_url.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/sendAudio"))
        self.assertIn(b"@destination", request.data)
        self.assertIn(b'name="audio"', request.data)
        self.assertIn(b"audio-bytes", request.data)

    @patch("src.publisher.TelegramClient")
    async def test_telethon_disconnects_on_failure(self, client_class):
        client = AsyncMock()
        client.send_file.side_effect = RuntimeError("upload failed")
        client_class.return_value = client
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digest.mp3"
            path.write_bytes(b"audio-bytes")
            self.assertFalse(await TelegramPublisher(api_id=123, api_hash="test").publish_audio("@destination", path))
        self.assertEqual(client.send_file.call_args.args[0], "@destination")
        client.disconnect.assert_awaited_once()

    async def test_missing_audio_is_not_uploaded(self):
        self.assertFalse(await TelegramPublisher(bot_token="test").publish_audio("@destination", Path("nonexistent.mp3")))
