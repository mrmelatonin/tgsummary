import argparse
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import main


class PipelineAudioTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, dry_run=False, enabled=True, text_success=True):
        args = argparse.Namespace(config="config.yaml", channels=None, hours=None,
                                  use_cache=True, use_cached_response=False, dry_run=dry_run, audio_output=None)
        speech = MagicMock()
        publisher = MagicMock()
        publisher.publish_summary = AsyncMock(return_value=text_success)
        publisher.publish_audio = AsyncMock(return_value=True)
        def synthesize(text, path):
            path.write_bytes(b"audio")
            return path
        speech.synthesize.side_effect = synthesize
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake", "TELEGRAM_BOT_TOKEN": "fake"}, clear=True), \
             patch.object(main, "load_dotenv"), \
             patch.object(main, "load_config", return_value={"speech": {"enabled": enabled}, "target_chat": "@same-channel"}), \
             patch.object(main, "save_cached_response"), \
             patch.object(main, "load_cached_posts", return_value=[{"text": "test"}]), \
             patch.object(main, "GeminiSummarizer") as summarizer, \
             patch.object(main, "AzureSpeechSynthesizer", return_value=speech) as factory, \
             patch.object(main, "TelegramPublisher", return_value=publisher):
            summarizer.return_value.summarize_posts.return_value = "Digest"
            if not text_success and not dry_run:
                with self.assertRaises(SystemExit):
                    await main.run_pipeline(args)
            else:
                await main.run_pipeline(args)
        return speech, publisher, factory

    async def test_audio_sent_to_same_channel_and_deleted(self):
        speech, publisher, _ = await self.run_case()
        publisher.publish_summary.assert_awaited_once_with(target_chat="@same-channel", text="Digest")
        audio_args = publisher.publish_audio.call_args.args
        self.assertEqual(audio_args[0], "@same-channel")
        self.assertFalse(Path(audio_args[1]).exists())
        speech.synthesize.assert_called_once()

    async def test_dry_run_skips_azure_and_telegram(self):
        speech, publisher, factory = await self.run_case(dry_run=True)
        factory.assert_not_called()
        speech.synthesize.assert_not_called()
        publisher.publish_summary.assert_not_awaited()
        publisher.publish_audio.assert_not_awaited()

    async def test_disabled_speech_preserves_text_only(self):
        speech, publisher, factory = await self.run_case(enabled=False)
        factory.assert_not_called()
        publisher.publish_summary.assert_awaited_once()
        publisher.publish_audio.assert_not_awaited()

    async def test_failed_text_does_not_generate_audio(self):
        speech, publisher, _ = await self.run_case(text_success=False)
        speech.synthesize.assert_not_called()
        publisher.publish_audio.assert_not_awaited()
