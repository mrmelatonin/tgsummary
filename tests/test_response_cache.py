import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
from src.cache_manager import (
    RESPONSE_CACHE_FILE_NAME, load_cached_response, save_cached_response,
)


class ResponseCacheTests(unittest.TestCase):
    def test_round_trip_and_invalid_response_preserves_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(save_cached_response("Digest: Україна", directory))
            self.assertEqual(load_cached_response(directory), "Digest: Україна")
            self.assertEqual(save_cached_response("  ", directory), "")
            self.assertEqual(load_cached_response(directory), "Digest: Україна")

    def test_missing_corrupt_and_invalid_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_cached_response(directory), "")
            for contents in ('{', '[]', '{"summary_text": 4}', '{"summary_text": " "}'):
                Path(directory, RESPONSE_CACHE_FILE_NAME).write_text(contents)
                self.assertEqual(load_cached_response(directory), "")


class CachedResponsePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def run_cached(self, response):
        args = argparse.Namespace(config="config.yaml", channels=None, hours=None,
                                  use_cache=False, use_cached_response=True,
                                  dry_run=True, audio_output=None)
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(main, "load_dotenv"), \
             patch.object(main, "load_config", return_value={}), \
             patch.object(main, "load_cached_response", return_value=response), \
             patch.object(main, "TelegramFetcher") as fetcher, \
             patch.object(main, "GeminiSummarizer") as summarizer, \
             patch.object(main, "TelegramPublisher") as publisher, \
             patch("builtins.print") as output:
            if response:
                await main.run_pipeline(args)
                output.assert_called_once_with("\n" + response + "\n")
            else:
                with self.assertRaises(SystemExit) as error:
                    await main.run_pipeline(args)
                self.assertEqual(error.exception.code, 1)
            fetcher.assert_not_called()
            summarizer.assert_not_called()
            publisher.assert_not_called()

    async def test_replay_without_credentials_or_channels(self):
        await self.run_cached("Cached digest")

    async def test_missing_cache_never_falls_back_to_api(self):
        await self.run_cached("")
