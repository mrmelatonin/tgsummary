import unittest
from unittest.mock import MagicMock, patch
from src.summarizer import GeminiSummarizer

class GeminiSummarizerTests(unittest.TestCase):
    def test_missing_api_key_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY is required"):
            GeminiSummarizer(api_key="")

    def test_empty_posts_returns_default_message(self):
        summarizer = GeminiSummarizer(api_key="fake-key")
        result = summarizer.summarize_posts([])
        self.assertIn("No new posts were found", result)

    def test_http_429_quota_exhaustion_aborts_immediately(self):
        summarizer = GeminiSummarizer(api_key="fake-key", max_retries=5)
        mock_client = MagicMock()
        err = RuntimeError("429 Resource has been exhausted (quota exceeded)")
        err.code = 429
        mock_resp = MagicMock()
        mock_resp.headers = {"retry-after": "85427"}
        err.response = mock_resp
        mock_client.models.generate_content.side_effect = err
        summarizer.client = mock_client

        posts = [{"channel_title": "Test", "channel_username": "test", "date": "today", "link": "http", "text": "Hello"}]
        with self.assertRaisesRegex(RuntimeError, "quota exceeded"):
            summarizer.summarize_posts(posts)
        # Should abort on attempt 1 without repeating 5 times
        self.assertEqual(mock_client.models.generate_content.call_count, 1)

    def test_http_429_small_retry_after_retries_and_succeeds(self):
        summarizer = GeminiSummarizer(api_key="fake-key", max_retries=3)
        mock_client = MagicMock()
        err = RuntimeError("429 rate limit")
        err.code = 429
        mock_resp = MagicMock()
        mock_resp.headers = {"retry-after": "0.01"}
        err.response = mock_resp

        mock_success = MagicMock()
        mock_success.text = "Summary result"
        mock_client.models.generate_content.side_effect = [err, mock_success]
        summarizer.client = mock_client

        posts = [{"channel_title": "Test", "channel_username": "test", "date": "today", "link": "http", "text": "Hello"}]
        res = summarizer.summarize_posts(posts)
        self.assertEqual(res, "Summary result")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

if __name__ == "__main__":
    unittest.main()

