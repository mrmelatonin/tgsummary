import unittest
from unittest.mock import MagicMock, patch
from google.genai import errors
from src.summarizer import GeminiSummarizer, generate, DEFAULT_MODELS, MODELS


class GeminiSummarizerTests(unittest.TestCase):
    def test_missing_api_key_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY is required"):
            GeminiSummarizer(api_key="")

    def test_empty_posts_returns_default_message(self):
        summarizer = GeminiSummarizer(api_key="fake-key")
        result = summarizer.summarize_posts([])
        self.assertIn("No new posts were found", result)

    def test_models_ordering_and_precedence(self):
        # Default models
        s1 = GeminiSummarizer(api_key="fake-key")
        self.assertEqual(s1.models, DEFAULT_MODELS)
        self.assertEqual(s1.model_name, "gemini-flash-latest")

        # Custom primary model prepended to default fallbacks
        s2 = GeminiSummarizer(api_key="fake-key", model_name="gemini-3.1-flash-lite")
        self.assertEqual(s2.models[0], "gemini-3.1-flash-lite")
        self.assertIn("gemini-flash-latest", s2.models)
        self.assertEqual(s2.model_name, "gemini-3.1-flash-lite")

        # Explicit models list overrides completely
        s3 = GeminiSummarizer(api_key="fake-key", models=["custom-1", "custom-2"])
        self.assertEqual(s3.models, ["custom-1", "custom-2"])

        # Custom primary model with custom fallback models
        s4 = GeminiSummarizer(
            api_key="fake-key",
            model_name="primary",
            fallback_models=["fb-1", "fb-2"]
        )
        self.assertEqual(s4.models, ["primary", "fb-1", "fb-2"])

    def test_http_429_fallback_to_second_model_succeeds(self):
        summarizer = GeminiSummarizer(
            api_key="fake-key",
            models=["model-a", "model-b"]
        )
        mock_client = MagicMock()
        err_429 = errors.APIError(429, {"error": {"message": "Resource has been exhausted (quota exceeded)"}})
        mock_success = MagicMock()
        mock_success.text = "Digest summary"

        mock_client.models.generate_content.side_effect = [err_429, mock_success]
        summarizer.client = mock_client

        posts = [{"channel_title": "News", "channel_username": "news", "date": "today", "link": "http", "text": "Post"}]
        result = summarizer.summarize_posts(posts)

        self.assertEqual(result, "Digest summary")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        call_models = [call.kwargs.get("model") for call in mock_client.models.generate_content.call_args_list]
        self.assertEqual(call_models, ["model-a", "model-b"])

    def test_http_503_fallback_to_second_model_succeeds(self):
        summarizer = GeminiSummarizer(
            api_key="fake-key",
            models=["model-a", "model-b"]
        )
        mock_client = MagicMock()
        err_503 = errors.APIError(503, {"error": {"message": "The service is temporarily unavailable"}})
        mock_success = MagicMock()
        mock_success.text = "Digest 503 fallback"

        mock_client.models.generate_content.side_effect = [err_503, mock_success]
        summarizer.client = mock_client

        posts = [{"channel_title": "News", "channel_username": "news", "date": "today", "link": "http", "text": "Post"}]
        result = summarizer.summarize_posts(posts)

        self.assertEqual(result, "Digest 503 fallback")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

    def test_all_models_busy_raises_runtime_error(self):
        summarizer = GeminiSummarizer(
            api_key="fake-key",
            models=["model-1", "model-2"]
        )
        mock_client = MagicMock()
        err_429 = errors.APIError(429, {"error": {"message": "rate limit"}})
        err_503 = errors.APIError(503, {"error": {"message": "busy"}})
        mock_client.models.generate_content.side_effect = [err_429, err_503]
        summarizer.client = mock_client

        posts = [{"channel_title": "News", "channel_username": "news", "date": "today", "link": "http", "text": "Post"}]
        with self.assertRaisesRegex(RuntimeError, "All models busy"):
            summarizer.summarize_posts(posts)
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

    def test_non_retryable_error_aborts_immediately_without_fallback(self):
        summarizer = GeminiSummarizer(
            api_key="fake-key",
            models=["model-a", "model-b"]
        )
        mock_client = MagicMock()
        err_400 = errors.APIError(400, {"error": {"message": "Invalid argument"}})
        mock_client.models.generate_content.side_effect = err_400
        summarizer.client = mock_client

        posts = [{"channel_title": "News", "channel_username": "news", "date": "today", "link": "http", "text": "Post"}]
        with self.assertRaises(errors.APIError):
            summarizer.summarize_posts(posts)
        # Should NOT try model-b on 400 error
        self.assertEqual(mock_client.models.generate_content.call_count, 1)

    def test_standalone_generate_function(self):
        mock_client = MagicMock()
        err_429 = errors.APIError(429, {"error": {"message": "rate limit"}})
        mock_resp = MagicMock()
        mock_resp.text = "Generated content"
        mock_client.models.generate_content.side_effect = [err_429, mock_resp]

        resp = generate("Hello", client=mock_client, models=["m1", "m2"])
        self.assertEqual(resp.text, "Generated content")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

    def test_standalone_generate_all_busy(self):
        mock_client = MagicMock()
        err = errors.APIError(429, {"error": {"message": "rate limit"}})
        mock_client.models.generate_content.side_effect = err

        with self.assertRaisesRegex(RuntimeError, "All models busy"):
            generate("Hello", client=mock_client, models=["m1"])
        self.assertEqual(mock_client.models.generate_content.call_count, 1)


if __name__ == "__main__":
    unittest.main()
