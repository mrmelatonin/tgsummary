import logging
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types, errors

logger = logging.getLogger(__name__)

# Default candidate models in fallback order
DEFAULT_MODELS = [
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
]
MODELS = DEFAULT_MODELS


def generate(
    prompt: str,
    client: Optional[genai.Client] = None,
    models: Optional[List[str]] = None,
    config: Optional[Any] = None,
) -> Any:
    """
    Generate content with automatic model fallback on 429 (rate limit / quota) or 503 (busy / high demand).
    """
    target_models = models or MODELS
    if client is None:
        client = genai.Client()

    last_error = None
    for m in target_models:
        try:
            logger.info(f"Calling Gemini API (model: {m})...")
            kwargs: Dict[str, Any] = {"model": m, "contents": prompt}
            if config is not None:
                kwargs["config"] = config
            response = client.models.generate_content(**kwargs)
            logger.info(f"Successfully generated response from Gemini using model '{m}'.")
            return response
        except errors.APIError as e:
            if getattr(e, "code", None) in (429, 503):
                last_error = e
                logger.warning(
                    f"Gemini model '{m}' returned APIError {e.code} ({getattr(e, 'message', str(e))}). "
                    "Falling back to next model..."
                )
                continue
            raise
        except Exception as e:
            code = getattr(e, "code", None) or getattr(e, "status_code", None)
            if code is None and hasattr(e, "response") and hasattr(e.response, "status_code"):
                code = e.response.status_code

            err_msg = str(e).lower()
            is_busy = (
                code in (429, 503)
                or any(term in err_msg for term in ("429", "503", "quota", "resource has been exhausted", "rate limit"))
            )
            if is_busy:
                last_error = e
                logger.warning(f"Gemini model '{m}' returned busy error ({e}). Falling back to next model...")
                continue
            raise

    raise RuntimeError("All models busy") from last_error


class GeminiSummarizer:
    def __init__(
        self,
        api_key: str,
        model_name: Optional[str] = None,
        models: Optional[List[str]] = None,
        fallback_models: Optional[List[str]] = None,
        max_retries: int = 5,
    ):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for summarization.")
        http_options = {
            "retry_options": {
                "attempts": 1,
                "max_delay": 5.0,
            },
            "timeout": 120000,
        }
        self.client = genai.Client(api_key=api_key, http_options=http_options)

        if models:
            self.models = list(models)
        elif model_name:
            fallback = list(fallback_models) if fallback_models is not None else list(DEFAULT_MODELS)
            self.models = [model_name] + [m for m in fallback if m != model_name]
        elif fallback_models:
            self.models = list(fallback_models)
        else:
            self.models = list(DEFAULT_MODELS)

        self.model_name = self.models[0] if self.models else "gemini-flash-latest"
        self.max_retries = max_retries

    def generate(self, prompt: str) -> Any:
        """
        Generate content using self.client with automatic fallback across self.models.
        """
        return generate(
            prompt=prompt,
            client=self.client,
            models=self.models,
            config=types.GenerateContentConfig(
                temperature=0.3,
            ),
        )

    def summarize_posts(
        self,
        posts: List[Dict[str, Any]],
        language: str = "English",
        custom_instruction: str = ""
    ) -> str:
        """
        Synthesizes multiple channel posts into a single topic-grouped summary digest using Gemini.
        Includes automatic fallback across candidate models on rate-limit (429) or service unavailable (503).
        """
        if not posts:
            return "No new posts were found in the monitored channels for the selected period."

        logger.info(f"Preparing {len(posts)} posts for Gemini topic summarization...")

        # Format input posts into structured text block
        formatted_posts = []
        for idx, post in enumerate(posts, 1):
            post_block = (
                f"[{idx}] Channel: {post['channel_title']} (@{post['channel_username']})\n"
                f"Date: {post['date']}\n"
                f"Link: {post['link']}\n"
                f"Content:\n{post['text']}\n"
            )
            formatted_posts.append(post_block)

        posts_payload = "\n---\n".join(formatted_posts)
        custom_inst_text = f"CUSTOM INSTRUCTIONS:\n{custom_instruction}" if custom_instruction else ""

        prompt = f"""
You are an expert news editor and intelligence analyst.
TASK:
1. Synthesize all posts into a clean, minimal daily summary digest.
2. Group the news by major KEY TOPICS / THEMES.
3. For each topic header, use a relevant emoji icon followed by a bold topic title (e.g., 📌 **Головні новини** or 💻 **Технології**).
4. Under each topic, provide a clean, non-bold bullet list (`- text`) of the key points.
5. Do NOT include sources, channel names, links, or channel tags inside the bullet points.
6. Do NOT use excessive formatting inside the bullet text (keep the bullet points normal, unformatted text).
7. Write the entire summary in {language}.

{custom_inst_text}

POSTS TO SUMMARIZE:
---
{posts_payload}
---
""".strip()

        response = self.generate(prompt)
        summary_text = response.text.strip() if response and hasattr(response, "text") and response.text else ""
        if not summary_text:
            raise RuntimeError("Gemini returned an empty summary.")
        logger.info("Successfully generated summary digest from Gemini.")
        return summary_text
