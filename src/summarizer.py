import time
import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class GeminiSummarizer:
    def __init__(self, api_key: str, model_name: str = "gemini-flash-latest", max_retries: int = 5):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for summarization.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries

    def summarize_posts(
        self,
        posts: List[Dict[str, Any]],
        language: str = "English",
        custom_instruction: str = ""
    ) -> str:
        """
        Synthesizes multiple channel posts into a single topic-grouped summary digest using Gemini.
        Includes automatic retry logic with exponential backoff for high demand / transient API errors.
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

        retry_delays = [5, 10, 20, 40, 60]  # Delays in seconds for retries 1 through 5

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Calling Gemini API (model: {self.model_name}, attempt {attempt}/{self.max_retries})...")
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                    )
                )

                summary_text = response.text.strip()
                logger.info("Successfully generated summary digest from Gemini.")
                return summary_text

            except Exception as e:
                err_msg = str(e)
                logger.warning(f"Gemini API attempt {attempt}/{self.max_retries} failed: {err_msg}")

                if attempt < self.max_retries:
                    delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
                    logger.info(f"Retrying Gemini API call in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} attempts to call Gemini API failed.")
                    raise RuntimeError(f"Failed to generate summary with Gemini after {self.max_retries} retries: {e}")
