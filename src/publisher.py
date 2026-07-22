import logging
import urllib.request
import urllib.parse
import json
from typing import List
from telethon import TelegramClient
from telethon.sessions import StringSession

logger = logging.getLogger(__name__)

MAX_TELEGRAM_MSG_LEN = 4000  # Safe boundary below 4096 char limit

def split_message(text: str, max_length: int = MAX_TELEGRAM_MSG_LEN) -> List[str]:
    """Splits text into chunks fitting Telegram message length limits."""
    if len(text) <= max_length:
        return [text]

    chunks = []
    current_chunk = ""

    # Split by paragraphs
    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= max_length:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                # Paragraph itself exceeds max_length, split by line
                lines = paragraph.split("\n")
                for line in lines:
                    if len(current_chunk) + len(line) + 1 <= max_length:
                        current_chunk += ("\n" if current_chunk else "") + line
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = line

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

class TelegramPublisher:
    def __init__(
        self,
        bot_token: str = "",
        api_id: int = 0,
        api_hash: str = "",
        session_string: str = ""
    ):
        self.bot_token = bot_token
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_string = session_string

    async def publish_summary(self, target_chat: str, text: str) -> bool:
        """
        Publishes the summary digest to the target Telegram chat/channel.
        Uses Bot API if bot_token is provided, otherwise falls back to Telethon client.
        """
        chunks = split_message(text)
        logger.info(f"Publishing summary ({len(chunks)} message chunk(s)) to {target_chat}...")

        if self.bot_token:
            return self._publish_via_bot(target_chat, chunks)
        elif self.api_id and self.api_hash:
            return await self._publish_via_telethon(target_chat, chunks)
        else:
            raise ValueError("Either TELEGRAM_BOT_TOKEN or TELEGRAM_API_ID/HASH must be provided to publish.")

    def _send_bot_request(self, target_chat: str, text: str, parse_mode: str = "Markdown") -> bool:
        """Helper to send HTTP request to Telegram Bot API."""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": target_chat,
            "text": text,
            "disable_web_page_preview": True
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                if res_data.get("ok"):
                    return True
                logger.warning(f"Bot API error response (parse_mode={parse_mode}): {res_data}")
                return False
        except Exception as e:
            logger.warning(f"Bot API request exception (parse_mode={parse_mode}): {e}")
            return False

    def _publish_via_bot(self, target_chat: str, chunks: List[str]) -> bool:
        """Send message via Telegram Bot API with parse_mode fallback protection."""
        for i, chunk in enumerate(chunks, 1):
            header = f"**Daily Telegram News Summary** ({i}/{len(chunks)})\n\n" if len(chunks) > 1 else ""
            full_text = header + chunk

            # Try sending with Markdown formatting
            success = self._send_bot_request(target_chat, full_text, parse_mode="Markdown")

            # Fallback to plain text if Markdown entity parsing failed
            if not success:
                logger.info(f"Retrying message part {i}/{len(chunks)} as plain text fallback...")
                success = self._send_bot_request(target_chat, full_text, parse_mode=None)

            if not success:
                logger.error(f"Failed to post part {i}/{len(chunks)} via Bot API.")
                return False

            logger.info(f"Successfully posted part {i}/{len(chunks)} via Bot API.")

        return True

    async def _publish_via_telethon(self, target_chat: str, chunks: List[str]) -> bool:
        """Send message via Telethon user client with parse_mode fallback protection."""
        session = StringSession(self.session_string) if self.session_string else "anon_session"
        client = TelegramClient(session, self.api_id, self.api_hash)

        await client.start()
        try:
            for i, chunk in enumerate(chunks, 1):
                header = f"**Daily Telegram News Summary** ({i}/{len(chunks)})\n\n" if len(chunks) > 1 else ""
                full_text = header + chunk
                try:
                    await client.send_message(target_chat, full_text, parse_mode="md")
                except Exception as parse_err:
                    logger.warning(f"Telethon Markdown parsing error ({parse_err}). Falling back to plain text...")
                    await client.send_message(target_chat, full_text, parse_mode=None)

                logger.info(f"Successfully posted part {i}/{len(chunks)} via Telethon.")
            await client.disconnect()
            return True
        except Exception as e:
            logger.error(f"Telethon publishing error: {e}")
            await client.disconnect()
            return False
