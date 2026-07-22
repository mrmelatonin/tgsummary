import os
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from telethon import TelegramClient
from telethon.sessions import StringSession

logger = logging.getLogger(__name__)

def sanitize_channel_username(channel: str) -> str:
    """Extract clean username from link or handle."""
    channel = channel.strip()
    if channel.startswith("https://t.me/"):
        channel = channel.replace("https://t.me/", "")
    elif channel.startswith("t.me/"):
        channel = channel.replace("t.me/", "")
    if channel.startswith("@"):
        channel = channel[1:]
    return channel

class TelegramFetcher:
    def __init__(self, api_id: int, api_hash: str, session_string: str = ""):
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_string = session_string

    async def fetch_recent_posts(
        self,
        channels: List[str],
        lookback_hours: int = 24,
        max_posts_per_channel: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Connect to Telegram via Telethon and fetch posts from public channels
        published within the specified lookback window.
        """
        session = StringSession(self.session_string) if self.session_string else "anon_session"
        client = TelegramClient(session, self.api_id, self.api_hash)

        all_posts = []
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

        logger.info(f"Connecting to Telegram... Fetching posts since {cutoff_time.isoformat()}")

        await client.start()

        for channel_raw in channels:
            channel_name = sanitize_channel_username(channel_raw)
            if not channel_name:
                continue

            logger.info(f"Fetching posts from channel: @{channel_name}")
            try:
                entity = await client.get_entity(channel_name)
                channel_title = getattr(entity, 'title', channel_name)

                post_count = 0
                async for message in client.iter_messages(entity, limit=max_posts_per_channel):
                    if not message.date or message.date < cutoff_time:
                        # Message is older than cutoff window
                        break

                    # Retrieve text content
                    text = message.text or message.message or ""
                    if not text.strip():
                        continue

                    post_count += 1
                    all_posts.append({
                        "channel_username": channel_name,
                        "channel_title": channel_title,
                        "message_id": message.id,
                        "date": message.date.strftime("%Y-%m-%d %H:%M UTC"),
                        "text": text.strip(),
                        "link": f"https://t.me/{channel_name}/{message.id}"
                    })

                logger.info(f"Retrieved {post_count} posts from @{channel_name}")

            except Exception as e:
                logger.error(f"Error fetching posts from channel @{channel_name}: {e}")

        await client.disconnect()
        return all_posts
