import os
import sys
import argparse
import asyncio
import logging
import tempfile
from pathlib import Path
import yaml
from dotenv import load_dotenv

from src.telegram_fetcher import TelegramFetcher
from src.summarizer import GeminiSummarizer
from src.publisher import TelegramPublisher
from src.speech import AzureSpeechSynthesizer, UaTtsSynthesizer, GeminiSpeechSynthesizer, create_speech_synthesizer
from src.cache_manager import save_cached_posts, load_cached_posts, save_cached_response, load_cached_response

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configure logging
initial_log_level = (
    logging.DEBUG
    if os.getenv("DEBUG", "").lower() in ("1", "true", "yes")
    or os.getenv("LOG_LEVEL", "").upper() == "DEBUG"
    else logging.INFO
)
logging.basicConfig(
    level=initial_log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("main")

def load_config(config_path: str = "config.yaml") -> dict:
    """Loads configuration from YAML file."""
    if not os.path.exists(config_path):
        logger.warning(f"Config file '{config_path}' not found. Using default settings.")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def generate_string_session():
    """Interactive helper to generate a Telethon StringSession for headless runs."""
    from telethon.sync import TelegramClient
    from telethon.sessions import StringSession

    load_dotenv()
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")

    if not api_id or not api_hash:
        print("Error: TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables must be set.")
        sys.exit(1)

    print("\n--- Telethon String Session Generator ---")
    print("Log in with your Telegram account to generate a session string for GitHub Actions.\n")

    with TelegramClient(StringSession(), int(api_id), api_hash) as client:
        session_str = client.session.save()
        print("\nSUCCESS! Your TELEGRAM_SESSION_STRING is:\n")
        print(session_str)
        print("\nSave this string in your .env file or GitHub Repository Secrets as TELEGRAM_SESSION_STRING.\n")

async def run_pipeline(args):
    """Main execution pipeline."""
    if getattr(args, "debug", False):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        for handler in root_logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logging.getLogger("src.speech").setLevel(logging.DEBUG)
        logging.getLogger("ua_tts").setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled via command-line flag.")

    # Load environment variables (.env)
    load_dotenv()

    # Load YAML config
    config = load_config(args.config)

    # Resolve settings
    channels = args.channels.split(",") if args.channels else config.get("channels", [])
    lookback_hours = args.hours or config.get("lookback_hours", 24)
    max_posts = config.get("max_posts_per_channel", 25)
    target_chat = os.getenv("TARGET_CHAT") or config.get("target_chat", "@my_summary_channel")

    summarizer_cfg = config.get("summarizer", {})
    model_name = summarizer_cfg.get("model_name", "gemini-flash-latest")
    language = summarizer_cfg.get("language", "English")
    custom_instruction = summarizer_cfg.get("custom_instruction", "")

    # Environment secrets
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    session_string = os.getenv("TELEGRAM_SESSION_STRING", "")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    azure_speech_key = os.getenv("AZURE_SPEECH_KEY", "")
    azure_speech_region = os.getenv("AZURE_SPEECH_REGION", "")

    speech_cfg = config.get("speech", {})
    speech = None
    if speech_cfg.get("enabled", False) and (not args.dry_run or args.audio_output):
        provider = str(speech_cfg.get("provider", "")).lower().strip()
        voice = str(speech_cfg.get("voice", "")).strip()
        is_ua_voice = voice.lower() in ("tetiana", "dmytro", "mykyta", "lada", "oleksa", "olena", "olga")
        is_gemini_voice = voice.lower() in ("kore", "puck", "charon", "fenrir", "aoede") or voice.startswith("voice_")

        if provider in ("gemini", "gemini-tts", "google") or is_gemini_voice:
            logger.info("Initializing Gemini TTS synthesizer (voice: '%s')...", voice or "Kore")
            speech = GeminiSpeechSynthesizer(
                api_key=gemini_api_key,
                voice=voice or "Kore",
                model=speech_cfg.get("model", "gemini-3.8-flash-tts"),
                style=speech_cfg.get("style"),
            )
        elif provider in ("ua-tts", "ua_tts", "local") or is_ua_voice:
            logger.info("Initializing ua-tts synthesizer (voice: '%s')...", voice or "dmytro")
            speech = UaTtsSynthesizer(
                voice=voice or "dmytro",
                speed=float(speech_cfg.get("speed", 1.0)),
                backend=speech_cfg.get("backend", "auto"),
                device=speech_cfg.get("device", "cpu"),
                stress=speech_cfg.get("stress", "dictionary"),
                model_path=speech_cfg.get("model_path"),
                cache_dir=speech_cfg.get("cache_dir"),
            )
        else:
            logger.info("Initializing Azure speech synthesizer (voice: '%s')...", voice or "uk-UA-PolinaNeural")
            speech = AzureSpeechSynthesizer(
                api_key=azure_speech_key,
                region=azure_speech_region,
                voice=voice or "uk-UA-PolinaNeural",
            )
    if args.audio_output and (not args.dry_run or not speech_cfg.get("enabled", False)):
        raise ValueError("--audio-output requires --dry-run and speech.enabled: true.")

    if args.use_cached_response:
        summary_text = load_cached_response()
        if not summary_text:
            logger.error("No valid cached Gemini response. Run without --use-cached-response to generate one.")
            sys.exit(1)
    else:
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY is required.")
            sys.exit(1)

        posts = []

        # Check if user requested using local cache
        if args.use_cache:
            logger.info("Using local cached posts as requested (--use-cache)...")
            posts = load_cached_posts()
            if not posts:
                logger.error("No valid local cache found. Falling back to fetching from Telegram.")

        # Fetch from Telegram if not loading from cache or cache was empty
        if not posts:
            if not channels:
                logger.error("No channels configured! Add channels to config.yaml or pass --channels.")
                sys.exit(1)

            if not api_id or not api_hash:
                logger.error("TELEGRAM_API_ID and TELEGRAM_API_HASH are required to fetch posts.")
                sys.exit(1)

            logger.info(f"Starting news aggregation for {len(channels)} channels (last {lookback_hours} hours)...")

            fetcher = TelegramFetcher(
                api_id=int(api_id),
                api_hash=api_hash,
                session_string=session_string
            )

            posts = await fetcher.fetch_recent_posts(
                channels=channels,
                lookback_hours=lookback_hours,
                max_posts_per_channel=max_posts
            )

            if posts:
                # Cache the freshly fetched posts locally
                save_cached_posts(posts)

        if not posts:
            logger.info("No posts found or retrieved. Exiting.")
            return

        logger.info(f"Loaded {len(posts)} total post(s). Proceeding to topic summarization...")

        # Step 2: Summarize with Gemini (includes 5 automatic retries with exponential backoff)
        summarizer = GeminiSummarizer(api_key=gemini_api_key, model_name=model_name, max_retries=5)
    
        try:
            summary_text = summarizer.summarize_posts(
                posts=posts,
                language=language,
                custom_instruction=custom_instruction
            )
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            logger.info("Note: Posts remain safely saved in local cache ('cache/fetched_posts_cache.json').")
            logger.info("You can retry summarization without refetching from Telegram by running: python main.py --use-cache")
            sys.exit(1)

        save_cached_response(summary_text)

    # Step 3: Output or Publish
    if args.dry_run:
        logger.info("=== DRY RUN MODE: Generated Summary Digest ===")
        print("\n" + summary_text + "\n")
        if speech and args.audio_output:
            logger.info("Synthesizing speech preview to %s...", args.audio_output)
            output = await asyncio.to_thread(speech.synthesize, summary_text, Path(args.audio_output))
            logger.info("Saved speech preview to %s", output)
        logger.info("Dry run completed. Summary was not sent to Telegram.")
    else:
        logger.info(f"Publishing summary digest to {target_chat}...")
        publisher = TelegramPublisher(
            bot_token=bot_token,
            api_id=int(api_id) if api_id else 0,
            api_hash=api_hash,
            session_string=session_string
        )
        success = await publisher.publish_summary(target_chat=target_chat, text=summary_text)

        if success:
            logger.info("Daily Telegram news digest published successfully!")
            if speech:
                try:
                    logger.info("Synthesizing audio digest for publishing...")
                    with tempfile.TemporaryDirectory(prefix="tgsummary-") as directory:
                        audio_path = await asyncio.to_thread(
                            speech.synthesize, summary_text, Path(directory) / "digest.mp3"
                        )
                        if not await publisher.publish_audio(target_chat, audio_path):
                            raise RuntimeError("Audio upload failed.")
                    logger.info("Audio digest published successfully!")
                except Exception as error:
                    logger.error("Text was published, but audio failed (%s). Rerunning will repost text.", type(error).__name__)
                    sys.exit(1)
        else:
            logger.error("Failed to publish summary digest.")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Telegram Channel News Aggregator & Gemini Summarizer")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--debug", "-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and summarize without sending to Telegram")
    parser.add_argument("--audio-output", help="With --dry-run and speech enabled, save a speech preview MP3 to this path")
    parser.add_argument("--use-cache", action="store_true", help="Use locally cached posts instead of refetching from Telegram")
    parser.add_argument("--use-cached-response", action="store_true", help="Replay the latest cached Gemini summary, skipping fetching and Google API calls")
    parser.add_argument("--hours", type=int, help="Override lookback hours period")
    parser.add_argument("--channels", type=str, help="Comma-separated channel usernames override")
    parser.add_argument("--generate-session", action="store_true", help="Generate a Telethon StringSession")

    args = parser.parse_args()

    if args.generate_session:
        generate_string_session()
        return

    asyncio.run(run_pipeline(args))

if __name__ == "__main__":
    main()
