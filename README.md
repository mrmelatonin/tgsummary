# Telegram Channel News Aggregator & Gemini Summarizer

An automated Python tool that fetches daily posts from specified public Telegram channels, synthesizes them into a single topic-grouped news digest using **Google Gemini AI**, and posts the summary digest to a target Telegram channel. Designed for seamless execution on **GitHub Actions (Free Tier)**.

---

## Features

- **Public Channel Monitoring**: Monitor any number of public Telegram channels configured in `config.yaml`.
- **Google Gemini Topic Synthesis**: Uses `google-genai` (`gemini-2.5-flash`) to organize all channel posts into cohesive key topic summaries rather than disconnected channel updates.
- **Telethon & StringSession Support**: Headless execution in cloud runners without interactive prompt blockages.
- **Telegram Character Limit Handling**: Automatically splits long digests into readable multi-part messages.
- **GitHub Actions Ready**: Pre-configured workflow optimized for GitHub Free Tier (takes ~30 seconds per run, well under the 2,000 free monthly minutes for private repos, and completely free for public repos).

---

## Setup & Configuration

### 1. Prerequisites & API Keys

You will need the following keys:

1. **Telegram API Credentials**:
   - Go to [my.telegram.org](https://my.telegram.org) and log in.
   - Click **API development tools** and create an app to get your `API_ID` and `API_HASH`.

2. **Google Gemini API Key**:
   - Get a free API key from [Google AI Studio](https://aistudio.google.com).

3. **Telegram Bot Token** (Optional but recommended for posting):
   - Create a bot via [@BotFather](https://t.me/BotFather) on Telegram and get the `BOT_TOKEN`.
   - Add your bot as an **Administrator** in your target private channel so it can post messages.

---

### 2. Local Installation & Session Generation

1. Clone or download this repository.
2. Install Python 3.10+ and dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your `TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, and `GEMINI_API_KEY`:
   ```bash
   cp .env.example .env
   ```
4. Generate your **Telethon String Session** (needed for GitHub Actions):
   ```bash
   python main.py --generate-session
   ```
   Follow the on-screen Telegram login prompt. Copy the generated `TELEGRAM_SESSION_STRING` output and save it in your `.env` file.

---

### 3. Configuring Monitored Channels (`config.yaml`)

Edit `config.yaml` to specify your public Telegram channels and summary options:

```yaml
channels:
  - "durov"
  - "telegram"
  - "techcrunch"
  - "bloomberg"

lookback_hours: 24
target_chat: "@my_summary_channel" # Or chat ID integer

summarizer:
  model_name: "gemini-2.5-flash"
  language: "English"
```

---

### 4. Local Test (Dry Run)

Test post retrieval and Gemini summarization locally without sending to Telegram:

```bash
python main.py --dry-run
```

To run a live test sending to your destination channel:

```bash
python main.py
```

---

## Setting Up GitHub Actions Automation (Free Tier)

This repository includes `.github/workflows/daily_summary.yml` which runs automatically every day at 08:00 UTC.

### Adding Secrets in GitHub:

1. Push this project to your GitHub repository.
2. Go to your repository **Settings** -> **Secrets and variables** -> **Actions**.
3. Add the following **Repository Secrets**:

| Secret Name | Value |
|---|---|
| `TELEGRAM_API_ID` | Your Telegram API ID integer |
| `TELEGRAM_API_HASH` | Your Telegram API Hash string |
| `TELEGRAM_SESSION_STRING` | Generated string from `python main.py --generate-session` |
| `TELEGRAM_BOT_TOKEN` | Your Telegram Bot token from @BotFather |
| `GEMINI_API_KEY` | Your Google Gemini API Key |
| `TARGET_CHAT` | Username or Chat ID of your summary channel (e.g., `@my_channel`) |

> 💡 **GitHub Free Tier Note**:
> - Public repos: GitHub Actions run minutes are **unlimited and 100% free**.
> - Private repos: GitHub Actions includes **2,000 free minutes/month**. This daily script uses ~15-30 minutes per month in total.

---

## File Overview

- `main.py`: Entry point CLI script supporting `--dry-run` and `--generate-session`.
- `config.yaml`: Public channels list and summary settings.
- `src/telegram_fetcher.py`: Retrieves recent posts from public channels via Telethon.
- `src/summarizer.py`: Gemini topic summarizer using official `google-genai` SDK.
- `src/publisher.py`: Formats and sends summary messages to Telegram chat/channel.
- `.github/workflows/daily_summary.yml`: GitHub Actions daily automation workflow.
