# Telegram Channel News Aggregator & Gemini Summarizer

An automated Python tool that fetches daily posts from specified public Telegram channels, synthesizes them into a single topic-grouped news digest using **Google Gemini AI**, and posts the summary digest to a target Telegram channel. Designed for seamless execution on **GitHub Actions (Free Tier)**.

---

## Features

- **Public Channel Monitoring**: Monitor any number of public Telegram channels configured in `config.yaml`.
- **Google Gemini Topic Synthesis**: Uses `google-genai` (`gemini-flash-latest`) to organize all channel posts into cohesive key topic summaries rather than disconnected channel updates.
- **Local Ukrainian Neural Audio (`ua-tts`)**: Offline, high-fidelity neural speech generation with natural Ukrainian voices (`dmytro`, `tetiana`, `mykyta`, `lada`, `oleksa`).
- **Offline Stanza Cache Interception**: Transparently discovers and reuses local Stanza linguistic resources, eliminating redundant re-downloads during synthesis.
- **Granular Debug Logging**: Detailed diagnostics for text chunking, model discovery, chunk-by-chunk synthesis RTF, and audio waveform metrics via `--debug` or `DEBUG=1`.
- **Telethon & StringSession Support**: Headless execution in cloud runners without interactive prompt blockages.
- **Telegram Character Limit Handling**: Automatically splits long digests into readable multi-part messages.
- **GitHub Actions Ready**: Pre-configured workflow optimized for GitHub Free Tier (takes ~30 seconds per run, well under the 2,000 free monthly minutes for private repos, and completely free for public repos).

---

## Setup & Configuration

### Optional audio digest (Local Neural ua-tts or Azure Speech)

The pipeline can narrate the generated digest and upload a single MP3 to the same
`target_chat` after publishing the text. It supports two speech providers:

1. **Local Ukrainian Neural TTS (`ua-tts`)** *(Default & Recommended)*:
   - High-fidelity offline neural speech generation tailored specifically for Ukrainian text.
   - Built directly into the repository (`ua_tts/` with local weights in `models/ua_tts/`).
   - Requires no external cloud accounts, API keys, or monthly spending limits.
   - Available natural voices: `dmytro` (Male), `tetiana` (Female), `mykyta` (Male), `lada` (Female), `oleksa` (Male), `olena`, `olga`.
   - Supports CPU and GPU (`cuda`) acceleration, adjustable speed, and dictionary stress placement.
   - Uses local cached Stanza language models automatically.

2. **Microsoft Azure Speech (`azure`)**:
   - Cloud-based REST API for multilingual or Azure neural voices (`uk-UA-PolinaNeural`, `uk-UA-OstapNeural`).
   - Requires `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` set in `.env` (e.g. `westeurope`).

#### Configuration (`config.yaml`)

```yaml
speech:
  enabled: true
  provider: "ua-tts"    # "ua-tts" (local neural) or "azure" (cloud REST API)
  voice: "dmytro"       # ua-tts: dmytro, tetiana, mykyta, lada, oleksa, olena, olga
                        # azure: uk-UA-OstapNeural, uk-UA-PolinaNeural
  speed: 1.0            # Playback speed (0.5 to 2.0)
  device: "cpu"         # "cpu" or "cuda"
  backend: "auto"       # "auto", "ukrainian_tts", "silero", "synth"
  stress: "dictionary"  # "dictionary" or "model"
  model_path: "models/ua_tts"
```

Preview locally without publishing:
```bash
python main.py --use-cache --dry-run --audio-output preview.mp3
```

Replay from cached Gemini response:
```bash
python main.py --use-cached-response --dry-run --audio-output preview.mp3
```

Replay with verbose debug logging enabled:
```bash
python main.py --use-cached-response --dry-run --audio-output preview.mp3 --debug
```

Offline test checks:
```bash
python -m unittest discover -s tests -v
```

---

### Verbose Debug Logging (`--debug` / `-v` / `DEBUG=1`)

You can enable verbose diagnostic logs dynamically to inspect every phase of text processing, neural speech generation, and audio export:

- **Command-line flag**: Add `--debug`, `-v`, or `--verbose` to any CLI command:
  ```bash
  python main.py --dry-run --debug
  ```
- **Environment variable**: Set `DEBUG=1` or `LOG_LEVEL=DEBUG` in your `.env` file or shell environment:
  ```bash
  DEBUG=1 python main.py --dry-run
  ```
- **GitHub Actions**: Set `DEBUG` as a repository secret or repository variable (`1`), or enable the **Enable verbose debug logging** checkbox when triggering via `workflow_dispatch`.

**What debug mode logs:**
- **Text Preprocessing & Chunking**: Character & word counts, sentence boundary splitting, and per-chunk text breakdown.
- **Model Discovery & Stanza Cache**: Candidate cache paths inspected, archive extraction details, and offline model validation.
- **Synthesis Diagnostics**: Per-chunk accented text transformations, time elapsed per chunk, Real-Time Factor (RTF), and resampling metrics (e.g., 22,050 Hz to 24,000 Hz).
- **Audio Waveform Diagnostics**: Sample counts, audio duration, peak min/max, RMS, and silence/clipping detection.
- **Audio File Export**: Export backend, target format, bitrate, and final file sizes.

---

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
3. Copy `.env.example` to `.env` and fill in your credentials:
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

### Reuse the cached Google Gemini response

Successful summaries are automatically saved as response text in
`cache/gemini_response_cache.json` before publishing or generating audio.
Preview the latest saved summary without fetching posts or calling Gemini:

```bash
python main.py --use-cached-response --dry-run
```

Add `--audio-output preview.mp3` to generate audio from that cached text when
speech is enabled. Omit `--dry-run` to publish the cached
summary again. Replay does not require `GEMINI_API_KEY`; publishing and speech
still require their respective credentials.

This replays the latest summary regardless of current channels, hours, model,
language, or prompt settings. Missing or invalid response caches cause an error
without calling Gemini. Each successful fresh summary replaces the cache;
the `cache/` directory is ignored by Git and is local to the current checkout.
The existing `--use-cache` option reuses fetched posts but still calls Gemini.
If both options are supplied, `--use-cached-response` takes precedence.

---

## Setting Up GitHub Actions Automation (Free Tier)

This repository includes `.github/workflows/daily_summary.yml` which can run automatically on schedule or manually via `workflow_dispatch`.

### Adding Secrets in GitHub:

1. Push this project to your GitHub repository.
2. Go to your repository **Settings** -> **Secrets and variables** -> **Actions**.
3. Add the following **Repository Secrets**:

| Secret / Variable Name | Value | Description |
|---|---|---|
| `TELEGRAM_API_ID` | Telegram API ID integer | Required for fetching posts |
| `TELEGRAM_API_HASH` | Telegram API Hash string | Required for fetching posts |
| `TELEGRAM_SESSION_STRING` | Generated string from `python main.py --generate-session` | Required for headless Telethon auth |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot token from @BotFather | Required for publishing text & audio |
| `GEMINI_API_KEY` | Google Gemini API Key | Required for summarization |
| `TARGET_CHAT` | Channel username or Chat ID (e.g., `@my_channel`) | Destination channel override |
| `AZURE_SPEECH_KEY` | Azure Speech key *(Optional)* | Only if `provider: azure` |
| `AZURE_SPEECH_REGION` | Azure Speech region *(Optional)* | Only if `provider: azure` |
| `DEBUG` | `1` or `true` *(Optional)* | Enables verbose debug diagnostics |

> 💡 **GitHub Free Tier Note**:
> - Public repos: GitHub Actions run minutes are **unlimited and 100% free**.
> - Private repos: GitHub Actions includes **2,000 free minutes/month**. This daily script uses ~15-30 minutes per month in total.

---

## File Overview

- `main.py`: Entry point CLI coordinating retrieval, Gemini summarization, speech synthesis, and publishing, with CLI debug flags (`--debug`, `-v`).
- `config.yaml`: Public channels list, lookback interval, and summarizer/speech settings.
- `src/telegram_fetcher.py`: Retrieves recent posts from public channels via Telethon.
- `src/summarizer.py`: Gemini topic summarizer using official `google-genai` SDK.
- `src/publisher.py`: Formats and sends summary text messages and audio digests to Telegram.
- `src/speech.py`: Speech synthesis interface supporting `UaTtsSynthesizer` (offline neural) and `AzureSpeechSynthesizer`.
- `src/cache_manager.py`: Utilities for caching and replaying fetched posts and Gemini summary responses.
- `ua_tts/`: Local Ukrainian neural text-to-speech engine with Stanza caching, sentence chunking, and audio export.
- `models/ua_tts/`: Bundled model configurations, phoneme features, and speaker vector statistics for `ua-tts`.
- `tests/`: Comprehensive unit test suite covering retrieval, summarization, caching, audio generation, and offline Stanza cache interception.
- `.github/workflows/daily_summary.yml`: GitHub Actions daily automation workflow with debug toggle.
