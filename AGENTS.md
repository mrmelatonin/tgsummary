# Repository Guidelines

## Project Structure & Module Organization

`main.py` is the command-line entry point and coordinates the daily digest pipeline. Keep Telegram retrieval, Gemini summarization, publishing, and cache handling in their focused modules under `src/` (`telegram_fetcher.py`, `summarizer.py`, `publisher.py`, and `cache_manager.py`). `config.yaml` contains non-secret runtime settings; `.env.example` documents required local variables. The scheduled GitHub Actions workflow lives in `.github/workflows/daily_summary.yml`. The `cache/` directory stores fetched-post data for local use. There is no dedicated test or assets directory currently.

## Build, Test, and Development Commands

Install dependencies with `python -m pip install -r requirements.txt`. Run a local summary without publishing with `python main.py --dry-run`; run the full pipeline with `python main.py`. Generate a Telethon StringSession interactively with `python main.py --generate-session`. Use `python main.py --help` to inspect CLI options. No separate build, formatter, or test command is configured.

## Coding Style & Naming Conventions

Use Python 3.10+ and follow the existing four-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Keep module responsibilities narrow and use type hints for public methods, matching the existing `src/` code. Use the standard `logging` module for operational messages; avoid printing secrets or credentials. Keep configuration keys descriptive and consistent with the YAML structure consumed by `main.py`.

## Testing Guidelines

There is currently no test suite or coverage requirement. For changes to the pipeline, run the dry-run command with valid local credentials and representative channel configuration. Avoid a live publishing run unless you intend to send a message to the configured Telegram destination. If adding tests, place them in a `tests/` directory and use `test_*.py` names.

## Commit & Pull Request Guidelines

This checkout does not include Git metadata, so repository-specific commit conventions could not be confirmed. Use concise imperative commit subjects (for example, `Add retry handling for Telegram fetches`). Pull requests should explain the behavior change, list configuration or secret changes, link related issues when applicable, and include relevant command output for operational changes. Never include `.env`, session strings, API keys, or other credentials.

## Security & Configuration Tips

Keep secrets in local `.env` or GitHub Actions secrets; commit only placeholders in `.env.example`. Do not commit generated session strings or private cache data. Put channel lists and summarizer preferences in `config.yaml`, and verify the target chat before running the publishing mode.
