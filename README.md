# Subway Telegram Bot

FastAPI service that bridges a Telegram bot with NYC subway data. Incoming Telegram messages are interpreted with Claude, mapped to GTFS stop IDs via the bundled `stops.txt`, and answered using live train arrival data from RealTimeRail.

## Prerequisites

- Python 3.11+
- Telegram bot token (replace the hard-coded sample token in `main.py` before deploying)
- Anthropic API key (for Claude)
- Outbound HTTPS access so the bot can reach Telegram, RealTimeRail, and Anthropic

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Set the required secrets in your shell or hosting platform:

- `ANTHROPIC_API_KEY` – Used by `ClaudeQueryResolver` to resolve free-form rider text.
- Optionally adjust the hard-coded Telegram token in `main.py`.

## Local Development

1. Start the bot:
   ```bash
   export ANTHROPIC_API_KEY="your-claude-key"
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
2. Send a message to the Telegram bot. The long-polling loop fetches updates, logs each request, and replies with live arrivals when available.

## Architecture Overview

- **Telegram ingestion** – `main.py` launches a long-polling task (`start_telegram_polling`) against `getUpdates`, logs each message, and sends replies with `_send_telegram_message`.
- **Claude-driven workflow** – `subway/query.py` embeds the full `stops.txt` CSV in Claude's context and exposes `get_realtime_stop_data` as a tool backed by `RealtimeRailClient`. Claude picks stop IDs, calls the tool, parses `stopTimes`, and returns a formatted rider response (or asks for clarification).
- **Real-time data lookup** – `subway/client.py` implements `RealtimeRailClient`, which fetches `https://realtimerail.nyc/transiter/v0.6/systems/us-ny-subway/stops/{stop_id}` and normalizes arrivals so they can be passed back to Claude.
- **Bot orchestration** – `subway/service.py` delegates each Telegram message to Claude and decides whether to forward the model's clarification prompt or final answer to the user.

## Key Implementation Files

- `main.py` – boots the FastAPI app, starts the Telegram polling loop, and forwards chat messages into the query pipeline.
- `subway/query.py` – defines `ClaudeQueryResolver`, including the tool-calling loop that routes RealTimeRail requests through Claude.
- `subway/client.py` – houses `RealtimeRailClient.stop_snapshot`, the entry point for calling the RealTimeRail API to fetch live train arrivals.
- `subway/service.py` – mediates between Telegram and Claude, returning either a clarification prompt or the final rider-facing summary.

## Data Flow Summary

1. **Claude interprets the station** – The resolver feeds the full `stops.txt` CSV to Claude so it can choose the most relevant stop IDs from the raw data.
2. **Claude calls the RealTimeRail tool** – For each stop ID it needs, Claude invokes `get_realtime_stop_data`, which proxies to `RealtimeRailClient.stop_snapshot`. Claude parses the `stopTimes` array to surface the next trains and formats the reply for the rider.

## Endpoints

- `GET /healthz` – Lightweight health check for deployment platforms.

## Next Steps

- Replace the bundled Telegram token before releasing.
- Persist station selections if you want to support multi-turn disambiguation.
- Harden the Claude prompt or add lightweight heuristics around station matching if the CSV proves too noisy for the model.
