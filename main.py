import asyncio
import logging
from typing import Any, Dict, Optional
import os

import httpx
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

from subway.rail_client import RealtimeRailClient
from subway.query import ClaudeQueryResolver
from subway.service import QueryOutcome, process_subway_query


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telegram-bot")

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", "8180821757:AAEIbhpmgXnCHQENFWBM8cgcnceHN_a6lDk")
TELEGRAM_API_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

app = FastAPI(title="Subway Telegram Bot", version="0.6.0")
_realtime_client = RealtimeRailClient()
_query_resolver = ClaudeQueryResolver(realtime_client=_realtime_client)
_polling_task: Optional[asyncio.Task] = None
_http_client: Optional[httpx.AsyncClient] = None

# Store last query per chat for repeat functionality
# Structure: {chat_id: {"text": str, "lat": Optional[str], "lon": Optional[str]}}
_last_queries: Dict[int, Dict[str, Optional[str]]] = {}

# In-memory database for storing user home locations
# Structure: {chat_id: {"lat": str, "lon": str}}
_home_locations: Dict[int, Dict[str, str]] = {}

# Track users who are in "setting home" mode (waiting for location share)
_setting_home_mode: set[int] = set()


@app.on_event("startup")
async def start_telegram_polling() -> None:
    """Initialize the long-polling loop that listens for Telegram updates."""
    global _http_client, _polling_task

    if _http_client is None:
        _http_client = httpx.AsyncClient(
            base_url=TELEGRAM_API_BASE_URL, timeout=20.0)

    if _polling_task is None or _polling_task.done():
        logger.info("Starting Telegram long-polling loop.")
        _polling_task = asyncio.create_task(_poll_telegram_updates())


@app.on_event("shutdown")
async def stop_telegram_polling() -> None:
    """Stop the Telegram polling loop and close the HTTP client."""
    global _http_client, _polling_task

    if _polling_task:
        _polling_task.cancel()
        try:
            await _polling_task
        except asyncio.CancelledError:
            pass
        finally:
            _polling_task = None

    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


@app.get("/healthz", response_class=PlainTextResponse)
async def healthcheck() -> str:
    """Simple healthcheck endpoint for deployment monitoring."""
    return "ok"


def _save_home_location(chat_id: int, lat: str, lon: str) -> None:
    """Save a user's home location to the in-memory database."""
    _home_locations[chat_id] = {"lat": lat, "lon": lon}
    logger.info("Saved home location for chat %s: (%s, %s)", chat_id, lat, lon)


def _get_home_location(chat_id: int) -> Optional[Dict[str, str]]:
    """Retrieve a user's home location from the in-memory database."""
    return _home_locations.get(chat_id)


async def _handle_query(message_body: str, lat: Optional[str], lon: Optional[str]) -> Optional[QueryOutcome]:
    outcome = await process_subway_query(
        message_body,
        lat, lon,
        resolver=_query_resolver,
    )

    return outcome


async def _poll_telegram_updates() -> None:
    """Continuously fetch updates from Telegram using long polling."""
    if _http_client is None:
        logger.error(
            "Telegram HTTP client is not initialized; polling cannot start.")
        return

    offset = 0
    while True:
        try:
            response = await _http_client.get(
                "/getUpdates",
                params={"timeout": 30, "offset": offset},
            )
            response.raise_for_status()
            payload = response.json()

            if not payload.get("ok", False):
                logger.error(
                    "Telegram getUpdates returned an error payload: %s", payload)
                await asyncio.sleep(5)
                continue

            updates = payload.get("result", [])
            for update in updates:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    offset = max(offset, update_id + 1)
                await _handle_update(update)
        except asyncio.CancelledError:
            logger.info("Telegram polling task cancelled.")
            raise
        except httpx.HTTPError as exc:
            logger.error("HTTP error while polling Telegram updates: %s", exc)
            await asyncio.sleep(5)
        except Exception as exc:
            logger.exception(
                "Unexpected error while polling Telegram updates: %s", exc)
            await asyncio.sleep(5)


async def _handle_update(update: Dict[str, Any]) -> None:
    """Process a single Telegram update."""
    # Handle both regular messages and callback queries (button clicks)
    if "callback_query" in update:
        await _handle_callback_query(update["callback_query"])
        return

    message = update.get("message") or update.get("edited_message")
    if not message:
        logger.debug("Ignoring update without message content: %s", update)
        return

    chat = message.get("chat") or {}
    chat_id = chat.get("id")

    if not chat_id:
        logger.debug("Ignoring message without chat id: %s", message)
        return

    # Handle location messages
    latitude = None
    longitude = None
    if "location" in message:
        location = message["location"]
        latitude = location.get("latitude")
        longitude = location.get("longitude")

    text = message.get("text") or ""

    if not text and not latitude:
        logger.info("Ignoring empty message from chat %s.", chat_id)
        await _send_telegram_message(
            chat_id,
            "Send me a subway stop or station name to get train times, or share your location to find nearby stations.",
        )
        return

    # Handle slash commands
    if text.startswith("/"):
        await _handle_command(chat_id, text)
        return

    # Check if user is in "setting home" mode and shared a location
    if latitude and longitude and chat_id in _setting_home_mode:
        # Save the location as home
        _save_home_location(chat_id, str(latitude), str(longitude))
        _setting_home_mode.remove(chat_id)

        # Send confirmation message
        await _send_telegram_message(
            chat_id,
            "✅ Thank you! Your home location has been saved.\n\n"
            "Now finding trains near your home..."
        )

        # Continue to process the query with the location
        text = ""  # Empty query will use location-based filtering

    logger.info("Incoming Telegram message | chat_id=%s | text=%s | lat=%s | lon=%s",
                chat_id, text, latitude, longitude)

    # Convert lat/lon to strings for the query
    lat_str = str(latitude) if latitude else None
    lon_str = str(longitude) if longitude else None

    # Store last query for repeat functionality (including location if provided)
    _last_queries[chat_id] = {
        "text": text,
        "lat": lat_str,
        "lon": lon_str
    }

    outcome = await _handle_query(text, lat_str, lon_str)
    reply = outcome.message if outcome else "Thanks! Message received."

    await _send_telegram_message(chat_id, reply)


async def _handle_command(chat_id: int, command: str) -> None:
    """Handle slash commands."""
    if command == "/hello":
        await _send_telegram_message(chat_id, "Hello! 👋")
    elif command == "/home":
        # Check if user has a saved home location
        home_location = _get_home_location(chat_id)

        if home_location:
            # User has saved home - query with saved location
            logger.info(
                "User %s requested home, using saved location: %s", chat_id, home_location)
            lat = home_location["lat"]
            lon = home_location["lon"]

            # Process query with empty text but with home coordinates
            outcome = await _handle_query("", lat, lon)
            reply = outcome.message if outcome else "Here are the trains near your home location."

            await _send_telegram_message(chat_id, reply)
        else:
            # User doesn't have saved home - prompt to share location
            logger.info(
                "User %s requested home but no saved location, prompting to set", chat_id)
            _setting_home_mode.add(chat_id)

            await _send_telegram_message_with_location_button(
                chat_id,
                "🏠 Set Your Home Location\n\n"
                "You haven't set a home location yet. Share your location below to save it as your home.\n\n"
                "Next time you tap 'Home', I'll show you trains near this location!"
            )
    elif command == "/start":
        # Add location request button for /start command
        await _send_telegram_message_with_location_button(
            chat_id,
            "🚇 Welcome to the NYC Subway Bot!\n\n"
            "I can help you find train times at any NYC subway station.\n\n"
            "📍 Share your location to find nearby stations, or just type a station name!"
        )
        return
    elif command == "/repeat":
        last_query = _last_queries.get(chat_id)
        if last_query:
            text = last_query.get("text", "")
            lat = last_query.get("lat")
            lon = last_query.get("lon")

            logger.info("Repeating last query for chat %s: text=%s, lat=%s, lon=%s",
                        chat_id, text, lat, lon)

            # Execute the same query again (with location if it was provided)
            outcome = await _handle_query(text, lat, lon)
            reply = outcome.message if outcome else "Thanks! Message received."
            await _send_telegram_message(chat_id, reply)
        else:
            await _send_telegram_message(
                chat_id,
                "No previous query to repeat. Send me a station name or location first!"
            )
    else:
        await _send_telegram_message(
            chat_id,
            "Send me a subway stop or station name to get train times.",
        )


async def _handle_callback_query(callback_query: Dict[str, Any]) -> None:
    """Handle button clicks (callback queries)."""
    callback_id = callback_query.get("id")
    data = callback_query.get("data", "")
    message = callback_query.get("message", {})
    chat_id = message.get("chat", {}).get("id")

    if not chat_id:
        return

    # Handle location button specially
    if data == "/location":
        # Answer callback and prompt user to share location
        if _http_client:
            try:
                await _http_client.post(
                    "/answerCallbackQuery",
                    json={
                        "callback_query_id": callback_id,
                        "text": "Please use the button below to share your location",
                        "show_alert": False
                    }
                )
            except Exception as exc:
                logger.error("Failed to answer callback query: %s", exc)

        # Send message with location request button
        await _send_telegram_message_with_location_button(
            chat_id,
            "📍 Tap the button below to share your current location:"
        )
        return

    # Answer the callback query to remove loading state
    if _http_client:
        try:
            await _http_client.post(
                "/answerCallbackQuery",
                json={"callback_query_id": callback_id}
            )
        except Exception as exc:
            logger.error("Failed to answer callback query: %s", exc)

    # Handle the button action as if it were a command
    logger.info("Button clicked | chat_id=%s | data=%s", chat_id, data)
    await _handle_command(chat_id, data)


async def _send_telegram_message(chat_id: int, text: str) -> None:
    """Send a message with inline keyboard buttons and set persistent reply keyboard."""
    if _http_client is None:
        logger.warning(
            "Telegram HTTP client not initialized; unable to send message to chat %s.", chat_id)
        return

    # Create inline keyboard with Home, Repeat, and Current Location buttons
    # Inline keyboards appear directly below the message
    inline_keyboard = {
        "inline_keyboard": [
            [
                {"text": "🏠 Home", "callback_data": "/home"},
                {"text": "🔄 Repeat", "callback_data": "/repeat"}
            ],
            [
                {"text": "📍 Current Location", "callback_data": "/location"}
            ]
        ]
    }

    payload = {
        "chat_id": chat_id,
        "text": text,
        "reply_markup": inline_keyboard
    }

    try:
        response = await _http_client.post(
            "/sendMessage",
            json=payload,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error(
            "Failed to send Telegram message to chat %s: %s", chat_id, exc)


async def _send_telegram_message_with_location_button(chat_id: int, text: str) -> None:
    """Send a message with a button to request user's location."""
    if _http_client is None:
        logger.warning(
            "Telegram HTTP client not initialized; unable to send message to chat %s.", chat_id)
        return

    # Create reply keyboard with location request button
    # This creates a button at the bottom of the chat that requests location when tapped
    reply_keyboard = {
        "keyboard": [
            [
                {"text": "📍 Share My Location", "request_location": True}
            ]
        ],
        "resize_keyboard": True,  # Makes the button smaller
        "one_time_keyboard": True  # Hides keyboard after use
    }

    payload = {
        "chat_id": chat_id,
        "text": text,
        "reply_markup": reply_keyboard
    }

    try:
        response = await _http_client.post(
            "/sendMessage",
            json=payload,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error(
            "Failed to send Telegram message to chat %s: %s", chat_id, exc)
