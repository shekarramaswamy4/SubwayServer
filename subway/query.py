from __future__ import annotations

import asyncio
import csv
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic, RateLimitError

from .rail_client import RealtimeRailClient, RealtimeRailError


logger = logging.getLogger("subway-query")

STOPS_CSV_PATH = Path(__file__).with_name("stops.txt")
DEBUG_LOG_DIR = Path(__file__).parent.parent / "debug_logs"


class ClaudeConfigurationError(Exception):
    """Raised when the Claude client is not properly configured."""


class StationResolutionError(Exception):
    """Raised when the model output cannot be interpreted."""


class DebugLogger:
    """Logs all query details to a debug file for troubleshooting."""

    def __init__(self, query: str, lat: Optional[str], lon: Optional[str]):
        """Initialize debug logger with a unique file per query."""
        DEBUG_LOG_DIR.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log_file = DEBUG_LOG_DIR / f"query_{timestamp}.log"
        self.sections: List[str] = []

        # Add initial query
        initial = f"Query: {query}\nLatitude: {lat}\nLongitude: {lon}"
        self._add_section("INPUT QUERY", initial)

    def _add_section(self, title: str, content: str) -> None:
        """Add a section to the debug log."""
        separator = "=" * 80
        section = f"\n{separator}\n{title}\n{separator}\n{content}\n"
        self.sections.append(section)

    def log_tool_call(self, tool_name: str, tool_input: Dict[str, Any], tool_output: str) -> None:
        """Log a tool call with its input and output."""
        content = f"Tool: {tool_name}\nInput: {json.dumps(tool_input, indent=2)}\n\nOutput:\n{tool_output}"
        self._add_section("TOOL CALL", content)

    def log_api_usage(self, iteration: int, usage: Any) -> None:
        """Log API token usage statistics."""
        content = (
            f"Iteration: {iteration}\n"
            f"Input tokens: {getattr(usage, 'input_tokens', 0)}\n"
            f"Cache creation tokens: {getattr(usage, 'cache_creation_input_tokens', 0)}\n"
            f"Cache read tokens: {getattr(usage, 'cache_read_input_tokens', 0)}\n"
            f"Output tokens: {getattr(usage, 'output_tokens', 0)}"
        )
        self._add_section("API USAGE", content)

    def log_raw_response(self, response_text: str) -> None:
        """Log raw Claude response."""
        self._add_section("RAW CLAUDE RESPONSE", response_text)

    def log_parsed_result(self, status: str, message: Optional[str], clarification: Optional[str]) -> None:
        """Log the final parsed result."""
        content = (
            f"Status: {status}\n"
            f"Message: {message}\n"
            f"Clarification: {clarification}"
        )
        self._add_section("PARSED RESULT", content)

    def log_error(self, error: Exception) -> None:
        """Log an error that occurred."""
        content = f"Error Type: {type(error).__name__}\nError Message: {str(error)}"
        self._add_section("ERROR", content)

    def write(self) -> None:
        """Write all sections to the debug file."""
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("".join(self.sections))
            logger.info("Debug log written to: %s", self.log_file)
        except Exception as exc:
            logger.error("Failed to write debug log: %s", exc)


@dataclass(slots=True)
class ResolverResult:
    status: str
    message: Optional[str]
    clarification: Optional[str]


class ClaudeQueryResolver:
    """Uses the Claude Agent SDK with tool-calling to answer subway arrival questions."""

    SYSTEM_PROMPT = (
        "You help riders find NYC subway trains. Given stops.txt CSV, map queries to stop IDs.\n"
        "\n"
        "If query is unrelated to trains, respond: {\"status\":\"ok\",\"message\":\"This is not what I'm meant to do!\"}\n"
        "\n"
        "Steps:\n"
        "1. Identify relevant stop IDs from CSV\n"
        "2. Call get_realtime_stop_data ONCE with ALL stop IDs as array: ['R20N','R20S','L03N','L03S']\n"
        "3. Parse JSON 'stops' array with 'trains' grouped by route/destination\n"
        "4. Return pure JSON (no markdown):\n"
        "   {\"status\":\"ok\"|\"clarify\",\"message\":string|null,\"clarification\":string|null}\n"
        "\n"
        "status=\"clarify\" if need more info, else \"ok\" with concise train summary (3-4 trains)."
    )

    TOOL_DEFINITION = [
        {
            "name": "get_realtime_stop_data",
            "description": "Fetch real-time trains for NYC stop IDs. Pass all IDs as array: ['R20N','R20S']",
            "input_schema": {
                "type": "object",
                "properties": {
                    "stop_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Stop IDs from stops.txt",
                    }
                },
                "required": ["stop_ids"],
            },
        }
    ]

    def __init__(
        self,
        realtime_client: RealtimeRailClient,
        *,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.1,
        max_tokens: int = 1024,  # Reduced from 2048 - responses are typically short
        max_tool_iterations: int = 10,
    ) -> None:
        self.realtime_client = realtime_client
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tool_iterations = max_tool_iterations
        self.stops_csv = self._load_stops_csv()
        self._client = AsyncAnthropic(
            api_key=self.api_key) if self.api_key else None

    @staticmethod
    def _load_stops_csv() -> str:
        try:
            return STOPS_CSV_PATH.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - defensive
            logger.error("Unable to read stops.csv data: %s", exc)
            return ""

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance in kilometers between two points
        on the earth (specified in decimal degrees).
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * \
            math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        # Radius of earth in kilometers
        r = 6371

        return c * r

    def _filter_stops_by_location(self, lat: float, lon: float, max_distance_km: float = 0.8, max_stops: int = 4) -> str:
        """
        Filter stops.txt to only include stops within max_distance_km of the given coordinates.
        If more than max_stops are found, returns only the closest ones.

        IMPORTANT: All stop IDs with the same stop name are kept together.
        The limit is applied at the stop name level, not individual stop IDs.
        Returns a filtered CSV string.
        """
        if not self.stops_csv:
            return ""

        lines = self.stops_csv.split("\n")
        if not lines:
            return ""

        # Keep header
        header = lines[0]

        # Parse and collect stops with their distances, grouped by name
        stops_by_name = {}  # stop_name -> [(distance, row), ...]
        reader = csv.DictReader(lines)
        for row in reader:
            try:
                stop_lat = float(row["stop_lat"])
                stop_lon = float(row["stop_lon"])

                distance = self._haversine_distance(
                    lat, lon, stop_lat, stop_lon)

                if distance <= max_distance_km:
                    stop_name = row["stop_name"]
                    if stop_name not in stops_by_name:
                        stops_by_name[stop_name] = []
                    stops_by_name[stop_name].append((distance, row))
            except (ValueError, KeyError):
                # Skip malformed rows
                continue

        # Sort stop names by their closest stop's distance
        sorted_stop_names = sorted(
            stops_by_name.keys(),
            key=lambda name: min(dist for dist, _ in stops_by_name[name])
        )

        # Apply limit at the stop name level
        if len(sorted_stop_names) > max_stops:
            logger.info("Limiting location-based results from %d to %d unique stop names",
                        len(sorted_stop_names), max_stops)
            sorted_stop_names = sorted_stop_names[:max_stops]

        # Reconstruct CSV, including all stop IDs for each selected stop name
        filtered_lines = [header]
        for stop_name in sorted_stop_names:
            for distance, row in stops_by_name[stop_name]:
                filtered_lines.append(",".join([
                    row["stop_id"],
                    row["stop_name"],
                    row["stop_lat"],
                    row["stop_lon"],
                    row["location_type"],
                    row["parent_station"]
                ]))

        return "\n".join(filtered_lines)

    def _filter_stops_by_query(self, query: str, max_stops: int = 4) -> str:
        """
        Intelligently filter stops.txt based on the query text.
        Uses fuzzy matching to find relevant station names.
        If more than max_stops are found, returns only the most relevant ones.

        IMPORTANT: All stop IDs with the same stop name are kept together.
        The limit is applied at the stop name level, not individual stop IDs.
        Returns a filtered CSV string.
        """
        if not self.stops_csv:
            return ""

        lines = self.stops_csv.split("\n")
        if not lines:
            return ""

        # Keep header
        header = lines[0]

        # Normalize query for matching
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        # Extract key words that might be station names
        # Remove common words that are unlikely to be station names
        stop_words = {"train", "trains", "subway", "station", "stop", "next", "when",
                      "what", "where", "how", "the", "a", "an", "to", "from", "at",
                      "is", "are", "time", "times", "arriving", "arrives"}
        search_tokens = query_tokens - stop_words

        # Parse and filter data rows, tracking match quality per stop name
        # Match quality: 1 = exact substring, 2 = token match, 3 = numeric match
        reader = csv.DictReader(lines)
        matched_stops_by_name = {}  # stop_name -> (quality, [rows])

        for row in reader:
            try:
                stop_name = row["stop_name"]
                stop_name_lower = stop_name.lower()

                # Determine match quality for this stop name
                quality = None

                # Strategy 1: Exact substring match (highest priority)
                if any(token in stop_name_lower for token in search_tokens if len(token) > 2):
                    quality = 1
                # Strategy 2: Check if any word in stop name appears in query
                elif set(stop_name_lower.split()) & search_tokens:
                    quality = 2
                # Strategy 3: Fuzzy match on numbers (e.g., "14th street", "14 st")
                else:
                    query_numbers = set(re.findall(r'\d+', query_lower))
                    stop_numbers = set(re.findall(r'\d+', stop_name_lower))
                    if query_numbers and stop_numbers and query_numbers & stop_numbers:
                        quality = 3

                # If this stop name matched, add it to our collection
                if quality is not None:
                    if stop_name not in matched_stops_by_name:
                        matched_stops_by_name[stop_name] = (quality, [])
                    matched_stops_by_name[stop_name][1].append(row)

            except (ValueError, KeyError):
                # Skip malformed rows
                continue

        logger.info("Filtered stops by query '%s': %d unique stop names matched",
                    query, len(matched_stops_by_name))

        # Sort stop names by match quality (lower is better) and limit at the name level
        sorted_stop_names = sorted(
            matched_stops_by_name.keys(),
            key=lambda name: matched_stops_by_name[name][0]
        )

        if len(sorted_stop_names) > max_stops:
            logger.info("Limiting query-based results from %d to %d unique stop names",
                        len(sorted_stop_names), max_stops)
            sorted_stop_names = sorted_stop_names[:max_stops]

        # Reconstruct CSV, including all stop IDs for each selected stop name
        filtered_lines = [header]
        for stop_name in sorted_stop_names:
            _quality, rows = matched_stops_by_name[stop_name]
            for row in rows:
                filtered_lines.append(",".join([
                    row["stop_id"],
                    row["stop_name"],
                    row["stop_lat"],
                    row["stop_lon"],
                    row["location_type"],
                    row["parent_station"]
                ]))

        return "\n".join(filtered_lines)

    @property
    def is_configured(self) -> bool:
        return self._client is not None and bool(self.stops_csv)

    @staticmethod
    def _parse_json_response(text_output: str) -> Dict[str, Any]:
        """
        Parse JSON from Claude's response, handling markdown code blocks and reasoning text.

        Args:
            text_output: Raw text response from Claude

        Returns:
            Parsed JSON payload

        Raises:
            StationResolutionError: If JSON cannot be parsed
        """
        # Strip markdown code blocks if present (e.g., ```json ... ```)
        text_output = text_output.strip()
        if text_output.startswith("```"):
            # Remove opening code fence (```json or ```\n)
            lines = text_output.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove closing code fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text_output = "\n".join(lines).strip()

        # Try to parse the output as-is first
        try:
            return json.loads(text_output)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract the last valid JSON object
            # This handles cases where Claude includes reasoning before the JSON
            json_start = text_output.rfind("{")
            if json_start != -1:
                # Find the matching closing brace
                brace_count = 0
                json_end = -1
                for i in range(json_start, len(text_output)):
                    if text_output[i] == "{":
                        brace_count += 1
                    elif text_output[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break

                if json_end != -1:
                    json_str = text_output[json_start:json_end]
                    try:
                        payload = json.loads(json_str)
                        logger.info(
                            "Extracted JSON from position %d-%d after initial parse failure",
                            json_start,
                            json_end,
                        )
                        return payload
                    except json.JSONDecodeError as exc:
                        raise StationResolutionError(
                            f"Claude response was not valid JSON: {exc}. Raw output: {text_output}"
                        ) from exc
                else:
                    raise StationResolutionError(
                        f"Could not find valid JSON object in response. Raw output: {text_output}"
                    )
            else:
                raise StationResolutionError(
                    f"No JSON object found in response. Raw output: {text_output}"
                )

    async def resolve(self, query: str, lat: Optional[str], lon: Optional[str]) -> ResolverResult:
        # Track total resolution time
        resolve_start_time = time.time()

        # Initialize debug logger
        debug_log = DebugLogger(query, lat, lon)

        try:
            if not self._client:
                raise ClaudeConfigurationError(
                    "ANTHROPIC_API_KEY is not configured.")

            if not self.stops_csv:
                raise StationResolutionError("Stops CSV data is unavailable.")

            # Filter stops based on whether lat/lon is provided or just query
            filtered_stops = self.stops_csv  # Default to full CSV

            if lat is not None and lon is not None:
                if query == "":
                    query = "Given the input stops, find the next trains for each stop."
                # Filter by geographic location (0.8 km radius)
                try:
                    lat_float = float(lat)
                    lon_float = float(lon)
                    filtered_stops = self._filter_stops_by_location(
                        lat_float, lon_float)
                    logger.info(
                        "Filtered stops by location (%.6f, %.6f)", lat_float, lon_float)
                except ValueError:
                    logger.warning(
                        "Invalid lat/lon values (%s, %s), falling back to query-based filtering", lat, lon)
                    filtered_stops = self._filter_stops_by_query(query)
            else:
                # Filter by intelligent query matching
                filtered_stops = self._filter_stops_by_query(query)

            current_time_unix = int(time.time())

            debug_log.log_raw_response(
                "Filtered stops data:\n" + filtered_stops)

            messages: List[Dict[str, Any]] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Current time (Unix seconds): {current_time_unix}\n\nUser Query:\n{query.strip()}",
                        },
                        {
                            "type": "text",
                            "text": f"Reference Data (stops.txt):\n{filtered_stops}",
                        },
                    ],
                }
            ]

            # Track tool calls to detect repetitive behavior
            tool_call_history: List[str] = []

            for iteration in range(self.max_tool_iterations):
                # Retry with exponential backoff on rate limit errors
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        # Track API call latency
                        api_start_time = time.time()

                        response = await self._client.messages.create(
                            model=self.model,
                            max_tokens=self.max_tokens,
                            temperature=self.temperature,
                            system=self.SYSTEM_PROMPT,
                            messages=messages,
                            tools=self.TOOL_DEFINITION,
                        )

                        # Calculate and log latency
                        api_latency = time.time() - api_start_time
                        logger.info(
                            "Claude API latency: %.2f seconds (iteration %d)",
                            api_latency,
                            iteration + 1
                        )

                        break  # Success, exit retry loop
                    except RateLimitError:
                        if retry < max_retries - 1:
                            wait_time = 2 ** retry  # Exponential backoff: 1s, 2s, 4s
                            logger.warning(
                                "Rate limit hit, retrying in %d seconds (attempt %d/%d)",
                                wait_time,
                                retry + 1,
                                max_retries,
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            logger.error(
                                "Rate limit exceeded after %d retries", max_retries)
                            raise

                # Log cache usage for monitoring
                usage = response.usage
                logger.info(
                    "API usage - input: %d, cache_creation: %d, cache_read: %d, output: %d",
                    getattr(usage, 'input_tokens', 0),
                    getattr(usage, 'cache_creation_input_tokens', 0),
                    getattr(usage, 'cache_read_input_tokens', 0),
                    getattr(usage, 'output_tokens', 0),
                )

                # Log to debug file
                debug_log.log_api_usage(iteration + 1, usage)

                messages.append(
                    {"role": "assistant", "content": response.content})

                tool_uses = [
                    block for block in response.content if block.type == "tool_use"]
                if tool_uses:
                    # Check for repetitive tool calls (same tool with same inputs)
                    current_calls = [
                        f"{tc.name}:{tc.input.get('stop_id', '') if hasattr(tc, 'input') and tc.input else ''}"
                        for tc in tool_uses
                    ]

                    # Detect if we're calling the same tool repeatedly (potential infinite loop)
                    if len(tool_call_history) >= 3:
                        recent_calls = tool_call_history[-3:]
                        if all(call in recent_calls for call in current_calls):
                            logger.warning(
                                "Detected repetitive tool calling pattern. Recent: %s, Current: %s",
                                recent_calls,
                                current_calls,
                            )
                            raise StationResolutionError(
                                "Claude appears to be stuck in a loop. Unable to resolve query."
                            )

                    tool_call_history.extend(current_calls)

                    tool_results = []
                    for tool_call in tool_uses:
                        tool_input = tool_call.input if hasattr(
                            tool_call, 'input') else {}
                        tool_result = await self._execute_tool(tool_call)

                        # Log tool call to debug file
                        debug_log.log_tool_call(
                            tool_call.name, tool_input, tool_result)

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": tool_result,
                            }
                        )

                    messages.append(
                        {
                            "role": "user",
                            "content": tool_results,
                        }
                    )
                    continue

                text_output = "".join(
                    block.text for block in response.content if block.type == "text"
                )

                # Always log the raw response for debugging
                logger.info("Raw Claude response: %s", text_output)
                debug_log.log_raw_response(text_output)

                if not text_output:
                    raise StationResolutionError(
                        "Claude returned an empty response.")

                # Parse the JSON response using the dedicated method
                payload = self._parse_json_response(text_output)

                status = (payload.get("status") or "").lower()
                message = payload.get("message")
                clarification = payload.get("clarification")

                if status not in {"ok", "clarify"}:
                    raise StationResolutionError(
                        f"Claude returned an unknown status '{payload.get('status')}'. Payload: {payload}"
                    )

                # Log parsed result
                debug_log.log_parsed_result(status, message, clarification)
                debug_log.write()

                # Log total resolution time
                total_latency = time.time() - resolve_start_time
                logger.info(
                    "Total query resolution time: %.2f seconds (query='%s', lat=%s, lon=%s)",
                    total_latency,
                    query[:50] if query else "",
                    lat,
                    lon
                )

                return ResolverResult(status=status, message=message, clarification=clarification)

            raise StationResolutionError(
                f"Exceeded maximum tool iterations ({self.max_tool_iterations}) without a final response."
            )
        except Exception as error:
            debug_log.log_error(error)
            debug_log.write()
            raise

    async def _execute_tool(self, tool_call: Any) -> str:
        tool_name = tool_call.name if hasattr(
            tool_call, 'name') else tool_call.get("name")
        if tool_name != "get_realtime_stop_data":
            logger.error(
                "Claude attempted to use unsupported tool '%s'.", tool_name)
            return json.dumps(
                {
                    "error": f"Unsupported tool {tool_name}.",
                }
            )

        # Handle both typed objects (from SDK) and dicts (from message history)
        if hasattr(tool_call, 'input'):
            tool_input = tool_call.input
        else:
            tool_input = tool_call.get("input", {})

        stop_ids = tool_input.get("stop_ids") if tool_input else None

        # Validate input
        if not stop_ids:
            return json.dumps({"error": "stop_ids must be provided as a list"})

        if not isinstance(stop_ids, list):
            return json.dumps({"error": "stop_ids must be a list of strings"})

        if not all(isinstance(sid, str) for sid in stop_ids):
            return json.dumps({"error": "All stop_ids must be strings"})

        try:
            # Fetch all stops concurrently
            logger.info("Fetching real-time data for %d stop IDs: %s",
                        len(stop_ids), stop_ids)
            payloads = await self.realtime_client.fetch_multiple_stops(stop_ids)

            if not payloads:
                return json.dumps({
                    "error": "Failed to fetch data for all requested stops",
                    "stop_ids": stop_ids
                })

            # Process and filter each payload
            filtered_results = []
            for stop_id, payload in payloads.items():
                try:
                    filtered_payload = self._filter_realtime_payload(payload)
                    filtered_results.append(filtered_payload)
                except Exception as exc:
                    logger.warning(
                        "Failed to process payload for stop %s: %s", stop_id, exc)
                    filtered_results.append({
                        "stop_id": stop_id,
                        "error": f"Failed to process: {exc}"
                    })

            # Return aggregated results
            result = {
                "stops": filtered_results,
                "total_stops": len(filtered_results),
                "requested_stops": len(stop_ids)
            }

            return json.dumps(result)

        except RealtimeRailError as exc:
            logger.warning(
                "RealtimeRail tool call failed for stops %s: %s", stop_ids, exc)
            return json.dumps({"error": str(exc), "stop_ids": stop_ids})
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "Unexpected error during RealTimeRail tool execution: %s", exc)
            return json.dumps({"error": f"Unhandled error: {exc}", "stop_ids": stop_ids})

    @staticmethod
    def _filter_realtime_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter RealTimeRail payload to only include relevant upcoming train data.

        The payload contains an array of stop time objects, each representing a train arrival.
        We organize trains by route name and destination, then sort by arrival time.
        """
        current_time = int(time.time())
        max_future_seconds = 1800  # Only trains within 30 minutes

        # Get stop times array - this is the main data structure
        stop_times = payload.get("stopTimes", [])

        # If the payload is a single stop time object (not an array), wrap it
        if isinstance(payload, dict) and "stop" in payload and "trip" in payload:
            stop_times = [payload]

        # Get stop info from first stop time (they all refer to the same stop)
        stop_info = None
        if stop_times and len(stop_times) > 0:
            stop_info = stop_times[0].get("stop", {})

        filtered = {
            "stop_id": stop_info.get("id", "") if stop_info else "",
            "stop_name": stop_info.get("name", "") if stop_info else "",
            "trains_by_route_and_destination": {}
        }

        from datetime import datetime

        # Iterate through each stop time and extract train information
        for idx, stop_time in enumerate(stop_times):
            try:
                # Extract arrival/departure time
                arrival = stop_time.get("arrival", {})
                departure = stop_time.get("departure", {})
                arrival_time = arrival.get("time") or departure.get("time")
                if not arrival_time:
                    continue
                arrival_time = int(arrival_time)

                if not arrival_time or not isinstance(arrival_time, (int, float)):
                    continue

                # Filter: only future trains within time window
                if not (current_time < arrival_time < current_time + max_future_seconds):
                    continue

                # Extract train route name (e.g., "F", "A", "1")
                trip = stop_time.get("trip", {})
                route = trip.get("route", {})
                route_name = route.get("id", "Unknown")

                # Extract destination name
                destination = stop_time.get("destination", {})
                destination_name = destination.get("name", "Unknown")

                # Extract headsign (direction indicator)
                headsign = stop_time.get("headsign", "")

                # Create a key for grouping: route_name + destination
                group_key = f"{route_name}_{destination_name}"

                # Initialize the group if it doesn't exist
                if group_key not in filtered["trains_by_route_and_destination"]:
                    filtered["trains_by_route_and_destination"][group_key] = {
                        "route": route_name,
                        "destination": destination_name,
                        "headsign": headsign,
                        "arrivals": []
                    }

                # Add this arrival to the group
                minutes_away = (arrival_time - current_time) // 60
                filtered["trains_by_route_and_destination"][group_key]["arrivals"].append({
                    "arrival_time": arrival_time,
                    "minutes_away": minutes_away
                })

            except (KeyError, TypeError, ValueError) as e:
                # Skip malformed stop times
                logger.debug("Skipping malformed stop time: %s", e)
                continue

        # Sort arrivals within each group by arrival time (earliest first)
        for group in filtered["trains_by_route_and_destination"].values():
            group["arrivals"].sort(key=lambda x: x["arrival_time"])

        # Convert to a list format for cleaner output
        trains_list = []
        for group in filtered["trains_by_route_and_destination"].values():
            trains_list.append({
                "route": group["route"],
                "destination": group["destination"],
                "headsign": group["headsign"],
                "arrivals": group["arrivals"]
            })

        # Sort groups by the earliest arrival time in each group
        trains_list.sort(
            key=lambda x: x["arrivals"][0]["arrival_time"] if x["arrivals"] else float('inf'))

        filtered["trains"] = trains_list
        # Remove the intermediate dictionary structure
        del filtered["trains_by_route_and_destination"]

        return filtered
