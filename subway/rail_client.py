from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx


LOGGER = logging.getLogger("realtimerail-client")

REALTIME_BASE_URL = "https://realtimerail.nyc/transiter/v0.6"


class RealtimeRailError(Exception):
    """Base error for issues communicating with the RealTimeRail API."""


class RealtimeRailRequestError(RealtimeRailError):
    """Raised when the RealTimeRail API request fails."""


class RealtimeRailResponseError(RealtimeRailError):
    """Raised when the RealTimeRail API returns an unexpected payload."""


@dataclass(slots=True)
class SubwayArrival:
    route_id: str
    direction: Optional[str]
    headsign: Optional[str]
    arrival_time: Optional[datetime]
    is_real_time: bool
    raw: Dict[str, Any]


@dataclass(slots=True)
class StopRealtimeData:
    stop_id: str
    stop_name: Optional[str]
    routes: Sequence[str]
    arrivals: List[SubwayArrival]


class RealtimeRailClient:
    """Client wrapper around the RealTimeRail Transit API."""

    def __init__(self, *, base_url: str = REALTIME_BASE_URL, timeout_seconds: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def fetch_stop_payload(self, stop_id: str) -> Dict[str, Any]:
        """Fetch the raw RealTimeRail JSON payload for a stop identifier."""
        if not stop_id:
            raise ValueError("stop_id must be provided.")

        url = f"{self.base_url}/systems/us-ny-subway/stops/{stop_id}"
        print(url)

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(url)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            LOGGER.warning(
                "RealTimeRail returned HTTP %s for stop %s", exc.response.status_code, stop_id
            )
            raise RealtimeRailRequestError(
                f"RealTimeRail returned HTTP {exc.response.status_code} for stop {stop_id}"
            ) from exc
        except httpx.HTTPError as exc:
            LOGGER.error(
                "Unable to reach RealTimeRail for stop %s: %s", stop_id, exc)
            raise RealtimeRailRequestError(
                f"Unable to reach RealTimeRail for stop {stop_id}: {exc}") from exc

        try:
            return response.json()
        except ValueError as exc:
            LOGGER.error(
                "RealTimeRail response was not valid JSON for stop %s", stop_id)
            raise RealtimeRailResponseError(
                "RealTimeRail response was not valid JSON.") from exc

    async def stop_snapshot(self, stop_id: str) -> StopRealtimeData:
        """Fetch live data for a specific stop identifier and normalize arrivals."""
        payload = await self.fetch_stop_payload(stop_id)

        try:
            return _parse_stop_payload(payload)
        except Exception as exc:
            LOGGER.debug(
                "Unexpected payload while parsing RealTimeRail data for stop %s: %s", stop_id, exc)
            raise RealtimeRailResponseError(
                f"Unexpected RealTimeRail payload structure: {exc}") from exc

    async def fetch_multiple_stops(self, stop_ids: List[str]) -> Dict[str, Any]:
        """
        Fetch real-time data for multiple stop IDs concurrently.
        Returns a dictionary mapping stop_id to their payload.
        Failed requests are logged but don't fail the entire batch.
        """
        import asyncio

        async def fetch_one(stop_id: str) -> tuple[str, Optional[Dict[str, Any]]]:
            try:
                payload = await self.fetch_stop_payload(stop_id)
                return (stop_id, payload)
            except (RealtimeRailRequestError, RealtimeRailResponseError) as exc:
                LOGGER.warning("Failed to fetch stop %s: %s", stop_id, exc)
                return (stop_id, None)

        # Fetch all stops concurrently
        results = await asyncio.gather(*[fetch_one(stop_id) for stop_id in stop_ids])

        # Return only successful fetches
        return {stop_id: payload for stop_id, payload in results if payload is not None}


def _parse_stop_payload(payload: Dict[str, Any]) -> StopRealtimeData:
    stop_info = payload.get("stop") or {}
    stop_id = str(stop_info.get("id") or stop_info.get("stop_id")
                  or payload.get("id") or payload.get("stop_id") or "")
    stop_name = stop_info.get("name") or stop_info.get("stop_name")

    routes: List[str] = []
    arrivals: List[SubwayArrival] = []

    for route_entry in payload.get("routes", []):
        if not isinstance(route_entry, dict):
            continue

        route_label = _extract_route_label(route_entry.get("route"))
        if route_label:
            routes.append(route_label)

        arrivals.extend(_extract_arrivals_from_route(route_entry, route_label))

    unique_routes = _deduplicate_preserve_order(routes)
    arrivals.sort(key=_arrival_sort_key)

    return StopRealtimeData(
        stop_id=stop_id or "unknown",
        stop_name=stop_name,
        routes=unique_routes,
        arrivals=arrivals,
    )


def _extract_route_label(route_info: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(route_info, dict):
        return None

    for key in ("short_name", "id", "name"):
        value = route_info.get(key)
        if value:
            return str(value).upper()
    return None


def _extract_arrivals_from_route(route_entry: Dict[str, Any], route_label: Optional[str]) -> List[SubwayArrival]:
    arrivals: List[SubwayArrival] = []

    for direction_entry in _extract_direction_entries(route_entry):
        direction_meta = (
            direction_entry.get("direction")
            or direction_entry.get("route_direction")
            or direction_entry.get("routeDirection")
            or {}
        )

        direction_name = _first_truthy(
            direction_meta.get("name"),
            direction_meta.get("long_name"),
            direction_meta.get("description"),
            direction_meta.get("id"),
        )

        headsign_hint = _first_truthy(
            direction_meta.get("headsign"),
            direction_meta.get("long_name"),
            direction_meta.get("description"),
        )

        for trip in _extract_trip_blocks(direction_entry):
            headsign = _first_truthy(
                trip.get("headsign"),
                (trip.get("trip") or {}).get("headsign"),
                headsign_hint,
            )

            for arrival_entry in _collect_arrival_entries(trip):
                arrival_time, is_real_time = _extract_arrival_time(
                    arrival_entry)
                if arrival_time is None:
                    continue

                arrivals.append(
                    SubwayArrival(
                        route_id=route_label or "",
                        direction=direction_name,
                        headsign=headsign,
                        arrival_time=arrival_time,
                        is_real_time=is_real_time,
                        raw=arrival_entry,
                    )
                )

    return arrivals


def _extract_direction_entries(route_entry: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for key in ("route_directions", "stop_directions", "directions"):
        candidates = route_entry.get(key)
        if isinstance(candidates, list):
            return (entry for entry in candidates if isinstance(entry, dict))
    return ()


def _extract_trip_blocks(direction_entry: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []

    for key in ("trips", "stop_times", "departures"):
        value = direction_entry.get(key)
        if isinstance(value, list):
            blocks.extend([item for item in value if isinstance(item, dict)])

    return blocks


def _collect_arrival_entries(trip: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    for key in ("stop_time_updates", "stop_time_update", "stop_times"):
        value = trip.get(key)
        if isinstance(value, list):
            entries.extend([item for item in value if isinstance(item, dict)])
        elif isinstance(value, dict):
            entries.append(value)

    if isinstance(trip.get("arrival"), dict):
        entries.append({"arrival": trip["arrival"]})

    for key in ("predicted_arrival_time", "scheduled_arrival_time", "arrival_time"):
        if key in trip:
            entries.append(
                {"time": trip[key], "predicted": key.startswith("predicted")})

    return entries


def _extract_arrival_time(entry: Dict[str, Any]) -> Tuple[Optional[datetime], bool]:
    if not isinstance(entry, dict):
        return None, False

    candidates: List[Tuple[Any, bool]] = []

    arrival = entry.get("arrival")
    if isinstance(arrival, dict):
        candidates.append(
            (arrival.get("time") or arrival.get("timestamp"), True))

    for key, realtime_flag in (
        ("predicted_arrival_time", True),
        ("predicted_time", True),
        ("expected_arrival_time", True),
        ("scheduled_arrival_time", False),
        ("scheduled_time", False),
        ("arrival_time", True),
        ("time", True),
    ):
        if key in entry:
            candidates.append((entry.get(key), realtime_flag))

    for raw_value, realtime_flag in candidates:
        if raw_value is None:
            continue
        dt = _parse_datetime(raw_value)
        if dt is not None:
            return dt, realtime_flag

    return None, False


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        if value <= 0:
            return None
        return datetime.fromtimestamp(value, tz=timezone.utc)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return datetime.fromtimestamp(int(text), tz=timezone.utc)

        # Handle ISO 8601 timestamps, optionally suffixed with Z
        text = text.replace("Z", "+00:00") if text.endswith("Z") else text
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    return None


def _arrival_sort_key(arrival: SubwayArrival) -> datetime:
    if arrival.arrival_time is None:
        return datetime.max.replace(tzinfo=timezone.utc)
    return arrival.arrival_time


def _first_truthy(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if value:
            return str(value)
    return None


def _deduplicate_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []

    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)

    return result
