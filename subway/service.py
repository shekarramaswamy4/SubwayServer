import logging
from dataclasses import dataclass
from typing import Optional

from .query import ClaudeConfigurationError, ClaudeQueryResolver, ResolverResult, StationResolutionError


logger = logging.getLogger("subway-service")


@dataclass(slots=True)
class QueryOutcome:
    message: str
    needs_clarification: bool


async def process_subway_query(
    query: str,
    lat: Optional[str],
    lon: Optional[str],
    *,
    resolver: ClaudeQueryResolver,
) -> Optional[QueryOutcome]:
    """
    Core workflow: delegate the user's message to Claude. Claude is responsible for mapping the
    station, calling the RealTimeRail tool, and formatting the rider-facing response.
    """
    if (not query or not query.strip()) and (not lat and not lon):
        return None
    normalized_query = query.strip()

    print("Received subway query:", query, lat, lon)

    try:
        result = await resolver.resolve(normalized_query, lat, lon)
    except ClaudeConfigurationError:
        logger.debug(
            "Skipping subway lookup; ANTHROPIC_API_KEY is not configured.")
        return None
    except StationResolutionError as exc:
        logger.warning(
            "Unable to resolve station from query '%s': %s", normalized_query, exc)
        return QueryOutcome(
            message="I couldn't figure out which station you meant. Could you rephrase it?",
            needs_clarification=True,
        )

    if _needs_clarification(result):
        clarification = result.clarification or f"Which station did you mean when you said '{normalized_query}'?"
        return QueryOutcome(
            message=clarification,
            needs_clarification=True,
        )

    return QueryOutcome(
        message=result.message or "Thanks! Message received.",
        needs_clarification=False,
    )


def _needs_clarification(result: ResolverResult) -> bool:
    return result.status == "clarify"
