"""Their day, read from DayFlow's live link: the relay address (ending in /relay) and its secret, pasted under
Configure. Only the school timetable, their own tasks and the phone calendar survive the reading (see planner.py).

Nothing here knows about Home Assistant, so it can be tested on its own.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import logging
from typing import Any

from .planner import DATE, Planner, build_brief, describe_day, from_snapshot

_LOGGER = logging.getLogger(__name__)

# A copy of the planner is used for this long before it is read again.
REFRESH_SECONDS = 10 * 60


class DayflowError(Exception):
    """The planner could not be read. `not_shared`: DayFlow answered but shares no planner (the live link is off)."""

    def __init__(self, message: str, status: int | None = None, not_shared: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.not_shared = not_shared


async def fetch_planner(http: Any, url: str, secret: str, now: float) -> Planner:
    """Reads the planner from the relay. `http` is an aiohttp client session."""
    import aiohttp  # Home Assistant always has it; imported here so the rest can be tested without it.

    try:
        async with http.get(
            f"{url.rstrip('/')}/data",
            headers={"Authorization": f"Bearer {secret}"},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            status = response.status
            try:
                body = await response.json(content_type=None)
            except ValueError:
                body = None
    except (aiohttp.ClientError, TimeoutError) as err:
        raise DayflowError("Could not reach DayFlow") from err
    if status == 501:
        raise DayflowError("The live link is switched off in DayFlow", status, True)
    if status in (401, 403):
        raise DayflowError("DayFlow did not accept the secret", status)
    if status >= 400:
        raise DayflowError(f"DayFlow answered {status}", status)
    # With the live link off the relay still answers, with no data. That is "not shared", never an empty week.
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict):
        raise DayflowError("DayFlow is not sharing its planner (is the live link on?)", status, True)
    return from_snapshot(data, now)


class Schedule:
    """Their schedule as Claude asks for it, read from DayFlow now and then."""

    def __init__(self, *, http: Any, url: str, secret: str, now: Callable[[], datetime]) -> None:
        self._http = http
        self._url = url
        self._secret = secret
        self._now = now
        self._planner: Planner | None = None

    async def _current(self) -> Planner:
        moment = self._now().timestamp()
        if self._planner is None or moment - self._planner.fetched_at > REFRESH_SECONDS:
            self._planner = await fetch_planner(self._http, self._url, self._secret, moment)
        return self._planner

    async def text(self, day: str | None = None) -> str:
        """The brief for today and the week ahead, or one day in full; or what went wrong, for Claude to pass on."""
        if day and not DATE.match(day):
            return "The day must be written as YYYY-MM-DD."
        try:
            planner = await self._current()
        except DayflowError as err:
            _LOGGER.warning("The planner could not be read: %s", err)
            return f"Their planner could not be read: {err}. Tell them so in a few words."
        return describe_day(planner, day) if day else build_brief(planner, self._now())
