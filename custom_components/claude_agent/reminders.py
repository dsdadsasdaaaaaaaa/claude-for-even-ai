"""Reminders the wearer sets by voice: "remind me at six to call mum". Each is delivered as a notification on their
phone (and the glasses, when the Even app shows those) at its time.

The agent adds and cancels them; Home Assistant asks every so often which have come due and sends them. Nothing here
knows about Home Assistant, so it can be tested on its own.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import secrets
from typing import Any, Protocol

MAX_REMINDERS = 50
MAX_TEXT_CHARS = 200
# Sent reminders are kept this long, so a late look at the list still shows them.
SENT_KEPT_SECONDS = 24 * 60 * 60

Reminder = dict[str, Any]


class Store(Protocol):
    async def read(self) -> Any: ...

    async def write(self, state: dict[str, Any]) -> None: ...


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def read_reminders(raw: Any) -> list[Reminder]:
    """Whatever was stored, coerced into the shape relied on."""
    data = raw if isinstance(raw, dict) else {}
    stored = data.get("reminders")
    return [
        {"id": one["id"], "text": one["text"], "at": float(one["at"]), "sent": one.get("sent") is True}
        for one in (stored if isinstance(stored, list) else [])
        if isinstance(one, dict) and isinstance(one.get("id"), str) and isinstance(one.get("text"), str) and _number(one.get("at"))
    ][-MAX_REMINDERS:]


class Reminders:
    """The reminders, kept in `store`. `deliver(reminder)` is called for each as it comes due."""

    def __init__(self, *, store: Store, now: Callable[[], float], deliver: Callable[[Reminder], Awaitable[None]] | None = None) -> None:
        self._store = store
        self._now = now
        self.deliver = deliver
        self._all: list[Reminder] = []
        self._loaded = False
        self._lock = asyncio.Lock()

    async def _load(self) -> None:
        if not self._loaded:
            self._all = read_reminders(await self._store.read())
            self._loaded = True

    async def _save(self) -> None:
        since = self._now() - SENT_KEPT_SECONDS
        self._all = [one for one in self._all if not one["sent"] or one["at"] >= since][-MAX_REMINDERS:]
        await self._store.write({"reminders": self._all})

    async def add(self, text: str, at: float) -> Reminder:
        """Sets a reminder for `at` (epoch seconds). The same text at the same minute is not set twice."""
        async with self._lock:
            await self._load()
            text = " ".join(text.split())[:MAX_TEXT_CHARS]
            for one in self._all:
                if not one["sent"] and one["text"].lower() == text.lower() and abs(one["at"] - at) < 60:
                    return dict(one)
            reminder: Reminder = {"id": secrets.token_hex(4), "text": text, "at": at, "sent": False}
            self._all.append(reminder)
            await self._save()
            return dict(reminder)

    async def pending(self) -> list[Reminder]:
        """The reminders still to come, soonest first."""
        async with self._lock:
            await self._load()
            return sorted((dict(one) for one in self._all if not one["sent"]), key=lambda one: one["at"])

    async def cancel(self, number: int) -> Reminder | None:
        """Drops the `number`th pending reminder (counting from 1, soonest first). None when there is no such one."""
        async with self._lock:
            await self._load()
            waiting = sorted((one for one in self._all if not one["sent"]), key=lambda one: one["at"])
            if number < 1 or number > len(waiting):
                return None
            gone = waiting[number - 1]
            self._all = [one for one in self._all if one is not gone]
            await self._save()
            return dict(gone)

    async def due(self) -> list[Reminder]:
        """Delivers every reminder whose time has come, and returns them."""
        async with self._lock:
            await self._load()
            now = self._now()
            ready = [one for one in self._all if not one["sent"] and one["at"] <= now]
            if not ready:
                return []
            for one in ready:
                one["sent"] = True
            await self._save()
        for one in ready:
            if self.deliver:
                await self.deliver(dict(one))
        return [dict(one) for one in ready]
