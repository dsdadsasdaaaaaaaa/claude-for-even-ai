"""Where the agent keeps things in Home Assistant's own storage.

The notes and conversations live in `.storage/claude_agent.memory` and the
laptop's jobs in `.storage/claude_agent.laptop`. One instance of each is shared,
so the settings screen and the agent never hold different copies.
"""

from __future__ import annotations

from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN


class Memory:
    """Something kept in Home Assistant's own storage."""

    def __init__(self, hass: HomeAssistant, name: str) -> None:
        self._store: Store[dict[str, Any]] = Store(hass, 1, f"{DOMAIN}.{name}", private=True)

    async def read(self) -> Any:
        return await self._store.async_load()

    async def write(self, state: dict[str, Any]) -> None:
        await self._store.async_save(state)


def shared_memory(hass: HomeAssistant, name: str = "memory") -> Memory:
    """The one instance for `name`, made the first time it is asked for."""
    shared = hass.data.setdefault(DOMAIN, {})
    key = f"store_{name}"
    if key not in shared:
        shared[key] = Memory(hass, name)
    return shared[key]
