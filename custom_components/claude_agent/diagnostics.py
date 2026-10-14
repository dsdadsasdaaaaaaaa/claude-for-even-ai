"""The download offered on the integration's page, for working out a problem.

It holds the settings, the outline of the last few requests from the Even
app, how many notes and earlier conversations are kept, when the laptop's jobs
ran, and the session app's own log. It also holds the last 10 things the
session app heard, so mishearings can be spotted. It leaves out every key, the
profile, the notes themselves, the conversations, and what the laptop was asked
to do.
"""

from __future__ import annotations

from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .agent import read_state
from .const import CONF_EFFORT, CONF_FOLLOWUP, CONF_MODEL, CONF_NOTIFY, CONF_PROFILE, DEFAULT_EFFORT, DEFAULT_FOLLOWUP, DEFAULT_MODEL, DOMAIN, NO_NOTIFY
from .memory import shared_memory


async def async_get_config_entry_diagnostics(hass: HomeAssistant, entry: ConfigEntry) -> dict[str, Any]:
    shared = hass.data.get(DOMAIN, {})
    laptop = shared.get("laptop")
    memory = read_state(await shared_memory(hass).read())
    return {
        "running": shared.get("agent") is not None,
        "model": entry.options.get(CONF_MODEL, DEFAULT_MODEL),
        "effort": entry.options.get(CONF_EFFORT, DEFAULT_EFFORT),
        "followup_minutes": entry.options.get(CONF_FOLLOWUP, DEFAULT_FOLLOWUP),
        "profile_characters": len(entry.options.get(CONF_PROFILE, "")),
        "notes": len(memory["notes"]),
        "conversation_turns": len(memory["thread"]["turns"]) if memory["thread"] else 0,
        "earlier_conversations": len(memory["earlier"]),
        "speech_to_text": {"provider": shared.get("stt_provider"), "key_set": bool(shared.get("stt_key"))},
        "open_sessions": len(memory["sessions"]),
        "recent_requests": list(shared.get("shapes", [])),
        # How long each recent answer took and what went into it (searches, tools), without what was said.
        "recent_answers": list(shared.get("timings", [])),
        # How each recent request from the glasses app went (answered, nothing heard, wrong key, and so on), with what
        # speech-to-text heard, so a mishearing can be spotted.
        "session_app_requests": list(shared.get("session_requests", [])),
        # The glasses app's own log: what it did on the glasses (presses, the mic, recordings), never what was said.
        "session_app_log": list(shared.get("session_app_log", [])),
        # How the laptop's jobs went, without what they asked for or found.
        "laptop": {
            "switched_on": bool(shared.get("laptop_on")),
            "notify": entry.options.get(CONF_NOTIFY, "(not chosen yet: the first phone)"),
            "notify_available": hass.services.has_service("notify", entry.options[CONF_NOTIFY])
            if entry.options.get(CONF_NOTIFY, NO_NOTIFY) != NO_NOTIFY
            else None,
            "online": laptop.online() if laptop else False,
            "last_seen": laptop.seen if laptop else None,
            "jobs": [
                {
                    "status": job["status"],
                    "follow_up": job["follow_up"],
                    "at": job["at"],
                    "started": job["started"],
                    "finished": job["finished"],
                    **job["stats"],
                }
                for job in (await laptop.recent(24 * 60 * 60) if laptop else [])
            ],
        },
    }
