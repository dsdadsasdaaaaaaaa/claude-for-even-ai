"""Claude for Even AI: your own Claude behind the Even AI button, served from Home Assistant.

The Even app is given an address on this Home Assistant server and a key.
What you say on the glasses arrives here, is answered by Claude through the
Claude API, and goes back to the glasses.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from functools import partial
import logging
import secrets
import time
from typing import Any

import anthropic

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.event import async_track_time_change, async_track_time_interval
from homeassistant.util import dt as dt_util

from .agent import Agent
from .const import (
    CONF_DAYFLOW_SECRET,
    CONF_DAYFLOW_URL,
    CONF_EFFORT,
    CONF_FOLLOWUP,
    CONF_LAPTOP,
    CONF_LAPTOP_TOKEN,
    CONF_MODEL,
    CONF_NOTIFY,
    CONF_PROFILE,
    CONF_STT_KEY,
    CONF_STT_PROVIDER,
    CONF_TOKEN,
    DEFAULT_EFFORT,
    DEFAULT_FOLLOWUP,
    DEFAULT_MODEL,
    DOMAIN,
    NO_NOTIFY,
)
from .dayflow import Schedule
from .laptop import DONE, EXPIRED, FAILED, Laptop
from .memory import shared_memory
from .reminders import Reminders
from .speech import DEFAULT_PROVIDER
from .views import register_views

_LOGGER = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Enough for the glasses' notification view; the whole outcome is still there to ask the agent about.
NOTIFY_CHARS = 300
# How often reminders that have come due are sent, and when the night's look over the day's conversations runs.
REMINDER_CHECK_SECONDS = 20
REFLECT_AT = {"hour": 3, "minute": 0, "second": 0}


def claude_client(api_key: str) -> anthropic.AsyncAnthropic:
    """A client for the Claude API. Building one reads certificates from disk, so call this off the event loop."""
    return anthropic.AsyncAnthropic(api_key=api_key, max_retries=1, timeout=45.0)


def make_ask(client: anthropic.AsyncAnthropic) -> Callable[[dict[str, Any], Callable[[str | None], Awaitable[None]]], Awaitable[dict[str, Any]]]:
    """How the agent sends one request to Claude."""

    async def ask(request: dict[str, Any], on_text: Callable[[str | None], Awaitable[None]]) -> dict[str, Any]:
        # Newer request fields go in the body as they are, so this works with whichever
        # version of the Anthropic package this Home Assistant has installed.
        extra_body: dict[str, Any] = {"output_config": {"effort": request["effort"]}}
        options: dict[str, Any] = {}
        if request["fallback"]:
            extra_body["fallbacks"] = "default"
            options["betas"] = [FALLBACK_BETA]
        if request.get("tool_choice"):
            options["tool_choice"] = request["tool_choice"]
        async with client.beta.messages.stream(
            model=request["model"],
            max_tokens=request["max_tokens"],
            system=request["system"],
            tools=request["tools"],
            messages=request["messages"],
            extra_body=extra_body,
            **options,
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    await on_text(event.text)
                elif event.type == "content_block_start" and event.content_block.type == "text":
                    # A new stretch of text, as after a search.
                    await on_text(None)
            message = await stream.get_final_message()
        return {
            "model": message.model,
            "stop_reason": message.stop_reason,
            "content": message.content,
            "tool_calls": [{"id": block.id, "name": block.name, "input": block.input} for block in message.content if block.type == "tool_use"],
            "searches": sum(1 for block in message.content if block.type == "server_tool_use"),
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                # What the prompt cache saved and what it stored, so the diagnostics show whether it works.
                "cache_read_input_tokens": getattr(message.usage, "cache_read_input_tokens", None) or 0,
                "cache_creation_input_tokens": getattr(message.usage, "cache_creation_input_tokens", None) or 0,
            },
        }

    return ask


def first_phone(hass: HomeAssistant) -> str | None:
    """The phone that Configure offers first, used until a choice is saved there."""
    phones = sorted(name for name in hass.services.async_services().get("notify", {}) if name.startswith("mobile_app_"))
    return phones[0] if phones else None


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


async def send_notification(hass: HomeAssistant, chosen: str | None, title: str, message: str) -> None:
    """A notification on the wearer's phone, which the glasses can show. `chosen` is the service picked under
    Configure, or None when nothing was saved there yet."""
    service = chosen if chosen is not None else first_phone(hass)
    if service is None or service == NO_NOTIFY:
        return
    if not hass.services.has_service("notify", service):
        # The phone app was signed out or renamed: say so, since this is the only way they hear.
        _LOGGER.warning("A notification was not sent: notify.%s is not available", service)
        return
    try:
        await hass.services.async_call("notify", service, {"title": title, "message": _cut(message, NOTIFY_CHARS)})
    except Exception as err:  # noqa: BLE001 - a missed notification must not lose what it was about
        _LOGGER.warning("Could not send a notification: %s", err)


def notifier(hass: HomeAssistant, chosen: str | None) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """How the wearer hears that a laptop job ended."""
    titles = {DONE: "Laptop: done", FAILED: "Laptop: didn't work", EXPIRED: "Laptop: job dropped"}

    async def notify(job: dict[str, Any]) -> None:
        text = job["result"]
        if job["status"] != DONE:
            # Messages written for the job, rather than by Claude, need to say which job they are about.
            text = f'"{_cut(" ".join(job["task"].split()), 80)}": {text}'
        await send_notification(hass, chosen, titles.get(job["status"], "Laptop"), text)

    return notify


def reminder_notifier(hass: HomeAssistant, chosen: str | None) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """How a reminder reaches the wearer when its time comes."""

    async def deliver(reminder: dict[str, Any]) -> None:
        await send_notification(hass, chosen, "Reminder", reminder["text"])

    return deliver


async def _close_later(client: anthropic.AsyncAnthropic) -> None:
    """Closes a client the settings replaced, once any answer it was still giving has had time to finish."""
    await asyncio.sleep(120)
    await client.close()


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Start answering with the key and settings that were entered."""
    if not entry.data.get(CONF_LAPTOP_TOKEN):
        # Set up before the laptop could be used; give it its own key now.
        hass.config_entries.async_update_entry(entry, data={**entry.data, CONF_LAPTOP_TOKEN: secrets.token_urlsafe(32)})
    client = await hass.async_add_executor_job(partial(claude_client, entry.data[CONF_API_KEY]))
    shared = hass.data.setdefault(DOMAIN, {})
    # The queue outlives a change of settings, so a job waiting for the laptop is not lost.
    laptop: Laptop = shared.get("laptop") or Laptop(store=shared_memory(hass, "laptop"), now=time.time)
    laptop.on_finished = notifier(hass, entry.options.get(CONF_NOTIFY))
    reminders: Reminders = shared.get("reminders") or Reminders(store=shared_memory(hass, "reminders"), now=time.time)
    reminders.deliver = reminder_notifier(hass, entry.options.get(CONF_NOTIFY))
    dayflow_url = (entry.options.get(CONF_DAYFLOW_URL) or "").strip()
    dayflow_secret = (entry.options.get(CONF_DAYFLOW_SECRET) or "").strip()
    schedule = Schedule(http=async_get_clientsession(hass), url=dayflow_url, secret=dayflow_secret, now=dt_util.now) if dayflow_url and dayflow_secret else None
    laptop_on = entry.options.get(CONF_LAPTOP, True)
    if not laptop_on:
        # Switched off: stop what is running and drop what is waiting, as the setting promises.
        await laptop.cancel("Called off: the laptop was switched off under Configure.")
    shared.update(
        laptop=laptop,
        reminders=reminders,
        laptop_on=laptop_on,
        laptop_token=entry.data[CONF_LAPTOP_TOKEN],
        stt_provider=entry.options.get(CONF_STT_PROVIDER, DEFAULT_PROVIDER),
        stt_key=entry.options.get(CONF_STT_KEY, ""),
        client=client,
    )
    shared["agent"] = Agent(
        ask_claude=make_ask(client),
        token=entry.data[CONF_TOKEN],
        store=shared_memory(hass),
        now=dt_util.now,
        model=entry.options.get(CONF_MODEL, DEFAULT_MODEL),
        effort=entry.options.get(CONF_EFFORT, DEFAULT_EFFORT),
        profile=entry.options.get(CONF_PROFILE, ""),
        thread_seconds=60 * float(entry.options.get(CONF_FOLLOWUP, DEFAULT_FOLLOWUP)),
        laptop=laptop if laptop_on else None,
        reminders=reminders,
        schedule=schedule,
        log=_LOGGER.debug,
        # Shared with the settings screen, so a hand edit of the notes never crosses an answer in progress.
        lock=shared.setdefault("turn_lock", asyncio.Lock()),
    )
    # Addresses cannot be taken off the web server again, so they are added once and look the agent up each time.
    if not shared.get("views"):
        register_views(hass)
        shared["views"] = True
    # The clocks, likewise: they look up whatever is running when they go off.
    if not shared.get("clocks"):
        shared["clocks"] = True

        async def deliver_due(now: datetime) -> None:
            waiting = hass.data.get(DOMAIN, {}).get("reminders")
            if waiting is None:
                return
            try:
                await waiting.due()
            except Exception as err:  # noqa: BLE001 - a reminder that fails to send must not stop the next check
                _LOGGER.warning("Could not send a reminder: %s", err)

        async def reflect(now: datetime) -> None:
            running = hass.data.get(DOMAIN, {}).get("agent")
            if running is None:
                return
            try:
                outcome = await running.reflect()
            except Exception as err:  # noqa: BLE001 - the night's look over the day is a nicety
                _LOGGER.warning("The look over the day's conversations failed: %s", err)
                return
            if outcome["looked_at"]:
                _LOGGER.info("Looked over %s conversation(s) from the day and kept %s note(s)", outcome["looked_at"], len(outcome["added"]))

        async_track_time_interval(hass, deliver_due, timedelta(seconds=REMINDER_CHECK_SECONDS))
        async_track_time_change(hass, reflect, **REFLECT_AT)
    entry.async_on_unload(entry.add_update_listener(_settings_changed))
    return True


async def _settings_changed(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Stop answering. The addresses stay up and report that nothing is set up."""
    shared = hass.data.get(DOMAIN, {})
    shared.pop("agent", None)
    shared["laptop_on"] = False
    client = shared.pop("client", None)
    if client is not None:
        hass.async_create_background_task(_close_later(client), "claude_agent: close the old Claude client")
    return True
