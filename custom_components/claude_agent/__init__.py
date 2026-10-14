"""Claude for Even AI: your own Claude behind the Even AI button, served from Home Assistant.

The Even app is given an address on this Home Assistant server and a key.
What you say on the glasses arrives here, is answered by Claude through the
Claude API, and goes back to the glasses.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
import logging
import secrets
import time
from typing import Any

import anthropic

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .agent import Agent
from .const import (
    CONF_EFFORT,
    CONF_FOLLOWUP,
    CONF_LAPTOP,
    CONF_LAPTOP_TOKEN,
    CONF_MODEL,
    CONF_NOTIFY,
    CONF_PROFILE,
    CONF_STREAM,
    CONF_STT_KEY,
    CONF_STT_PROVIDER,
    CONF_TOKEN,
    DEFAULT_EFFORT,
    DEFAULT_FOLLOWUP,
    DEFAULT_MODEL,
    DOMAIN,
    NO_NOTIFY,
)
from .laptop import DONE, EXPIRED, FAILED, Laptop
from .memory import shared_memory
from .speech import DEFAULT_PROVIDER
from .views import register_views

_LOGGER = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Enough for the glasses' notification view; the whole outcome is still there to ask the agent about.
NOTIFY_CHARS = 300


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
        if request.get("thinking"):
            extra_body["thinking"] = request["thinking"]
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


def notifier(hass: HomeAssistant, chosen: str | None) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """How the wearer hears that a laptop job ended: a notification on their phone, which the glasses can show.

    `chosen` is the service picked under Configure, or None when nothing was saved there yet.
    """
    titles = {DONE: "Laptop: done", FAILED: "Laptop: didn't work", EXPIRED: "Laptop: job dropped"}

    async def notify(job: dict[str, Any]) -> None:
        service = chosen if chosen is not None else first_phone(hass)
        if service is None or service == NO_NOTIFY:
            return
        if not hass.services.has_service("notify", service):
            # The phone app was signed out or renamed: say so, since this is the only way they hear how a job went.
            _LOGGER.warning("The laptop's result was not sent: notify.%s is not available", service)
            return
        text = job["result"]
        if job["status"] != DONE:
            # Messages written for the job, rather than by Claude, need to say which job they are about.
            text = f'"{_cut(" ".join(job["task"].split()), 80)}": {text}'
        try:
            await hass.services.async_call("notify", service, {"title": titles.get(job["status"], "Laptop"), "message": _cut(text, NOTIFY_CHARS)})
        except Exception as err:  # noqa: BLE001 - a missed notification must not lose the job's outcome
            _LOGGER.warning("Could not send the laptop's notification: %s", err)

    return notify


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
    laptop_on = entry.options.get(CONF_LAPTOP, True)
    if not laptop_on:
        # Switched off: stop what is running and drop what is waiting, as the setting promises.
        await laptop.cancel("Called off: the laptop was switched off under Configure.")
    shared.update(
        laptop=laptop,
        laptop_on=laptop_on,
        laptop_token=entry.data[CONF_LAPTOP_TOKEN],
        stream_always=entry.options.get(CONF_STREAM, False),
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
        log=_LOGGER.debug,
        # Shared with the settings screen, so a hand edit of the notes never crosses an answer in progress.
        lock=shared.setdefault("turn_lock", asyncio.Lock()),
    )
    # Addresses cannot be taken off the web server again, so they are added once and look the agent up each time.
    if not shared.get("views"):
        register_views(hass)
        shared["views"] = True
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
