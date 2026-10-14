"""Claude for Even AI: your own Claude behind the Even AI button, served from Home Assistant.

The Even app is given an address on this Home Assistant server and a key.
What you say on the glasses arrives here, is answered by Claude through the
Claude API, and goes back to the glasses.
"""

from __future__ import annotations

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
            "usage": {"input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens},
        }

    return ask


def notifier(hass: HomeAssistant, service: str) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """How the wearer hears that a laptop job ended: a notification on their phone, which the glasses can show."""
    titles = {DONE: "Laptop: done", FAILED: "Laptop: didn't work", EXPIRED: "Laptop: job dropped"}

    async def notify(job: dict[str, Any]) -> None:
        if service == NO_NOTIFY or not hass.services.has_service("notify", service):
            return
        message = job["result"] if len(job["result"]) <= NOTIFY_CHARS else job["result"][: NOTIFY_CHARS - 1].rstrip() + "…"
        try:
            await hass.services.async_call("notify", service, {"title": titles.get(job["status"], "Laptop"), "message": message})
        except Exception as err:  # noqa: BLE001 - a missed notification must not lose the job's outcome
            _LOGGER.warning("Could not send the laptop's notification: %s", err)

    return notify


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Start answering with the key and settings that were entered."""
    if not entry.data.get(CONF_LAPTOP_TOKEN):
        # Set up before the laptop could be used; give it its own key now.
        hass.config_entries.async_update_entry(entry, data={**entry.data, CONF_LAPTOP_TOKEN: secrets.token_urlsafe(32)})
    client = await hass.async_add_executor_job(partial(claude_client, entry.data[CONF_API_KEY]))
    shared = hass.data.setdefault(DOMAIN, {})
    # The queue outlives a change of settings, so a job waiting for the laptop is not lost.
    laptop: Laptop = shared.get("laptop") or Laptop(store=shared_memory(hass, "laptop"), now=time.time)
    laptop.on_finished = notifier(hass, entry.options.get(CONF_NOTIFY, NO_NOTIFY))
    laptop_on = entry.options.get(CONF_LAPTOP, True)
    shared.update(
        laptop=laptop,
        laptop_on=laptop_on,
        laptop_token=entry.data[CONF_LAPTOP_TOKEN],
        stream_always=entry.options.get(CONF_STREAM, False),
        stt_provider=entry.options.get(CONF_STT_PROVIDER, DEFAULT_PROVIDER),
        stt_key=entry.options.get(CONF_STT_KEY, ""),
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
    return True
