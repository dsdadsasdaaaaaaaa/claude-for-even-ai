"""Setting the integration up from Home Assistant's own screens."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import time
from typing import Any

import anthropic
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigFlow, OptionsFlow
from homeassistant.const import CONF_API_KEY
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.network import NoURLAvailableError, get_url
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CHAT_PATH,
    CONF_EFFORT,
    CONF_FOLLOWUP,
    CONF_LAPTOP,
    CONF_LAPTOP_TOKEN,
    CONF_MODEL,
    CONF_NEW_LAPTOP_TOKEN,
    CONF_NEW_TOKEN,
    CONF_NOTES,
    CONF_NOTIFY,
    CONF_PROFILE,
    CONF_STT_KEY,
    CONF_STT_PROVIDER,
    CONF_TOKEN,
    DEFAULT_EFFORT,
    DEFAULT_FOLLOWUP,
    DEFAULT_MODEL,
    DOMAIN,
    EFFORTS,
    KEY_FILE,
    MODELS,
    NO_NOTIFY,
)
from .memory import shared_memory
from .speech import DEFAULT_PROVIDER, PROVIDERS
from .state import apply_note_edits, notes_text, read_state

_LOGGER = logging.getLogger(__name__)

TITLE = "Claude for Even AI"


def server(hass: HomeAssistant) -> str:
    """This server's address: the remote one when there is one, so it works away from home."""
    try:
        base = get_url(hass, allow_internal=False, prefer_cloud=True)
    except NoURLAvailableError:
        try:
            base = get_url(hass)
        except NoURLAvailableError:
            base = "http://homeassistant.local:8123"
    return base.rstrip("/")


def endpoint(hass: HomeAssistant) -> str:
    """The address to give the Even app."""
    return f"{server(hass)}{CHAT_PATH}"


def phones(hass: HomeAssistant) -> dict[str, str]:
    """The phones with the Home Assistant app that can be sent a notification."""
    found = sorted(name for name in hass.services.async_services().get("notify", {}) if name.startswith("mobile_app_"))
    return {NO_NOTIFY: "Don't send notifications", **{name: name.removeprefix("mobile_app_").replace("_", " ") for name in found}}


async def check_key(hass: HomeAssistant, api_key: str) -> str | None:
    """Ask Anthropic whether the key works. Returns what went wrong, or None."""

    def _client() -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(api_key=api_key, max_retries=0, timeout=15.0)

    try:
        client = await hass.async_add_executor_job(_client)
        await client.models.list(limit=1)
    except anthropic.AuthenticationError:
        return "invalid_auth"
    except (anthropic.APIConnectionError, TimeoutError):
        return "cannot_connect"
    except anthropic.APIStatusError as err:
        _LOGGER.warning("Anthropic answered %s while checking the key", err.status_code)
        return "invalid_auth" if err.status_code in (401, 403) else "unknown"
    except Exception:  # noqa: BLE001 - shown to the user as "unknown", with the detail in the log
        _LOGGER.exception("Could not check the Anthropic key")
        return "unknown"
    return None


def _read_key_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _remove_key_file(path: str) -> None:
    with contextlib.suppress(OSError):
        os.remove(path)


def _choices(options: dict[str, str]) -> SelectSelector:
    return SelectSelector(
        SelectSelectorConfig(
            options=[{"value": value, "label": label} for value, label in options.items()],
            mode=SelectSelectorMode.DROPDOWN,
        )
    )


class ClaudeAgentConfigFlow(ConfigFlow, domain=DOMAIN):
    """Asks for the Anthropic key, then shows what to enter in the Even app."""

    VERSION = 1

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        # A key left in the config folder saves typing a long one on a phone.
        waiting = await self.hass.async_add_executor_job(_read_key_file, self.hass.config.path(KEY_FILE))

        errors: dict[str, str] = {}
        if user_input is not None:
            api_key = (user_input.get(CONF_API_KEY) or "").strip() or waiting
            problem = await check_key(self.hass, api_key) if api_key else "no_key"
            if problem is None:
                # The key the Even app will present. Made here so it is long and never typed by hand.
                self._data = {CONF_API_KEY: api_key, CONF_TOKEN: secrets.token_urlsafe(32), CONF_LAPTOP_TOKEN: secrets.token_urlsafe(32)}
                return await self.async_step_connect()
            errors["base"] = problem

        box = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({(vol.Optional if waiting else vol.Required)(CONF_API_KEY): box}),
            errors=errors,
            description_placeholders={
                "waiting": "A key is already waiting on this server, so you can leave the box empty and press Submit." if waiting else ""
            },
        )

    async def async_step_connect(self, user_input: dict[str, Any] | None = None):
        if user_input is not None:
            # The key now lives in Home Assistant's own settings; do not leave a copy lying in a file.
            await self.hass.async_add_executor_job(_remove_key_file, self.hass.config.path(KEY_FILE))
            return self.async_create_entry(title=TITLE, data=self._data)
        return self.async_show_form(
            step_id="connect",
            data_schema=vol.Schema({}),
            description_placeholders={"endpoint": endpoint(self.hass), "token": self._data[CONF_TOKEN]},
        )

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None):
        """Replace the Anthropic key, keeping everything else, the Even app's key and the laptop's included."""
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            api_key = (user_input.get(CONF_API_KEY) or "").strip()
            problem = await check_key(self.hass, api_key) if api_key else "no_key"
            if problem is None:
                return self.async_update_reload_and_abort(entry, data_updates={CONF_API_KEY: api_key})
            errors["base"] = problem
        box = TextSelector(TextSelectorConfig(type=TextSelectorType.PASSWORD))
        return self.async_show_form(step_id="reconfigure", data_schema=vol.Schema({vol.Required(CONF_API_KEY): box}), errors=errors)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return ClaudeAgentOptionsFlow(config_entry)


class ClaudeAgentOptionsFlow(OptionsFlow):
    """What Claude knows about you and how it answers; also shows the Even app details again."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        self._entry = config_entry
        # The notes as the box showed them, so only the lines changed in it are applied.
        self._notes_shown = ""

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        memory = shared_memory(self.hass)
        if user_input is not None:
            # The notes belong to the agent's memory, not the settings; an emptied box clears them. Only the lines
            # changed in the box are applied, under the agent's own lock, so a note Claude saved or removed while
            # this screen was open is kept as it is.
            submitted = user_input.pop(CONF_NOTES, "") or ""
            lock = self.hass.data.setdefault(DOMAIN, {}).setdefault("turn_lock", asyncio.Lock())
            async with lock:
                state = read_state(await memory.read())
                state["notes"] = apply_note_edits(self._notes_shown, submitted, state["notes"], time.time())
                await memory.write(state)
            # A key that may have got out is replaced; the reload that follows saving uses the new one.
            renewed = {}
            if user_input.pop(CONF_NEW_TOKEN, False):
                renewed[CONF_TOKEN] = secrets.token_urlsafe(32)
            if user_input.pop(CONF_NEW_LAPTOP_TOKEN, False):
                renewed[CONF_LAPTOP_TOKEN] = secrets.token_urlsafe(32)
            if renewed:
                self.hass.config_entries.async_update_entry(self._entry, data={**self._entry.data, **renewed})
            return self.async_create_entry(title="", data=user_input)

        notes = read_state(await memory.read())["notes"]
        self._notes_shown = notes_text(notes)
        current = self._entry.options
        targets = phones(self.hass)
        # A phone chosen earlier stays selectable even while its app is not connected.
        notify = current.get(CONF_NOTIFY, next((name for name in targets if name != NO_NOTIFY), NO_NOTIFY))
        targets.setdefault(notify, notify)
        schema = vol.Schema(
            {
                vol.Optional(CONF_PROFILE, description={"suggested_value": current.get(CONF_PROFILE, "")}): TextSelector(TextSelectorConfig(multiline=True)),
                vol.Required(CONF_MODEL, default=current.get(CONF_MODEL, DEFAULT_MODEL)): _choices(MODELS),
                vol.Required(CONF_EFFORT, default=current.get(CONF_EFFORT, DEFAULT_EFFORT)): _choices(EFFORTS),
                vol.Required(CONF_FOLLOWUP, default=current.get(CONF_FOLLOWUP, DEFAULT_FOLLOWUP)): NumberSelector(
                    NumberSelectorConfig(min=1, max=60, step=1, mode=NumberSelectorMode.BOX, unit_of_measurement="min")
                ),
                vol.Optional(CONF_NOTES, description={"suggested_value": self._notes_shown}): TextSelector(TextSelectorConfig(multiline=True)),
                vol.Required(CONF_STT_PROVIDER, default=current.get(CONF_STT_PROVIDER, DEFAULT_PROVIDER)): _choices(
                    {name: preset["label"] for name, preset in PROVIDERS.items()}
                ),
                vol.Optional(CONF_STT_KEY, description={"suggested_value": current.get(CONF_STT_KEY, "")}): TextSelector(
                    TextSelectorConfig(type=TextSelectorType.PASSWORD)
                ),
                vol.Required(CONF_LAPTOP, default=current.get(CONF_LAPTOP, True)): BooleanSelector(),
                vol.Required(CONF_NOTIFY, default=notify): _choices(targets),
                vol.Required(CONF_NEW_TOKEN, default=False): BooleanSelector(),
                vol.Required(CONF_NEW_LAPTOP_TOKEN, default=False): BooleanSelector(),
            }
        )
        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            description_placeholders={
                "endpoint": endpoint(self.hass),
                "token": self._entry.data[CONF_TOKEN],
                "server": server(self.hass),
                "laptop_token": self._entry.data.get(CONF_LAPTOP_TOKEN, "(restart Home Assistant to make one)"),
            },
        )
