"""The web addresses the Even app talks to.

They sit under /api/claude_agent on Home Assistant's own web server, so they
are reachable wherever Home Assistant is, including through its remote address.
Home Assistant's login is not used here: the Even app presents the key that
was shown when the integration was set up.
"""

from __future__ import annotations

import asyncio
from collections import deque
import hmac
from http import HTTPStatus
import json
import logging
import re
import time
from typing import Any

from aiohttp import web

from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from . import speech
from .agent import FAILED, LIMIT, NO_USAGE, Agent, BadRequest, chunk, completion, convert_messages, new_id, over_limit
from .const import (
    API_BASE,
    CHAT_PATH,
    DOMAIN,
    LAPTOP_DONE_PATH,
    LAPTOP_NEXT_PATH,
    LAPTOP_START_PATH,
    LAPTOP_WATCH_PATH,
    MODELS_PATH,
    SESSION_END_PATH,
    SESSION_TURN_PATH,
)
from .laptop import Laptop

_LOGGER = logging.getLogger(__name__)

# How many request outlines, and answer timings, are kept for the diagnostics download.
SHAPES_KEPT = 5
TIMINGS_KEPT = 30
# How long the laptop's request for a job is held open; short enough for any proxy on the way.
LAPTOP_WAIT = 25


def _agent(hass: HomeAssistant) -> Agent | None:
    return hass.data.get(DOMAIN, {}).get("agent")


def _presented(request: web.Request) -> str:
    """The key the app sent, wherever it put it."""
    header = request.headers.get("Authorization", "")
    if header[:7].lower() == "bearer ":
        return header[7:].strip()
    return request.headers.get("x-api-key") or request.headers.get("api-key") or header.strip()


def _error(message: str, status: HTTPStatus) -> web.Response:
    return web.json_response({"error": {"message": message}}, status=status)


async def _refuse() -> web.Response:
    # A wrong key is answered slowly. Nothing is reported to Home Assistant's own
    # login guard, because through the remote address that could lock the owner out.
    await asyncio.sleep(1)
    return _error("Invalid API key", HTTPStatus.UNAUTHORIZED)


def _outline(request: web.Request, body: dict[str, Any]) -> dict[str, Any]:
    """What a request looked like, without what was said in it.

    The Even app's request format is not documented, so its outline is kept for
    the diagnostics download: field names, the app's own instructions, and the
    size of each message, but not the wearer's words.
    """
    messages = body.get("messages")
    tools = body.get("tools")
    return {
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "headers": {
            name: ("(present)" if name.lower() in ("authorization", "x-api-key", "api-key", "cookie") else value)
            for name, value in request.headers.items()
        },
        "fields": {name: value for name, value in body.items() if name not in ("messages", "tools") and isinstance(value, (str, int, float, bool))},
        "messages": [
            {
                "role": message.get("role"),
                "characters": len(str(message.get("content", ""))),
                **({"content": str(message.get("content", ""))[:2000]} if message.get("role") in ("system", "developer") else {}),
            }
            for message in (messages if isinstance(messages, list) else [])
            if isinstance(message, dict)
        ],
        "tools": [
            (tool.get("function") or {}).get("name") or tool.get("name") or tool.get("type")
            for tool in (tools if isinstance(tools, list) else [])
            if isinstance(tool, dict)
        ],
    }


class HealthView(HomeAssistantView):
    """Says the agent is running. Nothing secret, so it can be opened in a browser to check the address."""

    url = API_BASE
    name = "api:claude_agent"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def get(self, request: web.Request) -> web.Response:
        if _agent(self._hass) is None:
            return web.Response(text="Your agent is installed but not set up yet.\n", status=HTTPStatus.SERVICE_UNAVAILABLE)
        return web.Response(text=f"Your agent is running. Even AI talks to it at {CHAT_PATH}.\n")


class ModelsView(HomeAssistantView):
    """The model list some apps ask for before they will save an endpoint."""

    url = MODELS_PATH
    name = "api:claude_agent:models"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def get(self, request: web.Request) -> web.Response:
        agent = _agent(self._hass)
        if agent is None:
            return _error("The agent is not set up", HTTPStatus.SERVICE_UNAVAILABLE)
        if not agent.authorized(_presented(request)):
            return await _refuse()
        return web.json_response({"object": "list", "data": [{"id": agent.model, "object": "model", "owned_by": "you"}]})


class ChatView(HomeAssistantView):
    """Where the Even app sends what was said, and reads Claude's answer."""

    url = CHAT_PATH
    name = "api:claude_agent:chat"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def post(self, request: web.Request) -> web.StreamResponse:
        agent = _agent(self._hass)
        if agent is None:
            return _error("The agent is not set up", HTTPStatus.SERVICE_UNAVAILABLE)
        if not agent.authorized(_presented(request)):
            return await _refuse()
        try:
            body = await request.json()
        except ValueError:
            return _error("The request body is not JSON", HTTPStatus.BAD_REQUEST)
        if not isinstance(body, dict):
            return _error("The request body is not an object", HTTPStatus.BAD_REQUEST)

        self._hass.data[DOMAIN].setdefault("shapes", deque(maxlen=SHAPES_KEPT)).append(_outline(request, body))
        if not convert_messages(body.get("messages"))[0]:
            return _error("No user message in the request", HTTPStatus.BAD_REQUEST)

        started = time.monotonic()
        completion_id = new_id()
        created = int(time.time())
        # Word by word when the app asks for it, or when switched on under Configure to try it with an app that does not ask.
        streamed = bool(body.get("stream")) or bool(self._hass.data[DOMAIN].get("stream_always"))
        timings = self._hass.data[DOMAIN].setdefault("timings", deque(maxlen=TIMINGS_KEPT))

        def record(result: dict[str, Any] | None, first_text: float | None, error: str = "") -> None:
            """How long this answer took and what went into it, without what was said."""
            stats = (result or {}).get("stats") or {}
            timings.append(
                {
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "seconds": round(time.monotonic() - started, 2),
                    "first_words_seconds": round(first_text - started, 2) if first_text else None,
                    "word_by_word": streamed,
                    "claude_requests": stats.get("requests", 0),
                    "web_searches": stats.get("searches", 0),
                    "tools": stats.get("tools", []),
                    "stop": (result or {}).get("stop_reason", ""),
                    "error": error,
                }
            )

        if not streamed:
            try:
                result = await agent.reply(body)
            except BadRequest as err:
                return _error(str(err), HTTPStatus.BAD_REQUEST)
            except Exception as err:  # noqa: BLE001 - whatever went wrong, the app needs an answer
                _LOGGER.warning("Claude could not be reached: %s", err)
                record(None, None, type(err).__name__)
                if over_limit(err):
                    # Trying again will not help here, so the wearer is told what will.
                    return web.json_response(completion(completion_id, created, agent.model, LIMIT, NO_USAGE))
                return _error("The assistant could not be reached", HTTPStatus.BAD_GATEWAY)
            record(result, None)
            return web.json_response(completion(completion_id, created, agent.model, result["text"], result["usage"]))

        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
        await response.prepare(request)
        listening = True
        sent = False
        first_text: float | None = None

        async def send(payload: dict[str, Any] | str) -> None:
            nonlocal listening
            if not listening:
                return
            data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            try:
                await response.write(f"data: {data}\n\n".encode())
            except (ConnectionError, RuntimeError):
                listening = False  # The glasses stopped listening.

        async def on_text(delta: str) -> None:
            nonlocal sent, first_text
            sent = True
            first_text = first_text or time.monotonic()
            await send(chunk(completion_id, created, agent.model, {"content": delta}))

        await send(chunk(completion_id, created, agent.model, {"role": "assistant", "content": ""}))
        try:
            result = await agent.reply(body, on_text)
        except Exception as err:  # noqa: BLE001 - whatever went wrong, leave the wearer a line to read
            _LOGGER.warning("Claude could not be reached: %s", err)
            record(None, first_text, type(err).__name__)
            if not sent:
                await send(chunk(completion_id, created, agent.model, {"content": LIMIT if over_limit(err) else FAILED}))
            await send(chunk(completion_id, created, agent.model, {}, "stop"))
        else:
            record(result, first_text)
            await send(chunk(completion_id, created, agent.model, {}, "stop"))
            if isinstance(body.get("stream_options"), dict) and body["stream_options"].get("include_usage"):
                await send({**chunk(completion_id, created, agent.model, {}), "choices": [], "usage": result["usage"]})
        await send("[DONE]")
        if listening:
            try:
                await response.write_eof()
            except (ConnectionError, RuntimeError):
                pass
        return response


def _laptop(hass: HomeAssistant, request: web.Request) -> Laptop | None:
    """The job queue, when the laptop is switched on and presented the right key."""
    shared = hass.data.get(DOMAIN, {})
    token = shared.get("laptop_token") or ""
    if not token or not hmac.compare_digest(_presented(request).encode(), token.encode()):
        return None
    return shared.get("laptop")


class LaptopNextView(HomeAssistantView):
    """Where the laptop helper asks for its next job. Held open for a while, so a new job reaches it at once."""

    url = LAPTOP_NEXT_PATH
    name = "api:claude_agent:laptop:next"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def get(self, request: web.Request) -> web.Response:
        laptop = _laptop(self._hass, request)
        if laptop is None:
            return await _refuse()
        if request.query.get("check"):
            # The helper's setup making sure it has the right address and key, without taking a job.
            return web.Response(status=HTTPStatus.NO_CONTENT)
        if not self._hass.data[DOMAIN].get("laptop_on"):
            # Switched off under Configure: the helper is told to wait, and nothing is handed out.
            await asyncio.sleep(LAPTOP_WAIT)
            return web.Response(status=HTTPStatus.NO_CONTENT)
        job = await laptop.next(LAPTOP_WAIT)
        if job is None:
            return web.Response(status=HTTPStatus.NO_CONTENT)
        return web.json_response({"id": job["id"], "task": job["task"], "follow_up": job["follow_up"]})


async def _job_id(request: web.Request) -> str | None:
    try:
        body = await request.json()
    except ValueError:
        return None
    return body["id"] if isinstance(body, dict) and isinstance(body.get("id"), str) else None


class LaptopStartView(HomeAssistantView):
    """Where the laptop helper says it is starting a job it was handed, so the job is not handed out again."""

    url = LAPTOP_START_PATH
    name = "api:claude_agent:laptop:start"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def post(self, request: web.Request) -> web.Response:
        laptop = _laptop(self._hass, request)
        if laptop is None:
            return await _refuse()
        job_id = await _job_id(request)
        if job_id is None:
            return _error("The request needs the job's id", HTTPStatus.BAD_REQUEST)
        if not await laptop.start(job_id):
            return _error("That job was called off", HTTPStatus.CONFLICT)
        return web.json_response({"ok": True})


class LaptopWatchView(HomeAssistantView):
    """Where the laptop helper, while working, waits to hear whether the job was called off."""

    url = LAPTOP_WATCH_PATH
    name = "api:claude_agent:laptop:watch"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def get(self, request: web.Request) -> web.Response:
        laptop = _laptop(self._hass, request)
        if laptop is None:
            return await _refuse()
        status = await laptop.watch(request.query.get("id", ""), LAPTOP_WAIT)
        if status is None:
            return _error("No such job", HTTPStatus.NOT_FOUND)
        return web.json_response({"status": status})


class LaptopDoneView(HomeAssistantView):
    """Where the laptop helper reports how a job went."""

    url = LAPTOP_DONE_PATH
    name = "api:claude_agent:laptop:done"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def post(self, request: web.Request) -> web.Response:
        laptop = _laptop(self._hass, request)
        if laptop is None:
            return await _refuse()
        try:
            body = await request.json()
        except ValueError:
            return _error("The request body is not JSON", HTTPStatus.BAD_REQUEST)
        if not isinstance(body, dict) or not isinstance(body.get("id"), str):
            return _error("The report needs the job's id", HTTPStatus.BAD_REQUEST)
        result = body.get("result") if isinstance(body.get("result"), str) else ""
        if not await laptop.finish(body["id"], body.get("ok") is True, result, body.get("stats")):
            return _error("No such job is running", HTTPStatus.NOT_FOUND)
        return web.json_response({"ok": True})


SESSION_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")

# The glasses app runs in a web view inside the Even app, so its requests are cross-site. The key, not a
# cookie, is what lets a request in, so any origin may make one.
CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Authorization, Content-Type, X-Session-Id",
    "Access-Control-Max-Age": "86400",
}


def _cors(response: web.StreamResponse) -> web.StreamResponse:
    response.headers.update(CORS)
    return response


async def _preflight(request: web.Request) -> web.Response:
    """The browser asking whether the app may make its request."""
    return web.Response(status=HTTPStatus.NO_CONTENT, headers=CORS)


class SessionTurnView(HomeAssistantView):
    """One turn of a session from the glasses app.

    The body is the recording made while the ring was held (a WAV file), or
    {"text": ...} as JSON. The answer streams back as server-sent events:
    {"type": "heard", "text"} with what was understood, then {"type": "text",
    "text"} pieces of the answer as it is written, {"type": "error", "message"}
    if something went wrong, and finally {"type": "done"}.
    """

    url = SESSION_TURN_PATH
    name = "api:claude_agent:session:turn"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def options(self, request: web.Request) -> web.Response:
        return await _preflight(request)

    async def post(self, request: web.Request) -> web.StreamResponse:
        agent = _agent(self._hass)
        if agent is None:
            return _cors(_error("The agent is not set up", HTTPStatus.SERVICE_UNAVAILABLE))
        if not agent.authorized(_presented(request)):
            return _cors(await _refuse())
        session_id = request.headers.get("X-Session-Id", "")
        if not SESSION_ID.match(session_id):
            return _cors(_error("The request needs an X-Session-Id header", HTTPStatus.BAD_REQUEST))
        shared = self._hass.data[DOMAIN]
        started = time.monotonic()
        typed = None
        audio = b""
        if request.content_type == "application/json":
            try:
                body = await request.json()
            except ValueError:
                return _cors(_error("The request body is not JSON", HTTPStatus.BAD_REQUEST))
            typed = body.get("text") if isinstance(body, dict) and isinstance(body.get("text"), str) else ""
        else:
            audio = await request.read()
            if len(audio) > speech.MAX_AUDIO_BYTES:
                return _cors(_error("The recording is too long", HTTPStatus.REQUEST_ENTITY_TOO_LARGE))

        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache", "X-Accel-Buffering": "no", **CORS}
        )
        await response.prepare(request)
        listening = True

        async def send(event: dict[str, Any]) -> None:
            nonlocal listening
            if not listening:
                return
            try:
                await response.write(f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n".encode())
            except (ConnectionError, RuntimeError):
                listening = False  # The app stopped listening.

        heard_at = None
        first_text: float | None = None
        try:
            if typed is None:
                words = speech.vocabulary(agent.profile, await agent.vocabulary_sources())
                said = await speech.transcribe(async_get_clientsession(self._hass), shared.get("stt_provider", ""), shared.get("stt_key", ""), audio, words)
                if speech.probably_silence(said, audio):
                    said = ""
            else:
                said = " ".join(typed.split())
            heard_at = time.monotonic()
            await send({"type": "heard", "text": said})
            if said:

                async def on_text(delta: str) -> None:
                    nonlocal first_text
                    first_text = first_text or time.monotonic()
                    await send({"type": "text", "text": delta})

                result = await agent.session_reply(session_id, said, on_text)
                stats = result.get("stats") or {}
                shared.setdefault("timings", deque(maxlen=TIMINGS_KEPT)).append(
                    {
                        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "session": True,
                        "seconds": round(time.monotonic() - started, 2),
                        "speech_to_text_seconds": round(heard_at - started, 2) if typed is None else None,
                        "first_words_seconds": round(first_text - started, 2) if first_text else None,
                        "claude_requests": stats.get("requests", 0),
                        "web_searches": stats.get("searches", 0),
                        "tools": stats.get("tools", []),
                        "stop": result.get("stop_reason", ""),
                        "error": "",
                    }
                )
        except speech.SpeechError as err:
            await send({"type": "error", "message": str(err)})
        except Exception as err:  # noqa: BLE001 - whatever went wrong, the glasses get a line to show
            _LOGGER.warning("A session turn failed: %s", err)
            await send({"type": "error", "message": LIMIT if over_limit(err) else FAILED})
        await send({"type": "done"})
        if listening:
            try:
                await response.write_eof()
            except (ConnectionError, RuntimeError):
                pass
        return response


class SessionEndView(HomeAssistantView):
    """Where the glasses app says a session is over, so it is kept with the earlier conversations."""

    url = SESSION_END_PATH
    name = "api:claude_agent:session:end"
    requires_auth = False

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def options(self, request: web.Request) -> web.Response:
        return await _preflight(request)

    async def post(self, request: web.Request) -> web.Response:
        agent = _agent(self._hass)
        if agent is None:
            return _cors(_error("The agent is not set up", HTTPStatus.SERVICE_UNAVAILABLE))
        if not agent.authorized(_presented(request)):
            return _cors(await _refuse())
        session_id = await _job_id(request)
        if not session_id or not SESSION_ID.match(session_id):
            return _cors(_error("The request needs the session's id", HTTPStatus.BAD_REQUEST))
        await agent.end_session(session_id)
        return _cors(web.json_response({"ok": True}))


def register_views(hass: HomeAssistant) -> None:
    """Add the addresses to Home Assistant's web server. They stay for as long as it runs."""
    hass.http.register_view(HealthView(hass))
    hass.http.register_view(ModelsView(hass))
    hass.http.register_view(ChatView(hass))
    hass.http.register_view(LaptopNextView(hass))
    hass.http.register_view(LaptopStartView(hass))
    hass.http.register_view(LaptopWatchView(hass))
    hass.http.register_view(LaptopDoneView(hass))
    hass.http.register_view(SessionTurnView(hass))
    hass.http.register_view(SessionEndView(hass))
