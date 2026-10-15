"""The web addresses the Even app talks to.

They sit under /api/claude_agent on Home Assistant's own web server, so they
are reachable wherever Home Assistant is, including through its remote address.
Home Assistant's login is not used here: the Even app presents the key that
was shown when the integration was set up.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Coroutine
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
from .agent import Agent, AnswerStopped, BadRequest, problem_text, same_key
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
    SESSION_JOB_PATH,
    SESSION_LOG_PATH,
    SESSION_TURN_PATH,
)
from .formats import NO_USAGE, chunk, completion, convert_messages, new_id
from .laptop import Laptop

_LOGGER = logging.getLogger(__name__)

# How many request outlines, and answer timings, are kept for the diagnostics download.
SHAPES_KEPT = 5
TIMINGS_KEPT = 30
# How long the laptop's request for a job is held open. Short, because networks on the way (school Wi-Fi, the
# remote-access proxy) cut connections that sit idle much longer, and each cut costs a reconnect.
LAPTOP_WAIT = 15
SESSION_REQUESTS_KEPT = 10
SESSION_LOG_KEPT = 150
# While a session turn is being answered, a comment is sent this often, so a slow answer can be told from a dead connection.
KEEPALIVE_SECONDS = 10
# Request details kept word for word in the diagnostics; anything else is only noted as present.
SHOWN_HEADERS = {"user-agent", "content-type", "accept", "accept-encoding", "content-length"}
SHOWN_FIELDS = {"model", "stream", "temperature", "max_tokens", "stream_options"}


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


async def _carry_through(hass: HomeAssistant, answer: Coroutine[Any, Any, dict[str, Any]], abandoned: Callable[[asyncio.Task], None]) -> dict[str, Any]:
    """Waits for `answer`, which runs to its end even when the app hangs up first.

    When the app hangs up, Home Assistant cancels the request, which would stop a
    laptop job from going out and the turn from being kept. Instead it carries on,
    and `abandoned` hears how it ended.
    """
    task = hass.async_create_task(answer, "claude_agent answer")
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.add_done_callback(abandoned)
        raise


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
        # Only harmless headers word for word: others can carry the remote address, a device or an account.
        "headers": {name: (value if name.lower() in SHOWN_HEADERS else "(present)") for name, value in request.headers.items()},
        "fields": {
            name: (value if name in SHOWN_FIELDS else type(value).__name__)
            for name, value in body.items()
            if name not in ("messages", "tools") and isinstance(value, (str, int, float, bool, dict))
        },
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
        timings = self._hass.data[DOMAIN].setdefault("timings", deque(maxlen=TIMINGS_KEPT))

        def record(result: dict[str, Any] | None, error: str = "") -> None:
            """How long this answer took and what went into it, without what was said."""
            stats = (result or {}).get("stats") or {}
            timings.append(
                {
                    "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "seconds": round(time.monotonic() - started, 2),
                    "claude_requests": stats.get("requests", 0),
                    "web_searches": stats.get("searches", 0),
                    "tools": stats.get("tools", []),
                    "cache_read": stats.get("cache_read", 0),
                    "cache_written": stats.get("cache_written", 0),
                    "model": stats.get("model", ""),
                    "stop": (result or {}).get("stop_reason", ""),
                    "error": error,
                }
            )

        def abandoned(task: asyncio.Task) -> None:
            """Records an answer the app stopped waiting for."""
            if task.cancelled():
                record(None, "app gave up, then cancelled")
            elif task.exception() is not None:
                record(None, f"app gave up, then {type(task.exception()).__name__}")
            else:
                record(task.result(), "app gave up before it was ready")

        try:
            result = await _carry_through(self._hass, agent.reply(body), abandoned)
        except BadRequest as err:
            return _error(str(err), HTTPStatus.BAD_REQUEST)
        except Exception as err:  # noqa: BLE001 - whatever went wrong, the app needs an answer
            _LOGGER.warning("Claude could not be reached: %s", err)
            record(None, type(err).__name__)
            # A sentence the glasses can show, saying what to do about it, rather than an error the Even app
            # shows its own way, if at all.
            text, usage = problem_text(err), NO_USAGE
        else:
            record(result)
            text, usage = result["text"], result["usage"]
        if body.get("stream"):
            # The Even app never asks for a stream, and shows "AI server error" for one it did not ask for. An app that
            # does ask gets the whole answer at once, in the streamed format it reads.
            include_usage = isinstance(body.get("stream_options"), dict) and body["stream_options"].get("include_usage") is True
            return web.Response(
                text=_as_stream(completion_id, created, agent.model, text, usage if include_usage else None),
                content_type="text/event-stream",
                charset="utf-8",
                headers={"Cache-Control": "no-cache"},
            )
        return web.json_response(completion(completion_id, created, agent.model, text, usage))


def _as_stream(completion_id: str, created: int, model: str, text: str, usage: dict[str, int] | None) -> str:
    """A whole answer as server-sent events in the chat-completions format: one piece, the end, and the usage if asked for."""
    pieces: list[dict[str, Any] | str] = [
        chunk(completion_id, created, model, {"role": "assistant", "content": text}),
        chunk(completion_id, created, model, {}, "stop"),
    ]
    if usage is not None:
        pieces.append({**chunk(completion_id, created, model, {}), "choices": [], "usage": usage})
    pieces.append("[DONE]")
    return "".join(f"data: {piece if isinstance(piece, str) else json.dumps(piece, ensure_ascii=False, separators=(',', ':'))}\n\n" for piece in pieces)


def _laptop(hass: HomeAssistant, request: web.Request) -> Laptop | None:
    """The job queue, when the laptop is switched on and presented the right key."""
    shared = hass.data.get(DOMAIN, {})
    token = shared.get("laptop_token") or ""
    if not token or not same_key(_presented(request), token):
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
            return web.Response(status=HTTPStatus.NO_CONTENT, headers=WAIT_HEADER)
        if not self._hass.data[DOMAIN].get("laptop_on"):
            # Switched off under Configure: the helper is told to wait, and nothing is handed out.
            await asyncio.sleep(LAPTOP_WAIT)
            return web.Response(status=HTTPStatus.NO_CONTENT, headers=WAIT_HEADER)
        job = await laptop.next(LAPTOP_WAIT)
        if job is None:
            return web.Response(status=HTTPStatus.NO_CONTENT, headers=WAIT_HEADER)
        return web.json_response({"id": job["id"], "task": job["task"], "follow_up": job["follow_up"], "model": job.get("model")}, headers=WAIT_HEADER)


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
        return web.json_response({"status": status}, headers=WAIT_HEADER)


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


# Tells the helper how long a request is held open, so it can tell a dropped connection from a quiet one sooner.
WAIT_HEADER = {"X-Laptop-Wait": str(LAPTOP_WAIT)}


def _session_id(request: web.Request) -> str:
    """The session a request belongs to: in the address, so no custom header has to pass the browser's cross-site check."""
    return request.query.get("session") or request.headers.get("X-Session-Id", "")


class SessionTurnView(HomeAssistantView):
    """One turn of a session from the glasses app.

    The body is the recording made while the ring was held (a WAV file, or Opus
    in an Ogg file, with how long and how loud it was as `seconds` and `loudness`
    in the address), or {"text": ...} as JSON. The answer streams back as
    server-sent events: {"type": "heard", "text"} with what was understood, then
    {"type": "text", "text"} pieces of the answer as it is written, {"type":
    "job", "id"} for each laptop job sent, {"type": "error", "message"} if
    something went wrong, and finally {"type": "done"}.
    """

    url = SESSION_TURN_PATH
    name = "api:claude_agent:session:turn"
    requires_auth = False
    # The glasses app runs in a web view inside the Even app, so its requests are cross-site. The key, not a
    # cookie, is what lets a request in, so Home Assistant may accept them from any origin.
    cors_allowed = True

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def post(self, request: web.Request) -> web.StreamResponse:
        shared = self._hass.data.setdefault(DOMAIN, {})
        started = time.monotonic()

        def note(outcome: str, **details: Any) -> None:
            """Keeps how each request from the glasses app went, without what was said, for the diagnostics."""
            shared.setdefault("session_requests", deque(maxlen=SESSION_REQUESTS_KEPT)).append(
                {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "outcome": outcome, "seconds": round(time.monotonic() - started, 2), **details}
            )

        agent = _agent(self._hass)
        if agent is None:
            note("agent not set up")
            return _error("The agent is not set up", HTTPStatus.SERVICE_UNAVAILABLE)
        if not agent.authorized(_presented(request)):
            note("wrong key")
            return await _refuse()
        session_id = _session_id(request)
        if not SESSION_ID.match(session_id):
            note("no session id")
            return _error("The request needs the session's id", HTTPStatus.BAD_REQUEST)
        typed = None
        audio = b""
        audio_type = request.content_type or "audio/wav"
        # A compressed recording cannot be measured here, so the app says how long and how loud it was.
        measured = {name: _float(request.query.get(name)) for name in ("seconds", "loudness")}
        if request.content_type == "application/json":
            try:
                body = await request.json()
            except ValueError:
                note("not JSON")
                return _error("The request body is not JSON", HTTPStatus.BAD_REQUEST)
            typed = body.get("text") if isinstance(body, dict) and isinstance(body.get("text"), str) else ""
        else:
            audio = await request.read()
            if len(audio) > speech.MAX_AUDIO_BYTES:
                note("recording too long", bytes=len(audio))
                return _error("The recording is too long", HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        seconds = measured["seconds"] if measured["seconds"] is not None else speech.seconds_of(audio)
        kind = {"kind": "typed" if typed is not None else "recording", **({"audio_seconds": round(seconds, 1), "audio": audio_type} if typed is None else {})}

        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
        await response.prepare(request)
        listening = True
        writing = asyncio.Lock()

        async def write(data: bytes) -> None:
            nonlocal listening
            if not listening:
                return
            try:
                async with writing:
                    await response.write(data)
            except (ConnectionError, RuntimeError):
                listening = False  # The app stopped listening.

        async def send(event: dict[str, Any]) -> None:
            await write(f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n".encode())

        async def keepalive() -> None:
            # A comment line, which the app's reader skips; it only shows that the connection is alive.
            while listening:
                await asyncio.sleep(KEEPALIVE_SECONDS)
                await write(b": keepalive\n\n")

        beating = asyncio.create_task(keepalive())
        heard_at = None
        first_text: float | None = None
        try:
            if typed is None:
                words = speech.vocabulary(agent.profile, await agent.vocabulary_sources())
                said = await speech.transcribe(
                    async_get_clientsession(self._hass), shared.get("stt_provider", ""), shared.get("stt_key", ""), audio, words, content_type=audio_type
                )
                if speech.probably_silence(said, audio, words, seconds=measured["seconds"], loud=measured["loudness"]):
                    said = ""
            else:
                said = " ".join(typed.split())
            heard_at = time.monotonic()
            await send({"type": "heard", "text": said})
            # What was heard is kept with the last few requests, so the speech recognizer's mistakes can be seen and fixed.
            kind["heard"] = said
            if not said:
                note("connection check" if typed == "" else "nothing heard", **kind)
                if typed == "" and not shared.get("stt_key"):
                    # The app's connection check: it reached Home Assistant, but holding the ring would fail.
                    await send({"type": "error", "message": "No speech-to-text key is set under Configure"})
            else:

                async def on_text(delta: str) -> None:
                    nonlocal first_text
                    if not listening:
                        # Nobody is reading: stop Claude writing an answer no one will see.
                        raise AnswerStopped
                    first_text = first_text or time.monotonic()
                    await send({"type": "text", "text": delta})

                result = await agent.session_reply(session_id, said, on_text)
                # The app watches each laptop job sent, so its outcome can be shown in the session when it is done.
                for job_id in result.get("jobs") or []:
                    await send({"type": "job", "id": job_id})
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
                        "cache_read": stats.get("cache_read", 0),
                        "cache_written": stats.get("cache_written", 0),
                        "model": stats.get("model", ""),
                        "stop": result.get("stop_reason", ""),
                        "error": "",
                    }
                )
                note("app stopped listening" if result.get("stopped") else "answered", **kind)
        except speech.SpeechError as err:
            note(f"speech-to-text: {err}", **kind)
            await send({"type": "error", "message": str(err)})
        except asyncio.CancelledError:
            note("app went away", **kind)
            raise
        except Exception as err:  # noqa: BLE001 - whatever went wrong, the glasses get a line to show
            _LOGGER.warning("A session turn failed: %s", err)
            note(f"failed: {type(err).__name__}", **kind)
            await send({"type": "error", "message": problem_text(err)})
        finally:
            beating.cancel()
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
    cors_allowed = True

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def post(self, request: web.Request) -> web.Response:
        agent = _agent(self._hass)
        if agent is None:
            return _error("The agent is not set up", HTTPStatus.SERVICE_UNAVAILABLE)
        if not agent.authorized(_presented(request)):
            return await _refuse()
        session_id = await _job_id(request)
        if not session_id or not SESSION_ID.match(session_id):
            return _error("The request needs the session's id", HTTPStatus.BAD_REQUEST)
        await agent.end_session(session_id)
        return web.json_response({"ok": True})


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


class SessionJobView(HomeAssistantView):
    """Where the glasses app asks how a laptop job sent from a session is going, so its outcome can be shown there."""

    url = SESSION_JOB_PATH
    name = "api:claude_agent:session:job"
    requires_auth = False
    cors_allowed = True

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def get(self, request: web.Request) -> web.Response:
        agent = _agent(self._hass)
        if agent is None:
            return _error("The agent is not set up", HTTPStatus.SERVICE_UNAVAILABLE)
        if not agent.authorized(_presented(request)):
            return await _refuse()
        laptop = self._hass.data[DOMAIN].get("laptop")
        job_id = request.query.get("id", "")
        job = await laptop.get(job_id) if laptop and job_id else None
        if job is None:
            return _error("No such job", HTTPStatus.NOT_FOUND)
        return web.json_response({"status": job["status"], "result": job["result"]})


class SessionLogView(HomeAssistantView):
    """Where the glasses app passes on its log, so a problem on the glasses can be read in the diagnostics."""

    url = SESSION_LOG_PATH
    name = "api:claude_agent:session:log"
    requires_auth = False
    cors_allowed = True

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def post(self, request: web.Request) -> web.Response:
        agent = _agent(self._hass)
        if agent is None:
            return _error("The agent is not set up", HTTPStatus.SERVICE_UNAVAILABLE)
        if not agent.authorized(_presented(request)):
            return await _refuse()
        try:
            body = await request.json()
        except ValueError:
            return _error("The request body is not JSON", HTTPStatus.BAD_REQUEST)
        lines = body.get("lines") if isinstance(body, dict) else None
        if not isinstance(lines, list):
            return _error("The request needs lines", HTTPStatus.BAD_REQUEST)
        kept = self._hass.data[DOMAIN].setdefault("session_app_log", deque(maxlen=SESSION_LOG_KEPT))
        for line in lines[-SESSION_LOG_KEPT:]:
            if isinstance(line, str):
                kept.append(line[:300])
        return web.json_response({"ok": True})


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
    hass.http.register_view(SessionJobView(hass))
    hass.http.register_view(SessionLogView(hass))
