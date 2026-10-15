"""Your own Claude behind Even AI: what the Even app sends in, Claude's answer out.

The Even app speaks the OpenAI chat-completions format and sends only the
newest thing the wearer said. This turns that into a conversation with Claude
that feels like the Claude app: a follow-up within a few minutes carries on,
anything later starts fresh (earlier conversations stay on hand if they refer
back), and lasting notes about the wearer are kept over time.

Nothing here knows about Home Assistant, so it can be tested on its own.

What Claude is told is in prompts.py, which laptop jobs need the laptop named in gate.py, the Even app's format in
formats.py, and the shape of what is kept in state.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import date, datetime
import hmac
import json
import re
from typing import Any, Protocol

from .formats import MAX_TURNS, NO_USAGE, Message, convert_messages, recent
from .gate import MENTIONS_LAPTOP, changes_things
from .prompts import (
    BUSY,
    CANCEL_LAPTOP,
    CANCEL_REMINDER,
    CUT_OFF,
    EARLIER,
    EMPTY,
    FAILED,
    FIRST_NOTE,
    FORGET,
    FRESH,
    HAND_OFF,
    INSTRUCTIONS,
    KEY_REFUSED,
    LAPTOP,
    LAPTOP_INSTRUCTIONS,
    LAPTOP_STATUS,
    LIMIT,
    NO_CREDIT,
    NOTHING_ASKED,
    QUICK_NOTE,
    REFLECT_INSTRUCTIONS,
    REFUSED,
    REMEMBER,
    REMIND,
    REMINDER_INSTRUCTIONS,
    SCHEDULE,
    SCHEDULE_INSTRUCTIONS,
    SESSION_NOTE,
)
from .state import EARLIER_KEPT, MAX_NOTE_CHARS, MAX_NOTES, parse_day, read_state

# Tools that can share a round with a laptop job without needing Claude to say more once they work.
# Skipping that extra request makes a laptop request quicker and cheaper.
NO_FOLLOW_UP = {"laptop", "remember", "forget"}
# How long a reply waits for the laptop's answer to a question about it: a quick question through Even AI cannot wait
# long, while a session in the glasses app streams and can.
RESULT_WAIT_SECONDS = 8
SESSION_RESULT_WAIT_SECONDS = 20
# How much of the day's conversations the night's look over them reads.
REFLECT_CHARS = 40_000

# A web search that pauses or a tool being used each cost one more request to Claude.
MAX_REQUESTS = 6
# Past MAX_TURNS a conversation is cut back to this many turns in one go, so the front of it stays the same for
# several turns and can be served from the prompt cache.
KEEP_TURNS = 8
# A press this long after the last one starts a new conversation, like opening a new chat.
THREAD_SECONDS = 3 * 60
# Earlier conversations kept for the wearer to refer back to: those from the last day, up to a size (and a count,
# EARLIER_KEPT).
EARLIER_SECONDS = 24 * 60 * 60
EARLIER_STORED_CHARS = 60_000
# How much of them the earlier_conversation tool shows at once.
EARLIER_CHARS = 6000
NOTES_PROMPT_CHARS = 16_000
# How many changes to the notes are recalled within one conversation.
MAX_CHANGES = 20
MAX_TOKENS = 4096
# How far back the laptop's jobs are reported, and how much of each outcome.
LAPTOP_SECONDS = 24 * 60 * 60
LAPTOP_JOBS_SHOWN = 6
LAPTOP_RESULT_CHARS = 600
# Without the laptop being mentioned, a job that changes things is only sent this soon after the last one, as a follow-on.
LAPTOP_FOLLOW_SECONDS = 5 * 60
# A session in the glasses app keeps this much of itself, and is put away after this long untouched.
SESSION_TURNS = 40
SESSION_KEEP_TURNS = 28
SESSION_IDLE_SECONDS = 4 * 60 * 60

# Said at the start, any of these starts a new chat; whatever follows is the first question in it.
# "Start over" and "clear chat" only count on their own or before a pause, so "start over the video" is left alone.
NEW_CHAT = re.compile(
    r"^\W*(?:ok(?:ay)?\W+|hey\W+)?"
    r"(?:(?:start\W+(?:a\W+)?)?new\W+(?:chat|conversation|topic)\b|(?:start\W+over|clear\W+(?:the\W+)?chat)(?=\W*$|\s*[,.!?;:]))\W*",
    re.IGNORECASE,
)

# A safety classifier can decline a harmless question; these models let the API retry it on a substitute.
FALLBACK_MODELS = re.compile(r"^claude-(opus|sonnet)-5-5")

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"
# The model setting that picks for itself: quick questions start on Haiku, which hands what needs more to Sonnet, and
# sessions are on Sonnet, or Opus when they ask it to think hard.
AUTO = "auto"
LAPTOP_MODELS = ("haiku", "sonnet", "opus")
# Asking for the deepest thinking, in so many words: "think hard about", "think really hard", "take your time with this",
# "use Opus". Everyday phrasing does not count: "I think hard work pays off", "I really think so", "take your time".
THINK_HARD = re.compile(
    r"\bthink(?:ing)?\s+(?:really\s+|very\s+|extra\s+)?(?:hard(?:er)?|deeply|carefully)\b(?=\s*(?:about|on|over|through|before|first)\b|\s*[.,!?;:]|\s*$)"
    r"|\b(?:deep\s+think(?:ing)?|ultra\s*think)\b"
    r"|\btake\s+(?:your|you're|ur)\s+time\s+(?:with|on|over)\b"
    r"|\b(?:use|using|with|switch\s+to|ask)\s+opus\b",
    re.IGNORECASE,
)

# Questions the quick model would only hand on, because a good answer needs the web. (Anything for the laptop is
# spotted with MENTIONS_LAPTOP.) These go to Sonnet straight away, which saves the hand-off's second.
NEEDS_MORE = re.compile(
    r"\b(?:search(?:\s+(?:for|the\s+web|online))?|look\s+(?:it\s+|that\s+|this\s+)?up|google|latest|news|headlines?|weather|forecast"
    r"|scores?|who\s+won|price\s+of|stocks?|exchange\s+rate|right\s+now|currently|at\s+the\s+moment"
    r"|(?:today|tonight|this\s+week)'?s\s+(?:news|weather|game|match|forecast))\b",
    re.IGNORECASE,
)

OnText = Callable[[str], Awaitable[None]]

# The end of a sentence, allowing for a closing quote or bracket after it.
SENTENCE_END = re.compile(r"""[.!?:]["')\]]*$""")


def joint(before: str, after: str, new_request: bool) -> str:
    """What belongs between two stretches of a reply that Claude wrote separately.

    Text on either side of a search or a saved note is two thoughts. Text split
    only to attach a source to part of a sentence is one, and must stay as it is.
    """
    if not before or not after or before[-1].isspace() or after[0].isspace():
        return ""
    if new_request or (SENTENCE_END.search(before) and after[0].isalnum()):
        return " "
    return ""


class Store(Protocol):
    """Where the notes and the conversations are kept."""

    async def read(self) -> Any: ...

    async def write(self, state: dict[str, Any]) -> None: ...


class Laptop(Protocol):
    """The queue of jobs for the laptop (see laptop.py)."""

    seen: float | None

    def online(self) -> bool: ...

    async def submit(self, task: str, *, follow_up: bool = False, model: str | None = None) -> dict[str, Any]: ...

    async def cancel(self) -> list[dict[str, Any]]: ...

    async def recent(self, within: float) -> list[dict[str, Any]]: ...

    async def outcome(self, job_id: str, wait: float) -> dict[str, Any] | None: ...


class Reminders(Protocol):
    """The reminders they set (see reminders.py)."""

    async def add(self, text: str, at: float) -> dict[str, Any]: ...

    async def cancel(self, number: int) -> dict[str, Any] | None: ...

    async def pending(self) -> list[dict[str, Any]]: ...


class Schedule(Protocol):
    """Their day from their planner (see dayflow.py)."""

    async def text(self, day: str | None = None) -> str: ...


class BadRequest(Exception):
    """The request cannot be answered as it was sent."""


class AnswerStopped(Exception):
    """Raised by `on_text` when nobody is listening any more, so Claude stops writing an answer no one will see."""


def over_limit(err: BaseException) -> bool:
    """Whether Claude turned the request down because the account's own spending limit was reached."""
    return "usage limits" in str(err).lower()


def problem_text(err: BaseException) -> str:
    """What to tell the wearer when Claude could not answer: what to do about it, when trying again will not help."""
    status = getattr(err, "status_code", None)
    message = str(err).lower()
    if over_limit(err):
        return LIMIT
    if "credit balance" in message:
        return NO_CREDIT
    if status in (401, 403) or "invalid x-api-key" in message or "authentication_error" in message:
        return KEY_REFUSED
    if status == 529 or "overloaded" in message:
        return BUSY
    return FAILED


def same_key(presented: str, key: str) -> bool:
    """Whether the key presented is `key`, in constant time. A header that is not valid text is simply wrong."""
    try:
        return hmac.compare_digest(presented.encode(), key.encode())
    except UnicodeError:
        return False


def trimmed(turns: list[Message], limit: int, keep: int) -> tuple[list[Message], list[Message]]:
    """A conversation cut back to its last `keep` turns once it passes `limit`, and the turns cut off."""
    if len(turns) <= limit:
        return turns, []
    kept = recent(turns, keep)
    return kept, turns[: len(turns) - len(kept)]


def _size(turns: list[Message]) -> int:
    return sum(len(turn["content"]) for turn in turns)


def ago(seconds: float) -> str:
    """How long ago, in words: "40 seconds", "2 minutes"."""
    if seconds < 90:
        return f"{max(1, round(seconds))} seconds"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)} minutes"
    return f"{round(seconds / 3600)} hours"


def _blocks(content: Any) -> list[dict[str, Any]]:
    """A message's content as a list of blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return list(content) if isinstance(content, list) else []


def with_context(messages: list[Message], context: str) -> list[Message]:
    """The messages as sent to Claude: what changes with every request rides on the newest message, and the
    conversation before it is marked for the prompt cache, since the next turn sends it again unchanged."""
    sent = list(messages)
    sent[-1] = {"role": "user", "content": [{"type": "text", "text": f"<context>\n{context}\n</context>"}, *_blocks(sent[-1]["content"])]}
    for index in range(len(sent) - 2, -1, -1):
        if sent[index]["role"] == "assistant":
            blocks = _blocks(sent[index]["content"])
            if blocks and isinstance(blocks[-1], dict) and blocks[-1].get("type") == "text" and blocks[-1].get("text"):
                blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
                sent[index] = {"role": "assistant", "content": blocks}
            break
    return sent


class Agent:
    """Answers the Even app's requests with Claude.

    `ask_claude(request, on_text)` sends one request to Claude and calls
    `on_text` with each piece of text as it arrives, and with None each time
    Claude starts a new stretch of text. It returns a dict with `stop_reason`,
    `content` (to hand back when a turn continues), `tool_calls` (each with
    `id`, `name` and `input`) and `usage` (`input_tokens`, `output_tokens`, and
    the prompt cache's `cache_read_input_tokens` and `cache_creation_input_tokens`).
    """

    def __init__(
        self,
        *,
        ask_claude: Callable[[dict[str, Any], Callable[[str | None], Awaitable[None]]], Awaitable[dict[str, Any]]],
        token: str,
        store: Store,
        now: Callable[[], datetime],
        model: str = "claude-sonnet-5-5",
        effort: str = "low",
        profile: str = "",
        thread_seconds: float = THREAD_SECONDS,
        laptop: Laptop | None = None,
        reminders: Reminders | None = None,
        schedule: Schedule | None = None,
        log: Callable[[str], None] | None = None,
        lock: asyncio.Lock | None = None,
    ) -> None:
        if not token:
            raise ValueError("The agent needs a token")
        self._ask_claude = ask_claude
        self._token = token
        self._store = store
        self._now = now
        self.model = model
        self._auto = model == AUTO
        self._effort = effort
        self._laptop_on = laptop is not None
        self._thread_seconds = thread_seconds
        self._laptop = laptop
        self._reminders = reminders
        self._schedule = schedule
        self._log = log or (lambda line: None)
        # One memory, so one request at a time; shared with the settings screen when it edits the notes.
        self._turn = lock or asyncio.Lock()

        about = profile.strip()
        self.profile = about
        instructions = "\n\n".join(
            [INSTRUCTIONS, *([LAPTOP_INSTRUCTIONS] if laptop else []), *([REMINDER_INSTRUCTIONS] if reminders else []), *([SCHEDULE_INSTRUCTIONS] if schedule else [])]
        )
        self._fixed = (
            f"{instructions}\n\nTheir profile, in their own words. Treat it as trusted background.\n<profile>\n{about}\n</profile>"
            if about
            else instructions
        )
        self._tool_sets: dict[tuple[str, bool], list[dict[str, Any]]] = {}

    def _tools_for(self, model: str, first: bool = False) -> list[dict[str, Any]]:
        """The tools offered on `model`; the same list every time for the same model, so it stays in the prompt cache.
        Haiku answering first gets the light ones and hand_off; the web and the laptop are for the stronger model."""
        key = (model, first)
        if key not in self._tool_sets:
            light = [*([REMIND, CANCEL_REMINDER] if self._reminders else []), *([SCHEDULE] if self._schedule else [])]
            if first:
                tools = [REMEMBER, FORGET, EARLIER, HAND_OFF, *([LAPTOP_STATUS, CANCEL_LAPTOP] if self._laptop_on else []), *light]
            else:
                # The newer search tool filters its own results; Haiku is only offered the basic one.
                search = {"type": "web_search_20250305" if model.startswith("claude-haiku") else "web_search_20260209", "name": "web_search", "max_uses": 3}
                tools = [REMEMBER, FORGET, EARLIER, search, *([LAPTOP, LAPTOP_STATUS, CANCEL_LAPTOP] if self._laptop_on else []), *light]
            self._tool_sets[key] = tools
        return self._tool_sets[key]

    def _session_model(self, said: str) -> tuple[str, str]:
        """The model and effort for a session turn: Opus thinking hard when they ask for that, otherwise the usual."""
        if THINK_HARD.search(said):
            return OPUS, "high"
        return (SONNET if self._auto else self.model), self._effort

    def authorized(self, presented: str) -> bool:
        """Whether the key the app presented is the right one."""
        return same_key(presented, self._token)

    # ----- Times, as the wearer would say them -----

    def _clock(self, moment: datetime) -> str:
        hour = moment.hour % 12 or 12
        return f"{hour}:{moment.minute:02d} {'AM' if moment.hour < 12 else 'PM'}"

    def _when(self, at: float) -> str:
        """A time earlier today as "2:05 PM", or an earlier one with its day."""
        now = self._now()
        moment = datetime.fromtimestamp(at, now.tzinfo)
        return self._clock(moment) if moment.date() == now.date() else f"{moment:%a} {self._clock(moment)}"

    def _date(self, day: date) -> str:
        return f"{day:%b} {day.day}, {day.year}"

    # ----- What Claude is sent -----

    def _notes_block(self, notes: list[dict[str, Any]]) -> str:
        """The saved notes as Claude sees them. Each keeps its number even when older ones no longer fit."""
        if not notes:
            return "Your saved notes about them: none at the moment."
        zone = self._now().tzinfo
        lines: list[str] = []
        used = 0
        first_shown = len(notes)
        for index in range(len(notes) - 1, -1, -1):
            if used >= NOTES_PROMPT_CHARS:
                break
            note = notes[index]
            saved = datetime.fromtimestamp(note["at"], zone)
            until = f" (until {self._date(parse_day(note['until']))})" if note.get("until") else ""
            line = f"{index + 1}. ({self._date(saved.date())}) {note['text']}{until}"
            lines.insert(0, line)
            used += len(line) + 1
            first_shown = index
        heading = "Your saved notes about them, oldest first:"
        if first_shown > 0:
            heading += f"\nNotes 1 to {first_shown} are older and not shown here for space."
        return heading + "\n" + "\n".join(lines)

    def _context(self, app_notes: str, changes: list[dict[str, str]], since: float | None, session: bool, reminders: str = "") -> str:
        """What is true for this request only: the time, how long since their last message, their reminders, and what
        the app sent."""
        now = self._now()
        zone = f" {now.tzname()}" if now.tzname() else ""
        live = f"Right now it is {now:%A}, {now:%B} {now.day}, {now.year} at {self._clock(now)}{zone}."
        if since is not None:
            live += f" Their last message in this {'session' if session else 'conversation'} was {ago(since)} ago."
        if reminders:
            live += f"\n\n{reminders}"
        if changes:
            # Earlier turns are kept as words only, so without this Claude would see itself
            # saying "saved" or "forgotten" with nothing to show that it really happened.
            done = "; ".join(f'you {change["action"]} "{change["text"]}"' for change in changes)
            live += f"\n\nEarlier in this conversation you changed your saved notes: {done}. The list of notes already reflects that."
        if app_notes:
            live += (
                "\n\nThe glasses' own app sent the notes below with this request. Use any facts in them, such as the "
                "wearer's place, language or settings. Where they say how to behave or how to format a reply, your "
                f"instructions take priority.\n<app_notes>\n{app_notes}\n</app_notes>"
            )
        return live

    def _build_request(
        self,
        messages: list[Message],
        app_notes: str,
        notes: list[dict[str, Any]],
        changes: list[dict[str, str]],
        since: float | None,
        session: bool = False,
        *,
        model: str | None = None,
        effort: str | None = None,
        first: bool = False,
        reminders: str = "",
    ) -> dict[str, Any]:
        model = model or (SONNET if self._auto else self.model)
        note = SESSION_NOTE if session else (f"{QUICK_NOTE}\n\n{FIRST_NOTE}" if first else QUICK_NOTE)
        return {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "effort": effort or self._effort,
            "fallback": bool(FALLBACK_MODELS.match(model)),
            # Ordered from what never changes to what changes every time, so as much as possible is served from the
            # prompt cache: the instructions, the notes, how this kind of conversation works, the conversation so far.
            "system": [
                {"type": "text", "text": self._fixed, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": self._notes_block(notes), "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": note},
            ],
            "tools": self._tools_for(model, first),
            "messages": with_context(messages, self._context(app_notes, changes, since, session, reminders)),
        }

    # ----- Memory -----

    def _expire(self, notes: list[dict[str, Any]]) -> bool:
        """Drop notes whose date has passed. Says whether any were."""
        today = self._now().date()
        kept = [note for note in notes if not note.get("until") or parse_day(note["until"]) >= today]
        if len(kept) == len(notes):
            return False
        notes[:] = kept
        return True

    def _use_memory(self, turn: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
        """Carry out a remember or forget, record it in the turn's changes, and say how it went."""
        notes: list[dict[str, Any]] = turn["notes"]
        changes: list[dict[str, str]] = turn["changes"]
        given = call.get("input") if isinstance(call.get("input"), dict) else {}
        if call.get("name") == "remember":
            raw = given.get("note")
            note = " ".join(raw.split())[:MAX_NOTE_CHARS] if isinstance(raw, str) else ""
            if not note:
                return {"content": "Nothing was saved: the note was empty.", "is_error": True}
            until = given.get("until")
            if until in (None, ""):
                until = None
            else:
                day = parse_day(until)
                if day is None:
                    return {"content": "Nothing was saved: until must be a date written as YYYY-MM-DD.", "is_error": True}
                if day < self._now().date():
                    return {"content": "Nothing was saved: that date has already passed.", "is_error": True}
            existing = next((saved for saved in notes if saved["text"].lower() == note.lower()), None)
            if existing is not None:
                if until and existing.get("until") != until:
                    existing["until"] = until
                    changes.append({"action": "saved", "text": f"{existing['text']} (until {until})"})
                    self._log(f"new date on a note: {existing['text']} (until {until})")
                    return {"content": "Updated the date on that note."}
                return {"content": "That was already saved."}
            if len(notes) >= MAX_NOTES:
                # Room is made by letting go of a note that was going to expire anyway, never a lasting one.
                dated = [saved for saved in notes if saved.get("until")]
                if not dated:
                    return {
                        "content": f"Nothing was saved: there are already {MAX_NOTES} notes. Forget one that is out of date, then save this again.",
                        "is_error": True,
                    }
                oldest = min(dated, key=lambda saved: saved["at"])
                notes[:] = [saved for saved in notes if saved is not oldest]
                changes.append({"action": "removed", "text": oldest["text"]})
            saved_note: dict[str, Any] = {"at": self._now().timestamp(), "text": note}
            if until:
                saved_note["until"] = until
            notes.append(saved_note)
            changes.append({"action": "saved", "text": note})
            self._log(f"remembered: {note}")
            return {"content": "Saved."}
        if call.get("name") == "forget":
            number = given.get("number")
            index = number - 1 if isinstance(number, int) and not isinstance(number, bool) else -1
            # Numbers are those of the list Claude was shown at the start of this turn, however many changed since.
            numbered: list[dict[str, Any]] = turn["numbered"]
            if index < 0 or index >= len(numbered):
                return {"content": "There is no note with that number.", "is_error": True}
            target = numbered[index]
            if not any(saved is target for saved in notes):
                return {"content": "That note was already removed."}
            notes[:] = [saved for saved in notes if saved is not target]
            changes.append({"action": "removed", "text": target["text"]})
            self._log(f"forgot: {target['text']}")
            return {"content": f'Removed "{target["text"]}".'}
        return {"content": "There is no such tool.", "is_error": True}

    # ----- Conversations -----

    def _file_away(self, state: dict[str, Any], thread: dict[str, Any] | None, kind: str) -> None:
        """Keep a finished conversation with the earlier ones, so it can still be looked up."""
        since = self._now().timestamp() - EARLIER_SECONDS
        kept = [old for old in state["earlier"] if old["at"] >= since]
        if thread and thread["at"] >= since and thread["turns"]:
            kept.append({"started": thread["started"], "at": thread["at"], "turns": thread["turns"], "kind": kind})
        kept.sort(key=lambda old: old["at"])
        kept = kept[-EARLIER_KEPT:]
        while len(kept) > 1 and sum(_size(old["turns"]) for old in kept) > EARLIER_STORED_CHARS:
            kept.pop(0)
        state["earlier"] = kept

    def _retire(self, state: dict[str, Any]) -> None:
        """Put the quick chat in progress away with the earlier ones."""
        thread = state["thread"]
        state["thread"] = None
        self._file_away(state, thread, "chat")

    def _retire_sessions(self, state: dict[str, Any], *, only: str | None = None) -> None:
        """Put away one session, or every session left untouched for hours (the app was closed without ending it)."""
        now = self._now().timestamp()
        for session_id in list(state["sessions"]):
            if session_id == only or (only is None and now - state["sessions"][session_id]["at"] > SESSION_IDLE_SECONDS):
                self._file_away(state, state["sessions"].pop(session_id), "session")

    def _lookback(self, state: dict[str, Any], session_id: str | None) -> list[dict[str, Any]]:
        """Every conversation the earlier_conversation tool can show, besides the one being answered: those filed away,
        sessions still open, and from a session, the quick chat too."""
        found = [dict(old) for old in state["earlier"]]
        for other, session in state["sessions"].items():
            if other != session_id:
                found.append({"started": session["started"], "at": session["at"], "turns": session["turns"], "kind": "session", "open": True})
        if session_id is not None and state["thread"]:
            thread = state["thread"]
            found.append({"started": thread["started"], "at": thread["at"], "turns": thread["turns"], "kind": "chat"})
        found.sort(key=lambda old: old["at"])
        return found

    def _earlier_text(self, earlier: list[dict[str, Any]]) -> str:
        """The conversations, newest first. Where one is too long, its end is kept: that is what they most likely mean."""
        since = self._now().timestamp() - EARLIER_SECONDS
        recent_ones = [old for old in reversed(earlier) if old["at"] >= since]
        if not recent_ones:
            return "There are no earlier conversations from the last day."
        parts: list[str] = []
        left = EARLIER_CHARS
        for old in recent_ones:
            what = "Session in their glasses app" if old.get("kind") == "session" else "Conversation"
            heading = f"{what} from {self._when(old['started'])} to {self._when(old['at'])}" + (", still open" if old.get("open") else "") + ":"
            lines = [f"{'They' if turn['role'] == 'user' else 'You'} said: {turn['content']}" for turn in old["turns"]]
            text = "\n".join([heading, *lines])
            if len(text) <= left:
                parts.append(text)
                left -= len(text) + 2
                continue
            # Too long for what is left: keep its last turns that fit, and say the start is missing.
            marker = "(The start of it is left out.)"
            kept: list[str] = []
            room = left - len(heading) - len(marker) - 2
            for line in reversed(lines):
                if len(line) + 1 > room:
                    if not kept and room > 200:
                        # Even the last turn is too long: its end, which is where it got to.
                        kept.insert(0, "…" + line[-(room - 2) :])
                    break
                kept.insert(0, line)
                room -= len(line) + 1
            if kept:
                parts.append("\n".join([heading, marker, *kept]))
            break
        if not parts:
            return "There are no earlier conversations from the last day."
        return "Earlier conversations, newest first.\n\n" + "\n\n".join(parts)

    # ----- The laptop -----

    async def _laptop_status(self) -> str:
        """Whether the laptop is on and how its recent jobs went."""
        if self._laptop is None:
            return "There is no laptop connected."
        if self._laptop.online():
            lines = ["Their laptop is on and ready for jobs."]
        elif self._laptop.seen is None:
            lines = ["Their laptop has not been connected yet, so a job will wait until it is."]
        else:
            lines = [f"Their laptop is off or asleep; it was last on at {self._when(self._laptop.seen)}. A job waits up to 12 hours for it."]
        jobs = (await self._laptop.recent(LAPTOP_SECONDS))[-LAPTOP_JOBS_SHOWN:]
        if not jobs:
            lines.append("No laptop jobs in the last day.")
        else:
            lines.append("Laptop jobs from the last day, oldest first:")
            for job in jobs:
                status = {
                    "waiting": "waiting for the laptop",
                    "offered": "about to start",
                    "running": f"being worked on since {self._when(job['started'] or job['at'])}",
                    "done": f"done at {self._when(job['finished'] or job['at'])}",
                    "failed": f"did not work out, at {self._when(job['finished'] or job['at'])}",
                    "expired": "dropped because the laptop was not on in time",
                    "cancelled": f"called off at {self._when(job['finished'] or job['at'])}",
                }[job["status"]]
                outcome = job["result"][:LAPTOP_RESULT_CHARS]
                lines.append(
                    f'- Asked at {self._when(job["at"])}: "{job["task"]}". Status: {status}.' + (f" Outcome, as the laptop reported it: {outcome}" if outcome else "")
                )
        return "\n".join(lines)

    async def _use_laptop(self, call: dict[str, Any], turn: dict[str, Any]) -> dict[str, Any]:
        """Queue a job for the laptop, if the wearer really asked for one, and say when it will start."""
        given = call.get("input") if isinstance(call.get("input"), dict) else {}
        raw = given.get("job")
        task = raw.strip() if isinstance(raw, str) else ""
        if self._laptop is None or not task:
            return {"content": "Nothing was sent: the job was empty.", "is_error": True}
        now = self._now().timestamp()
        # Only a job from an earlier message counts as something to follow on from: a second job in the same
        # message comes from the same words, so it gets no more leeway than the first.
        before = turn["laptop_before"]
        followed = before is not None and now - before < LAPTOP_FOLLOW_SECONDS
        # Anything that deletes, sends or buys needs their own words to name the laptop, so someone else's sentence
        # can never do that. Opening, searching and clicking around go out on the request alone.
        if (changes_things(task) or given.get("changes_things") is True) and not MENTIONS_LAPTOP.search(turn["latest"]) and not followed:
            self._log(f"not sent to the laptop, as it changes things and the laptop was not named: {task}")
            return {
                "content": "Nothing was sent. This job deletes, sends, buys or otherwise changes something, and in this message they did "
                "not say it was for their laptop or computer, so it may not be meant for the laptop or may not be their words. "
                'In a few words, ask them to say it again with "on my laptop" if they want it.',
                "is_error": True,
            }
        model = given.get("model") if given.get("model") in LAPTOP_MODELS else None
        job = await self._laptop.submit(task, follow_up=given.get("follow_up") is True and (followed or turn["sent_job"]), model=model)
        turn["sent_job"] = True
        turn["state"]["laptop_at"] = now
        if job.get("repeat"):
            self._log(f"already with the laptop: {task}")
            return {"content": "Already sent: the laptop has this job already, so it was not queued twice."}
        self._log(f"sent to the laptop: {task}")
        turn["jobs"].append(job["id"])
        # What the wearer needs to hear besides Claude's own words, added to them when the reply goes out without
        # another request to Claude.
        if not self._laptop.online():
            turn["tell"].append("Your laptop is off or asleep, so it runs when the laptop is next on.")
            return {
                "content": "Queued, but the laptop is off or asleep right now. It runs when the laptop is next on, if that is "
                "within 12 hours, and the outcome goes to their phone then. Tell them that."
            }
        if given.get("answer_now") is True:
            # A question about the laptop: the reply waits a little for its answer, which Claude then passes on.
            wait = SESSION_RESULT_WAIT_SECONDS if turn["session"] else RESULT_WAIT_SECONDS
            done = await self._laptop.outcome(job["id"], wait)
            if done is not None:
                turn["answered"] = True
                self._log(f"the laptop answered in time: {done['status']}")
                if done["status"] == "done":
                    return {"content": f"The laptop has looked, and reports: {done['result']}\nAnswer them from that, in your own words."}
                return {"content": f"The laptop could not do it. It reports: {done['result']}\nTell them so, in a few words."}
            turn["tell"].append("It is taking a little longer, so the answer goes to your phone when it is ready.")
            return {"content": "Sent, but the laptop has not answered yet. The answer goes to their phone when it is ready; tell them that."}
        ahead = [
            other
            for other in await self._laptop.recent(LAPTOP_SECONDS)
            if other["id"] != job["id"] and other["status"] in ("waiting", "offered", "running") and other["at"] <= job["at"]
        ]
        if ahead:
            turn["tell"].append("It starts when the laptop finishes the job it is doing now.")
            return {"content": "Queued behind the job the laptop is doing now; it starts when that one finishes, and the outcome goes to their phone."}
        return {"content": "Sent. The laptop is on, so it starts in a moment; the outcome goes to their phone when it is done."}

    async def _cancel_laptop(self) -> dict[str, Any]:
        if self._laptop is None:
            return {"content": "There is no laptop connected.", "is_error": True}
        cancelled = await self._laptop.cancel()
        if not cancelled:
            return {"content": "Nothing to call off: no laptop job is waiting or running. Check laptop_status if they want to know how the last one went."}
        self._log(f"called off {len(cancelled)} laptop job(s)")
        return {"content": "Called off: " + "; ".join(f'"{job["task"][:200]}"' for job in cancelled) + "."}

    # ----- Answering -----

    async def _use_tool(self, call: dict[str, Any], turn: dict[str, Any]) -> dict[str, Any]:
        name = call.get("name")
        if name in ("remember", "forget"):
            return self._use_memory(turn, call)
        if name == "earlier_conversation":
            return {"content": self._earlier_text(turn["earlier"])}
        if name == "laptop":
            return await self._use_laptop(call, turn)
        if name == "laptop_status":
            return {"content": await self._laptop_status()}
        if name == "cancel_laptop_job":
            return await self._cancel_laptop()
        if name == "remind":
            return await self._use_reminder(call)
        if name == "cancel_reminder":
            return await self._cancel_reminder(call)
        if name == "schedule":
            return await self._use_schedule(call)
        return {"content": "There is no such tool.", "is_error": True}

    # ----- Reminders, and their day -----

    async def _reminders_text(self) -> str:
        """The reminders still to come, as Claude is told them with each request."""
        if self._reminders is None:
            return ""
        waiting = await self._reminders.pending()
        if not waiting:
            return ""
        lines = [f"{index + 1}. {self._when(one['at'])}: {one['text']}" for index, one in enumerate(waiting)]
        return "Reminders they have set, soonest first (cancel_reminder takes the number):\n" + "\n".join(lines)

    async def _use_reminder(self, call: dict[str, Any]) -> dict[str, Any]:
        if self._reminders is None:
            return {"content": "There is no such tool.", "is_error": True}
        given = call.get("input") if isinstance(call.get("input"), dict) else {}
        text = " ".join(given["text"].split()) if isinstance(given.get("text"), str) else ""
        if not text:
            return {"content": "Nothing was set: the reminder was empty.", "is_error": True}
        moment = None
        if isinstance(given.get("at"), str):
            for pattern in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
                try:
                    moment = datetime.strptime(given["at"].strip(), pattern)
                    break
                except ValueError:
                    continue
        if moment is None:
            return {"content": "Nothing was set: the time must be written as YYYY-MM-DD HH:MM, in their local time.", "is_error": True}
        now = self._now()
        moment = moment.replace(tzinfo=now.tzinfo)
        if moment <= now:
            return {"content": "Nothing was set: that time has already passed. Work the time out from the time given to you.", "is_error": True}
        reminder = await self._reminders.add(text, moment.timestamp())
        self._log(f"reminder set for {self._when(reminder['at'])}: {text}")
        return {"content": f"Set for {self._when(reminder['at'])}. It goes to their phone then; confirm it to them in a few words."}

    async def _cancel_reminder(self, call: dict[str, Any]) -> dict[str, Any]:
        if self._reminders is None:
            return {"content": "There is no such tool.", "is_error": True}
        given = call.get("input") if isinstance(call.get("input"), dict) else {}
        number = given.get("number")
        gone = await self._reminders.cancel(number) if isinstance(number, int) and not isinstance(number, bool) else None
        if gone is None:
            return {"content": "There is no reminder with that number.", "is_error": True}
        self._log(f"reminder cancelled: {gone['text']}")
        return {"content": f'Cancelled "{gone["text"]}".'}

    async def _use_schedule(self, call: dict[str, Any]) -> dict[str, Any]:
        if self._schedule is None:
            return {"content": "There is no such tool.", "is_error": True}
        given = call.get("input") if isinstance(call.get("input"), dict) else {}
        day = given["day"].strip() if isinstance(given.get("day"), str) and given["day"].strip() else None
        return {"content": await self._schedule.text(day)}

    def _new_turn(self, state: dict[str, Any], current: dict[str, Any] | None, latest: str, session_id: str | None) -> dict[str, Any]:
        """What one question works with, and what changes while it is answered."""
        return {
            "state": state,
            "notes": state["notes"],
            # The list as Claude is shown it, so a note is forgotten by the number Claude saw.
            "numbered": list(state["notes"]),
            "changes": list(current["changes"]) if current else [],
            "earlier": self._lookback(state, session_id),
            "latest": latest,
            "laptop_before": state["laptop_at"],
            "sent_job": False,
            "tell": [],
            # The laptop jobs sent in this turn, and whether one answered in time, so Claude writes the answer itself.
            "jobs": [],
            "answered": False,
            "session": session_id is not None,
            "reminders": "",
            # The answer so far, kept should the app stop listening part-way.
            "partial": "",
        }

    async def _run(self, request: dict[str, Any], state: dict[str, Any], turn: dict[str, Any], on_text: OnText | None) -> dict[str, Any]:
        """Run one question to its end and return everything Claude wrote."""
        messages = list(request["messages"])
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # What the answer took, without what was said: for the diagnostics.
        stats: dict[str, Any] = {"requests": 0, "searches": 0, "tools": [], "cache_read": 0, "cache_written": 0}
        text = ""
        stop_reason = ""
        for attempt in range(MAX_REQUESTS):
            new_request = True
            new_stretch = True
            written_before = len(text)

            async def forward(delta: str | None) -> None:
                nonlocal text, new_request, new_stretch
                if delta is None:
                    new_stretch = True
                    return
                if not delta:
                    return
                if new_stretch:
                    delta = joint(text, delta, new_request) + delta
                new_request = new_stretch = False
                text += delta
                turn["partial"] = text
                if on_text:
                    await on_text(delta)

            # The last request allowed gets no tools, so Claude answers with what it has rather than starting
            # something (a laptop job, a note) it would never get to tell them about.
            asked = {**request, "messages": messages}
            if attempt == MAX_REQUESTS - 1:
                asked["tool_choice"] = {"type": "none"}
            result = await self._ask_claude(asked, forward)
            spent = result.get("usage") or {}
            read = spent.get("cache_read_input_tokens") or 0
            written = spent.get("cache_creation_input_tokens") or 0
            usage["prompt_tokens"] += (spent.get("input_tokens") or 0) + read + written
            usage["completion_tokens"] += spent.get("output_tokens") or 0
            stats["requests"] += 1
            stats["searches"] += result.get("searches") or 0
            stats["cache_read"] += read
            stats["cache_written"] += written
            stop_reason = result.get("stop_reason") or ""

            # A web search that ran long: hand the turn back unchanged and the API resumes it.
            if stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": result["content"]})
                continue
            # Only a turn that ended on purpose is acted on; one cut off mid-request is not.
            calls = (result.get("tool_calls") or []) if stop_reason == "tool_use" else []
            if not calls:
                break
            # Answering first, Claude passed the question on: nothing else from this round is carried out.
            handing = next((call for call in calls if call.get("name") == "hand_off"), None)
            if handing is not None:
                why = handing["input"].get("why") if isinstance(handing.get("input"), dict) else None
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                return {"text": "", "usage": usage, "stop_reason": stop_reason, "stats": stats, "hand_off": why if why in ("thinking", "web", "laptop") else "thinking"}
            messages.append({"role": "assistant", "content": result["content"]})
            outcomes = [await self._use_tool(call, turn) for call in calls]
            # A tool that turned the call down is marked, so the diagnostics show a laptop job that did not go out.
            stats["tools"] += [f"{call.get('name')}{' (turned down)' if outcome.get('is_error') else ''}" for call, outcome in zip(calls, outcomes)]
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": call["id"], **outcome} for call, outcome in zip(calls, outcomes)],
                }
            )
            # Kept straight away, so a note or a laptop job survives whatever happens to the rest of the turn.
            await self._store.write(state)
            # A laptop job that went out needs no second request, as what to tell them is already known:
            # the words Claude wrote alongside the call, or else the "say" it gave with the job.
            sent_job = any(call.get("name") == "laptop" for call in calls)
            if sent_job and not turn["answered"] and all(call.get("name") in NO_FOLLOW_UP for call in calls) and not any(outcome.get("is_error") for outcome in outcomes):
                if text[written_before:].strip():
                    break
                said = " ".join(
                    call["input"]["say"].strip()
                    for call in calls
                    if call.get("name") == "laptop" and isinstance(call.get("input"), dict) and isinstance(call["input"].get("say"), str)
                ).strip()
                if said:
                    await forward(None)
                    await forward(" ".join([said, *dict.fromkeys(turn["tell"])]))
                    break
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return {"text": text.strip(), "usage": usage, "stop_reason": stop_reason, "stats": stats}

    async def _answer(
        self,
        state: dict[str, Any],
        turn: dict[str, Any],
        messages: list[Message],
        app_notes: str,
        since: float | None,
        on_text: OnText | None,
        *,
        session: bool = False,
    ) -> dict[str, Any]:
        """Asks Claude, carrying out any tools it uses, and returns its answer.

        With the model set to pick for itself, a quick question goes to Haiku first, held back rather than shown as it
        is written, so that when Haiku hands the question on, nothing it wrote is seen; Sonnet then answers it.
        """
        spent: dict[str, Any] | None = None
        reminders = turn["reminders"]
        if session:
            model, effort = self._session_model(turn["latest"])
            used = model.split("-")[1]
        elif self._auto and not self._needs_more(turn["latest"]):
            first = self._build_request(messages, app_notes, state["notes"], turn["changes"], since, model=HAIKU, effort=self._effort, first=True, reminders=reminders)
            tried = await self._run(first, state, turn, None)
            if not tried.get("hand_off") and tried["stop_reason"] != "refusal" and tried["text"]:
                tried["stats"]["model"] = "haiku"
                if on_text:
                    await on_text(tried["text"])
                return tried
            # Haiku passed it on, or could not answer (it has no fallback of its own when a classifier declines).
            why = tried.get("hand_off") or ("refused" if tried["stop_reason"] == "refusal" else "no answer")
            spent = tried
            model = SONNET
            effort = "medium" if why == "thinking" and self._effort == "low" else self._effort
            used = f"haiku, then sonnet ({why})"
            turn["partial"] = ""
        else:
            # The chosen model; or, with the model picking for itself, a question Haiku would only hand on.
            model, effort = (SONNET if self._auto else self.model), self._effort
            used = model.split("-")[1]
        request = self._build_request(messages, app_notes, state["notes"], turn["changes"], since, session, model=model, effort=effort, reminders=reminders)
        result = await self._run(request, state, turn, on_text)
        result["stats"]["model"] = used
        if spent:
            # What the first try cost counts too.
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                result["usage"][name] += spent["usage"][name]
            for name in ("requests", "searches", "cache_read", "cache_written"):
                result["stats"][name] += spent["stats"][name]
            result["stats"]["tools"] = spent["stats"]["tools"] + result["stats"]["tools"]
        if result["stop_reason"] == "refusal":
            # A classifier stopped the answer: whatever came before is dropped, as the API asks.
            partial = result["text"]
            result["text"] = REFUSED
            result["refused"] = True
            if on_text:
                await on_text(f"\n\n{REFUSED}" if partial else REFUSED)
        elif not result["text"]:
            result["text"] = EMPTY
            if on_text:
                await on_text(EMPTY)
        return result

    def _needs_more(self, said: str) -> bool:
        """Whether a quick question is one the quick model would only hand on: for the laptop, or needing the web."""
        return (self._laptop_on and MENTIONS_LAPTOP.search(said) is not None) or NEEDS_MORE.search(said) is not None

    @staticmethod
    def _worth_keeping(result: dict[str, Any]) -> bool:
        """Whether an exchange becomes part of the conversation. Words not meant for Claude, and a question a classifier
        declined, are left out, so they are not sent again with every later question."""
        return not result.get("refused") and result["text"].strip() != NOTHING_ASKED

    async def _keep_cut_off(self, state: dict[str, Any], turn: dict[str, Any], keep: Callable[[str], None]) -> None:
        """Records an answer the app stopped listening to: what they asked and what Claude got as far as writing,
        so asking again does not send a laptop job twice and Claude knows they did not see the rest."""
        partial = turn["partial"].strip()
        keep(f"{partial} {CUT_OFF}" if partial else CUT_OFF)
        await asyncio.shield(self._store.write(state))

    async def reply(self, body: dict[str, Any], on_text: OnText | None = None) -> dict[str, Any]:
        """Answer one request. Returns the reply's `text` and its `usage`.

        `on_text` is called with each piece of the reply as it grows. Raises
        BadRequest when there is nothing to answer, and passes on whatever
        `ask_claude` raises when Claude cannot be reached.
        """
        incoming, app_notes = convert_messages(body.get("messages"))
        if not incoming:
            raise BadRequest("No user message in the request")

        async with self._turn:
            started = self._now().timestamp()
            state = read_state(await self._store.read())
            self._expire(state["notes"])
            latest = incoming[-1]["content"]
            thread = state["thread"]
            same_conversation = thread is not None and started - thread["at"] < self._thread_seconds

            # "New chat", on its own or with the first question of the new one after it.
            fresh = NEW_CHAT.match(latest) if len(incoming) == 1 else None
            if fresh:
                same_conversation = False
                latest = latest[fresh.end() :].strip()
                if not re.search(r"\w", latest):
                    self._retire(state)
                    await self._store.write(state)
                    if on_text:
                        await on_text(FRESH)
                    return {"text": FRESH, "usage": dict(NO_USAGE), "stats": {"requests": 0, "searches": 0, "tools": []}}
                incoming = [{"role": "user", "content": latest}]
            if not same_conversation:
                self._retire(state)

            self._retire_sessions(state)

            # When the app sends only the newest line, carry the conversation on from where it was left.
            carried_on = len(incoming) == 1 and same_conversation
            messages = [*thread["turns"], *incoming] if carried_on else incoming
            since = started - thread["at"] if carried_on else None
            current = thread if same_conversation else None
            turn = self._new_turn(state, current, latest, None)
            turn["reminders"] = await self._reminders_text()

            def keep(answer: str) -> None:
                turns, _ = trimmed([*messages, {"role": "assistant", "content": answer}], MAX_TURNS, KEEP_TURNS)
                state["thread"] = {
                    "started": current["started"] if current else started,
                    "at": self._now().timestamp(),
                    "turns": turns,
                    "changes": turn["changes"][-MAX_CHANGES:],
                }

            try:
                result = await self._answer(state, turn, messages, app_notes, since, on_text)
            except AnswerStopped:
                await self._keep_cut_off(state, turn, keep)
                return {"text": turn["partial"].strip(), "usage": dict(NO_USAGE), "stats": {"requests": 0, "searches": 0, "tools": []}, "stopped": True}
            except asyncio.CancelledError:
                await self._keep_cut_off(state, turn, keep)
                raise
            if self._worth_keeping(result):
                keep(result["text"])
            elif current:
                # The conversation stays as it was, and its clock is not restarted; only a change to the notes is kept.
                current["changes"] = turn["changes"][-MAX_CHANGES:]
                state["thread"] = current
            await self._store.write(state)
            result["jobs"] = turn["jobs"]
            return result

    async def session_reply(self, session_id: str, said: str, on_text: OnText | None = None) -> dict[str, Any]:
        """Answer one turn of a session in the glasses app, which lasts until it is ended."""
        said = " ".join(said.split())
        if not said:
            raise BadRequest("Nothing was said")
        async with self._turn:
            now = self._now().timestamp()
            state = read_state(await self._store.read())
            self._expire(state["notes"])
            self._retire_sessions(state)
            # A quick chat that has gone quiet is filed away, as the next quick question would, so it can be looked up.
            if state["thread"] and now - state["thread"]["at"] >= self._thread_seconds:
                self._retire(state)

            fresh = NEW_CHAT.match(said)
            if fresh:
                self._retire_sessions(state, only=session_id)
                said = said[fresh.end() :].strip()
                if not re.search(r"\w", said):
                    await self._store.write(state)
                    if on_text:
                        await on_text(FRESH)
                    return {"text": FRESH, "usage": dict(NO_USAGE), "stats": {"requests": 0, "searches": 0, "tools": []}}

            session = state["sessions"].get(session_id)
            messages = [*(session["turns"] if session else []), {"role": "user", "content": said}]
            since = now - session["at"] if session else None
            turn = self._new_turn(state, session, said, session_id)
            turn["reminders"] = await self._reminders_text()

            def keep(answer: str) -> None:
                turns, dropped = trimmed([*messages, {"role": "assistant", "content": answer}], SESSION_TURNS, SESSION_KEEP_TURNS)
                started = session["started"] if session else now
                if dropped:
                    # The part cut off is filed with the earlier conversations, so "what did we say at the start" still works.
                    self._file_away(state, {"started": started, "at": now, "turns": dropped}, "session")
                state["sessions"][session_id] = {
                    "started": started,
                    "at": self._now().timestamp(),
                    "turns": turns,
                    "changes": turn["changes"][-MAX_CHANGES:],
                }

            try:
                result = await self._answer(state, turn, messages, "", since, on_text, session=True)
            except AnswerStopped:
                await self._keep_cut_off(state, turn, keep)
                return {"text": turn["partial"].strip(), "usage": dict(NO_USAGE), "stats": {"requests": 0, "searches": 0, "tools": []}, "stopped": True}
            except asyncio.CancelledError:
                await self._keep_cut_off(state, turn, keep)
                raise
            if self._worth_keeping(result):
                keep(result["text"])
            elif session:
                session["changes"] = turn["changes"][-MAX_CHANGES:]
            await self._store.write(state)
            result["jobs"] = turn["jobs"]
            return result

    async def end_session(self, session_id: str) -> None:
        """The wearer closed the session: keep it with the earlier conversations."""
        async with self._turn:
            state = read_state(await self._store.read())
            if session_id in state["sessions"]:
                self._retire_sessions(state, only=session_id)
                await self._store.write(state)

    # ----- The night's look over the day -----

    async def reflect(self) -> dict[str, Any]:
        """Goes over the conversations since the last time, on the quick model, and saves what is worth keeping about
        them as notes. Meant to run once a night; it costs a fraction of a cent."""
        async with self._turn:
            state = read_state(await self._store.read())
            self._expire(state["notes"])
            since = state["reflected_at"] or 0.0
            now = self._now().timestamp()
            found = [old for old in state["earlier"] if old["at"] > since]
            found += [
                {"started": one["started"], "at": one["at"], "turns": one["turns"], "kind": "session"}
                for one in state["sessions"].values()
                if one["at"] > since
            ]
            thread = state["thread"]
            if thread and thread["at"] > since:
                found.append({"started": thread["started"], "at": thread["at"], "turns": thread["turns"], "kind": "chat"})
            found.sort(key=lambda old: old["at"])
            if not found:
                return {"added": [], "looked_at": 0}
            request = {
                "model": HAIKU,
                "max_tokens": 1024,
                "effort": "low",
                "fallback": False,
                "system": [{"type": "text", "text": f"{REFLECT_INSTRUCTIONS}\n\n{self._notes_block(state['notes'])}"}],
                "tools": [],
                "messages": [{"role": "user", "content": f"{self._conversations_text(found, REFLECT_CHARS)}\n\nReply with the JSON only."}],
            }
            pieces: list[str] = []

            async def collect(delta: str | None) -> None:
                if delta:
                    pieces.append(delta)

            await self._ask_claude(request, collect)
            added = self._add_notes(state["notes"], "".join(pieces), now)
            state["reflected_at"] = now
            await self._store.write(state)
            self._log(f"looked over {len(found)} conversation(s) and kept {len(added)} note(s)")
            return {"added": added, "looked_at": len(found)}

    def _conversations_text(self, found: list[dict[str, Any]], limit: int) -> str:
        """The conversations, oldest first, cut from the front when there is more than `limit`."""
        parts: list[str] = []
        for old in found:
            what = "Session in their glasses app" if old.get("kind") == "session" else "Conversation"
            heading = f"{what} from {self._when(old['started'])} to {self._when(old['at'])}:"
            lines = [f"{'They' if turn['role'] == 'user' else 'You'} said: {turn['content']}" for turn in old["turns"]]
            parts.append("\n".join([heading, *lines]))
        text = "\n\n".join(parts)
        if len(text) > limit:
            text = "(The start is left out.)\n" + text[-limit:]
        return text

    def _add_notes(self, notes: list[dict[str, Any]], written: str, now: float) -> list[str]:
        """Saves the notes in Claude's JSON reply that are new, up to the room there is. Returns them."""
        start, end = written.find("{"), written.rfind("}")
        try:
            parsed = json.loads(written[start : end + 1]) if start >= 0 and end > start else None
        except ValueError:
            parsed = None
        given = parsed.get("notes") if isinstance(parsed, dict) else None
        if not isinstance(given, list):
            self._log("the look over the day gave no usable list of notes")
            return []
        added: list[str] = []
        for raw in given:
            note = " ".join(raw.split())[:MAX_NOTE_CHARS] if isinstance(raw, str) else ""
            if not note or any(saved["text"].lower() == note.lower() for saved in notes) or len(notes) >= MAX_NOTES:
                continue
            notes.append({"at": now, "text": note})
            added.append(note)
            self._log(f"remembered from the day: {note}")
        return added

    async def edit_notes(self, edit: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]) -> None:
        """Change the notes by hand, between questions, so an answer in progress cannot write over the edit."""
        async with self._turn:
            state = read_state(await self._store.read())
            state["notes"] = edit(state["notes"])
            await self._store.write(state)

    async def vocabulary_sources(self) -> list[dict[str, Any]]:
        """The saved notes, for the speech recognizer's list of names."""
        return read_state(await self._store.read())["notes"]
