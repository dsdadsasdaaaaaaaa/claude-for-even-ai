"""What the agent keeps between questions, read back into a shape it can rely on whatever was stored: the notes
about the wearer, the conversation in progress, open sessions and earlier conversations. Also the notes as they are
shown and edited by hand under Configure.
"""

from __future__ import annotations

from datetime import date
import re
from typing import Any

from .formats import Message

# How many earlier conversations are kept, at most.
EARLIER_KEPT = 20
MAX_NOTES = 200
MAX_NOTE_CHARS = 300
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_day(value: Any) -> date | None:
    """A YYYY-MM-DD date, or None when it is not one."""
    if not isinstance(value, str) or not ISO_DATE.match(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _turns(stored: Any) -> list[Message]:
    return [
        {"role": turn["role"], "content": turn["content"]}
        for turn in (stored if isinstance(stored, list) else [])
        if isinstance(turn, dict) and turn.get("role") in ("user", "assistant") and isinstance(turn.get("content"), str)
    ]


def _thread(thread: Any) -> dict[str, Any] | None:
    """A conversation in progress, quick chat or session, or None when there is nothing usable."""
    if not isinstance(thread, dict) or not _number(thread.get("at")):
        return None
    turns = _turns(thread.get("turns"))
    if not turns:
        return None
    stored_changes = thread.get("changes")
    changes = [
        {"action": change["action"], "text": change["text"]}
        for change in (stored_changes if isinstance(stored_changes, list) else [])
        if isinstance(change, dict) and change.get("action") in ("saved", "removed") and isinstance(change.get("text"), str)
    ]
    return {
        "started": float(thread["started"]) if _number(thread.get("started")) else float(thread["at"]),
        "at": float(thread["at"]),
        "turns": turns,
        "changes": changes,
        # Kept here before 0.6.0; read only to find when the last laptop job went out.
        "laptop_at": float(thread["laptop_at"]) if _number(thread.get("laptop_at")) else None,
    }


def read_state(raw: Any) -> dict[str, Any]:
    """Whatever was stored, coerced into the shape the agent relies on."""
    data = raw if isinstance(raw, dict) else {}
    stored_notes = data.get("notes")
    notes = []
    for note in stored_notes if isinstance(stored_notes, list) else []:
        if isinstance(note, dict) and isinstance(note.get("text"), str) and _number(note.get("at")):
            kept = {"at": float(note["at"]), "text": note["text"]}
            if parse_day(note.get("until")):
                kept["until"] = note["until"]
            notes.append(kept)

    stored_sessions = data.get("sessions")
    sessions = {}
    for session_id, session in (stored_sessions.items() if isinstance(stored_sessions, dict) else []):
        kept_session = _thread(session)
        if isinstance(session_id, str) and kept_session:
            sessions[session_id] = kept_session

    stored_earlier = data.get("earlier")
    earlier = [
        {
            "started": float(old["started"]),
            "at": float(old["at"]),
            "turns": _turns(old.get("turns")),
            "kind": "session" if old.get("kind") == "session" else "chat",
        }
        for old in (stored_earlier if isinstance(stored_earlier, list) else [])
        if isinstance(old, dict) and _number(old.get("started")) and _number(old.get("at")) and _turns(old.get("turns"))
    ]
    thread = _thread(data.get("thread"))
    # When the last laptop job went out, from a quick chat or a session alike.
    if _number(data.get("laptop_at")):
        laptop_at: float | None = float(data["laptop_at"])
    else:
        laptop_at = max((one["laptop_at"] for one in [thread, *sessions.values()] if one and one["laptop_at"]), default=None)
    return {"notes": notes, "thread": thread, "sessions": sessions, "earlier": earlier[-EARLIER_KEPT:], "laptop_at": laptop_at}


NOTE_LINE = re.compile(r"^(?P<text>.*?)\s*\(until (?P<until>\d{4}-\d{2}-\d{2})\)$")


def notes_text(notes: list[dict[str, Any]]) -> str:
    """The notes as the wearer sees and edits them under Configure: one per line."""
    return "\n".join(f"{note['text']} (until {note['until']})" if note.get("until") else note["text"] for note in notes)


def edited_notes(text: str, old: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
    """The notes as edited by hand, one per line. A note left as it was keeps the day it was saved."""
    saved_at = {note["text"].lower(): note["at"] for note in old}
    notes: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        match = NOTE_LINE.match(line)
        until = match["until"] if match and parse_day(match["until"]) else None
        body = (match["text"] if until else line).strip(" -•*")[:MAX_NOTE_CHARS]
        if not body or any(note["text"].lower() == body.lower() for note in notes):
            continue
        note: dict[str, Any] = {"at": saved_at.get(body.lower(), now), "text": body}
        if until:
            note["until"] = until
        notes.append(note)
    return notes[-MAX_NOTES:]


def apply_note_edits(shown: str, submitted: str, notes: list[dict[str, Any]], now: float) -> list[dict[str, Any]]:
    """The notes after a hand edit under Configure.

    Only the lines changed in the box count, so a note Claude saved or removed
    while the screen was open stays saved or removed.
    """
    if submitted.strip() == shown.strip():
        return notes

    def key(note: dict[str, Any]) -> tuple[str, str | None]:
        return note["text"].lower(), note.get("until")

    before = {key(note) for note in edited_notes(shown, notes, now)}
    after = edited_notes(submitted, notes, now)
    removed = before - {key(note) for note in after}
    result = [note for note in notes if key(note) not in removed]
    for note in after:
        if key(note) in before:
            continue
        same = next((kept for kept in result if kept["text"].lower() == note["text"].lower()), None)
        if same is None:
            result.append(note)
        elif note.get("until"):
            same["until"] = note["until"]
        else:
            same.pop("until", None)
    return result[-MAX_NOTES:]
