"""The OpenAI chat-completions format the Even app speaks: its messages in, a reply out."""

from __future__ import annotations

import secrets
import time
from typing import Any

Message = dict[str, Any]

# The most turns of a conversation sent to Claude.
MAX_TURNS = 12
# The most of the Even app's own notes passed on to Claude.
MAX_APP_NOTES = 4000
NO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def text_of(content: Any) -> str:
    """The text of a message, whether it came as a string or as a list of parts."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


def recent(messages: list[Message], limit: int = MAX_TURNS) -> list[Message]:
    """The last few turns, starting on one of the wearer's."""
    kept = messages[-limit:]
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    return kept


def convert_messages(incoming: Any) -> tuple[list[Message], str]:
    """Turn OpenAI-style messages into the alternating list Claude expects.

    System messages are the Even app's own notes; they are returned separately.
    """
    notes: list[str] = []
    messages: list[Message] = []
    for item in incoming if isinstance(incoming, list) else []:
        if not isinstance(item, dict):
            continue
        content = text_of(item.get("content")).strip()
        if not content:
            continue
        role = item.get("role")
        if role in ("system", "developer"):
            notes.append(content)
        elif role in ("user", "assistant"):
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += f"\n{content}"
            else:
                messages.append({"role": role, "content": content})
    while messages and messages[-1]["role"] != "user":
        messages.pop()
    return recent(messages), "\n\n".join(notes)[:MAX_APP_NOTES]


def new_id() -> str:
    """An id for one reply, in the style the Even app expects."""
    return f"chatcmpl-{int(time.time()):x}{secrets.token_hex(4)}"


def completion(completion_id: str, created: int, model: str, text: str, usage: dict[str, int]) -> dict[str, Any]:
    """A whole reply in the chat-completions format."""
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": usage,
    }


def chunk(completion_id: str, created: int, model: str, delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    """One piece of a streamed reply in the chat-completions format."""
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
