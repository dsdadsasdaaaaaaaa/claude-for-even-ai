"""Speech to text for sessions from the glasses app: a recording in, the words out.

The app sends what was said while the ring was held; this hands it to an
OpenAI-compatible transcription service (Groq's Whisper by default), along with
the names and words the wearer uses, so those come out spelled right.

Nothing here knows about Home Assistant, so it can be tested on its own.
"""

from __future__ import annotations

import re
from typing import Any

PROVIDERS = {
    "groq": {"label": "Groq (Whisper, fast)", "base_url": "https://api.groq.com/openai/v1", "model": "whisper-large-v3-turbo"},
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini-transcribe"},
}
DEFAULT_PROVIDER = "groq"
# Whisper reads at most about 224 tokens of hints, so only this much of the vocabulary is sent.
MAX_VOCABULARY_CHARS = 700
# A recording longer than this is turned down before it is sent anywhere.
MAX_AUDIO_BYTES = 4 * 1024 * 1024

MISHEAR_LINE = re.compile(r"words you might mishear:?\s*(.+)", re.IGNORECASE)


class SpeechError(Exception):
    """The recording could not be turned into words."""


def is_wav(audio: bytes) -> bool:
    return len(audio) > 44 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"


# What Whisper tends to "hear" in a silent or near-silent recording.
SILENCE_WORDS = {"", "you", "thank you", "thanks", "thank you for watching", "thanks for watching", "bye", "okay", "uh", "um", "hmm"}


def seconds_of(audio: bytes) -> float:
    """How long a 16 kHz, 16-bit mono WAV recording is."""
    return max(0, len(audio) - 44) / 32000


def probably_silence(text: str, audio: bytes) -> bool:
    """Whether the words are what Whisper makes of a recording with nobody talking: a press too short, or a stock phrase."""
    words = " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())
    return seconds_of(audio) < 0.5 or (words in SILENCE_WORDS and seconds_of(audio) < 4)


def vocabulary(profile: str, notes: list[dict[str, Any]]) -> str:
    """The names and words to steer recognition toward.

    The profile's "Words you might mishear" line when it has one, since it was
    written for exactly this; otherwise the capitalised names in the profile and notes.
    """
    found = MISHEAR_LINE.search(profile)
    if found:
        return found.group(1).strip()[:MAX_VOCABULARY_CHARS]
    text = " ".join([profile, *(note["text"] for note in notes)])
    names: list[str] = []
    for word in re.findall(r"\b[A-Z][a-zA-Z'’-]{2,}\b", text):
        if word not in names and word.lower() not in COMMON:
            names.append(word)
    return ", ".join(names)[:MAX_VOCABULARY_CHARS]


# Capitalised only because they start sentences or are always capitalised; no help to the recognizer.
COMMON = {
    "the", "they", "their", "this", "that", "these", "there", "then", "when", "what", "where", "which", "while", "with",
    "and", "but", "for", "from", "have", "has", "had", "her", "his", "she", "him", "you", "your", "our", "its", "are",
    "was", "were", "who", "how", "why", "not", "also", "about", "after", "before", "most", "some", "any", "all", "one",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march",
    "april", "may", "june", "july", "august", "september", "october", "november", "december", "english", "i'm", "i've",
}


async def transcribe(http: Any, provider: str, key: str, audio: bytes, words: str = "", language: str = "en") -> str:
    """The words in `audio`, a WAV recording. `http` is an aiohttp client session."""
    import aiohttp  # Home Assistant always has it; imported here so the rest can be tested without it.

    if not key:
        raise SpeechError("No speech-to-text key is set under Configure")
    if not is_wav(audio):
        raise SpeechError("The recording is not a WAV file")
    if len(audio) > MAX_AUDIO_BYTES:
        raise SpeechError("The recording is too long")
    preset = PROVIDERS.get(provider) or PROVIDERS[DEFAULT_PROVIDER]
    form = aiohttp.FormData()
    form.add_field("file", audio, filename="speech.wav", content_type="audio/wav")
    form.add_field("model", preset["model"])
    form.add_field("response_format", "json")
    if language:
        form.add_field("language", language)
    if words:
        form.add_field("prompt", words)
    try:
        async with http.post(
            f"{preset['base_url']}/audio/transcriptions",
            data=form,
            headers={"Authorization": f"Bearer {key}"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            body = await response.json(content_type=None)
            if response.status in (401, 403):
                raise SpeechError("The speech-to-text key was turned down")
            if response.status >= 400:
                detail = body.get("error") if isinstance(body, dict) else None
                message = detail.get("message") if isinstance(detail, dict) else detail
                raise SpeechError(str(message or f"The speech service answered {response.status}"))
    except (aiohttp.ClientError, TimeoutError) as err:
        raise SpeechError("Could not reach the speech service") from err
    text = body.get("text") if isinstance(body, dict) else None
    return text.strip() if isinstance(text, str) else ""
