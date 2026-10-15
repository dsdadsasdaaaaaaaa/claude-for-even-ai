"""Speech to text for sessions from the glasses app: a recording in, the words out.

The app sends what was said while the ring was held; this hands it to an
OpenAI-compatible transcription service (Groq's Whisper by default), along with
the names and words the wearer uses, so those come out spelled right.

Nothing here knows about Home Assistant, so it can be tested on its own.
"""

from __future__ import annotations

from array import array
import logging
import math
import re
from typing import Any

_LOGGER = logging.getLogger(__name__)

PROVIDERS = {
    # The full model, not the turbo one: a little slower, noticeably better with names and accents.
    "groq": {"label": "Groq (Whisper)", "base_url": "https://api.groq.com/openai/v1", "model": "whisper-large-v3"},
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini-transcribe"},
}
DEFAULT_PROVIDER = "groq"
# Whisper reads at most about 224 tokens of hints, so only this much of the vocabulary is sent.
MAX_VOCABULARY_CHARS = 700
# A recording longer than this is turned down before it is sent anywhere.
MAX_AUDIO_BYTES = 4 * 1024 * 1024

MISHEAR_LABEL = re.compile(r"words you might mishear\s*:?", re.IGNORECASE)
# The next "Something:" heading, which ends the list.
NEXT_LABEL = re.compile(r"^\s*[A-Z][\w ]{0,40}:(\s|$)")
# How a word sounds and what it gets misheard as are quoted, or in brackets; the real word is not.
QUOTED = re.compile(r"[\"“”][^\"“”]*[\"“”]|‘[^’]*’|(?<!\w)'[^']*'(?!\w)|\([^)]*\)")
# The words that tie the parts of an entry together: "Name, said "naym", may become "Nam" or ...".
JOINERS = re.compile(
    r"\b(said|says|pronounced|sounds?(\s+like)?|may|might|can|could|often|usually|becomes?|comes?\s+out(\s+as)?"
    r"|turns?\s+into|heard\s+as|written\s+as|spel(led|t)(\s+as)?|or|as|like)\b",
    re.IGNORECASE,
)
# A list marker at the start of a line.
BULLET = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+")


class SpeechError(Exception):
    """The recording could not be turned into words."""


def is_wav(audio: bytes) -> bool:
    return len(audio) > 44 and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"


# What Whisper tends to "hear" in a silent or near-silent recording.
SILENCE_WORDS = {"", "you", "thank you", "thanks", "thank you for watching", "thanks for watching", "bye", "okay", "uh", "um", "hmm"}


def seconds_of(audio: bytes) -> float:
    """How long a 16 kHz, 16-bit mono WAV recording is."""
    return max(0, len(audio) - 44) / 32000


# Louder than this somewhere in the recording, someone was talking to the glasses, so even "Okay" was really said.
SPOKEN_LOUDNESS = 0.08


def loudness(audio: bytes) -> float:
    """The loudest tenth of a second in a 16-bit WAV recording, from 0 to 1 (the same measure as the app's)."""
    samples = array("h")
    body = audio[44:]
    samples.frombytes(body[: len(body) // 2 * 2])
    window = 1600
    loudest = 0.0
    for start in range(0, len(samples), window):
        part = samples[start : start + window : 4]
        if part:
            loudest = max(loudest, math.sqrt(sum(sample * sample for sample in part) / len(part)) / 32768)
    return loudest


def probably_silence(text: str, audio: bytes, hints: str = "", *, seconds: float | None = None, loud: float | None = None) -> bool:
    """Whether the words are what Whisper makes of a recording with nobody talking to it: a press too short, a stock
    phrase from a quiet recording, or the list of names it was given, read back. A compressed recording cannot be
    measured here, so the app says how long and how loud it was."""
    words = " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())
    length = seconds if seconds is not None else seconds_of(audio)
    if length < 0.5:
        return True
    if words in SILENCE_WORDS and length < 4:
        return words == "" or (loud if loud is not None else loudness(audio)) < SPOKEN_LOUDNESS
    hint_words = set(re.findall(r"[\w']+", hints.lower()))
    said = words.split()
    return len(said) >= 3 and all(word in hint_words for word in said)


def _fit(terms: list[str]) -> str:
    """The terms as a comma list, as many as fit, never cut mid-word."""
    kept: list[str] = []
    for term in terms:
        if len(", ".join([*kept, term])) > MAX_VOCABULARY_CHARS:
            break
        kept.append(term)
    return ", ".join(kept)


def _mishear_entries(profile: str) -> list[str] | None:
    """The entries of the profile's "Words you might mishear" list, on one line or as a list under it; None without one."""
    found = MISHEAR_LABEL.search(profile)
    if not found:
        return None
    lines: list[str] = []
    for index, line in enumerate(profile[found.end() :].splitlines()):
        if index > 0 and (not line.strip() or NEXT_LABEL.match(line)):
            if lines:
                break
            continue
        lines.append(BULLET.sub("", line))
    entries: list[str] = []
    for line in lines:
        entries += [entry for entry in re.split(r"[;\n]", line) if entry.strip()]
    return entries


def _names(entry: str) -> list[str]:
    """The real words in one entry. One that says how a word sounds or what it becomes names only that word, before
    its first comma, bracket or joining word; one that does not is a plain list."""
    plain = QUOTED.sub(" ", entry)
    if JOINERS.search(plain):
        first = re.split(r"[,(\"“‘]", entry, maxsplit=1)[0]
        pieces = [JOINERS.split(first, maxsplit=1)[0]]
    else:
        pieces = plain.split(",")
    names = []
    for piece in pieces:
        # A full stop ends the list, but not one inside a name such as "Mr. Abrams" or "Node.js".
        term = " ".join(re.sub(r"(?<![A-Z][a-z])(?<!\b[A-Z])\.(?=\s|$)", " ", piece).split()).strip(" '’-:")
        if term and any(c.isalpha() for c in term) and len(term) <= 40:
            names.append(term)
    return names


def vocabulary(profile: str, notes: list[dict[str, Any]]) -> str:
    """The names and words to steer recognition toward, spelled right, and nothing else.

    From the profile's "Words you might mishear" list when it has one, keeping
    only the real words: the list also spells out how each sounds and what it
    gets misheard as, and handing the recognizer those would steer it toward
    the very mistakes the list warns about. Otherwise, the capitalised names in
    the profile and notes.
    """
    terms: list[str] = []
    entries = _mishear_entries(profile)
    if entries:
        for entry in entries:
            for term in _names(entry):
                if term.lower() not in (known.lower() for known in terms):
                    terms.append(term)
        if terms:
            return _fit(terms)
    text = " ".join([profile, *(note["text"] for note in notes)])
    for word in re.findall(r"\b[A-Z][a-zA-Z'’-]{2,}\b", text):
        if word not in terms and word.lower() not in COMMON:
            terms.append(word)
    return _fit(terms)


# Capitalised only because they start sentences or are always capitalised; no help to the recognizer.
COMMON = {
    "the", "they", "their", "this", "that", "these", "there", "then", "when", "what", "where", "which", "while", "with",
    "and", "but", "for", "from", "have", "has", "had", "her", "his", "she", "him", "you", "your", "our", "its", "are",
    "was", "were", "who", "how", "why", "not", "also", "about", "after", "before", "most", "some", "any", "all", "one",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march",
    "april", "may", "june", "july", "august", "september", "october", "november", "december", "english", "i'm", "i've",
}


# What a recording may come as: WAV, or Opus in an Ogg file, which is a fifteenth the size.
WAV_TYPES = ("audio/wav", "audio/x-wav", "audio/wave")
OGG_TYPES = ("audio/ogg", "audio/opus")


def is_ogg(audio: bytes) -> bool:
    return len(audio) > 28 and audio[:4] == b"OggS"


async def transcribe(http: Any, provider: str, key: str, audio: bytes, words: str = "", language: str = "en", content_type: str = "audio/wav") -> str:
    """The words in `audio`, a WAV recording or Opus in an Ogg file (`content_type` says which). `http` is an aiohttp
    client session."""
    import aiohttp  # Home Assistant always has it; imported here so the rest can be tested without it.

    if not key:
        raise SpeechError("No speech-to-text key is set under Configure")
    kind = content_type.split(";")[0].strip().lower() or "audio/wav"
    if kind in OGG_TYPES:
        if not is_ogg(audio):
            raise SpeechError("The recording is not an Ogg file")
        filename, sent_as = "speech.ogg", "audio/ogg"
    elif kind in WAV_TYPES:
        if not is_wav(audio):
            raise SpeechError("The recording is not a WAV file")
        filename, sent_as = "speech.wav", "audio/wav"
    else:
        raise SpeechError("The recording is in a format that is not accepted")
    if len(audio) > MAX_AUDIO_BYTES:
        raise SpeechError("The recording is too long")
    preset = PROVIDERS.get(provider) or PROVIDERS[DEFAULT_PROVIDER]
    form = aiohttp.FormData()
    form.add_field("file", audio, filename=filename, content_type=sent_as)
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
            try:
                body = await response.json(content_type=None)
            except ValueError:
                # An error page, or something else that is not the answer.
                body = None
            status = response.status
    except (aiohttp.ClientError, TimeoutError) as err:
        raise SpeechError("Could not reach the speech service") from err
    if status >= 400:
        detail = body.get("error") if isinstance(body, dict) else None
        message = detail.get("message") if isinstance(detail, dict) else detail
        # The service's own words go to the log; the glasses get a sentence that fits on them.
        _LOGGER.warning("The speech service answered %s: %s", status, message or "(no message)")
        if status in (401, 403):
            raise SpeechError("The speech-to-text key was turned down")
        if status == 413:
            raise SpeechError("The recording is too long")
        if status == 429:
            raise SpeechError("Speech-to-text is over its limit for now. Try again in a minute.")
        raise SpeechError("The speech service is having trouble. Try again.")
    if not isinstance(body, dict):
        raise SpeechError("The speech service sent back something unreadable")
    text = body.get("text")
    return text.strip() if isinstance(text, str) else ""
