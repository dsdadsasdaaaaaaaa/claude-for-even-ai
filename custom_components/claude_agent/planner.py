"""The wearer's day, as read from their DayFlow planner: the school timetable, their own tasks, and their phone
calendar.

A DayFlow snapshot holds a good deal more than that (client records, message threads, paid meetings). None of it
belongs here, so `from_snapshot` keeps the three things above and lets the rest fall away: it is never stored,
logged, shown, or sent to Claude. Client names are read for one purpose only, which is to keep out anything that
mentions one. They are matched the way DayFlow's own privacy screen matches them: by the whole name, and by its first
word and any later word that is not a describing one ("Sam Top" also hides "Text Sam back"), with accents ignored.

The rules are the same as Second Brain's (src/planner/planner.ts), kept in step with it.
"""

from __future__ import annotations

import calendar as _calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
import re
from typing import Any, Callable
import unicodedata

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class Entry:
    """One thing on one day."""

    id: str
    title: str
    date: str
    # Minutes from midnight, or None for an all-day item.
    start: int | None
    end: int | None
    kind: str  # class, task or event
    done: bool
    # The time is a best guess for an unusual school day, not taken from the bell sheet.
    approximate: bool


@dataclass
class Recurrence:
    freq: str  # daily, weekly, monthly or yearly
    interval: int
    # 0 = Sunday .. 6 = Saturday. Empty means the weekday of the first occurrence.
    weekdays: list[int]
    until: str | None


@dataclass
class Item:
    id: str
    title: str
    # For a repeating item, its first day.
    date: str
    start: int | None
    duration: int
    kind: str
    approximate: bool
    recurrence: Recurrence | None
    skips: set[str]
    completed: bool
    completions: set[str]


@dataclass
class Calendar:
    # 'ok' when the phone calendar was read; 'unavailable' when it is switched off or was never allowed; 'missing' when
    # none was sent.
    status: str
    # The days it covers, when DayFlow said.
    first: str | None
    last: str | None


@dataclass
class Planner:
    # When this copy was fetched, epoch seconds.
    fetched_at: float
    items: list[Item]
    # True for text that mentions a client. `full_names_only` checks whole names alone, for a timetable class, whose
    # subject can share a client's first name.
    names_client: Callable[[str, bool], bool]
    # DayFlow shares only the schedule and hides client names, so no text can be checked against them: only school
    # items are kept.
    names_hidden: bool
    calendar: Calendar
    # The day DayFlow starts its weeks on, for "every other week": 0 Sunday, 1 Monday.
    week_start: int
    # When DayFlow last refreshed what it shares (epoch seconds), if it said.
    stored_at: float | None


# ----- Dates (always local, always handled as YYYY-MM-DD keys) -----


def date_key(day: date) -> str:
    return day.strftime("%Y-%m-%d")


def from_key(key: str) -> date:
    return date.fromisoformat(key)


def add_days(key: str, days: int) -> str:
    return date_key(from_key(key) + timedelta(days=days))


def days_between(first: str, second: str) -> int:
    return (from_key(second) - from_key(first)).days


def _weekday(day: date) -> int:
    """0 Sunday .. 6 Saturday, as DayFlow counts."""
    return (day.weekday() + 1) % 7


def minutes_of_day(moment: datetime) -> int:
    return moment.hour * 60 + moment.minute


def clock(minutes: int) -> str:
    hour = minutes // 60 % 24
    minute = minutes % 60
    return f"{12 if hour % 12 == 0 else hour % 12}:{minute:02d} {'AM' if hour < 12 else 'PM'}"


def day_label(key: str) -> str:
    day = from_key(key)
    return f"{day:%a}, {day:%b} {day.day}"


# ----- Reading a snapshot -----


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def clean_title(raw: str) -> str:
    """A title as it may go into a prompt: one line, no brackets that could pass for markup, at most 80 characters."""
    clean = re.sub(r"[\x00-\x1f\x7f-\x9f  ]+", " ", raw)
    clean = re.sub(r"[<>\[\]]", "", clean)
    clean = " ".join(clean.split())
    return f"{clean[:79].rstrip()}…" if len(clean) > 80 else clean


def stamp(value: Any) -> float | None:
    """DayFlow's freshness stamp, as epoch seconds, whether it came as milliseconds or a date string."""
    number = _num(value)
    if number is not None:
        return number / 1000
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def from_snapshot(data: Any, fetched_at: float) -> Planner:
    """Builds the planner from DayFlow's `data` object. See the note at the top about what is left out."""
    snapshot = _record(data)
    items: list[Item] = []
    names_client, names_hidden = _client_matcher(snapshot)

    for raw in _list(snapshot.get("tasks")):
        task = _record(raw)
        # Paid meetings are client business, not the wearer's school or personal day.
        if task.get("meeting") is not None:
            continue
        day = _str(task.get("date"))
        item_id = _str(task.get("id"))
        title = clean_title(_str(task.get("title")))
        if not item_id or not title or not DATE.match(day):
            continue
        tags = [_str(tag) for tag in _list(task.get("tags"))]
        school = "school" in tags or "timetable" in tags
        # With client names hidden nothing can be checked, so only school items are trusted.
        if (not school) if names_hidden else names_client(title, "timetable" in tags):
            continue
        start = _num(task.get("startMinutes"))
        timed = task.get("allDay") is not True and start is not None
        source = next((tag for tag in tags if tag.startswith("times:")), None)
        # A lesson is on the timetable or belongs to a block. Fairs, closures and exam weeks are tagged as school too,
        # but they are events.
        lesson = timed and any(tag == "timetable" or tag.startswith("block:") or tag.startswith("special:") for tag in tags)
        duration = _num(task.get("durationMinutes"))
        items.append(
            Item(
                id=item_id,
                title=title,
                date=day,
                start=int(start) if timed and start is not None else None,
                duration=60 if duration is None else int(duration),
                kind="class" if lesson else "event" if "school" in tags else "task",
                approximate=source is not None and source != "times:sheet",
                recurrence=_read_recurrence(task.get("recurrence")),
                skips={_str(skip) for skip in _list(task.get("skips"))},
                completed=task.get("completed") is True,
                completions={day for day, done in _record(task.get("completions")).items() if done is True},
            )
        )

    calendar = _record(snapshot.get("calendar"))
    for index, raw in enumerate(_list(calendar.get("events"))):
        event = _record(raw)
        day = _str(event.get("date"))
        # DayFlow sends calendar titles only at its widest sharing. Without one, it is still busy time.
        titled = clean_title(_str(event.get("title")))
        title = titled or "Busy"
        if not DATE.match(day) or (titled and names_client(titled, False)):
            continue
        start = None if event.get("allDay") is True else _num(event.get("startMinutes"))
        end = _num(event.get("endMinutes"))
        items.append(
            Item(
                id=f"calendar-{index}",
                title=title,
                date=day,
                start=int(start) if start is not None else None,
                duration=int(end - start) if start is not None and end is not None and end > start else 60,
                kind="event",
                approximate=False,
                recurrence=None,
                skips=set(),
                completed=False,
                completions=set(),
            )
        )

    first = _str(calendar.get("from"))
    last = _str(calendar.get("to"))
    state = Calendar(
        status="missing" if snapshot.get("calendar") is None else "ok" if calendar.get("status") == "ok" else "unavailable",
        first=first if DATE.match(first) else None,
        last=last if DATE.match(last) else None,
    )
    week_start = 0 if _record(snapshot.get("settings")).get("weekStartsOn") == 0 else 1
    return Planner(
        fetched_at=fetched_at,
        items=items,
        names_client=names_client,
        names_hidden=names_hidden,
        calendar=state,
        week_start=week_start,
        stored_at=stamp(snapshot.get("storedAt")),
    )


# Words people use to describe a contact rather than name them ("Alex Guy", "Sam Top"), and the commonest English
# words. Only ever checked from the second word of a name on, as DayFlow does.
DESCRIBING = set(
    (
        "the and but not yes okay for from with about into over under out off new old big small little "
        "tall short long young good bad nice great best better top super high low hot cold cool fast slow "
        "early late first second third last next one two three four five six seven eight nine ten "
        "guy girl man woman boy lady dude bro sis mom dad wife husband friend buddy boss mate "
        "work home house car gym school office shop store bar club hotel condo apartment "
        "client lead maybe potential regular weekly monthly daily vip cash card paid owes free cheap "
        "weekend morning afternoon evening night today tomorrow monday tuesday wednesday thursday "
        "friday saturday sunday north south east west downtown uptown city town "
        "red blue green black white brown grey gray pink gold silver "
        "fat thin bald beard blonde ginger tattoo glasses loud quiet funny weird rude kind sweet "
        "real fake main other another same just only also very too more less most all any some "
        "every each this that these those here there when where what who why how "
        "can does did done have has had are was were been being get got give gave take took come came "
        "going gone see saw seen know knew think thought want need like love call text message phone "
        "number email address time day week month year hour minute deal price rate money pay tip "
        "deposit book booked booking meet meeting session appointment cancel cancelled flake flaky "
        "repeat returning referral online app site web telegram sms whatsapp insta instagram snap"
    ).split(" ")
)


def _fold(text: str) -> str:
    """Lower case with accents taken off, so "José" and "Jose" are one name."""
    return "".join(char for char in unicodedata.normalize("NFD", text) if not unicodedata.combining(char)).lower()


def _words_of(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", _fold(text))


def _client_matcher(snapshot: dict[str, Any]) -> tuple[Callable[[str, bool], bool], bool]:
    """A test for "does this text mention a client", and whether the names are hidden. Names come from the client
    list, paid meetings and the meeting log, live only inside the returned test, and decide nothing except what gets
    left out. At DayFlow's "schedule only" sharing the names arrive as person-1, person-2 and so on, which nothing can
    be checked against."""
    names: set[str] = set()
    hidden = snapshot.get("scope") == "schedule"

    def add(value: Any) -> None:
        nonlocal hidden
        name = _fold(_str(value).strip())
        if re.match(r"^person-\d+$", name):
            hidden = True
        # Shorter than three letters would match half the dictionary.
        elif len(name) >= 3:
            names.add(name)

    for raw in _list(snapshot.get("clients")):
        client = _record(raw)
        add(client.get("name"))
        add(client.get("displayName"))
    for raw in _list(snapshot.get("tasks")):
        add(_record(_record(raw).get("meeting")).get("client"))
    # A client typed only on a meeting whose task was later deleted lives on in the log.
    for raw in _list(snapshot.get("meetingLog")):
        add(_record(raw).get("client"))
    if hidden:
        return (lambda text, full_names_only=False: True), True
    if not names:
        return (lambda text, full_names_only=False: False), False

    whole: set[str] = set()
    phrases: list[list[str]] = []
    parts: set[str] = set()
    for name in names:
        words = _words_of(name)
        if len(words) == 1:
            whole.add(words[0])
        elif len(words) > 1:
            phrases.append(words)
        for index, word in enumerate(words):
            if len(word) >= 3 and (index == 0 or word not in DESCRIBING):
                parts.add(word)

    def names_client(text: str, full_names_only: bool = False) -> bool:
        words = _words_of(text)
        if any(word in whole or (not full_names_only and word in parts) for word in words):
            return True
        joined = f" {' '.join(words)} "
        return any(f" {' '.join(phrase)} " in joined for phrase in phrases)

    return names_client, False


def _read_recurrence(raw: Any) -> Recurrence | None:
    if raw is None:
        return None
    rule = _record(raw)
    if rule.get("freq") not in ("daily", "weekly", "monthly", "yearly"):
        return None
    until = _str(rule.get("until"))
    interval = _num(rule.get("interval"))
    return Recurrence(
        freq=rule["freq"],
        interval=max(1, int(interval) if interval is not None else 1),
        # DayFlow has only ever been seen to use 1-5 here; treat a 7 as Sunday in case it counts ISO-style.
        weekdays=[int(day) % 7 for day in _list(rule.get("weekdays")) if _num(day) is not None],
        until=until if DATE.match(until) else None,
    )


# ----- Asking what is on a day -----


def _week_of(key: str, week_start: int) -> str:
    """The first day of the week `key` falls in, for weeks starting on `week_start`."""
    return add_days(key, -((_weekday(from_key(key)) - week_start + 7) % 7))


def _occurs_on(item: Item, day: str, week_start: int) -> bool:
    """Whether `item` falls on `day`. The repeat rules are DayFlow's own."""
    if day in item.skips:
        return False
    rule = item.recurrence
    if rule is None:
        return item.date == day
    diff = days_between(item.date, day)
    if diff < 0 or (rule.until is not None and day > rule.until):
        return False
    first = from_key(item.date)
    this = from_key(day)
    if rule.freq == "daily":
        return diff % rule.interval == 0
    if rule.freq == "weekly":
        weekdays = rule.weekdays or [_weekday(first)]
        if _weekday(this) not in weekdays:
            return False
        # Count whole weeks from the week the series began, so "every 2 weeks" keeps its rhythm.
        weeks = round(days_between(_week_of(item.date, week_start), _week_of(day, week_start)) / 7)
        return weeks % rule.interval == 0
    if rule.freq == "monthly":
        # The 31st in a thirty-day month is the 30th, and February's last day stands in for the 29th to 31st.
        last_of_month = _calendar.monthrange(this.year, this.month)[1]
        if this.day != min(first.day, last_of_month):
            return False
        months = (this.year - first.year) * 12 + this.month - first.month
        return months % rule.interval == 0
    years = this.year - first.year
    if years % rule.interval != 0 or this.month != first.month:
        return False
    if this.day == first.day:
        return True
    # A 29 February start falls on the 28th in other years.
    return first.day == 29 and this.month == 2 and this.day == 28 and not _calendar.isleap(this.year)


def entries_on(planner: Planner, day: str) -> list[Entry]:
    """Everything on `day`, timed entries first in order, then all-day ones."""
    entries = [
        Entry(
            id=item.id,
            title=item.title,
            date=day,
            start=item.start,
            end=None if item.start is None else item.start + item.duration,
            kind=item.kind,
            done=day in item.completions if item.recurrence else item.completed,
            approximate=item.approximate,
        )
        for item in planner.items
        if _occurs_on(item, day, planner.week_start)
    ]
    return sorted(entries, key=lambda entry: (entry.start if entry.start is not None else math.inf, entry.title))


# ----- Words for the model -----


def _timed(entry: Entry) -> str:
    if entry.start is None or entry.end is None:
        return entry.title
    return f"{'~' if entry.approximate else ''}{clock(entry.start)}-{clock(entry.end)} {entry.title}"


def _calendar_silent_on(planner: Planner, day: str) -> bool:
    """Whether the phone calendar says nothing about `day`: switched off, or the day lies outside the days it covers.
    A day like that is never "free"."""
    calendar = planner.calendar
    if calendar.status == "missing":
        return False
    if calendar.status != "ok":
        return True
    return (calendar.first is not None and day < calendar.first) or (calendar.last is not None and day > calendar.last)


def describe_day(planner: Planner, day: str) -> str:
    """One line listing a day."""
    entries = [entry for entry in entries_on(planner, day) if not entry.done]
    unchecked = " (phone calendar not checked)" if _calendar_silent_on(planner, day) else ""
    if not entries:
        return f"{day_label(day)}: nothing scheduled{unchecked}"
    parts = [_timed(entry) for entry in entries if entry.start is not None]
    all_day = [entry.title for entry in entries if entry.start is None]
    if all_day:
        parts.append(f"all day: {', '.join(all_day)}")
    return f"{day_label(day)}: {' | '.join(parts)}{unchecked}"


# Lunch, spares, breaks and support blocks sit on the timetable but are not lessons.
FREE_PERIOD = re.compile(r"\b(lunch|spare|free|break|recess|support|advisory|assembly)\b", re.IGNORECASE)


def is_free_period(title: str) -> bool:
    return FREE_PERIOD.search(title) is not None


def now_and_next(planner: Planner, now: datetime) -> tuple[Entry | None, list[Entry], list[Entry]]:
    """What is on now (a lesson if one is, otherwise whatever began last), everything on now, and the timed entries
    still to come today, soonest first."""
    minute = minutes_of_day(now)
    today = [entry for entry in entries_on(planner, date_key(now.date())) if entry.start is not None and entry.end is not None and not entry.done]
    running = [entry for entry in today if entry.start <= minute < entry.end]  # type: ignore[operator]
    # A longer entry that began earlier (a study block, an all-morning event) must not hide the lesson that is on.
    lesson = next((entry for entry in running if entry.kind == "class" and not is_free_period(entry.title)), None)
    latest = max(running, key=lambda entry: entry.start, default=None)  # type: ignore[arg-type,return-value]
    return lesson or latest, running, [entry for entry in today if entry.start > minute]  # type: ignore[operator]


AHEAD_DAYS = 7
AHEAD_LIMIT = 14
# DayFlow refreshes what it shares every few minutes while it is open; a copy older than this may have missed changes.
STALE_SNAPSHOT_SECONDS = 6 * 60 * 60


def build_brief(planner: Planner, now: datetime) -> str:
    """A compact picture of the wearer's schedule: where they are in today, today and tomorrow in full, and the tests,
    deadlines and events of the week ahead."""
    today = date_key(now.date())
    lines: list[str] = []

    current, _running, upcoming = now_and_next(planner, now)
    minute = minutes_of_day(now)
    here = f"Right now: {current.title} until {clock(current.end)} ({current.end - minute} min left)." if current and current.end is not None else "Right now: nothing scheduled."
    nxt = f" Next: {upcoming[0].title} at {clock(upcoming[0].start)}." if upcoming and upcoming[0].start is not None else " Nothing else is scheduled today."
    lines.append(here + nxt)
    lines.append(f"Today, {describe_day(planner, today)}")
    lines.append(f"Tomorrow, {describe_day(planner, add_days(today, 1))}")

    ahead: list[str] = []
    listed: set[str] = set()
    for offset in range(2, AHEAD_DAYS + 1):
        if len(listed) >= AHEAD_LIMIT:
            break
        day = add_days(today, offset)
        # Lessons repeat every week; what is worth knowing in advance is everything else, each thing once.
        notable = [entry for entry in entries_on(planner, day) if entry.kind != "class" and not entry.done and entry.id not in listed][: AHEAD_LIMIT - len(listed)]
        for entry in notable:
            listed.add(entry.id)
        if notable:
            ahead.append(f"{day_label(day)}: {', '.join(_timed(entry) for entry in notable)}")
    if ahead:
        lines.append(f"Coming up: {'; '.join(ahead)}")

    if any(item.approximate for item in planner.items):
        lines.append("A time marked ~ is estimated for an unusual school day; say so when you quote it.")
    if planner.names_hidden:
        lines.append("DayFlow shares only school items at the moment, so personal tasks are not listed.")
    if planner.stored_at is not None and planner.fetched_at - planner.stored_at > STALE_SNAPSHOT_SECONDS:
        refreshed = datetime.fromtimestamp(planner.stored_at, now.tzinfo)
        lines.append(f"DayFlow last refreshed this at {clock(minutes_of_day(refreshed))} on {day_label(date_key(refreshed.date()))}; anything changed since is missing.")
    return "\n".join(lines)
