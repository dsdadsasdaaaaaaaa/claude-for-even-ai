"""Jobs the glasses hand to the laptop.

The agent queues a job here. The helper on the laptop asks for the next one,
says it is starting it, carries it out with Claude Code, and reports back. The
laptop always makes the connection, so it works from whatever network it is on,
and a job waits here while the laptop is asleep or away.

A job handed out but never started (the answer was lost on a bad connection)
is handed out again, and a report that arrives late still counts, so a network
blip neither loses a job nor runs it twice.

Nothing here knows about Home Assistant, so it can be tested on its own.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

# A laptop that asked for work this recently is counted as on.
ONLINE_SECONDS = 90
# A job the laptop has not picked up by then is dropped, so it never runs long after it was asked for.
WAIT_SECONDS = 12 * 60 * 60
# A job handed out but not started within this never reached the laptop, and is handed out again.
OFFER_SECONDS = 60
# The helper gives up on a job after 20 minutes; past this it is not coming back.
RUN_SECONDS = 25 * 60
MAX_JOBS = 20
# The same job asked for again this soon after it started is the same request arriving twice, not a new one.
REPEAT_SECONDS = 60
MAX_TASK_CHARS = 4000
MAX_RESULT_CHARS = 2000

# Which Claude the agent chose for a job, when it chose.
MODELS = ("haiku", "sonnet", "opus")

WAITING, OFFERED, RUNNING = "waiting", "offered", "running"
DONE, FAILED, EXPIRED, CANCELLED = "done", "failed", "expired", "cancelled"
STATUSES = (WAITING, OFFERED, RUNNING, DONE, FAILED, EXPIRED, CANCELLED)
UNFINISHED = (WAITING, OFFERED, RUNNING)

Job = dict[str, Any]


class Store(Protocol):
    """Where the jobs are kept."""

    async def read(self) -> Any: ...

    async def write(self, state: dict[str, Any]) -> None: ...


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _time(value: Any) -> float | None:
    return float(value) if _number(value) else None


def _stats(raw: Any) -> dict[str, float]:
    """How long a job took and how many steps, as the laptop reported it. Numbers only, nothing it saw or did."""
    given = raw if isinstance(raw, dict) else {}
    return {name: float(given[name]) for name in ("seconds", "turns") if _number(given.get(name))}


def read_jobs(raw: Any) -> tuple[list[Job], float | None]:
    """Whatever was stored, coerced into the shape the queue relies on."""
    data = raw if isinstance(raw, dict) else {}
    stored = data.get("jobs")
    jobs = [
        {
            "id": job["id"],
            "task": job["task"],
            "follow_up": job.get("follow_up") is True,
            "status": job["status"],
            "at": float(job["at"]),
            "offered": _time(job.get("offered")),
            "started": _time(job.get("started")),
            "finished": _time(job.get("finished")),
            "result": job["result"] if isinstance(job.get("result"), str) else "",
            "lost": job.get("lost") is True,
            "stats": _stats(job.get("stats")),
            "model": job["model"] if job.get("model") in MODELS else None,
        }
        for job in (stored if isinstance(stored, list) else [])
        if isinstance(job, dict)
        and isinstance(job.get("id"), str)
        and isinstance(job.get("task"), str)
        and job.get("status") in STATUSES
        and _number(job.get("at"))
    ]
    return jobs[-MAX_JOBS:], _time(data.get("seen"))


class Laptop:
    """The queue of jobs for the laptop.

    `on_finished(job)` is called each time a job ends, however it ends, so the
    wearer can be told; a job they called off themselves is not reported.
    """

    def __init__(self, *, store: Store, now: Callable[[], float], on_finished: Callable[[Job], Awaitable[None]] | None = None) -> None:
        self._store = store
        self._now = now
        self.on_finished = on_finished
        self._jobs: list[Job] = []
        self.seen: float | None = None
        self._loaded = False
        self._lock = asyncio.Lock()
        # Replaced each time something changes, so anyone waiting wakes and looks again.
        self._changed = asyncio.Event()

    def _touch(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    async def _load(self) -> None:
        if not self._loaded:
            self._jobs, self.seen = read_jobs(await self._store.read())
            self._loaded = True

    async def _save(self) -> None:
        del self._jobs[: max(0, len(self._jobs) - MAX_JOBS)]
        await self._store.write({"jobs": self._jobs, "seen": self.seen})
        self._touch()

    def _find(self, job_id: str) -> Job | None:
        return next((job for job in self._jobs if job["id"] == job_id), None)

    async def _end(self, job: Job, status: str, result: str, *, lost: bool = False) -> None:
        job["status"] = status
        job["finished"] = self._now()
        job["result"] = result[:MAX_RESULT_CHARS]
        job["lost"] = lost
        if self.on_finished and status != CANCELLED:
            await self.on_finished(dict(job))

    async def _tidy(self) -> bool:
        """Drop jobs that waited too long, hand lost ones out again, give up on stuck ones. Says whether anything changed."""
        now = self._now()
        changed = False
        for job in self._jobs:
            if job["status"] in (WAITING, OFFERED) and now - job["at"] > WAIT_SECONDS:
                await self._end(job, EXPIRED, "The laptop was not on in time, so this was dropped.")
                changed = True
            elif job["status"] == OFFERED and now - (job["offered"] or job["at"]) > OFFER_SECONDS:
                job["status"] = WAITING
                changed = True
            elif job["status"] == RUNNING and now - (job["started"] or job["at"]) > RUN_SECONDS:
                await self._end(job, FAILED, "The laptop stopped before finishing this.", lost=True)
                changed = True
        return changed

    def online(self) -> bool:
        return self.seen is not None and self._now() - self.seen < ONLINE_SECONDS

    async def submit(self, task: str, *, follow_up: bool = False, model: str | None = None) -> Job:
        """Queue a job and return it. A follow-up carries on from the laptop's previous job; `model` is which Claude
        should do it, when the agent chose one.

        The same job asked for again while it waits, or within a minute of starting, is not queued a second time:
        the existing one comes back with "repeat" set.
        """
        async with self._lock:
            await self._load()
            await self._tidy()
            wanted = " ".join(task[:MAX_TASK_CHARS].lower().split())
            for queued in self._jobs:
                if (
                    " ".join(queued["task"].lower().split()) == wanted
                    and queued["follow_up"] == follow_up
                    and (queued["status"] in (WAITING, OFFERED) or (queued["status"] == RUNNING and self._now() - (queued["started"] or queued["at"]) < REPEAT_SECONDS))
                ):
                    return {**queued, "repeat": True}
            job: Job = {
                "id": secrets.token_hex(6),
                "task": task[:MAX_TASK_CHARS],
                "follow_up": follow_up,
                "status": WAITING,
                "at": self._now(),
                "offered": None,
                "started": None,
                "finished": None,
                "result": "",
                "lost": False,
                "stats": {},
                "model": model if model in MODELS else None,
            }
            self._jobs.append(job)
            await self._save()
            return dict(job)

    async def next(self, wait: float) -> Job | None:
        """The laptop asking for work: hand over the oldest job, waiting up to `wait` seconds for one."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait
        while True:
            async with self._lock:
                await self._load()
                self.seen = self._now()
                changed = await self._tidy()
                for job in self._jobs:
                    # The helper reports a job before asking for the next, so one still running was cut
                    # short, by a restart or the laptop going to sleep.
                    if job["status"] == RUNNING:
                        await self._end(job, FAILED, "The laptop was interrupted before finishing this.", lost=True)
                        changed = True
                # One handed out but never started did not reach the laptop: hand it out again.
                job = next((job for job in self._jobs if job["status"] == OFFERED), None) or next(
                    (job for job in self._jobs if job["status"] == WAITING), None
                )
                if job:
                    job["status"] = OFFERED
                    job["offered"] = self._now()
                    changed = True
                if changed:
                    await self._save()
                if job:
                    return dict(job)
                changed_event = self._changed
            left = deadline - loop.time()
            if left <= 0:
                return None
            try:
                await asyncio.wait_for(changed_event.wait(), left)
            except TimeoutError:
                return None

    async def start(self, job_id: str) -> bool:
        """The laptop saying it is starting a job it was handed. False when it should not: the job was called off."""
        async with self._lock:
            await self._load()
            self.seen = self._now()
            job = self._find(job_id)
            if job is None:
                return False
            if job["status"] == RUNNING:
                # The laptop asked twice because the first answer was lost.
                return True
            if job["status"] not in (OFFERED, WAITING):
                return False
            job["status"] = RUNNING
            job["started"] = self._now()
            await self._save()
            return True

    async def finish(self, job_id: str, ok: bool, result: str, stats: Any = None) -> bool:
        """The laptop reporting how a job went. A late report for a job given up on still counts."""
        async with self._lock:
            await self._load()
            self.seen = self._now()
            job = self._find(job_id)
            if job is None or not (job["status"] == RUNNING or (job["status"] == FAILED and job["lost"])):
                return False
            job["stats"] = _stats(stats)
            await self._end(job, DONE if ok else FAILED, " ".join(result.split()) or ("Done." if ok else "It did not work."))
            await self._save()
            return True

    async def watch(self, job_id: str, wait: float) -> str | None:
        """The laptop, while working, waiting up to `wait` seconds to hear whether the job was called off. Returns its status."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait
        while True:
            async with self._lock:
                await self._load()
                self.seen = self._now()
                job = self._find(job_id)
                if job is None:
                    return None
                if job["status"] not in (OFFERED, RUNNING):
                    return job["status"]
                changed_event = self._changed
            left = deadline - loop.time()
            if left <= 0:
                return job["status"]
            try:
                await asyncio.wait_for(changed_event.wait(), left)
            except TimeoutError:
                pass

    async def cancel(self, reason: str = "Called off from the glasses.") -> list[Job]:
        """Call off every job that has not finished. Returns the ones called off."""
        async with self._lock:
            await self._load()
            cancelled = []
            for job in self._jobs:
                if job["status"] in UNFINISHED:
                    await self._end(job, CANCELLED, reason)
                    cancelled.append(dict(job))
            if cancelled:
                await self._save()
            return cancelled

    async def recent(self, within: float) -> list[Job]:
        """The jobs asked for in the last `within` seconds, oldest first."""
        async with self._lock:
            await self._load()
            if await self._tidy():
                await self._save()
            since = self._now() - within
            return [dict(job) for job in self._jobs if job["at"] >= since]
