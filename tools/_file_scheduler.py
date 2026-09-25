# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run test files concurrently within a fixed budget of slots.

``tools/conftest.py`` runs every test file in its own pytest process. Most of a process's life is spent
starting up -- booting Kit, importing, collecting -- rather than testing, so running several files at once
overlaps that fixed cost instead of paying it file after file. Splitting one file across ``pytest-xdist``
workers does the opposite for most files, since every worker pays the startup again; it only pays off for
the few long files whose tests dwarf their startup, which is why a file may ask for several slots.

The scheduler knows nothing about pytest. It is given :class:`TestFileJob` entries and a ``run`` callable,
and guarantees:

* at most ``max_slots`` slots are in use, a job holding :attr:`TestFileJob.slots` of them;
* jobs start in the order they are given, so a job that does not fit yet is not overtaken (a wide job
  would otherwise wait forever behind a stream of narrow ones);
* rendering jobs never overlap one another, though other jobs may overtake a rendering job waiting its turn.
  The first renderer in a fresh container compiles shaders into a cache the rest reuse, and several starting
  at once each compile the same shaders; renderers also share one log file, so overlapping ones would
  garble each other's diagnostics.

With ``max_slots <= 1`` jobs run one after the other on the calling thread, exactly as a plain loop would.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generic, TypeVar

R = TypeVar("R")


@dataclass(frozen=True)
class TestFileJob:
    """One test file to run."""

    __test__ = False  # not a pytest test class, despite the name

    path: str
    """Test file path; also the key of its result."""

    slots: int = 1
    """Slots the job holds while it runs, e.g. one per ``pytest-xdist`` worker; more than the budget holds them all."""

    renders: bool = False
    """Whether the job starts a renderer, and so never runs alongside another job that does."""


@dataclass(frozen=True)
class JobContext:
    """What the scheduler tells a job it starts."""

    renderer_cold: bool
    """Whether this job renders and no rendering job had finished starting before it, so it compiles shaders."""

    mark_started: Callable[[], None]
    """Call once the job's process has finished starting; a rendering job's start warms the shader cache."""


class _Scheduler(Generic[R]):
    def __init__(self, run: Callable[[TestFileJob, JobContext], R], max_slots: int):
        self._run = run
        self._max_slots = max_slots
        self._renderer_ready = threading.Event()
        self._finished: queue.Queue[tuple[TestFileJob, R | BaseException]] = queue.Queue()

    def _context(self, job: TestFileJob) -> JobContext:
        def mark_started():
            if job.renders:
                self._renderer_ready.set()

        return JobContext(renderer_cold=job.renders and not self._renderer_ready.is_set(), mark_started=mark_started)

    def run_serially(self, jobs: Iterable[TestFileJob]) -> dict[str, R]:
        results = {}
        for job in jobs:
            results[job.path] = self._run(job, self._context(job))
        return results

    def run_concurrently(self, jobs: Iterable[TestFileJob]) -> dict[str, R]:
        source = iter(jobs)
        pending: list[TestFileJob] = []
        running: list[TestFileJob] = []
        results: dict[str, R] = {}
        order: list[str] = []
        exhausted = False

        def free_slots() -> int:
            return self._max_slots - sum(self._cost(job) for job in running)

        def next_startable() -> TestFileJob | None:
            nonlocal exhausted
            index = 0
            while True:
                # Pull only as far as the first startable job: a shared work queue hands a claimed file to no one else.
                if index == len(pending):
                    if exhausted or free_slots() == 0:
                        return None
                    try:
                        pending.append(next(source))
                    except StopIteration:
                        exhausted = True
                        return None
                job = pending[index]
                if job.renders and any(other.renders for other in running):
                    index += 1
                    continue
                return job if self._cost(job) <= free_slots() else None

        with ThreadPoolExecutor(max_workers=self._max_slots, thread_name_prefix="test-file") as pool:
            while True:
                while (job := next_startable()) is not None:
                    pending.remove(job)
                    running.append(job)
                    order.append(job.path)
                    pool.submit(self._run_and_report, job, self._context(job))
                if not running:
                    break
                job, outcome = self._finished.get()
                running.remove(job)
                if isinstance(outcome, BaseException):
                    raise outcome
                results[job.path] = outcome
        return {path: results[path] for path in order}

    def _run_and_report(self, job: TestFileJob, context: JobContext) -> None:
        try:
            outcome = self._run(job, context)
        except BaseException as error:  # re-raised on the scheduling thread
            outcome = error
        self._finished.put((job, outcome))

    def _cost(self, job: TestFileJob) -> int:
        return max(1, min(job.slots, self._max_slots))



def run_test_files(
    jobs: Iterable[TestFileJob], run: Callable[[TestFileJob, JobContext], R], max_slots: int = 1
) -> dict[str, R]:
    """Run ``run`` for every job, at most ``max_slots`` slots at a time.

    Args:
        jobs: Jobs in the order they should start. Consumed lazily, so it may be a generator that claims
            files from a queue shared with other runners.
        run: Runs one job and returns its result. Called on a worker thread when ``max_slots > 1``.
        max_slots: Slots available; a job wider than this is given all of them.

    Returns:
        Each job's result keyed by :attr:`TestFileJob.path`, in the order the jobs started.

    Raises:
        BaseException: Whatever ``run`` raised, once the jobs already running have finished.
    """
    scheduler = _Scheduler(run, max_slots)
    if max_slots <= 1:
        return scheduler.run_serially(jobs)
    return scheduler.run_concurrently(jobs)
