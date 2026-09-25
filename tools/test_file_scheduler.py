# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``tools/_file_scheduler.py``."""

from __future__ import annotations

import threading

import pytest
from _file_scheduler import JobContext, TestFileJob, run_test_files

_TIMEOUT = 10.0
"""Seconds any wait in these tests may take before it counts as a deadlock."""


class _Harness:
    """A ``run`` callable whose jobs start and finish only when the test says so.

    Records which jobs are in flight, so a test can assert on concurrency at each step instead of relying
    on sleeps.
    """

    def __init__(self):
        self._lock = threading.Condition()
        self.running: dict[str, TestFileJob] = {}
        self.started: list[str] = []
        self.contexts: dict[str, JobContext] = {}
        self.peak_slots = 0
        self._release: dict[str, threading.Event] = {}

    def __call__(self, job: TestFileJob, context: JobContext) -> str:
        release = threading.Event()
        with self._lock:
            self.running[job.path] = job
            self.started.append(job.path)
            self.contexts[job.path] = context
            self._release[job.path] = release
            self.peak_slots = max(self.peak_slots, sum(j.slots for j in self.running.values()))
            self._lock.notify_all()
        assert release.wait(_TIMEOUT), f"{job.path} was never released"
        with self._lock:
            del self.running[job.path]
            self._lock.notify_all()
        return f"result:{job.path}"

    def wait_running(self, *paths: str) -> None:
        """Wait until exactly ``paths`` are in flight."""
        with self._lock:
            assert self._lock.wait_for(lambda: set(self.running) == set(paths), _TIMEOUT), (
                f"expected {sorted(paths)} running, have {sorted(self.running)}"
            )

    def finish(self, path: str) -> None:
        with self._lock:
            self._release[path].set()
            assert self._lock.wait_for(lambda: path not in self.running, _TIMEOUT)


def _run_in_background(jobs, harness, max_slots):
    outcome = {}

    def target():
        try:
            outcome["results"] = run_test_files(jobs, harness, max_slots=max_slots)
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, outcome


def test_one_slot_runs_jobs_in_order_on_the_calling_thread():
    seen = []

    def run(job, context):
        seen.append((job.path, threading.current_thread() is threading.main_thread()))
        return job.path.upper()

    results = run_test_files([TestFileJob("a"), TestFileJob("b", slots=4)], run, max_slots=1)

    assert seen == [("a", True), ("b", True)]
    assert results == {"a": "A", "b": "B"}


def test_jobs_fill_the_slots_and_the_next_starts_when_one_finishes():
    harness = _Harness()
    jobs = [TestFileJob(name) for name in "abcd"]
    thread, outcome = _run_in_background(jobs, harness, max_slots=3)

    harness.wait_running("a", "b", "c")
    harness.finish("b")
    harness.wait_running("a", "c", "d")
    for name in "acd":
        harness.finish(name)
    thread.join(_TIMEOUT)

    assert harness.peak_slots == 3
    assert outcome["results"] == {name: f"result:{name}" for name in "abcd"}


def test_a_wide_job_holds_its_slots_and_is_not_overtaken():
    harness = _Harness()
    jobs = [TestFileJob("narrow"), TestFileJob("wide", slots=3), TestFileJob("after")]
    thread, outcome = _run_in_background(jobs, harness, max_slots=3)

    # "wide" needs all three slots; "after" fits beside "narrow" but must not jump the queue.
    harness.wait_running("narrow")
    with harness._lock:
        assert not harness._lock.wait_for(lambda: "wide" in harness.started, timeout=0.2)
    harness.finish("narrow")
    harness.wait_running("wide")
    harness.finish("wide")
    harness.wait_running("after")
    harness.finish("after")
    thread.join(_TIMEOUT)

    assert harness.started == ["narrow", "wide", "after"]
    assert harness.peak_slots == 3


def test_a_job_wider_than_the_budget_gets_every_slot():
    harness = _Harness()
    thread, outcome = _run_in_background([TestFileJob("huge", slots=8), TestFileJob("next")], harness, max_slots=2)

    harness.wait_running("huge")
    harness.finish("huge")
    harness.wait_running("next")
    harness.finish("next")
    thread.join(_TIMEOUT)

    assert "error" not in outcome


def test_rendering_jobs_never_overlap_and_others_overtake_them():
    harness = _Harness()
    jobs = [
        TestFileJob("render-1", renders=True),
        TestFileJob("render-2", renders=True),
        TestFileJob("plain"),
        TestFileJob("sentinel"),
    ]
    thread, outcome = _run_in_background(jobs, harness, max_slots=2)

    harness.wait_running("render-1", "plain")
    assert harness.contexts["render-1"].renderer_cold
    assert not harness.contexts["plain"].renderer_cold

    # Starting up is not enough: the slot "plain" frees goes past the second renderer to the job behind it.
    harness.contexts["render-1"].mark_started()
    harness.finish("plain")
    harness.wait_running("render-1", "sentinel")
    harness.finish("render-1")
    harness.wait_running("render-2", "sentinel")
    assert not harness.contexts["render-2"].renderer_cold

    harness.finish("render-2")
    harness.finish("sentinel")
    thread.join(_TIMEOUT)
    assert "error" not in outcome


def test_a_renderer_that_ends_without_starting_releases_the_next_which_is_still_cold():
    harness = _Harness()
    jobs = [TestFileJob("render-1", renders=True), TestFileJob("render-2", renders=True)]
    thread, outcome = _run_in_background(jobs, harness, max_slots=2)

    harness.wait_running("render-1")
    harness.finish("render-1")  # e.g. killed by a startup hang, never reaching collection
    harness.wait_running("render-2")
    assert harness.contexts["render-2"].renderer_cold
    harness.finish("render-2")
    thread.join(_TIMEOUT)

    assert "error" not in outcome


def test_only_the_first_renderer_is_cold_when_run_serially():
    contexts = {}

    def run(job, context):
        contexts[job.path] = context
        context.mark_started()

    run_test_files([TestFileJob("render-1", renders=True), TestFileJob("render-2", renders=True)], run, max_slots=1)

    assert contexts["render-1"].renderer_cold
    assert not contexts["render-2"].renderer_cold


def test_jobs_are_pulled_only_when_they_can_start():
    harness = _Harness()
    pulled = []

    def claim():
        for name in "abc":
            pulled.append(name)
            yield TestFileJob(name)

    thread, outcome = _run_in_background(claim(), harness, max_slots=2)

    harness.wait_running("a", "b")
    assert pulled == ["a", "b"]
    harness.finish("a")
    harness.wait_running("b", "c")
    harness.finish("b")
    harness.finish("c")
    thread.join(_TIMEOUT)

    assert pulled == ["a", "b", "c"]


def test_an_error_in_one_job_is_raised_after_the_others_finish():
    finished = threading.Event()
    other_started = threading.Event()

    def run(job, context):
        if job.path == "bad":
            assert other_started.wait(_TIMEOUT)
            raise RuntimeError("boom")
        other_started.set()
        finished.wait(0.2)
        finished.set()
        return job.path

    with pytest.raises(RuntimeError, match="boom"):
        run_test_files([TestFileJob("bad"), TestFileJob("good")], run, max_slots=2)
    assert finished.is_set()
