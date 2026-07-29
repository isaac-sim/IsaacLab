# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Poll OSMO workflows and reconcile per-task states into ``dispatch.json``.

:data:`OSMO_STATE_TO_FAILURE_KIND` is the single source of truth for the
OSMO-state to failure-kind mapping.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from tools.odin.client import WorkflowSnapshot
from tools.odin.state import DispatchState, FailureInfo, JobEntry, write_dispatch_state

__all__ = [
    "OSMO_STATE_TO_FAILURE_KIND",
    "TERMINAL_OSMO_STATES",
    "classify_terminal_state",
    "is_terminal",
    "poll_until_terminal",
    "sync_once",
]

# ``COMPLETED`` is intentionally absent: the caller decides between success and
# ``malformed_bundle`` after validating the fetched bundle.
OSMO_STATE_TO_FAILURE_KIND: dict[str, str] = {
    "FAILED": "benchmark_crash",
    "FAILED_EXEC_TIMEOUT": "timeout",
    "FAILED_BACKEND_ERROR": "infrastructure",
    "FAILED_PREEMPTED": "infrastructure",
    "FAILED_EVICTED": "infrastructure",
    "FAILED_IMAGE_PULL": "infrastructure",
    "FAILED_START_ERROR": "infrastructure",
    "FAILED_START_TIMEOUT": "infrastructure",
    "FAILED_QUEUE_TIMEOUT": "infrastructure",
    "FAILED_SERVER_ERROR": "infrastructure",
    "FAILED_CANCELED": "infrastructure",
}

TERMINAL_OSMO_STATES: frozenset[str] = frozenset({"COMPLETED", *OSMO_STATE_TO_FAILURE_KIND})


class _StatusClient(Protocol):
    def status(self, workflow_id: str) -> WorkflowSnapshot: ...


def is_terminal(osmo_state: str) -> bool:
    """Return ``True`` iff *osmo_state* is a terminal task state.

    Any unrecognised ``FAILED*`` state counts as terminal, so OSMO version drift
    cannot wedge the poll loop.

    Args:
        osmo_state: OSMO task state string.

    Returns:
        Whether the state is terminal.
    """
    return osmo_state in TERMINAL_OSMO_STATES or osmo_state.startswith("FAILED")


def classify_terminal_state(osmo_state: str) -> str | None:
    """Return the failure kind for a terminal OSMO state.

    Args:
        osmo_state: OSMO task state string.

    Returns:
        One of the four failure kinds, or ``None`` for ``COMPLETED`` and for
        non-terminal states. Unrecognised ``FAILED*`` states fall back to
        ``infrastructure``.
    """
    if osmo_state == "COMPLETED":
        return None
    if osmo_state in OSMO_STATE_TO_FAILURE_KIND:
        return OSMO_STATE_TO_FAILURE_KIND[osmo_state]
    if osmo_state.startswith("FAILED"):
        return "infrastructure"
    return None


def _osmo_status_to_job_status(osmo_state: str) -> str:
    """Map an OSMO task state to a ``dispatch.json`` job status."""
    if osmo_state == "COMPLETED":
        return "completed"
    if osmo_state == "RUNNING":
        return "running"
    if is_terminal(osmo_state):
        return "failed"
    # SUBMITTING, WAITING, PROCESSING, SCHEDULING, INITIALIZING, RESCHEDULED.
    return "pending"


def sync_once(
    *,
    client: _StatusClient,
    state: DispatchState,
    dispatch_dir: Path,
    on_task_completed: Callable[[JobEntry], None],
    completed_seen: set[str] | None = None,
) -> None:
    """Run one reconciliation pass over every workflow and rewrite state.

    Idempotent: repeated calls with no remote change are a no-op.

    Args:
        client: Provides ``status(workflow_id) -> WorkflowSnapshot``.
        state: Mutated in place; must have ``osmo_workflow_ids`` populated.
        dispatch_dir: Directory holding ``dispatch.json``.
        on_task_completed: Called once per task that reaches ``COMPLETED``.
        completed_seen: Row keys that already fired the hook. Pass the same set
            across ticks so a completion is not re-enqueued every poll. Seeded
            from already-completed jobs when ``None``.

    Raises:
        ValueError: If ``state.osmo_workflow_ids`` is empty.
    """
    if not state.osmo_workflow_ids:
        raise ValueError("state.osmo_workflow_ids is required for OSMO polling")
    by_name = {job.osmo_task_name: job for job in state.jobs}
    if completed_seen is None:
        completed_seen = {job.row_key for job in state.jobs if job.status == "completed"}

    for workflow_id in state.osmo_workflow_ids:
        try:
            snapshot = client.status(workflow_id)
        except Exception as exc:  # noqa: BLE001 - any query failure is skippable
            # OSMO returns transient 4xx/5xx, and a query can race a workflow
            # finishing. Skip this workflow for this tick; the next one picks it
            # up. A persistent auth failure keeps surfacing here, which is the
            # operator's signal to fix credentials.
            print(f"[odin] osmo workflow query {workflow_id} failed; skipping this tick: {exc}", flush=True)
            continue

        for task in snapshot.tasks:
            job = by_name.get(task.name)
            if job is None:
                continue
            new_status = _osmo_status_to_job_status(task.status)
            if new_status == job.status:
                continue
            if new_status == "running":
                # OSMO exposes no assigned host; name the chunk workflow instead.
                job.transition_to("running", assigned_to=f"osmo:{workflow_id}")
            elif new_status == "completed":
                # The hook may reclassify a row it just completed, e.g. to
                # ``malformed_bundle``. OSMO keeps reporting COMPLETED, so
                # firing again would try an illegal failed -> completed move.
                if job.row_key in completed_seen:
                    continue
                completed_seen.add(job.row_key)
                job.transition_to("completed")
                on_task_completed(job)
            elif new_status == "failed":
                job.transition_to(
                    "failed",
                    failure=FailureInfo(
                        kind=classify_terminal_state(task.status) or "infrastructure",
                        message=f"OSMO task {task.name} reached {task.status} (exit={task.exit_code})",
                        details={"osmo_state": task.status, "exit_code": task.exit_code},
                    ),
                )
            else:
                job.transition_to("pending")

    write_dispatch_state(dispatch_dir, state)


def poll_until_terminal(
    *,
    client: _StatusClient,
    state: DispatchState,
    dispatch_dir: Path,
    on_task_completed: Callable[[JobEntry], None],
    poll_interval_s: float,
) -> None:
    """Drive every workflow to completion, rewriting ``dispatch.json`` each tick.

    Args:
        client: Provides ``status(workflow_id) -> WorkflowSnapshot``.
        state: Mutated in place.
        dispatch_dir: Directory holding ``dispatch.json``.
        on_task_completed: Called once per task that reaches ``COMPLETED``.
        poll_interval_s: Seconds between passes; ``0`` in tests.
    """
    completed_seen = {job.row_key for job in state.jobs if job.status == "completed"}
    while not _all_terminal(state):
        sync_once(
            client=client,
            state=state,
            dispatch_dir=dispatch_dir,
            on_task_completed=on_task_completed,
            completed_seen=completed_seen,
        )
        if _all_terminal(state):
            break
        if poll_interval_s > 0:
            time.sleep(poll_interval_s)


def _all_terminal(state: DispatchState) -> bool:
    return all(job.status in ("completed", "failed") for job in state.jobs)
