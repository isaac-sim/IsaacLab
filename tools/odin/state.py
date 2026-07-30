# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-disk state for one OdinV2 dispatch.

``dispatch.json`` lives at ``<dispatch_dir>/dispatch.json`` and is rewritten
atomically (temp file plus rename) after every state transition, so a concurrent
reader never observes a truncated file.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

__all__ = [
    "FAILURE_KINDS",
    "SCHEMA_VERSION",
    "DispatchState",
    "FailureInfo",
    "JobEntry",
    "read_dispatch_state",
    "reset_in_flight_to_pending",
    "write_dispatch_state",
]

SCHEMA_VERSION = "2.0"
_DISPATCH_FILENAME = "dispatch.json"

FAILURE_KINDS = frozenset({"benchmark_crash", "malformed_bundle", "play_failed", "timeout", "infrastructure"})


def _utc_now_iso() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FailureInfo:
    """Classified failure attached to a :class:`JobEntry` in ``failed`` status.

    Args:
        kind: One of :data:`FAILURE_KINDS`.
        message: Human-readable summary.
        details: Free-form diagnostic payload, e.g. the raw OSMO state.
    """

    kind: str
    message: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class JobEntry:
    """One dispatched row — the smallest unit of work.

    Args:
        row_key: Deterministic identity, ``{rl_library}_{physics}_{task_id}_seed{seed}``.
        task_id: Gym task id.
        rl_library: One of ``rsl_rl``, ``rl_games``, ``skrl``, ``sb3``.
        physics: Physics preset token passed as ``physics=<value>``, or ``None``
            for tasks that declare no physics preset.
        renderer: Renderer preset token, or ``None`` to run headless.
        presets: Domain preset tokens applied to the run.
        seed: Environment seed.
        num_envs: Parallel environment count, or ``None`` to use the task's
            shipped agent-config default.
        max_iterations: Training iterations, or ``None`` to use the task's
            shipped agent-config default.
        timeout_s: Wall-clock budget [s] used for chunk ordering, or ``None``
            to inherit the workflow-level ceiling.
        osmo_task_name: DNS-1123-safe OSMO task name.
        image_ref: Fully qualified image reference, digest-pinned.
        side: ``"a"`` or ``"b"``, identifying the image of an A/B pair.
    """

    row_key: str
    task_id: str
    rl_library: str
    physics: str | None
    renderer: str | None
    presets: tuple[str, ...]
    seed: int
    num_envs: int | None
    max_iterations: int | None
    timeout_s: int | None
    osmo_task_name: str
    image_ref: str
    side: str = "a"
    status: str = "pending"
    assigned_to: str | None = None
    failure: FailureInfo | None = None
    started_at: str | None = None
    ended_at: str | None = None

    # ``pending`` may go straight to a terminal state: OSMO can report a task as
    # finished without the poller ever observing it RUNNING (fast scheduling, an
    # instant image-pull failure).
    _ALLOWED_TRANSITIONS: ClassVar[dict[str, frozenset[str]]] = {
        "pending": frozenset({"running", "completed", "failed"}),
        "running": frozenset({"completed", "failed", "pending"}),
        "completed": frozenset({"pending"}),
        "failed": frozenset({"pending"}),
    }

    def transition_to(
        self,
        status: str,
        *,
        failure: FailureInfo | None = None,
        assigned_to: str | None = None,
    ) -> None:
        """Move this job to *status*, enforcing the target's field invariants.

        Transitioning to the current status is a no-op, so repeated poll ticks
        do not reset timestamps.

        Args:
            status: Target status.
            failure: Required when *status* is ``failed``.
            assigned_to: Required when *status* is ``running``; identifies the
                OSMO workflow, e.g. ``osmo:wf-1``.

        Raises:
            ValueError: If the transition is not allowed, or a field required by
                the target status is missing.
        """
        if status == self.status:
            return
        if status not in self._ALLOWED_TRANSITIONS.get(self.status, frozenset()):
            raise ValueError(f"illegal transition {self.status!r} -> {status!r} for row_key={self.row_key!r}")

        if status == "running":
            if assigned_to is None:
                raise ValueError(f"assigned_to is required to enter 'running' (row_key={self.row_key!r})")
            self.assigned_to = assigned_to
            self.started_at = _utc_now_iso()
        elif status in ("completed", "failed"):
            if status == "failed" and failure is None:
                raise ValueError(f"failure is required to enter 'failed' (row_key={self.row_key!r})")
            self.failure = failure
            self.ended_at = _utc_now_iso()
        elif status == "pending":
            self.assigned_to = None
            self.started_at = None
            self.ended_at = None
            self.failure = None

        self.status = status


@dataclass
class DispatchState:
    """Complete on-disk state for one dispatch.

    Args:
        schema_version: Schema version of this document.
        dispatch_id: ``YYYYMMDD-HHMMSS`` identifier.
        started_at: UTC ISO-8601 start timestamp.
        ended_at: UTC ISO-8601 end timestamp, or ``None`` while in flight.
        seeds: Seeds the dispatch was expanded across.
        images: Image reference per side, e.g. ``{"a": ..., "b": ...}``.
            Digest-pinned so a retag cannot change what an A/B compared.
        jobs: One entry per dispatched row.
        osmo_workflow_ids: One workflow id per submitted chunk.
        parent_dispatch_id: Set when this dispatch retries rows from another.
    """

    schema_version: str
    dispatch_id: str
    started_at: str
    ended_at: str | None
    seeds: list[int]
    images: dict[str, str]
    jobs: list[JobEntry]
    osmo_workflow_ids: list[str] = field(default_factory=list)
    parent_dispatch_id: str | None = None


def _job_to_dict(job: JobEntry) -> dict[str, Any]:
    failure = None
    if job.failure is not None:
        failure = {"kind": job.failure.kind, "message": job.failure.message, "details": job.failure.details}
    return {
        "row_key": job.row_key,
        "task_id": job.task_id,
        "rl_library": job.rl_library,
        "physics": job.physics,
        "renderer": job.renderer,
        "presets": list(job.presets),
        "seed": job.seed,
        "num_envs": job.num_envs,
        "max_iterations": job.max_iterations,
        "timeout_s": job.timeout_s,
        "osmo_task_name": job.osmo_task_name,
        "image_ref": job.image_ref,
        "side": job.side,
        "status": job.status,
        "assigned_to": job.assigned_to,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
        "failure": failure,
    }


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _job_from_dict(payload: dict[str, Any]) -> JobEntry:
    failure = None
    if payload.get("failure") is not None:
        failure = FailureInfo(
            kind=str(payload["failure"]["kind"]),
            message=str(payload["failure"].get("message", "")),
            details=dict(payload["failure"].get("details") or {}),
        )
    return JobEntry(
        row_key=str(payload["row_key"]),
        task_id=str(payload["task_id"]),
        rl_library=str(payload["rl_library"]),
        physics=_optional_str(payload.get("physics")),
        renderer=_optional_str(payload.get("renderer")),
        presets=tuple(payload.get("presets") or ()),
        seed=int(payload["seed"]),
        num_envs=_optional_int(payload.get("num_envs")),
        max_iterations=_optional_int(payload.get("max_iterations")),
        timeout_s=_optional_int(payload.get("timeout_s")),
        osmo_task_name=str(payload["osmo_task_name"]),
        image_ref=str(payload["image_ref"]),
        side=str(payload.get("side", "a")),
        status=str(payload.get("status", "pending")),
        assigned_to=_optional_str(payload.get("assigned_to")),
        failure=failure,
        started_at=_optional_str(payload.get("started_at")),
        ended_at=_optional_str(payload.get("ended_at")),
    )


def read_dispatch_state(dispatch_dir: Path) -> DispatchState | None:
    """Read ``<dispatch_dir>/dispatch.json``.

    Args:
        dispatch_dir: Directory holding the dispatch.

    Returns:
        The parsed state, or ``None`` when the file does not exist.

    Raises:
        ValueError: If the file declares a different schema version.
    """
    path = dispatch_dir / _DISPATCH_FILENAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    got = str(payload.get("schema_version", ""))
    if got != SCHEMA_VERSION:
        raise ValueError(f"unsupported dispatch.json schema_version {got!r} (expected {SCHEMA_VERSION!r})")
    return DispatchState(
        schema_version=got,
        dispatch_id=str(payload["dispatch_id"]),
        started_at=str(payload["started_at"]),
        ended_at=payload.get("ended_at"),
        seeds=[int(seed) for seed in payload.get("seeds") or []],
        images=dict(payload.get("images") or {}),
        jobs=[_job_from_dict(job) for job in payload.get("jobs") or []],
        osmo_workflow_ids=[str(x) for x in payload.get("osmo_workflow_ids") or []],
        parent_dispatch_id=payload.get("parent_dispatch_id"),
    )


def write_dispatch_state(dispatch_dir: Path, state: DispatchState) -> None:
    """Atomically rewrite ``<dispatch_dir>/dispatch.json``.

    Args:
        dispatch_dir: Directory holding the dispatch; created if missing.
        state: State to serialize.
    """
    dispatch_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": state.schema_version,
        "dispatch_id": state.dispatch_id,
        "started_at": state.started_at,
        "ended_at": state.ended_at,
        "seeds": list(state.seeds),
        "images": dict(state.images),
        "jobs": [_job_to_dict(job) for job in state.jobs],
        "osmo_workflow_ids": list(state.osmo_workflow_ids),
        "parent_dispatch_id": state.parent_dispatch_id,
    }
    handle_fd, tmp_name = tempfile.mkstemp(prefix=".dispatch_", suffix=".json.tmp", dir=str(dispatch_dir))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(handle_fd, "w") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp_path, dispatch_dir / _DISPATCH_FILENAME)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def reset_in_flight_to_pending(state: DispatchState) -> None:
    """Flip ``running`` jobs back to ``pending`` so a resumed dispatch re-polls them.

    ``completed`` and ``failed`` jobs are left alone; a failed row is only
    re-attempted through an explicit retry.

    Args:
        state: State mutated in place.
    """
    for job in state.jobs:
        if job.status == "running":
            job.transition_to("pending")
