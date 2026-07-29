# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Round-tripping, invariants, and transitions for ``dispatch.json``."""

from pathlib import Path

import pytest

from tools.odin.state import (
    SCHEMA_VERSION,
    DispatchState,
    FailureInfo,
    JobEntry,
    read_dispatch_state,
    reset_in_flight_to_pending,
    write_dispatch_state,
)


def _job(row_key: str = "rsl_rl_physx_Isaac-Ant_seed42") -> JobEntry:
    return JobEntry(
        row_key=row_key,
        task_id="Isaac-Ant",
        rl_library="rsl_rl",
        physics="physx",
        renderer=None,
        presets=(),
        seed=42,
        num_envs=4096,
        max_iterations=500,
        timeout_s=3600,
        osmo_task_name="rsl-rl-physx-isaac-ant-seed42",
        image_ref="nvcr.io/nvidian/x@sha256:abc",
        side="a",
    )


def _state(*jobs: JobEntry) -> DispatchState:
    return DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260728-120000",
        started_at="2026-07-28T12:00:00Z",
        ended_at=None,
        seeds=[42],
        images={"a": "nvcr.io/nvidian/x@sha256:abc"},
        jobs=list(jobs),
    )


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    state = _state(_job())
    state.osmo_workflow_ids = ["wf-1", "wf-2"]
    write_dispatch_state(tmp_path, state)

    loaded = read_dispatch_state(tmp_path)

    assert loaded is not None
    assert loaded.dispatch_id == "20260728-120000"
    assert loaded.images == {"a": "nvcr.io/nvidian/x@sha256:abc"}
    assert loaded.osmo_workflow_ids == ["wf-1", "wf-2"]
    assert loaded.jobs[0].row_key == "rsl_rl_physx_Isaac-Ant_seed42"
    assert loaded.jobs[0].renderer is None
    assert loaded.jobs[0].side == "a"
    assert loaded.jobs[0].num_envs == 4096


def test_optional_sizing_survives_the_round_trip(tmp_path: Path) -> None:
    # Run #1 omits sizing so each task runs at its shipped default.
    job = _job()
    job.num_envs = None
    job.max_iterations = None
    write_dispatch_state(tmp_path, _state(job))

    loaded = read_dispatch_state(tmp_path)

    assert loaded is not None
    assert loaded.jobs[0].num_envs is None
    assert loaded.jobs[0].max_iterations is None


def test_absent_physics_round_trips_as_none(tmp_path: Path) -> None:
    # Tasks that declare no physics preset carry physics=None. Reloading it as
    # the string "None" survives every is-not-None guard and renders a bogus
    # `physics=None` selector into a retry's workflow.
    job = _job()
    job.physics = None
    write_dispatch_state(tmp_path, _state(job))

    loaded = read_dispatch_state(tmp_path)

    assert loaded is not None
    assert loaded.jobs[0].physics is None


def test_read_returns_none_when_absent(tmp_path: Path) -> None:
    assert read_dispatch_state(tmp_path) is None


def test_a_different_schema_version_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "dispatch.json").write_text('{"schema_version": "1.6", "dispatch_id": "x"}')
    with pytest.raises(ValueError, match="schema_version"):
        read_dispatch_state(tmp_path)


def test_write_is_atomic_leaving_no_temp_files(tmp_path: Path) -> None:
    write_dispatch_state(tmp_path, _state(_job()))
    assert [p.name for p in tmp_path.iterdir()] == ["dispatch.json"]


def test_transition_to_running_sets_started_at() -> None:
    job = _job()
    job.transition_to("running", assigned_to="osmo:wf-1")
    assert job.status == "running"
    assert job.started_at is not None
    assert job.assigned_to == "osmo:wf-1"


def test_transition_to_running_requires_an_assignee() -> None:
    with pytest.raises(ValueError, match="assigned_to"):
        _job().transition_to("running")


def test_transition_to_failed_requires_failure_info() -> None:
    job = _job()
    job.transition_to("running", assigned_to="osmo:wf-1")
    with pytest.raises(ValueError, match="failure"):
        job.transition_to("failed")


def test_transition_to_failed_records_kind_and_ended_at() -> None:
    job = _job()
    job.transition_to("running", assigned_to="osmo:wf-1")
    job.transition_to("failed", failure=FailureInfo(kind="timeout", message="exceeded"))
    assert job.status == "failed"
    assert job.ended_at is not None
    assert job.failure is not None
    assert job.failure.kind == "timeout"


def test_pending_to_terminal_is_allowed_for_fast_osmo_tasks() -> None:
    # OSMO can report a task finished without the poller ever seeing it RUNNING.
    job = _job()
    job.transition_to("completed")
    assert job.status == "completed"
    assert job.ended_at is not None


def test_same_state_transition_is_a_noop() -> None:
    job = _job()
    job.transition_to("running", assigned_to="osmo:wf-1")
    first_started = job.started_at
    job.transition_to("running", assigned_to="osmo:wf-1")
    assert job.started_at == first_started


def test_illegal_transition_is_rejected() -> None:
    job = _job()
    job.transition_to("completed")
    with pytest.raises(ValueError, match="transition"):
        job.transition_to("running", assigned_to="osmo:wf-1")


def test_failure_details_round_trip(tmp_path: Path) -> None:
    job = _job()
    job.transition_to(
        "failed",
        failure=FailureInfo(kind="infrastructure", message="evicted", details={"osmo_state": "FAILED_EVICTED"}),
    )
    write_dispatch_state(tmp_path, _state(job))

    loaded = read_dispatch_state(tmp_path)

    assert loaded is not None
    assert loaded.jobs[0].failure is not None
    assert loaded.jobs[0].failure.details["osmo_state"] == "FAILED_EVICTED"


def test_reset_in_flight_returns_running_jobs_to_pending() -> None:
    running, done = _job("a"), _job("b")
    running.transition_to("running", assigned_to="osmo:wf-1")
    done.transition_to("completed")
    state = _state(running, done)

    reset_in_flight_to_pending(state)

    assert state.jobs[0].status == "pending"
    assert state.jobs[0].started_at is None
    assert state.jobs[0].assigned_to is None
    assert state.jobs[1].status == "completed"
