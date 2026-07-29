# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OSMO state reconciliation and failure classification."""

from pathlib import Path

import pytest

from tools.odin.client import TaskSnapshot, WorkflowSnapshot
from tools.odin.poller import PollError, classify_terminal_state, is_terminal, poll_until_terminal, sync_once
from tools.odin.state import SCHEMA_VERSION, DispatchState, FailureInfo, JobEntry


class _FakeClient:
    """Replays a scripted sequence of snapshots, holding the last one."""

    def __init__(self, *snapshots: WorkflowSnapshot) -> None:
        self._snapshots = list(snapshots)

    def status(self, workflow_id: str) -> WorkflowSnapshot:
        return self._snapshots.pop(0) if len(self._snapshots) > 1 else self._snapshots[0]


class _ExplodingClient:
    def status(self, workflow_id: str) -> WorkflowSnapshot:
        raise RuntimeError("osmo backend hiccup")


def _job() -> JobEntry:
    return JobEntry(
        row_key="rsl_rl_physx_Isaac-Ant_seed42",
        task_id="Isaac-Ant",
        rl_library="rsl_rl",
        physics="physx",
        renderer=None,
        presets=(),
        seed=42,
        num_envs=None,
        max_iterations=None,
        timeout_s=None,
        osmo_task_name="t1",
        image_ref="img@sha256:abc",
    )


def _state(job: JobEntry) -> DispatchState:
    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id="20260728-120000",
        started_at="2026-07-28T12:00:00Z",
        ended_at=None,
        seeds=[42],
        images={"a": "img@sha256:abc"},
        jobs=[job],
    )
    state.osmo_workflow_ids = ["wf-1"]
    return state


def _snapshot(status: str, exit_code: int | None = None) -> WorkflowSnapshot:
    return WorkflowSnapshot(
        workflow_id="wf-1",
        status="RUNNING",
        tasks=[TaskSnapshot(name="t1", status=status, exit_code=exit_code)],
    )


@pytest.mark.parametrize(
    ("osmo_state", "kind"),
    [
        ("FAILED", "benchmark_crash"),
        ("FAILED_EXEC_TIMEOUT", "timeout"),
        ("FAILED_EVICTED", "infrastructure"),
    ],
)
def test_terminal_states_map_to_failure_kinds(osmo_state: str, kind: str) -> None:
    assert classify_terminal_state(osmo_state) == kind


def test_completed_has_no_failure_kind() -> None:
    assert classify_terminal_state("COMPLETED") is None


def test_unknown_failed_state_defaults_to_infrastructure() -> None:
    # OSMO version drift must not wedge the loop or be misread as a crash.
    assert classify_terminal_state("FAILED_SOMETHING_NEW") == "infrastructure"
    assert is_terminal("FAILED_SOMETHING_NEW")


def test_running_state_is_not_terminal() -> None:
    assert not is_terminal("RUNNING")


def test_completed_task_fires_the_hook_once(tmp_path: Path) -> None:
    state = _state(_job())
    fired: list[str] = []
    client = _FakeClient(_snapshot("COMPLETED"))
    seen: set[str] = set()

    for _ in range(2):
        sync_once(
            client=client,
            state=state,
            dispatch_dir=tmp_path,
            on_task_completed=lambda job: fired.append(job.row_key),
            completed_seen=seen,
        )

    assert fired == ["rsl_rl_physx_Isaac-Ant_seed42"]
    assert state.jobs[0].status == "completed"


def test_a_row_reclassified_by_the_hook_is_not_completed_again(tmp_path: Path) -> None:
    # The bundle validator reclassifies a COMPLETED row with no readable bundle.
    # OSMO keeps reporting COMPLETED, so a second tick would attempt an illegal
    # failed -> completed transition and abandon every still-running row.
    state = _state(_job())
    client = _FakeClient(_snapshot("COMPLETED"))
    seen: set[str] = set()

    def reclassify(job: JobEntry) -> None:
        job.transition_to("pending")
        job.transition_to("failed", failure=FailureInfo(kind="malformed_bundle", message="nothing parseable"))

    for _ in range(3):
        sync_once(
            client=client,
            state=state,
            dispatch_dir=tmp_path,
            on_task_completed=reclassify,
            completed_seen=seen,
        )

    assert state.jobs[0].status == "failed"
    assert state.jobs[0].failure is not None
    assert state.jobs[0].failure.kind == "malformed_bundle"


def test_failed_task_records_the_osmo_state_in_details(tmp_path: Path) -> None:
    state = _state(_job())
    sync_once(
        client=_FakeClient(_snapshot("FAILED_EXEC_TIMEOUT", exit_code=137)),
        state=state,
        dispatch_dir=tmp_path,
        on_task_completed=lambda job: None,
    )
    failure = state.jobs[0].failure
    assert failure is not None
    assert failure.kind == "timeout"
    assert failure.details["osmo_state"] == "FAILED_EXEC_TIMEOUT"
    assert failure.details["exit_code"] == 137


def test_running_task_is_assigned_to_its_workflow(tmp_path: Path) -> None:
    state = _state(_job())
    sync_once(
        client=_FakeClient(_snapshot("RUNNING")),
        state=state,
        dispatch_dir=tmp_path,
        on_task_completed=lambda job: None,
    )
    assert state.jobs[0].status == "running"
    assert state.jobs[0].assigned_to == "osmo:wf-1"


def test_sync_writes_dispatch_json(tmp_path: Path) -> None:
    sync_once(
        client=_FakeClient(_snapshot("RUNNING")),
        state=_state(_job()),
        dispatch_dir=tmp_path,
        on_task_completed=lambda job: None,
    )
    assert (tmp_path / "dispatch.json").exists()


def test_unknown_task_names_are_ignored(tmp_path: Path) -> None:
    snapshot = WorkflowSnapshot(
        workflow_id="wf-1", status="RUNNING", tasks=[TaskSnapshot(name="stranger", status="COMPLETED", exit_code=0)]
    )
    state = _state(_job())
    sync_once(client=_FakeClient(snapshot), state=state, dispatch_dir=tmp_path, on_task_completed=lambda job: None)
    assert state.jobs[0].status == "pending"


def test_transient_client_error_is_tolerated(tmp_path: Path) -> None:
    # A 5xx, or a query racing a finishing workflow, must not abort the dispatch.
    state = _state(_job())
    sync_once(client=_ExplodingClient(), state=state, dispatch_dir=tmp_path, on_task_completed=lambda job: None)
    assert state.jobs[0].status == "pending"


def test_missing_workflow_ids_is_an_error(tmp_path: Path) -> None:
    state = _state(_job())
    state.osmo_workflow_ids = []
    with pytest.raises(ValueError, match="osmo_workflow_ids"):
        sync_once(
            client=_FakeClient(_snapshot("RUNNING")),
            state=state,
            dispatch_dir=tmp_path,
            on_task_completed=lambda job: None,
        )


def test_poll_until_terminal_returns_when_all_jobs_finish(tmp_path: Path) -> None:
    state = _state(_job())
    poll_until_terminal(
        client=_FakeClient(_snapshot("RUNNING"), _snapshot("COMPLETED")),
        state=state,
        dispatch_dir=tmp_path,
        on_task_completed=lambda job: None,
        poll_interval_s=0,
    )
    assert state.jobs[0].status == "completed"


def test_poll_gives_up_after_a_run_of_total_query_failures(tmp_path: Path) -> None:
    # Bad credentials fail every query forever. Without a ceiling the loop spins
    # silently instead of telling the operator what to fix.
    with pytest.raises(PollError, match="consecutive polls"):
        poll_until_terminal(
            client=_ExplodingClient(),
            state=_state(_job()),
            dispatch_dir=tmp_path,
            on_task_completed=lambda job: None,
            poll_interval_s=0,
            max_consecutive_failures=3,
        )


def test_a_recovered_query_resets_the_failure_budget(tmp_path: Path) -> None:
    class _FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def status(self, workflow_id: str) -> WorkflowSnapshot:
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("osmo backend hiccup")
            return _snapshot("COMPLETED")

    state = _state(_job())
    poll_until_terminal(
        client=_FlakyClient(),
        state=state,
        dispatch_dir=tmp_path,
        on_task_completed=lambda job: None,
        poll_interval_s=0,
        max_consecutive_failures=3,
    )
    assert state.jobs[0].status == "completed"


def test_poll_until_terminal_is_a_noop_when_already_terminal(tmp_path: Path) -> None:
    job = _job()
    job.transition_to("completed")
    poll_until_terminal(
        client=_ExplodingClient(),
        state=_state(job),
        dispatch_dir=tmp_path,
        on_task_completed=lambda j: None,
        poll_interval_s=0,
    )
