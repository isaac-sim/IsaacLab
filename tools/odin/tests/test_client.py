# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Subprocess wrapping and error classification for the ``osmo`` CLI."""

import json
import subprocess
from pathlib import Path

import pytest

from tools.odin.client import OsmoAuthError, OsmoClient, OsmoCliError, OsmoTransientError


class _FakeRun:
    """Records invocations and replays canned ``CompletedProcess`` results."""

    def __init__(self, *results: subprocess.CompletedProcess) -> None:
        self._results = list(results)
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        return self._results.pop(0) if len(self._results) > 1 else self._results[0]


def _cp(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _client(monkeypatch, fake: _FakeRun) -> OsmoClient:
    client = OsmoClient(profile="default")
    monkeypatch.setattr(client, "_run", fake)
    return client


def test_submit_parses_the_workflow_id(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeRun(_cp(stdout="Workflow ID  - wf-abc123\nStatus - SUBMITTED\n"))
    assert _client(monkeypatch, fake).submit(tmp_path / "wf.yaml") == "wf-abc123"


def test_submit_forwards_pool_and_priority(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeRun(_cp(stdout="Workflow ID  - wf-1\n"))
    _client(monkeypatch, fake).submit(tmp_path / "wf.yaml", pool="p1", priority="HIGH")
    assert "--pool" in fake.calls[0] and "p1" in fake.calls[0]
    assert "--priority" in fake.calls[0] and "HIGH" in fake.calls[0]


def test_submit_omits_optional_flags_when_unset(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeRun(_cp(stdout="Workflow ID  - wf-1\n"))
    _client(monkeypatch, fake).submit(tmp_path / "wf.yaml")
    assert "--pool" not in fake.calls[0]
    assert "--priority" not in fake.calls[0]


def test_unparseable_submit_output_raises(monkeypatch, tmp_path: Path) -> None:
    with pytest.raises(OsmoCliError, match="Workflow ID"):
        _client(monkeypatch, _FakeRun(_cp(stdout="nothing useful"))).submit(tmp_path / "wf.yaml")


def test_auth_failure_is_classified(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeRun(_cp(returncode=1, stderr="request failed with status code 403"))
    with pytest.raises(OsmoAuthError):
        _client(monkeypatch, fake).submit(tmp_path / "wf.yaml")


def test_server_error_is_classified_as_transient(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeRun(_cp(returncode=1, stderr="request failed with status code 503"))
    with pytest.raises(OsmoTransientError):
        _client(monkeypatch, fake).submit(tmp_path / "wf.yaml")


def test_connection_reset_is_classified_as_transient(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeRun(_cp(returncode=1, stderr="connection reset by peer"))
    with pytest.raises(OsmoTransientError):
        _client(monkeypatch, fake).submit(tmp_path / "wf.yaml")


# Shape of a real `osmo workflow query -t json` document, trimmed to the fields
# the client reads. Tasks are nested under groups[].tasks[]; there is no
# top-level "tasks" key, and the workflow is named by "name".
_QUERY_JSON = json.dumps(
    {
        "name": "wf-1",
        "status": "RUNNING",
        "groups": [
            {"name": "g-t1", "status": "COMPLETED", "tasks": [{"name": "t1", "status": "COMPLETED", "exit_code": 0}]},
            {"name": "g-t2", "status": "FAILED", "tasks": [{"name": "t2", "status": "FAILED", "exit_code": 137}]},
        ],
    }
)


def test_status_asks_for_json_with_the_flag_osmo_accepts(monkeypatch) -> None:
    # `osmo workflow query` takes -t/--format-type, not --output; the wrong flag
    # made every query fail and every exit_code come back None.
    fake = _FakeRun(_cp(stdout=_QUERY_JSON))
    _client(monkeypatch, fake).status("wf-1")
    assert fake.calls[0][-2:] == ["-t", "json"]
    assert "--output" not in fake.calls[0]


def test_status_flattens_the_tasks_nested_under_groups(monkeypatch) -> None:
    snapshot = _client(monkeypatch, _FakeRun(_cp(stdout=_QUERY_JSON))).status("wf-1")
    assert snapshot.status == "RUNNING"
    assert [(task.name, task.status, task.exit_code) for task in snapshot.tasks] == [
        ("t1", "COMPLETED", 0),
        ("t2", "FAILED", 137),
    ]


def test_failed_status_query_raises(monkeypatch) -> None:
    fake = _FakeRun(_cp(returncode=1, stderr="no such workflow"))
    with pytest.raises(OsmoCliError, match="no such workflow"):
        _client(monkeypatch, fake).status("wf-1")


def test_data_download_shells_out_to_osmo_data(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeRun(_cp())
    _client(monkeypatch, fake).data_download("s3://bucket/dispatch/row", tmp_path / "dest")
    assert fake.calls[0][:3] == ["osmo", "data", "download"]
    assert "s3://bucket/dispatch/row" in fake.calls[0]
    assert (tmp_path / "dest").exists()


def test_failed_data_download_raises(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeRun(_cp(returncode=1, stderr="no such key"))
    with pytest.raises(OsmoCliError, match="no such key"):
        _client(monkeypatch, fake).data_download("s3://bucket/missing", tmp_path / "dest")


def test_dataset_helpers_are_gone() -> None:
    # OSMO datasets were retired; leaving these around would invite their use.
    assert not hasattr(OsmoClient, "dataset_download")
    assert not hasattr(OsmoClient, "dataset_delete")


def test_validate_accepts_a_good_workflow(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeRun(_cp(stdout="Workflow validation succeeded."))
    _client(monkeypatch, fake).validate(tmp_path / "wf.yaml")
    assert fake.calls[0][:3] == ["osmo", "workflow", "validate"]


def test_validate_raises_on_schema_rejection(monkeypatch, tmp_path: Path) -> None:
    # OSMO rejects unknown task fields outright; catching that before submit is
    # the whole point of the preflight.
    fake = _FakeRun(_cp(returncode=1, stderr="Extra inputs are not permitted"))
    with pytest.raises(OsmoCliError, match="Extra inputs"):
        _client(monkeypatch, fake).validate(tmp_path / "wf.yaml")


def test_data_check_reports_pass(monkeypatch) -> None:
    fake = _FakeRun(_cp(stdout='{"status": "pass"}'))
    assert _client(monkeypatch, fake).data_check("swift://h/x") is True
    assert "-a" in fake.calls[0] and "WRITE" in fake.calls[0]


def test_data_check_reports_failure(monkeypatch) -> None:
    fake = _FakeRun(_cp(stdout='{"status": "fail"}'))
    assert _client(monkeypatch, fake).data_check("swift://h/x") is False
