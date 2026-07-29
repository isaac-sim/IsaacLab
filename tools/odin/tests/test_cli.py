# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end CLI wiring, exercised offline via ``--dry_run``."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from tools.odin.cli import main

_ODIN_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    shutil.copy(_ODIN_ROOT / "config" / "odin.yaml", tmp_path / "odin.yaml")
    shutil.copy(_ODIN_ROOT / "config" / "tasks.yaml", tmp_path / "tasks.yaml")
    return tmp_path


def _dispatch_argv(workspace: Path, *extra: str) -> list[str]:
    return [
        "dispatch",
        "--config",
        str(workspace / "odin.yaml"),
        "--tasks_yaml",
        str(workspace / "tasks.yaml"),
        "--image",
        "nvcr.io/nvidian/x@sha256:abc",
        "--seeds",
        "42",
        "--runs_root",
        str(workspace / "runs"),
        "--dry_run",
        *extra,
    ]


def _dispatch_dir(workspace: Path) -> Path:
    return next((workspace / "runs").iterdir())


def _state(workspace: Path) -> dict:
    return json.loads((_dispatch_dir(workspace) / "dispatch.json").read_text())


def test_no_subcommand_exits_non_zero() -> None:
    assert main([]) != 0


def test_dry_run_dispatch_exits_zero(workspace: Path) -> None:
    assert main(_dispatch_argv(workspace)) == 0


def test_dry_run_writes_workflow_yaml_per_chunk(workspace: Path) -> None:
    main(_dispatch_argv(workspace, "--chunk_size", "1"))
    rendered = sorted(_dispatch_dir(workspace).glob("workflow.*.yaml"))
    assert len(rendered) >= 2
    assert yaml.safe_load(rendered[0].read_text())["workflow"]["groups"]


def test_dry_run_writes_dispatch_json_with_pending_jobs(workspace: Path) -> None:
    main(_dispatch_argv(workspace))
    state = _state(workspace)
    assert state["schema_version"] == "2.0"
    assert state["jobs"]
    assert all(job["status"] == "pending" for job in state["jobs"])


def test_dry_run_submits_nothing(workspace: Path) -> None:
    # No OSMO call means no workflow ids recorded.
    main(_dispatch_argv(workspace))
    assert _state(workspace)["osmo_workflow_ids"] == []


def test_include_filter_narrows_the_row_set(workspace: Path) -> None:
    import fnmatch

    main(_dispatch_argv(workspace, "--include", "Isaac-Cartpole-*"))
    matched = {job["task_id"] for job in _state(workspace)["jobs"]}
    assert matched
    assert all(fnmatch.fnmatch(task_id, "Isaac-Cartpole-*") for task_id in matched)


def test_rows_carry_no_sizing_by_default(workspace: Path) -> None:
    # The seed list omits sizing so each task runs at its shipped default.
    main(_dispatch_argv(workspace))
    job = _state(workspace)["jobs"][0]
    assert job["num_envs"] is None
    assert job["max_iterations"] is None


def test_ab_mode_plans_both_sides(workspace: Path) -> None:
    main(_dispatch_argv(workspace, "--image_b", "nvcr.io/nvidian/x@sha256:def"))
    state = _state(workspace)
    assert {job["side"] for job in state["jobs"]} == {"a", "b"}
    assert state["images"] == {"a": "nvcr.io/nvidian/x@sha256:abc", "b": "nvcr.io/nvidian/x@sha256:def"}


def test_ab_row_keys_are_disambiguated_by_side(workspace: Path) -> None:
    main(_dispatch_argv(workspace, "--image_b", "nvcr.io/nvidian/x@sha256:def"))
    keys = [job["row_key"] for job in _state(workspace)["jobs"]]
    assert len(keys) == len(set(keys))


def test_ab_osmo_task_names_are_unique(workspace: Path) -> None:
    # Two OSMO tasks sharing a name inside one workflow is a submit-time error.
    main(_dispatch_argv(workspace, "--image_b", "nvcr.io/nvidian/x@sha256:def"))
    names = [job["osmo_task_name"] for job in _state(workspace)["jobs"]]
    assert len(names) == len(set(names))


def test_each_side_uses_its_own_image(workspace: Path) -> None:
    main(_dispatch_argv(workspace, "--image_b", "nvcr.io/nvidian/x@sha256:def"))
    by_side = {job["side"]: job["image_ref"] for job in _state(workspace)["jobs"]}
    assert by_side["a"] == "nvcr.io/nvidian/x@sha256:abc"
    assert by_side["b"] == "nvcr.io/nvidian/x@sha256:def"


def test_seeds_are_recorded(workspace: Path) -> None:
    main(_dispatch_argv(workspace))
    assert _state(workspace)["seeds"] == [42]


def test_multiple_seeds_expand(workspace: Path) -> None:
    main(_dispatch_argv(workspace, "--seeds", "42,43"))
    state = _state(workspace)
    assert state["seeds"] == [42, 43]
    assert {job["seed"] for job in state["jobs"]} == {42, 43}


def test_bad_config_path_exits_non_zero(workspace: Path, capsys) -> None:
    code = main(
        [
            "dispatch",
            "--config",
            str(workspace / "absent.yaml"),
            "--tasks_yaml",
            str(workspace / "tasks.yaml"),
            "--image",
            "img",
            "--seeds",
            "42",
            "--runs_root",
            str(workspace / "runs"),
            "--dry_run",
        ]
    )
    assert code == 1
    assert "could not read" in capsys.readouterr().err


def test_build_image_dry_run_writes_a_dockerfile(workspace: Path, tmp_path: Path) -> None:
    code = main(
        [
            "build-image",
            "--config",
            str(workspace / "odin.yaml"),
            "--ref",
            "HEAD",
            "--context_dir",
            str(tmp_path / "ctx"),
            "--dry_run",
        ]
    )
    assert code == 0
    assert (tmp_path / "ctx" / "Dockerfile").exists()


def test_build_image_rejects_an_unknown_ref(workspace: Path, tmp_path: Path, capsys) -> None:
    code = main(
        [
            "build-image",
            "--config",
            str(workspace / "odin.yaml"),
            "--ref",
            "definitely-not-a-ref-abc123",
            "--context_dir",
            str(tmp_path / "ctx"),
            "--dry_run",
        ]
    )
    assert code == 1
    assert "definitely-not-a-ref" in capsys.readouterr().err


def test_status_on_a_missing_dispatch_exits_non_zero(workspace: Path) -> None:
    assert main(["status", "--runs_root", str(workspace / "runs"), "20260101-000000"]) != 0


def test_status_reports_job_counts(workspace: Path, capsys) -> None:
    main(_dispatch_argv(workspace))
    dispatch_id = _dispatch_dir(workspace).name
    capsys.readouterr()

    code = main(["status", "--runs_root", str(workspace / "runs"), dispatch_id])

    assert code == 0
    assert "pending" in capsys.readouterr().out


def test_harvest_on_a_missing_dispatch_exits_non_zero(workspace: Path) -> None:
    assert main(["harvest", "--runs_root", str(workspace / "runs"), "20260101-000000"]) != 0


def test_harvest_writes_metadata_from_bundles(workspace: Path, tmp_path: Path) -> None:
    main(_dispatch_argv(workspace))
    dispatch_id = _dispatch_dir(workspace).name
    row_key = _state(workspace)["jobs"][0]["row_key"]
    row_dir = _dispatch_dir(workspace) / row_key
    row_dir.mkdir(parents=True, exist_ok=True)
    (row_dir / "benchmark_training_x_2026-07-29_12-00-00.json").write_text(
        json.dumps(
            {
                "schema_version": "1.2",
                "run": {
                    "task": "Isaac-Ant",
                    "framework": "rsl_rl",
                    "config": {"physics_backend": "newton_mjwarp"},
                    "seed": 42,
                    "duration_s": 120.0,
                    "status": "completed",
                    "num_envs": 4096,
                    "max_iterations": 500,
                },
                "runtime": {"total_wall_time_s": 120.0},
            }
        )
    )
    out = tmp_path / "task_metadata.yaml"

    code = main(
        [
            "harvest",
            "--runs_root",
            str(workspace / "runs"),
            dispatch_id,
            "--output",
            str(out),
            "--timeout_headroom",
            "2.0",
        ]
    )

    assert code == 0
    entry = yaml.safe_load(out.read_text())["tasks"][0]
    assert entry["num_envs"] == 4096
    assert entry["timeout_s"] == 240


def test_discover_writes_a_planner_ready_task_list(workspace: Path, tmp_path: Path, monkeypatch) -> None:
    # The registry walk needs Isaac Lab; the CLI wiring around it does not.
    from tools.odin import cli as cli_module
    from tools.odin.discover import DiscoveredTask

    monkeypatch.setattr(
        cli_module,
        "discover_tasks",
        lambda: [
            DiscoveredTask("Isaac-Ant", "core", ("rsl_rl",), (DiscoveredTask.Mode("newton_mjwarp", None, None),)),
            DiscoveredTask("IsaacContrib-Walk", "contrib", ("rsl_rl",), (DiscoveredTask.Mode("ovphysx", None, None),)),
        ],
    )
    out = tmp_path / "tasks.yaml"

    assert main(["discover", "--output", str(out)]) == 0

    payload = yaml.safe_load(out.read_text())
    assert payload["discovery"]["row_count"] == 2
    assert {row["scope"] for row in payload["tasks"]} == {"core", "contrib"}


def test_discover_scope_filter_can_empty_the_list(tmp_path: Path, monkeypatch) -> None:
    from tools.odin import cli as cli_module
    from tools.odin.discover import DiscoveredTask

    monkeypatch.setattr(
        cli_module,
        "discover_tasks",
        lambda: [DiscoveredTask("Isaac-Ant", "core", ("rsl_rl",), (DiscoveredTask.Mode(None, None, None),))],
    )
    assert main(["discover", "--scope", "contrib", "--output", str(tmp_path / "t.yaml")]) != 0


def _mark_failed(workspace: Path, kind: str = "benchmark_crash", limit: int = 2) -> str:
    """Fail the first *limit* jobs of the newest dispatch and return its id."""
    dispatch_dir = _dispatch_dir(workspace)
    state = json.loads((dispatch_dir / "dispatch.json").read_text())
    state["osmo_workflow_ids"] = ["wf-1"]
    for job in state["jobs"][:limit]:
        job["status"] = "failed"
        job["ended_at"] = "2026-07-29T12:00:00Z"
        job["failure"] = {"kind": kind, "message": "boom", "details": {}}
    (dispatch_dir / "dispatch.json").write_text(json.dumps(state))
    return dispatch_dir.name


def test_dispatch_requires_an_image_unless_resuming(workspace: Path, capsys) -> None:
    code = main(
        [
            "dispatch",
            "--config",
            str(workspace / "odin.yaml"),
            "--tasks_yaml",
            str(workspace / "tasks.yaml"),
            "--seeds",
            "42",
            "--runs_root",
            str(workspace / "runs"),
            "--dry_run",
        ]
    )
    assert code == 1
    assert "--image is required" in capsys.readouterr().err


def test_dispatch_requires_seeds_unless_retrying(workspace: Path, capsys) -> None:
    code = main(
        [
            "dispatch",
            "--config",
            str(workspace / "odin.yaml"),
            "--tasks_yaml",
            str(workspace / "tasks.yaml"),
            "--image",
            "img",
            "--runs_root",
            str(workspace / "runs"),
            "--dry_run",
        ]
    )
    assert code == 1
    assert "--seeds is required" in capsys.readouterr().err


def test_resume_rejects_a_dispatch_that_never_submitted(workspace: Path, capsys) -> None:
    # Nothing was submitted, so there is no workflow to re-attach to.
    main(_dispatch_argv(workspace))
    dispatch_id = _dispatch_dir(workspace).name
    code = main(
        [
            "dispatch",
            "--config",
            str(workspace / "odin.yaml"),
            "--runs_root",
            str(workspace / "runs"),
            "--resume",
            dispatch_id,
        ]
    )
    assert code == 1
    assert "never got past planning" in capsys.readouterr().err


def test_resume_rejects_an_unknown_dispatch(workspace: Path, capsys) -> None:
    code = main(
        [
            "dispatch",
            "--config",
            str(workspace / "odin.yaml"),
            "--runs_root",
            str(workspace / "runs"),
            "--resume",
            "20260101-000000",
        ]
    )
    assert code == 1
    assert "no dispatch.json" in capsys.readouterr().err


def test_resume_latest_with_no_dispatches_is_an_error(workspace: Path, capsys) -> None:
    (workspace / "runs").mkdir(parents=True, exist_ok=True)
    code = main(
        [
            "dispatch",
            "--config",
            str(workspace / "odin.yaml"),
            "--runs_root",
            str(workspace / "runs"),
            "--resume",
            "LATEST",
        ]
    )
    assert code == 1
    assert "no dispatch found" in capsys.readouterr().err


def test_retry_failed_plans_only_the_failed_rows(workspace: Path) -> None:
    main(_dispatch_argv(workspace, "--include", "Isaac-Cartpole-Direct"))
    parent = _mark_failed(workspace, limit=2)
    failed_ids = {
        job["task_id"]
        for job in json.loads((_dispatch_dir(workspace) / "dispatch.json").read_text())["jobs"]
        if job["status"] == "failed"
    }

    code = main(_retry_argv(workspace, parent))

    assert code == 0
    child = sorted((workspace / "runs").iterdir())[-1]
    state = json.loads((child / "dispatch.json").read_text())
    assert state["parent_dispatch_id"] == parent
    assert {job["task_id"] for job in state["jobs"]} <= failed_ids


def test_retry_failed_reuses_the_parent_seeds(workspace: Path) -> None:
    main(_dispatch_argv(workspace, "--seeds", "42,43", "--include", "Isaac-Cartpole-Direct"))
    parent = _mark_failed(workspace, limit=2)

    main(_retry_argv(workspace, parent))

    child = sorted((workspace / "runs").iterdir())[-1]
    assert json.loads((child / "dispatch.json").read_text())["seeds"]


def _retry_argv(workspace: Path, parent: str) -> list[str]:
    return [
        "dispatch",
        "--config",
        str(workspace / "odin.yaml"),
        "--image",
        "img@sha256:abc",
        "--runs_root",
        str(workspace / "runs"),
        "--retry_failed",
        parent,
        "--dry_run",
    ]


def _identity(job: dict) -> tuple:
    return (job["task_id"], job["rl_library"], job["physics"], job["seed"])


def test_retry_failed_pairs_each_identity_with_the_seeds_it_failed_on(workspace: Path) -> None:
    # Collapsing failures to (identities, union-of-seeds) re-expands a cross
    # product, so a seed that passed is billed a second GPU run.
    main(_dispatch_argv(workspace, "--seeds", "42,43", "--include", "Isaac-Cartpole-Direct"))
    dispatch_dir = _dispatch_dir(workspace)
    state = json.loads((dispatch_dir / "dispatch.json").read_text())
    state["osmo_workflow_ids"] = ["wf-1"]
    by_seed = {seed: [job for job in state["jobs"] if job["seed"] == seed] for seed in (42, 43)}
    # Two distinct identities, each failing on one seed only.
    victims = [by_seed[42][0], by_seed[43][1]]
    for job in victims:
        job["status"] = "failed"
        job["ended_at"] = "2026-07-29T12:00:00Z"
        job["failure"] = {"kind": "benchmark_crash", "message": "boom", "details": {}}
    (dispatch_dir / "dispatch.json").write_text(json.dumps(state))

    assert main(_retry_argv(workspace, dispatch_dir.name)) == 0

    child = json.loads((sorted((workspace / "runs").iterdir())[-1] / "dispatch.json").read_text())
    assert {_identity(job) for job in child["jobs"]} == {_identity(job) for job in victims}


def test_retry_of_an_ab_dispatch_is_refused(workspace: Path, capsys) -> None:
    # Retrying plans one row per identity against --image, so a side-B failure
    # would silently be re-run on side A's image.
    main(_dispatch_argv(workspace, "--include", "Isaac-Cartpole-Direct", "--image_b", "nvcr.io/nvidian/x@sha256:def"))
    parent = _mark_failed(workspace, limit=2)

    code = main(_retry_argv(workspace, parent))

    assert code == 1
    assert "A/B dispatch" in capsys.readouterr().err


def test_chunk_size_below_one_leaves_no_orphan_dispatch(workspace: Path, capsys) -> None:
    # Chunking runs after dispatch.json is written, so a late rejection would
    # leave a dispatch that LATEST resolves to.
    code = main(_dispatch_argv(workspace, "--chunk_size", "0"))

    assert code == 1
    assert "chunk_size" in capsys.readouterr().err
    assert not (workspace / "runs").exists()


def test_retry_failed_on_a_clean_dispatch_is_an_error(workspace: Path, capsys) -> None:
    main(_dispatch_argv(workspace))
    parent = _dispatch_dir(workspace).name
    code = main(_retry_argv(workspace, parent))
    assert code == 1
    assert "no failed rows" in capsys.readouterr().err


def test_resume_resets_in_flight_rows_and_polls(workspace: Path, monkeypatch, capsys) -> None:
    # The happy path: workflows exist, so resume must re-attach rather than
    # re-submit. Every other resume test returns early before reaching it.
    from tools.odin import cli as cli_module
    from tools.odin.client import TaskSnapshot, WorkflowSnapshot

    main(_dispatch_argv(workspace, "--include", "Isaac-Cartpole-Direct"))
    dispatch_dir = _dispatch_dir(workspace)
    state = json.loads((dispatch_dir / "dispatch.json").read_text())
    state["osmo_workflow_ids"] = ["wf-1"]
    for job in state["jobs"]:
        job["status"] = "running"
        job["assigned_to"] = "osmo:wf-1"
        job["started_at"] = "2026-07-29T12:00:00Z"
    (dispatch_dir / "dispatch.json").write_text(json.dumps(state))
    names = [job["osmo_task_name"] for job in state["jobs"]]

    class _Client:
        def __init__(self, *a, **k) -> None: ...

        def status(self, workflow_id):
            return WorkflowSnapshot(
                workflow_id=workflow_id,
                status="COMPLETED",
                tasks=[TaskSnapshot(name=n, status="COMPLETED", exit_code=0) for n in names],
            )

        def data_download(self, remote_uri, dest_dir):
            dest_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(cli_module, "OsmoClient", _Client)

    code = main(
        [
            "dispatch",
            "--config",
            str(workspace / "odin.yaml"),
            "--runs_root",
            str(workspace / "runs"),
            "--resume",
            "LATEST",
            "--poll_interval",
            "0",
        ]
    )

    out = capsys.readouterr().out
    assert "resuming" in out
    # Nothing parseable came back, so completed rows are reclassified.
    final = json.loads((dispatch_dir / "dispatch.json").read_text())
    assert all(job["status"] == "failed" for job in final["jobs"])
    assert all(job["failure"]["kind"] == "malformed_bundle" for job in final["jobs"])
    assert code == 1
