# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end CLI wiring, exercised offline via ``--dry-run``."""

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
        "--config", str(workspace / "odin.yaml"),
        "--tasks-yaml", str(workspace / "tasks.yaml"),
        "--image", "nvcr.io/nvidian/x@sha256:abc",
        "--seeds", "42",
        "--runs-root", str(workspace / "runs"),
        "--dry-run",
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
    main(_dispatch_argv(workspace, "--chunk-size", "1"))
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
    main(_dispatch_argv(workspace, "--include", "Isaac-Cartpole-*"))
    assert {job["task_id"] for job in _state(workspace)["jobs"]} == {"Isaac-Cartpole-Direct"}


def test_rows_carry_no_sizing_by_default(workspace: Path) -> None:
    # The seed list omits sizing so each task runs at its shipped default.
    main(_dispatch_argv(workspace))
    job = _state(workspace)["jobs"][0]
    assert job["num_envs"] is None
    assert job["max_iterations"] is None


def test_ab_mode_plans_both_sides(workspace: Path) -> None:
    main(_dispatch_argv(workspace, "--image-b", "nvcr.io/nvidian/x@sha256:def"))
    state = _state(workspace)
    assert {job["side"] for job in state["jobs"]} == {"a", "b"}
    assert state["images"] == {"a": "nvcr.io/nvidian/x@sha256:abc", "b": "nvcr.io/nvidian/x@sha256:def"}


def test_ab_row_keys_are_disambiguated_by_side(workspace: Path) -> None:
    main(_dispatch_argv(workspace, "--image-b", "nvcr.io/nvidian/x@sha256:def"))
    keys = [job["row_key"] for job in _state(workspace)["jobs"]]
    assert len(keys) == len(set(keys))


def test_ab_osmo_task_names_are_unique(workspace: Path) -> None:
    # Two OSMO tasks sharing a name inside one workflow is a submit-time error.
    main(_dispatch_argv(workspace, "--image-b", "nvcr.io/nvidian/x@sha256:def"))
    names = [job["osmo_task_name"] for job in _state(workspace)["jobs"]]
    assert len(names) == len(set(names))


def test_each_side_uses_its_own_image(workspace: Path) -> None:
    main(_dispatch_argv(workspace, "--image-b", "nvcr.io/nvidian/x@sha256:def"))
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
        ["dispatch", "--config", str(workspace / "absent.yaml"), "--tasks-yaml", str(workspace / "tasks.yaml"),
         "--image", "img", "--seeds", "42", "--runs-root", str(workspace / "runs"), "--dry-run"]
    )
    assert code == 1
    assert "could not read" in capsys.readouterr().err


def test_build_image_dry_run_writes_a_dockerfile(workspace: Path, tmp_path: Path) -> None:
    code = main([
        "build-image",
        "--config", str(workspace / "odin.yaml"),
        "--ref", "HEAD",
        "--profile", "newton",
        "--context-dir", str(tmp_path / "ctx"),
        "--dry-run",
    ])
    assert code == 0
    assert (tmp_path / "ctx" / "Dockerfile").exists()


def test_build_image_rejects_an_unknown_ref(workspace: Path, tmp_path: Path, capsys) -> None:
    code = main([
        "build-image",
        "--config", str(workspace / "odin.yaml"),
        "--ref", "definitely-not-a-ref-abc123",
        "--context-dir", str(tmp_path / "ctx"),
        "--dry-run",
    ])
    assert code == 1
    assert "definitely-not-a-ref" in capsys.readouterr().err


def test_status_on_a_missing_dispatch_exits_non_zero(workspace: Path) -> None:
    assert main(["status", "--runs-root", str(workspace / "runs"), "20260101-000000"]) != 0


def test_status_reports_job_counts(workspace: Path, capsys) -> None:
    main(_dispatch_argv(workspace))
    dispatch_id = _dispatch_dir(workspace).name
    capsys.readouterr()

    code = main(["status", "--runs-root", str(workspace / "runs"), dispatch_id])

    assert code == 0
    assert "pending" in capsys.readouterr().out
