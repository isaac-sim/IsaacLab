# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering of the OSMO workflow YAML."""

import dataclasses

import pytest
import yaml

from tools.odin.config_file import ImageSpec, OdinConfig, ResourcesSpec, RetrySpec
from tools.odin.plan import PlannedRow
from tools.odin.workflow import osmo_safe_task_name, render_workflow_yaml

_CFG = OdinConfig(
    osmo_profile="default",
    pool="isaac-dev-l40s-04",
    priority="NORMAL",
    image=ImageSpec(registry="nvcr.io/nvidian", repository="antoiner-isaac-lab", pull_credential="ngc_read_only"),
    resources=ResourcesSpec(cpu=8, gpu=1, memory="64Gi", storage="64Gi", platform="ovx-l40s"),
    exec_timeout_s=86400,
    queue_timeout_s=172800,
    chunk_size=25,
    results_uri="s3://isaac-odin/results",
    retry=RetrySpec(reschedule_codes="3001-3006", restart_codes=""),
)

_ROW = PlannedRow(
    row_key="rsl_rl_physx_Isaac-Ant_seed42",
    task_id="Isaac-Ant",
    rl_library="rsl_rl",
    physics="physx",
    renderer=None,
    seed=42,
    num_envs=None,
    max_iterations=None,
    timeout_s=None,
    uv_extras=("rsl-rl", "isaacsim"),
    osmo_task_name="rsl-rl-physx-isaac-ant-seed42",
)

_IMAGE = "nvcr.io/nvidian/antoiner-isaac-lab@sha256:abc"


def _render(rows=(_ROW,), publish=None, chunk_index=0):
    rows = list(rows)
    if publish is None:
        publish = {row.row_key: "echo published" for row in rows}
    return render_workflow_yaml(
        dispatch_id="20260728-120000",
        chunk_index=chunk_index,
        rows=rows,
        cfg=_CFG,
        image_ref=_IMAGE,
        publish_commands=publish,
    )


def _task(rendered: str, index: int = 0) -> dict:
    return yaml.safe_load(rendered)["workflow"]["groups"][index]["tasks"][0]


def _script(rendered: str, index: int = 0) -> str:
    return _task(rendered, index)["files"][0]["contents"]


def test_rendered_yaml_parses() -> None:
    assert yaml.safe_load(_render())["workflow"]["name"]


def test_each_task_gets_its_own_group() -> None:
    # OSMO's exec_timeout clock starts at each group's RUNNING transition. With
    # tasks at the workflow root they share one implicit group, so a
    # late-scheduled task is killed for other tasks' queue time.
    second = dataclasses.replace(_ROW, row_key="other", osmo_task_name="other")
    groups = yaml.safe_load(_render(rows=(_ROW, second)))["workflow"]["groups"]
    assert len(groups) == 2
    assert all(len(group["tasks"]) == 1 for group in groups)


def test_image_reference_is_the_digest_pinned_argument() -> None:
    assert _task(_render())["image"] == _IMAGE


def test_pull_credential_is_emitted_when_configured() -> None:
    assert _task(_render())["pull_credential"] == "ngc_read_only"


def test_timeouts_come_from_config() -> None:
    timeout = yaml.safe_load(_render())["workflow"]["timeout"]
    assert timeout["exec_timeout"] == "86400s"
    assert timeout["queue_timeout"] == "172800s"


def test_entry_script_invokes_the_upstream_benchmark_cli() -> None:
    script = _script(_render())
    assert "uv run" in script
    assert "--extra rsl-rl" in script
    assert "--extra isaacsim" in script
    assert "isaaclab benchmark training" in script
    assert "--rl_library rsl_rl" in script
    assert "--task Isaac-Ant" in script
    assert "--seed 42" in script
    assert "--benchmark_formatter schema" in script
    assert "physics=physx" in script


def test_entry_script_does_not_pass_headless() -> None:
    # --headless was removed upstream in favour of --visualizer/--viz; runs are
    # headless when neither is passed.
    assert "--headless" not in _script(_render())


def test_unsized_row_omits_the_sizing_flags() -> None:
    # Omitting them is what makes the task fall back to its shipped agent config.
    script = _script(_render())
    assert "--num_envs" not in script
    assert "--max_iterations" not in script


def test_sized_row_emits_the_sizing_flags() -> None:
    row = dataclasses.replace(_ROW, num_envs=4096, max_iterations=500)
    script = _script(_render(rows=(row,)))
    assert "--num_envs 4096" in script
    assert "--max_iterations 500" in script


def test_headless_row_omits_the_renderer_selector() -> None:
    assert "renderer=" not in _script(_render())


def test_renderer_selector_is_emitted_when_set() -> None:
    row = dataclasses.replace(_ROW, renderer="isaacsim_rtx")
    assert "renderer=isaacsim_rtx" in _script(_render(rows=(row,)))


def test_publish_command_is_spliced_in() -> None:
    script = _script(_render(publish={_ROW.row_key: "osmo data upload s3://b/x $OUT"}))
    assert "osmo data upload s3://b/x $OUT" in script


def test_entry_script_preserves_the_benchmark_exit_code() -> None:
    # Results must publish even on failure, so the script captures rc rather
    # than dying under `set -e`, then exits with it.
    script = _script(_render())
    assert "rc=$?" in script
    assert "exit $rc" in script


def test_exit_actions_come_from_retry_config() -> None:
    actions = _task(_render())["exitActions"]
    assert actions["RESCHEDULE"] == "3001-3006"
    assert "RESTART" not in actions


def test_workflow_name_is_unique_per_chunk() -> None:
    first = yaml.safe_load(_render(chunk_index=0))["workflow"]["name"]
    second = yaml.safe_load(_render(chunk_index=1))["workflow"]["name"]
    assert first != second


def test_missing_publish_command_is_rejected() -> None:
    with pytest.raises(KeyError, match=_ROW.row_key):
        _render(publish={})


def test_long_row_key_is_truncated_with_a_stable_hash() -> None:
    long_key = "rsl_rl_newton_mjwarp_" + "Isaac-Very-Long-Task-Name-" * 4 + "seed42"
    name = osmo_safe_task_name(long_key)
    assert len(name) <= 63
    assert name == osmo_safe_task_name(long_key)
