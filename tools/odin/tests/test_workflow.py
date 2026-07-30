# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rendering of the OSMO workflow YAML.

The rendered artefact is pinned whole by the goldens under ``golden/``. The
tests below cover the invariants that a golden diff would show but not explain.
"""

import dataclasses
import subprocess
from pathlib import Path

import yaml

from tools.odin.config_file import ImageSpec, OdinConfig, ResourcesSpec, RetrySpec
from tools.odin.plan import UV_EXTRAS, PlannedRow
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
    presets=(),
    play=False,
    keep_checkpoint=False,
    video_length=200,
    seed=42,
    num_envs=None,
    max_iterations=None,
    timeout_s=None,
    uv_extras=UV_EXTRAS,
    osmo_task_name="rsl-rl-physx-isaac-ant-seed42",
)

_IMAGE = "nvcr.io/nvidian/antoiner-isaac-lab@sha256:abc"


_OUTPUT_URI = "swift://h/AUTH_team-isaac/results/20260728-120000/"


def _render(rows=(_ROW,), chunk_index=0, output_uri=_OUTPUT_URI):
    return render_workflow_yaml(
        dispatch_id="20260728-120000",
        chunk_index=chunk_index,
        rows=list(rows),
        cfg=_CFG,
        image_ref=_IMAGE,
        output_uri=output_uri,
    )


def _task(rendered: str, index: int = 0) -> dict:
    return yaml.safe_load(rendered)["workflow"]["groups"][index]["tasks"][0]


def _script(rendered: str, index: int = 0) -> str:
    return _task(rendered, index)["files"][0]["contents"]


def test_default_row_renders_the_golden_workflow(assert_golden) -> None:
    assert_golden("workflow_default.yaml", _render())


def test_sized_camera_row_renders_the_golden_workflow(assert_golden) -> None:
    row = dataclasses.replace(
        _ROW,
        num_envs=4096,
        max_iterations=500,
        timeout_s=900,
        renderer="ovrtx",
        presets=("depth", "simple_shading_full_mdl"),
    )
    assert_golden("workflow_sized_renderer_presets.yaml", _render(rows=(row,)))


def test_play_and_keep_checkpoint_row_renders_the_golden_workflow(assert_golden) -> None:
    row = dataclasses.replace(_ROW, play=True, keep_checkpoint=True, video_length=120)
    assert_golden("workflow_play_keep_checkpoint.yaml", _render(rows=(row,)))


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


def test_pull_credential_maps_the_name_to_the_registry_host() -> None:
    # OSMO rejects a bare `pull_credential:` field, and its `credentials:` block
    # is keyed by credential name with the registry HOST as the value -- not the
    # namespaced registry path.
    assert _task(_render())["credentials"] == {"ngc_read_only": "nvcr.io"}


def test_entry_script_preserves_the_benchmark_exit_code() -> None:
    # Results must publish even on failure, so the script captures rc rather
    # than dying under `set -e`, then exits with it.
    script = _script(_render())
    assert "rc=$?" in script
    assert "exit $rc" in script


def test_artifact_listing_cannot_fail_the_task() -> None:
    # `head` closes the pipe early, so `find` dies of SIGPIPE and the pipeline
    # exits 141. Under `set -o pipefail` that ends the script before `exit $rc`,
    # turning a successful run with many output files into a failed row.
    assert 'find "$OUT" -type f | head -40 || true' in _script(_render())


def test_workflow_name_is_unique_per_chunk() -> None:
    first = yaml.safe_load(_render(chunk_index=0))["workflow"]["name"]
    second = yaml.safe_load(_render(chunk_index=1))["workflow"]["name"]
    assert first != second


def test_long_row_key_is_truncated_with_a_stable_hash() -> None:
    long_key = "rsl_rl_newton_mjwarp_" + "Isaac-Very-Long-Task-Name-" * 4 + "seed42"
    name = osmo_safe_task_name(long_key)
    assert len(name) <= 63
    assert name == osmo_safe_task_name(long_key)


def test_chained_play_is_skipped_when_training_fails() -> None:
    # Nothing to roll out, and a play error would mask the real verdict.
    script = _script(_render(rows=(dataclasses.replace(_ROW, play=True),)))
    assert 'if [ "$rc" -ne 0 ]; then' in script
    assert 'PLAY_STATUS="skipped_training_failed"' in script


def _run_exit_stanza(tmp_path: Path, *, rc: int, play_status: str, play_rc: str) -> int:
    """Execute the rendered script's final exit logic and return its exit code.

    Runs the stanza from the last ``set -e`` onwards with the state variables
    preset, so the exit contract is checked by running it rather than by
    matching source text.
    """
    script = _script(_render(rows=(dataclasses.replace(_ROW, play=True),)))
    stanza = script[script.rindex("set -e") :]
    preamble = f'OUT={tmp_path}\nrc={rc}\nPLAY_STATUS="{play_status}"\nPLAY_RC="{play_rc}"\n'
    return subprocess.run(["bash", "-c", preamble + stanza], capture_output=True, text=True).returncode


def test_a_training_failure_is_the_verdict(tmp_path: Path) -> None:
    assert _run_exit_stanza(tmp_path, rc=1, play_status="skipped_training_failed", play_rc="") == 1


def test_a_failed_play_fails_the_row(tmp_path: Path) -> None:
    # Videos are the deliverable: a row that trained but produced no video must
    # not report green, or a whole dispatch can silently yield no videos.
    assert _run_exit_stanza(tmp_path, rc=0, play_status="ran", play_rc="1") == 90


def test_a_successful_run_exits_zero(tmp_path: Path) -> None:
    assert _run_exit_stanza(tmp_path, rc=0, play_status="ran", play_rc="0") == 0


def test_a_row_without_play_exits_zero(tmp_path: Path) -> None:
    assert _run_exit_stanza(tmp_path, rc=0, play_status="not_requested", play_rc="") == 0


def test_chained_script_carries_the_same_presets_to_both_steps() -> None:
    row = dataclasses.replace(_ROW, play=True, presets=("depth",), renderer="ovrtx")
    script = _script(_render(rows=(row,)))
    assert script.count("presets=depth") == 2
    assert script.count("renderer=ovrtx") == 2


def test_entry_script_never_calls_system_python() -> None:
    # The base image is nvidia/cuda plus uv, which manages its own Python, so
    # there is no system interpreter on PATH. Invoking one silently yielded an
    # empty checkpoint and a misleading "no checkpoint" diagnosis.
    script = _script(_render(rows=(dataclasses.replace(_ROW, play=True),)))
    code = "\n".join(line for line in script.splitlines() if not line.lstrip().startswith("#"))
    assert "python3" not in code
    assert "/workspace/isaaclab/.venv/bin/python" in code
