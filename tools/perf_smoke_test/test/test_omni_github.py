# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free checks for the omni-github artifact emitted by the perf-smoke gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_GATE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _GATE_DIR.parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

import omni_github  # noqa: E402
from contracts import BenchResult  # noqa: E402
from gate_types import OracleVerdict  # noqa: E402
from oracle import Baseline, compare  # noqa: E402

_SCHEMA_PATH = _REPO_ROOT / ".github/actions/upload-omni-github-test-results/result-json.schema.json"


def _bench_result(*, task_id: str = "Isaac-Cartpole-Direct", fps: float | None = 100.0) -> BenchResult:
    launch_config = {
        "task_id": task_id,
        "backend": "physx",
        "backend_key": "physx",
        "physics_backend": "physx",
        "render_backend": None,
        "gpu_model": "l40s",
        "num_envs": 4096,
        "num_frames": 600,
    }
    return BenchResult(
        task_id=task_id,
        backend="physx",
        physics_backend="physx",
        render_backend=None,
        backend_key="physx",
        preset="default",
        was_retried=False,
        perf_smoke_test_info_present=fps is not None,
        raw_fps_mean=fps,
        raw_fps_std=1.0 if fps is not None else None,
        raw_fps_min=(fps - 1.0) if fps is not None else None,
        raw_fps_max=(fps + 1.0) if fps is not None else None,
        wall_time_s=42.5,
        startup_time_s=12.3,
        runtime_contract_hash="runtime-a",
        runtime_resources={
            "gpu_name": "NVIDIA L40S",
            "gpu_mem_used_mb": 1024.0,
            "system_ram_used_mb": 2048.0,
            "cuda_version": "12.4",
            "nvidia_driver_version": "550.90",
        },
        provenance={"software": {"isaacsim": "5.0.0", "isaaclab": "3.0.0", "warp": "1.13.0", "torch": "2.9.0"}},
        launch_config=launch_config,
        launch_config_hash="launch-a",
        benchmark_contract_hash="bench-a",
        baseline_epoch=1,
    )


def _rows():
    baseline = Baseline(median_fps=100.0, mad_fps=2.0, sample_count=5)
    pass_bench = _bench_result(fps=100.0)
    block_bench = _bench_result(task_id="Isaac-Ant-Direct", fps=80.0)
    return [
        (compare(pass_bench, baseline, []), pass_bench),
        (compare(block_bench, baseline, []), block_bench),
    ]


def test_build_result_shape_and_custom_diagnostics() -> None:
    """Every row carries the verdict plus hardware/software diagnostics under custom.perf_smoke."""
    result = omni_github.build_result(_rows(), platform="linux-x86_64", app_config="aggregate")

    assert result["test_tool_id"] == omni_github.TEST_TOOL_ID
    assert result["app"] == {"platform": "linux-x86_64", "config": "aggregate"}
    assert len(result["tests"]) == 2

    passed, blocked = result["tests"]
    assert passed["passed"] is True and "message" not in passed
    assert blocked["passed"] is False and blocked["message"].startswith("BLOCK")

    custom = passed["custom"]["perf_smoke"]
    assert custom["task_id"] == "Isaac-Cartpole-Direct"
    assert custom["verdict"] == "PASS"
    assert custom["physics_backend"] == "physx"
    assert custom["num_envs"] == 4096
    assert custom["num_frames"] == 600
    assert custom["wall_time_s"] == pytest.approx(42.5)
    assert custom["startup_time_s"] == pytest.approx(12.3)
    assert custom["fps_mean"] == pytest.approx(100.0)
    assert custom["regression_pct"] == pytest.approx(0.0)
    assert custom["vram_mb"] == pytest.approx(1024.0)
    assert custom["sysram_mb"] == pytest.approx(2048.0)
    # Trimmed to the dashboard core: deep diagnostics stay out of the artifact.
    assert "runtime_contract_hash" not in custom
    assert "warp_version" not in custom
    # None-valued diagnostics are dropped, never emitted as null.
    assert all(value is not None for value in custom.values())


def _relax_custom_value_oneof(schema: dict) -> dict:
    """Return the shared schema with ``customValue.oneOf`` rewritten to ``anyOf``.

    The committed schema defines custom values as a ``oneOf`` union that lists both
    ``integer`` and ``number``. Under JSON Schema draft 2020-12 an integral value
    (e.g. ``1`` or ``100.0``) matches *both* branches, so ``oneOf`` rejects every
    integral number even though omni-github ingests them as ``l_*``/``d_*`` leaves.
    We keep numeric diagnostics and validate against the intended union semantics.
    """
    custom_value = schema["$defs"]["customValue"]
    custom_value["anyOf"] = custom_value.pop("oneOf")
    for branch in custom_value["anyOf"]:
        items = branch.get("items")
        if isinstance(items, dict) and "oneOf" in items:
            items["anyOf"] = items.pop("oneOf")
    return schema


def test_write_artifact_matches_shared_schema(tmp_path) -> None:
    """The emitted result JSON validates against omni-github's committed schema."""
    jsonschema = pytest.importorskip("jsonschema")

    out = omni_github.write_artifact(_rows(), tmp_path, platform="linux-x86_64", app_config="aggregate")

    manifest = json.loads((out / omni_github.MANIFEST_NAME).read_text())
    assert manifest == {"schema_version": 1, "result_paths": [omni_github.RESULT_REL_PATH]}

    result = json.loads((out / omni_github.RESULT_REL_PATH).read_text())
    schema = _relax_custom_value_oneof(json.loads(_SCHEMA_PATH.read_text()))
    jsonschema.validate(instance=result, schema=schema)


def test_hard_failure_row_has_no_measured_fps() -> None:
    """A crashed benchmark (no FPS) still emits a valid failing row without null customs."""
    bench = _bench_result(fps=None)
    result_row = compare(bench, baseline=None, fps_mean_thresholds=[])
    assert result_row.verdict == OracleVerdict.HARD_FAILURE

    payload = omni_github.build_result([(result_row, bench)], platform="linux-x86_64", app_config="aggregate")
    row = payload["tests"][0]

    assert row["passed"] is False
    assert "fps_mean" not in row["custom"]["perf_smoke"]
    assert row["custom"]["perf_smoke"]["verdict"] == "HARD_FAILURE"
