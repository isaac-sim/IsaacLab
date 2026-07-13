# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free checks for the typed perf-smoke result and oracle path."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from baseline_manager import load_baseline, match_context_from_bench_result, update_baseline  # noqa: E402
from contracts import CONTRACT_SCHEMA_VERSION, BenchResult  # noqa: E402
from gate_types import FpsMeanThreshold, OracleVerdict, ThresholdSource  # noqa: E402
from hashing import stable_hash  # noqa: E402
from oracle import Baseline, compare  # noqa: E402


def _bench_result(*, fps: float | None = 100.0, info_present: bool = True, was_retried: bool = False) -> BenchResult:
    """Build the minimal typed result shape used by oracle and baseline tests."""
    launch_config = {
        "task_id": "Isaac-Cartpole-Direct",
        "backend": "physx",
        "backend_key": "physx",
        "physics_backend": "physx",
        "render_backend": None,
        "gpu_model": "l40s",
        "launch_config_hash": "launch-a",
        "benchmark_contract_hash": "bench-a",
        "baseline_epoch": 1,
    }
    return BenchResult(
        task_id="Isaac-Cartpole-Direct",
        backend="physx",
        physics_backend="physx",
        render_backend=None,
        backend_key="physx",
        preset="default",
        was_retried=was_retried,
        perf_smoke_test_info_present=info_present,
        raw_fps_mean=fps,
        raw_fps_std=1.0 if fps is not None else None,
        raw_fps_min=(fps - 1.0) if fps is not None else None,
        raw_fps_max=(fps + 1.0) if fps is not None else None,
        runtime_contract_hash="runtime-a",
        runtime_resources={"gpu_name": "NVIDIA L40S", "gpu_mem_used_mb": 1024.0},
        provenance={"software": {"isaaclab": "3.0.0", "warp": "1.13.0"}},
        launch_config=launch_config,
        launch_config_hash="launch-a",
        benchmark_contract_hash="bench-a",
        baseline_epoch=1,
    )


def test_bench_result_round_trips_with_schema_version() -> None:
    """Typed results serialize to the stable wire shape and reconstruct strictly."""
    result = _bench_result(fps=123.0)
    payload = result.to_dict()
    payload["ignored_future_field"] = "allowed"

    restored = BenchResult.from_dict(payload)

    assert restored.schema_version == CONTRACT_SCHEMA_VERSION
    assert restored.raw_fps_mean == 123.0
    assert not hasattr(restored, "ignored_future_field")


def test_bench_result_requires_identity_fields() -> None:
    """The typed schema catches producer bugs that omit required identity fields."""
    payload = _bench_result().to_dict()
    payload.pop("task_id")

    with pytest.raises(TypeError):
        BenchResult.from_dict(payload)


def test_stable_hash_is_key_order_independent() -> None:
    """Contract hashing must not depend on dict insertion order."""
    left = {"runtime": {"warp": "1.13.0", "torch": "2.9.0"}, "backend": "physx"}
    right = {"backend": "physx", "runtime": {"torch": "2.9.0", "warp": "1.13.0"}}

    assert stable_hash(left) == stable_hash(right)


def test_oracle_pass_warn_and_block_from_typed_result() -> None:
    """The post-migration oracle still scores typed BenchResult objects correctly."""
    baseline = Baseline(median_fps=100.0, mad_fps=2.0, sample_count=5)

    pass_result = compare(_bench_result(fps=100.0), baseline, [])
    warn_result = compare(_bench_result(fps=94.0), baseline, [])
    block_result = compare(_bench_result(fps=88.0), baseline, [])

    assert pass_result.verdict == OracleVerdict.PASS
    assert warn_result.verdict == OracleVerdict.WARN
    assert warn_result.threshold_source == ThresholdSource.ROLLING_WINDOW.value
    assert block_result.verdict == OracleVerdict.BLOCK
    assert block_result.regression_pct == pytest.approx(-12.0)


def test_oracle_threshold_floor_can_block_without_baseline() -> None:
    """Configured hard floors remain gating even before rolling baselines exist."""
    threshold = FpsMeanThreshold(name="hard-floor", value=50.0, verdict=OracleVerdict.BLOCK)

    result = compare(_bench_result(fps=40.0), baseline=None, fps_mean_thresholds=[threshold])

    assert result.verdict == OracleVerdict.BLOCK
    assert result.threshold_source == ThresholdSource.THRESHOLD.value
    assert result.hard_floor_fps == 50.0


def test_oracle_hard_fails_missing_benchmark_info() -> None:
    """A missing runtime bundle stays a hard failure instead of looking like no baseline."""
    result = compare(_bench_result(fps=None, info_present=False), baseline=None, fps_mean_thresholds=[])

    assert result.verdict == OracleVerdict.HARD_FAILURE
    assert result.measured_fps is None


def test_baseline_selection_filters_incompatible_contract_hashes(tmp_path: Path) -> None:
    """Seeder/gate matching only selects samples with compatible launch and runtime contracts."""
    baselines_dir = tmp_path / "baselines"
    compatible = _bench_result(fps=100.0).to_dict()
    incompatible = _bench_result(fps=50.0).to_dict()
    incompatible["runtime_contract_hash"] = "runtime-old"

    update_baseline(
        baselines_dir,
        "l40s",
        "Isaac-Cartpole-Direct",
        "physx",
        100.0,
        sample_metadata={
            "gpu_model": "l40s",
            "task_id": "Isaac-Cartpole-Direct",
            "backend_key": "physx",
            "launch_config_hash": compatible["launch_config_hash"],
            "benchmark_contract_hash": compatible["benchmark_contract_hash"],
            "runtime_contract_hash": compatible["runtime_contract_hash"],
            "baseline_epoch": compatible["baseline_epoch"],
            "fps": 100.0,
            "sample_id": "compatible",
        },
    )
    update_baseline(
        baselines_dir,
        "l40s",
        "Isaac-Cartpole-Direct",
        "physx",
        50.0,
        sample_metadata={
            "gpu_model": "l40s",
            "task_id": "Isaac-Cartpole-Direct",
            "backend_key": "physx",
            "launch_config_hash": incompatible["launch_config_hash"],
            "benchmark_contract_hash": incompatible["benchmark_contract_hash"],
            "runtime_contract_hash": incompatible["runtime_contract_hash"],
            "baseline_epoch": incompatible["baseline_epoch"],
            "fps": 50.0,
            "sample_id": "incompatible-runtime",
        },
    )

    context = match_context_from_bench_result(compatible, gpu_model="l40s")
    baseline = load_baseline(baselines_dir, "l40s", "Isaac-Cartpole-Direct", "physx", match_context=context)

    assert baseline is not None
    assert baseline.median_fps == 100.0
    assert baseline.sample_count == 1
    assert baseline.total_sample_count == 2
