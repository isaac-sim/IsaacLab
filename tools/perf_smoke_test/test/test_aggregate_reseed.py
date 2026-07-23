# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free checks for aggregate's post-gate under-filled-bucket detector.

The detector flags a task for targeted reseeding when this run produced a valid
measurement but the matching baseline window for its exact runtime_contract_hash
has fewer than MIN_BASELINE_SAMPLES samples. These tests drive aggregate.main()
in flat-file mode (no git) and assert the reseed_tasks GITHUB_OUTPUT signal that
the gate's reseed job consumes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

import aggregate  # noqa: E402
from baseline_manager import update_baseline  # noqa: E402
from contracts import BenchResult  # noqa: E402
from gate_config import MIN_BASELINE_SAMPLES  # noqa: E402
from gate_types import OracleVerdict  # noqa: E402

_RUNTIME_HASH = "runtime-a"


def _bench_result(*, fps: float | None = 100.0, info_present: bool = True) -> BenchResult:
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
        was_retried=False,
        perf_smoke_test_info_present=info_present,
        raw_fps_mean=fps,
        raw_fps_std=1.0 if fps is not None else None,
        raw_fps_min=(fps - 1.0) if fps is not None else None,
        raw_fps_max=(fps + 1.0) if fps is not None else None,
        runtime_contract_hash=_RUNTIME_HASH,
        runtime_resources={"gpu_name": "NVIDIA L40S"},
        provenance={"software": {"isaaclab": "3.0.0", "warp": "1.13.0"}},
        launch_config=launch_config,
        launch_config_hash="launch-a",
        benchmark_contract_hash="bench-a",
        baseline_epoch=1,
    )


def _write_artifact(artifacts_dir: Path, bench_result: BenchResult) -> None:
    task_dir = artifacts_dir / "bench-Isaac-Cartpole-Direct-physx"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "perf_smoke_test_result.json").write_text(json.dumps(bench_result.to_dict()))


def _seed_flat_baseline(baselines_dir: Path, count: int) -> None:
    for i in range(count):
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
                "launch_config_hash": "launch-a",
                "benchmark_contract_hash": "bench-a",
                "runtime_contract_hash": _RUNTIME_HASH,
                "baseline_epoch": 1,
                "fps": 100.0,
                "sample_id": f"seed-{i}",
            },
        )


def _run_aggregate(tmp_path: Path, monkeypatch, bench_result: BenchResult, baseline_count: int) -> dict[str, str]:
    artifacts_dir = tmp_path / "artifacts"
    baselines_dir = tmp_path / "baselines"
    output_file = tmp_path / "gh_output.txt"
    _write_artifact(artifacts_dir, bench_result)
    if baseline_count:
        _seed_flat_baseline(baselines_dir, baseline_count)
    else:
        baselines_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aggregate.py",
            "--artifacts_dir",
            str(artifacts_dir),
            "--gpu_model",
            "L40S",
            "--baselines_dir",
            str(baselines_dir),
            "--allow_baseline_update",
            "false",
        ],
    )
    aggregate.main()

    outputs: dict[str, str] = {}
    if output_file.exists():
        for line in output_file.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                outputs[key] = value
    return outputs


def test_reseed_flagged_when_no_baseline(tmp_path, monkeypatch) -> None:
    """A valid measurement with zero matching samples opens a bucket that needs reseeding."""
    outputs = _run_aggregate(tmp_path, monkeypatch, _bench_result(fps=100.0), baseline_count=0)

    assert outputs.get("reseed_tasks") == "Isaac-Cartpole-Direct"
    assert outputs.get("reseed_min_samples") == str(MIN_BASELINE_SAMPLES)


def test_reseed_flagged_when_window_insufficient(tmp_path, monkeypatch) -> None:
    """Fewer than MIN_BASELINE_SAMPLES matching samples still counts as under-filled."""
    outputs = _run_aggregate(tmp_path, monkeypatch, _bench_result(fps=100.0), baseline_count=MIN_BASELINE_SAMPLES - 1)

    assert outputs.get("reseed_tasks") == "Isaac-Cartpole-Direct"


def test_reseed_not_flagged_when_bucket_full(tmp_path, monkeypatch) -> None:
    """A fully populated bucket must not trigger a reseed."""
    outputs = _run_aggregate(tmp_path, monkeypatch, _bench_result(fps=100.0), baseline_count=MIN_BASELINE_SAMPLES)

    assert "reseed_tasks" not in outputs


def test_reseed_not_flagged_without_valid_measurement(tmp_path, monkeypatch) -> None:
    """A crashed run (no measurement) must not reseed even with an empty bucket."""
    outputs = _run_aggregate(tmp_path, monkeypatch, _bench_result(fps=None, info_present=False), baseline_count=0)

    assert "reseed_tasks" not in outputs


def test_sticky_summary_explains_results_in_reviewer_language() -> None:
    """The sticky comment leads with actionable guidance and keeps diagnostics available."""
    result = SimpleNamespace(
        task_id="Isaac-Cartpole-Direct",
        backend="physx",
        verdict=OracleVerdict.BLOCK,
        measured_fps=90.0,
        baseline_fps=100.0,
        regression_pct=-10.0,
        baseline_sample_count=5,
        threshold_source="rolling_window",
        hard_floor_fps=None,
        failure_phase=None,
        was_retried=True,
        note="block_confirmed(n=1)",
        crossed_thresholds=[],
    )

    summary = aggregate._build_summary_markdown([(result, _bench_result(fps=90.0))], blocking=False)

    assert "### Overall result" in summary
    assert "blocking-level performance regressions were detected" in summary
    assert "Advisory" in summary
    assert "### How to read this" in summary
    assert "Start with **BLOCK** and **HARD FAILURE**" in summary
    assert "| Isaac-Cartpole-Direct | physx | 🚫 BLOCK | 90.0 | 100.0 | -10.00% | 5 |" in summary
    assert "Blocking-level slowdown detected; result was retried" in summary
    assert "passed only after retry" not in summary
    assert "<summary>Technical details</summary>" in summary
    assert "block_confirmed(n=1)" in summary


def test_sticky_summary_identifies_baseline_warmup() -> None:
    """WARN rows clearly distinguish baseline collection from a regression."""
    result = SimpleNamespace(
        task_id="Isaac-Cartpole-Direct",
        backend="newton",
        verdict=OracleVerdict.WARN,
        measured_fps=100.0,
        baseline_fps=None,
        regression_pct=None,
        baseline_sample_count=2,
        threshold_source="insufficient_window",
        hard_floor_fps=None,
        failure_phase=None,
        was_retried=False,
        note="insufficient baseline samples",
        crossed_thresholds=[],
    )

    summary = aggregate._build_summary_markdown([(result, _bench_result())], blocking=True)

    assert "No confirmed regression" in summary
    assert "Baseline warming up (2/5 samples)" in summary
    assert "**Blocking:** BLOCK and HARD FAILURE results fail the check." in summary
