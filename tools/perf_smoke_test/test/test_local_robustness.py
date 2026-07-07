# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU-free robustness checks for the performance smoke oracle contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_GATE_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _GATE_DIR.parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from baseline_manager import load_baseline, match_context_from_bench_result, update_baseline  # noqa: E402
from gate_types import OracleVerdict, ThresholdSource  # noqa: E402
from oracle import compare  # noqa: E402

_TASK_ID = "Isaac-Cartpole-Direct"
_BACKEND = "physx"
_GPU_MODEL = "L40S"
_LAUNCH_CONFIG_HASH = "launch-contract"
_BENCHMARK_CONTRACT_HASH = "benchmark-contract"
_RUNTIME_CONTRACT_HASH = "runtime-era-a"


def _bench_result(*, runtime_hash: str = _RUNTIME_CONTRACT_HASH, info_present: bool = True) -> dict:
    """Return a minimal benchmark result matching the aggregate/oracle contract."""
    return {
        "task_id": _TASK_ID,
        "backend": _BACKEND,
        "backend_key": _BACKEND,
        "physics_backend": "physx",
        "render_backend": None,
        "attempt": 1,
        "was_retried": False,
        "failure_phase": None,
        "startup_time_s": 1.0,
        "wall_time_s": 2.0,
        "perf_smoke_test_info_present": info_present,
        "launch_config_hash": _LAUNCH_CONFIG_HASH,
        "benchmark_contract_hash": _BENCHMARK_CONTRACT_HASH,
        "runtime_contract_hash": runtime_hash,
        "baseline_epoch": 1,
        "launch_config": {
            "gpu_model": _GPU_MODEL,
            "task_id": _TASK_ID,
            "backend_key": _BACKEND,
            "launch_config_hash": _LAUNCH_CONFIG_HASH,
            "benchmark_contract_hash": _BENCHMARK_CONTRACT_HASH,
            "baseline_epoch": 1,
            "excluded_frames_raw": [],
        },
    }


def _write_perf_info(artifact_dir: Path, fps: float) -> None:
    """Write a minimal perf_smoke_test_info.json file consumed by the oracle."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "phase_name": "runtime",
            "measurements": [
                {
                    "name": f"{_TASK_ID} Step Frametimes",
                    "value": {"Environment step effective FPS": [fps] * 8},
                }
            ],
            "metadata": [],
        }
    ]
    (artifact_dir / "perf_smoke_test_info.json").write_text(json.dumps(payload), encoding="utf-8")


def _seed_baseline(
    baselines_dir: Path,
    *,
    runtime_hash: str = _RUNTIME_CONTRACT_HASH,
    commit_sha: str | None = None,
    sample_count: int = 5,
    fps: float = 1000.0,
) -> None:
    """Seed matching flat-file baseline samples for one task/backend."""
    for idx in range(sample_count):
        update_baseline(
            baselines_dir,
            _GPU_MODEL,
            _TASK_ID,
            _BACKEND,
            fps,
            sample_metadata={
                "schema_version": 1,
                "fps": fps,
                "timestamp": f"2026-01-01T00:00:{idx:02d}+00:00",
                "gpu_model": _GPU_MODEL,
                "task_id": _TASK_ID,
                "backend_key": _BACKEND,
                "commit_sha": commit_sha,
                "launch_config_hash": _LAUNCH_CONFIG_HASH,
                "benchmark_contract_hash": _BENCHMARK_CONTRACT_HASH,
                "runtime_contract_hash": runtime_hash,
                "baseline_epoch": 1,
                "sample_id": f"sample-{runtime_hash}-{commit_sha or 'none'}-{idx}",
            },
        )


def _evaluate(tmp_path: Path, baselines_dir: Path, *, fps: float, bench_result: dict):
    """Evaluate one synthetic artifact through baseline matching and the real oracle."""
    artifact_dir = tmp_path / "artifacts" / _TASK_ID / _BACKEND
    _write_perf_info(artifact_dir, fps)
    match_context = match_context_from_bench_result(bench_result, gpu_model=_GPU_MODEL)
    baseline = load_baseline(baselines_dir, _GPU_MODEL, _TASK_ID, _BACKEND, match_context=match_context)
    return compare(
        bench_result=bench_result,
        baseline=baseline,
        fps_mean_thresholds=[],
        excluded_frames=frozenset(),
        artifact_dir=artifact_dir,
    )


def test_noise_floor_widens_rolling_threshold_for_jittery_cells(tmp_path: Path) -> None:
    """A calibrated noise floor prevents a tiny MAD window from over-tightening BLOCK."""
    baselines_dir = tmp_path / "baselines"
    _seed_baseline(baselines_dir)
    artifact_dir = tmp_path / "artifacts" / _TASK_ID / _BACKEND
    _write_perf_info(artifact_dir, 960.0)
    bench_result = _bench_result()
    match_context = match_context_from_bench_result(bench_result, gpu_model=_GPU_MODEL)
    baseline = load_baseline(baselines_dir, _GPU_MODEL, _TASK_ID, _BACKEND, match_context=match_context)

    without_floor = compare(
        bench_result=bench_result,
        baseline=baseline,
        fps_mean_thresholds=[],
        excluded_frames=frozenset(),
        artifact_dir=artifact_dir,
    )
    with_floor = compare(
        bench_result=bench_result,
        baseline=baseline,
        fps_mean_thresholds=[],
        excluded_frames=frozenset(),
        artifact_dir=artifact_dir,
        noise_floor_pct=2.52,
    )

    assert without_floor.verdict == OracleVerdict.BLOCK
    assert without_floor.baseline_noise_pct == 0.0
    assert with_floor.verdict == OracleVerdict.PASS
    assert with_floor.effective_noise_pct == pytest.approx(2.52)
    assert with_floor.noise_floor_pct == pytest.approx(2.52)
    assert with_floor.note == "noise_floor=2.52%"


def _evaluate_with_attempts(tmp_path: Path, baselines_dir: Path, *, initial_fps: float, attempts: list[float]):
    """Evaluate one synthetic cell whose initial draw is annotated with confirmation attempts."""
    bench_result = _bench_result()
    bench_result["confirmation_fps_attempts"] = attempts
    return _evaluate(tmp_path, baselines_dir, fps=initial_fps, bench_result=bench_result)


def test_confirm_block_downgrades_when_median_does_not_reproduce(tmp_path: Path) -> None:
    """An initial BLOCK whose attempt median does not block is downgraded to WARN."""
    baselines_dir = tmp_path / "baselines"
    _seed_baseline(baselines_dir)  # median 1000, mad 0 -> BLOCK threshold at 1000

    result = _evaluate_with_attempts(tmp_path, baselines_dir, initial_fps=950.0, attempts=[950.0, 1000.0, 1000.0])

    assert result.verdict == OracleVerdict.WARN
    assert "block_not_reproduced(n=3)" in (result.note or "")
    assert result.measured_fps == pytest.approx(1000.0)  # median of attempts drives the verdict


def test_confirm_block_keeps_block_when_median_still_blocks(tmp_path: Path) -> None:
    """An initial BLOCK whose attempt median still blocks stays BLOCK (confirmed)."""
    baselines_dir = tmp_path / "baselines"
    _seed_baseline(baselines_dir)

    result = _evaluate_with_attempts(tmp_path, baselines_dir, initial_fps=950.0, attempts=[950.0, 940.0, 945.0])

    assert result.verdict == OracleVerdict.BLOCK
    assert "block_confirmed(n=3)" in (result.note or "")
    assert result.measured_fps == pytest.approx(945.0)


def test_confirm_block_failed_reruns_stay_blocking_and_visible(tmp_path: Path) -> None:
    """When reruns produce no usable FPS, the cell stays BLOCK and says so explicitly."""
    baselines_dir = tmp_path / "baselines"
    _seed_baseline(baselines_dir)

    # Only the initial attempt is recorded (both reruns failed operationally).
    result = _evaluate_with_attempts(tmp_path, baselines_dir, initial_fps=950.0, attempts=[950.0])

    assert result.verdict == OracleVerdict.BLOCK
    assert "block_unconfirmed(reruns_failed)" in (result.note or "")


def test_confirm_block_cell_records_initial_and_rerun_attempts(tmp_path: Path) -> None:
    """confirm_block_cell seeds the initial gate FPS, appends reruns, and persists them."""
    sys.path.insert(0, str(_GATE_DIR))
    from confirm import confirm_block_cell

    artifact_dir = tmp_path / "artifacts" / _TASK_ID / _BACKEND
    _write_perf_info(artifact_dir, 950.0)  # initial gate mean = 950
    result_path = artifact_dir / "perf_smoke_test_result.json"
    bench_result = _bench_result()
    result_path.write_text(json.dumps(bench_result), encoding="utf-8")

    calls: list[int] = []

    def fake_rerun(_bench_result: dict, _attempt_dir: Path, attempt: int) -> float:
        calls.append(attempt)
        return 1000.0

    attempts = confirm_block_cell(bench_result, artifact_dir, result_path, frozenset(), fake_rerun, 2)

    assert attempts == [pytest.approx(950.0), 1000.0, 1000.0]
    assert calls == [2, 3]
    saved = json.loads(result_path.read_text())
    assert saved["confirmation_fps_attempts"] == [pytest.approx(950.0), 1000.0, 1000.0]
    assert saved["confirmation_policy"]["completed_attempts"] == 3


def test_confirm_block_cell_skips_failed_reruns(tmp_path: Path) -> None:
    """A rerun that returns None (no usable FPS) is skipped, leaving only the initial."""
    sys.path.insert(0, str(_GATE_DIR))
    from confirm import confirm_block_cell

    artifact_dir = tmp_path / "artifacts" / _TASK_ID / _BACKEND
    _write_perf_info(artifact_dir, 950.0)
    result_path = artifact_dir / "perf_smoke_test_result.json"
    bench_result = _bench_result()
    result_path.write_text(json.dumps(bench_result), encoding="utf-8")

    attempts = confirm_block_cell(bench_result, artifact_dir, result_path, frozenset(), lambda *_args, **_kw: None, 2)

    assert attempts == [pytest.approx(950.0)]


@pytest.mark.parametrize(
    ("fps", "expected_verdict"),
    [
        (1000.0, OracleVerdict.PASS),
        (980.0, OracleVerdict.WARN),
        (950.0, OracleVerdict.BLOCK),
    ],
)
def test_oracle_classifies_pass_warn_block_boundaries(
    tmp_path: Path, fps: float, expected_verdict: OracleVerdict
) -> None:
    """Regression percentage floor prevents tiny drops from becoming BLOCK."""
    baselines_dir = tmp_path / "baselines"
    _seed_baseline(baselines_dir)

    result = _evaluate(tmp_path, baselines_dir, fps=fps, bench_result=_bench_result())

    assert result.verdict == expected_verdict
    assert result.baseline_sample_count == 5
    assert result.threshold_source == ThresholdSource.ROLLING_WINDOW.value


def test_runtime_contract_mismatch_ignores_old_era_baseline(tmp_path: Path) -> None:
    """A different runtime_contract_hash must not be treated as compatible."""
    baselines_dir = tmp_path / "baselines"
    _seed_baseline(baselines_dir, runtime_hash="runtime-era-old")

    result = _evaluate(tmp_path, baselines_dir, fps=500.0, bench_result=_bench_result(runtime_hash="runtime-era-new"))

    assert result.verdict == OracleVerdict.WARN
    assert result.note == "no_baseline"
    assert result.baseline_sample_count == 0


def test_base_sha_ancestry_filter_excludes_non_ancestor_baselines(tmp_path: Path) -> None:
    """Samples not reachable from the PR base SHA must not participate in comparison."""
    baselines_dir = tmp_path / "baselines"
    non_ancestor_sha = "0" * 40
    base_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()
    _seed_baseline(baselines_dir, commit_sha=non_ancestor_sha)
    bench_result = _bench_result()
    match_context = match_context_from_bench_result(bench_result, gpu_model=_GPU_MODEL, base_sha=base_sha)

    baseline = load_baseline(baselines_dir, _GPU_MODEL, _TASK_ID, _BACKEND, match_context=match_context)

    assert baseline is None


def test_missing_perf_info_is_hard_failure(tmp_path: Path) -> None:
    """A benchmark without the canonical perf info file is a hard failure, not a perf datapoint."""
    result = compare(
        bench_result=_bench_result(info_present=False),
        baseline=None,
        fps_mean_thresholds=[],
        excluded_frames=frozenset(),
        artifact_dir=tmp_path,
    )

    assert result.verdict == OracleVerdict.HARD_FAILURE
    assert result.threshold_source == ThresholdSource.NOT_APPLICABLE.value


def test_aggregate_applies_confirmation_attempts_end_to_end(tmp_path: Path) -> None:
    """aggregate.py must finalize a pre-annotated BLOCK cell on the median of attempts."""
    baselines_dir = tmp_path / "baselines"
    _seed_baseline(baselines_dir)  # median 1000, mad 0

    artifact_dir = tmp_path / "artifacts" / "bench-cell" / _TASK_ID / _BACKEND
    _write_perf_info(artifact_dir, 950.0)  # initial draw would BLOCK
    bench_result = _bench_result()
    bench_result["confirmation_fps_attempts"] = [950.0, 1000.0, 1000.0]  # median does not block
    (artifact_dir / "perf_smoke_test_result.json").write_text(json.dumps(bench_result), encoding="utf-8")

    summary_file = tmp_path / "summary.md"
    result = subprocess.run(
        [
            sys.executable,
            str(_GATE_DIR / "aggregate.py"),
            "--artifacts_dir",
            str(tmp_path / "artifacts"),
            "--baselines_dir",
            str(baselines_dir),
            "--gpu_model",
            _GPU_MODEL,
            "--confirm_rerun_mode",
            "none",
            "--summary_file",
            str(summary_file),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    summary = summary_file.read_text()
    assert "block_not_reproduced(n=3)" in summary
    assert "| WARN |" in summary


def test_aggregate_empty_artifacts_fails_clearly(tmp_path: Path) -> None:
    """aggregate.py should fail clearly when no perf_smoke_test_result.json files exist."""
    result = subprocess.run(
        [
            sys.executable,
            str(_GATE_DIR / "aggregate.py"),
            "--artifacts_dir",
            str(tmp_path / "missing-artifacts"),
            "--baselines_dir",
            str(tmp_path / "baselines"),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "No perf_smoke_test_result.json files found" in result.stdout
