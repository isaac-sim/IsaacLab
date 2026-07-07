# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Oracle layer for the CI performance smoke test"""

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .gate_config import DEFAULT_K_BLOCK, DEFAULT_K_WARN, MIN_BASELINE_SAMPLES, MIN_BLOCK_REGRESSION_PCT
    from .gate_types import (
        BisectVerdict,
        FailurePhase,
        FpsMeanThreshold,
        OracleVerdict,
        ThresholdSource,
        verdict_severity,
        worst_verdict,
    )
except ImportError:  # pragma: no cover (for direct scripting execution/import)
    from gate_config import DEFAULT_K_BLOCK, DEFAULT_K_WARN, MIN_BASELINE_SAMPLES, MIN_BLOCK_REGRESSION_PCT
    from gate_types import (
        BisectVerdict,
        FailurePhase,
        FpsMeanThreshold,
        OracleVerdict,
        ThresholdSource,
        verdict_severity,
        worst_verdict,
    )


@dataclass
class Baseline:
    """Rolling-window statistics for one compatible benchmark history"""

    median_fps: float
    mad_fps: float
    k_warn: float = DEFAULT_K_WARN
    k_block: float = DEFAULT_K_BLOCK
    sample_count: int = 0
    source: str = "unknown"
    total_sample_count: int | None = None


@dataclass
class OracleResult:
    """Full verdict record produced by :func:`compare`"""

    verdict: OracleVerdict
    bisect_verdict: str
    failure_phase: str | None
    measured_fps: float | None
    baseline_fps: float | None
    regression_pct: float | None
    fps_median: float | None
    fps_p5: float | None
    fps_p95: float | None
    gpu_mem_used_mb: float | None
    startup_time_s: float | None
    wall_time_s: float | None
    was_retried: bool
    task_id: str
    backend: str
    baseline_sample_count: int = 0
    baseline_source: str = "none"
    threshold_source: str = ThresholdSource.NO_BASELINE.value
    warn_threshold_fps: float | None = None
    block_threshold_fps: float | None = None
    hard_floor_fps: float | None = None
    min_block_regression_pct: float = MIN_BLOCK_REGRESSION_PCT
    baseline_noise_pct: float | None = None
    effective_noise_pct: float | None = None
    noise_floor_pct: float | None = None
    note: str | None = None
    crossed_thresholds: list[dict] = field(default_factory=list)


_BISECT_BAD_PHASES: frozenset[str] = frozenset({FailurePhase.INIT.value, FailurePhase.RUNTIME.value})


def _bisect_verdict(verdict: OracleVerdict, was_retried: bool, failure_phase: str | None) -> str:
    """Compute the bisect-friendly label for a given verdict."""
    if verdict == OracleVerdict.PASS:
        return BisectVerdict.SKIP.value if was_retried else BisectVerdict.GOOD.value
    if verdict == OracleVerdict.WARN:
        return BisectVerdict.SKIP.value
    if verdict == OracleVerdict.BLOCK:
        return BisectVerdict.BAD.value
    if failure_phase in _BISECT_BAD_PHASES:
        return BisectVerdict.BAD.value
    return BisectVerdict.SKIP.value


def _percentile(sorted_data: list[float], p: float) -> float:
    """Linear-interpolation percentile on a pre-sorted non-empty list."""
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    idx = p / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)


def apply_excluded_frames(fps_series: list[float], excluded_frames: frozenset[int]) -> list[float]:
    """Return fps_series with frames at 0-based indices listed in excluded_frames removed"""
    if not excluded_frames:
        return list(fps_series)
    return [fps for idx, fps in enumerate(fps_series) if idx not in excluded_frames]


def _hard_failure(
    bench_result: dict,
    failure_phase: str | None,
    was_retried: bool,
    gpu_mem_used_mb: float | None,
    *,
    note: str | None = None,
) -> OracleResult:
    verdict = OracleVerdict.HARD_FAILURE
    return OracleResult(
        verdict=verdict,
        bisect_verdict=_bisect_verdict(verdict, was_retried, failure_phase),
        failure_phase=failure_phase,
        measured_fps=None,
        baseline_fps=None,
        regression_pct=None,
        fps_median=None,
        fps_p5=None,
        fps_p95=None,
        gpu_mem_used_mb=gpu_mem_used_mb,
        startup_time_s=bench_result.get("startup_time_s"),
        wall_time_s=bench_result.get("wall_time_s"),
        was_retried=was_retried,
        task_id=bench_result["task_id"],
        backend=bench_result.get("backend_key") or bench_result["backend"],
        threshold_source=ThresholdSource.NOT_APPLICABLE.value,
        note=note,
    )


def _extract_fps_series(perf_smoke_test_info: list[dict]) -> list[float]:
    for phase in perf_smoke_test_info:
        if phase.get("phase_name") == "runtime":
            for measurement in phase.get("measurements", []):
                if measurement.get("name", "").endswith("Step Frametimes"):
                    value = measurement.get("value", {})
                    return list(value.get("Environment step effective FPS", []))
            break
    return []


def _append_note(note: str | None, extra: str) -> str:
    """Append semicolon-separated oracle context without dropping existing notes."""
    return f"{note}; {extra}" if note else extra


def compare(
    bench_result: dict,
    baseline: "Baseline | None",
    fps_mean_thresholds: "list[FpsMeanThreshold]",
    excluded_frames: "frozenset[int]",
    artifact_dir: "Path",
    *,
    min_block_regression_pct: float = MIN_BLOCK_REGRESSION_PCT,
    noise_floor_pct: float = 0.0,
) -> OracleResult:
    """Compare a benchmark result against its baseline and return an OracleResult.

    The final verdict is the most severe of the rolling-window/baseline verdict and
    the verdict from any crossed gating threshold (see :class:`~gate_types.FpsMeanThreshold`).
    Reporting-only thresholds that cross are recorded in ``crossed_thresholds`` without
    changing the verdict.
    """
    task_id: str = bench_result["task_id"]
    backend: str = bench_result.get("backend_key") or bench_result["backend"]
    failure_phase: str | None = bench_result.get("failure_phase")
    was_retried: bool = bool(bench_result.get("was_retried"))
    startup_time_s: float | None = bench_result.get("startup_time_s")
    wall_time_s: float | None = bench_result.get("wall_time_s")
    gpu_mem_used_mb: float | None = (bench_result.get("gpu_diag") or {}).get("gpu_mem_used_mb")

    config_mismatch = bench_result.get("config_mismatch")
    if config_mismatch or failure_phase == FailurePhase.CONFIG_MISMATCH.value:
        return _hard_failure(
            bench_result,
            FailurePhase.CONFIG_MISMATCH.value,
            was_retried,
            gpu_mem_used_mb,
            note=str(config_mismatch or "config_mismatch"),
        )

    if not bench_result.get("perf_smoke_test_info_present", False):
        return _hard_failure(bench_result, failure_phase, was_retried, gpu_mem_used_mb)

    perf_info_path = Path(artifact_dir) / "perf_smoke_test_info.json"
    with perf_info_path.open() as fh:
        perf_info = json.load(fh)

    filtered = apply_excluded_frames(_extract_fps_series(perf_info), excluded_frames)
    if not filtered:
        return _hard_failure(bench_result, failure_phase, was_retried, gpu_mem_used_mb, note="empty_fps_series")

    mean_fps = statistics.mean(filtered)
    sorted_filtered = sorted(filtered)
    fps_median = _percentile(sorted_filtered, 50.0)
    fps_p5 = _percentile(sorted_filtered, 5.0)
    fps_p95 = _percentile(sorted_filtered, 95.0)

    # Confirm-on-BLOCK: when an initially-blocking cell is re-run, the gate FPS of
    # each attempt (initial + reruns) is recorded in ``confirmation_fps_attempts``.
    # Comparing the *median* of attempts means a single slow draw cannot finalize a
    # BLOCK. The list is populated by confirm.py only for cells that initially block.
    confirmation_attempts = [
        float(value)
        for value in bench_result.get("confirmation_fps_attempts") or []
        if isinstance(value, (int, float)) and value > 0.0
    ]
    confirmation_requested = bool(bench_result.get("confirmation_fps_attempts"))
    if len(confirmation_attempts) >= 2:
        mean_fps = statistics.median(confirmation_attempts)

    baseline_fps = baseline.median_fps if baseline is not None else None
    baseline_sample_count = baseline.sample_count if baseline is not None else 0
    baseline_source = baseline.source if baseline is not None else "none"
    baseline_noise_pct = None
    effective_noise_fps = None
    effective_noise_pct = None
    applied_noise_floor_pct = None
    if baseline is not None and baseline.median_fps > 0.0:
        baseline_noise_pct = baseline.mad_fps / baseline.median_fps * 100.0
        noise_floor_fps = max(0.0, float(noise_floor_pct)) / 100.0 * baseline.median_fps
        effective_noise_fps = max(baseline.mad_fps, noise_floor_fps)
        effective_noise_pct = effective_noise_fps / baseline.median_fps * 100.0
        if noise_floor_fps > baseline.mad_fps:
            applied_noise_floor_pct = float(noise_floor_pct)
    regression_pct = None
    if baseline_fps:
        regression_pct = ((mean_fps - baseline_fps) / baseline_fps) * 100.0

    # --- Configured FPS thresholds (hard floors and named reference points) ---
    crossed = [t for t in fps_mean_thresholds if t.crosses(mean_fps)]
    crossed_thresholds = [
        {
            "threshold_name": t.name,
            "threshold": t.value,
            "threshold_verdict": t.verdict.value if t.verdict is not None else None,
            "gating": t.is_gating,
        }
        for t in crossed
    ]
    threshold_verdict = OracleVerdict.PASS
    for t in crossed:
        if t.is_gating:
            threshold_verdict = worst_verdict(threshold_verdict, t.verdict)
    block_values = [t.value for t in fps_mean_thresholds if t.verdict == OracleVerdict.BLOCK]
    hard_floor_fps = max(block_values) if block_values else None

    # --- Rolling-window / baseline verdict ---
    warn_threshold = None
    block_threshold = None
    baseline_threshold_source = ThresholdSource.NO_BASELINE.value
    notes: list[str] = []

    if baseline is None:
        baseline_verdict = OracleVerdict.WARN
        notes.append("no_baseline")
    elif baseline.sample_count < MIN_BASELINE_SAMPLES:
        baseline_verdict = OracleVerdict.WARN
        baseline_threshold_source = ThresholdSource.INSUFFICIENT_WINDOW.value
        notes.append(f"insufficient_baseline(n={baseline.sample_count},min={MIN_BASELINE_SAMPLES})")
    else:
        baseline_threshold_source = ThresholdSource.ROLLING_WINDOW.value
        threshold_noise = effective_noise_fps if effective_noise_fps is not None else baseline.mad_fps
        block_threshold = baseline.median_fps - baseline.k_block * threshold_noise
        warn_threshold = baseline.median_fps - baseline.k_warn * threshold_noise
        if applied_noise_floor_pct is not None:
            notes.append(f"noise_floor={applied_noise_floor_pct:.2f}%")
        if mean_fps < block_threshold:
            if regression_pct is None or regression_pct <= -float(min_block_regression_pct):
                baseline_verdict = OracleVerdict.BLOCK
            else:
                baseline_verdict = OracleVerdict.WARN
                notes.append("below_mad_block_but_inside_regression_floor")
        elif mean_fps < warn_threshold:
            baseline_verdict = OracleVerdict.WARN
        else:
            baseline_verdict = OracleVerdict.PASS

    # --- Combine: the most severe of the threshold and baseline verdicts wins ---
    verdict = worst_verdict(threshold_verdict, baseline_verdict)
    if threshold_verdict != OracleVerdict.PASS and verdict_severity(threshold_verdict) >= verdict_severity(
        baseline_verdict
    ):
        threshold_source = ThresholdSource.THRESHOLD.value
    else:
        threshold_source = baseline_threshold_source

    if verdict == OracleVerdict.PASS and was_retried:
        verdict = OracleVerdict.WARN
        notes.append("was_retried")

    note = "; ".join(notes) if notes else None

    # Apply the confirm-on-BLOCK policy after the verdict is computed from the
    # median attempt. A single slow draw can no longer finalize a block: an
    # initial block that does not reproduce is downgraded to WARN, while a failed
    # rerun stays blocking (fail-safe) but is surfaced explicitly so an
    # operational rerun failure can never silently look like "nothing happened".
    if confirmation_requested:
        attempt_count = len(confirmation_attempts)
        if attempt_count >= 2:
            if verdict == OracleVerdict.BLOCK:
                note = _append_note(note, f"block_confirmed(n={attempt_count})")
            else:
                verdict = OracleVerdict.WARN
                note = _append_note(note, f"block_not_reproduced(n={attempt_count})")
        else:
            note = _append_note(note, "block_unconfirmed(reruns_failed)")

    return OracleResult(
        verdict=verdict,
        bisect_verdict=_bisect_verdict(verdict, was_retried, failure_phase),
        failure_phase=failure_phase,
        measured_fps=mean_fps,
        baseline_fps=baseline_fps,
        regression_pct=regression_pct,
        fps_median=fps_median,
        fps_p5=fps_p5,
        fps_p95=fps_p95,
        gpu_mem_used_mb=gpu_mem_used_mb,
        startup_time_s=startup_time_s,
        wall_time_s=wall_time_s,
        was_retried=was_retried,
        task_id=task_id,
        backend=backend,
        baseline_sample_count=baseline_sample_count,
        baseline_source=baseline_source,
        threshold_source=threshold_source,
        warn_threshold_fps=warn_threshold,
        block_threshold_fps=block_threshold,
        hard_floor_fps=hard_floor_fps,
        min_block_regression_pct=float(min_block_regression_pct),
        baseline_noise_pct=baseline_noise_pct,
        effective_noise_pct=effective_noise_pct,
        noise_floor_pct=applied_noise_floor_pct,
        note=note,
        crossed_thresholds=crossed_thresholds,
    )
