# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Oracle layer for the CI performance regression gate.

Provides verdict computation (PASS / WARN / BLOCK / HARD_FAILURE) by comparing
a measured FPS sample against a rolling baseline.

Threshold model (hardened)
--------------------------
The gate draws WARN/BLOCK bands a number of robust spreads below the baseline
center (median). The spread is the median absolute deviation scaled to a
std-equivalent, **floored** at a percentage of the center::

    spread = max(MAD_TO_STD * mad, min_spread_pct/100 * center)
    WARN   when measured < center - k_warn  * spread
    BLOCK  when measured < center - k_block * spread

The spread floor is the anti-flap guardrail: without it, a freakishly stable
task (or a tiny window) drives ``mad -> 0``, collapsing the band onto the median
so that trivial measurement noise trips a BLOCK. The window must hold at least
:data:`MIN_WINDOW` samples before its median+MAD is trusted; below that the run
is a seed run and PASSes (there is no calibrated reference to regress against).

Per-task overrides (from ``baseline_overrides.json``, committed with the PR) can
adjust ``k_warn`` / ``k_block`` / ``min_spread_pct``, pin the center/spread for an
intended perf change (``pin_center_fps`` / ``pin_spread_fps``), ``skip`` a task,
or opt into an advisory tail check (``tail_p99_warn``).
"""

import json
import statistics
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Robust-threshold defaults (overridable per task/gpu via baseline_overrides.json).
DEFAULT_K_WARN = 3.0
DEFAULT_K_BLOCK = 6.0
DEFAULT_MIN_SPREAD_PCT = 1.5  # spread floor as % of center (guards tiny/stable windows)
MAD_TO_STD = 1.4826  # MAD -> std-equivalent for ~normal data
MIN_WINDOW = 5  # samples needed before the rolling median+MAD is trusted


class OracleVerdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"
    HARD_FAILURE = "HARD_FAILURE"


@dataclass
class Baseline:
    """Rolling-window statistics used as the comparison reference for a single (task, backend) pair.

    Args:
        median_fps: Median FPS computed from the baseline window
        mad_fps: Median absolute deviation of FPS in the baseline window
        k_warn: Number of spreads below the median that triggers a WARN verdict
        k_block: Number of spreads below the median that triggers a BLOCK verdict
        sample_count: Number of samples in the window used to compute the stats
    """

    median_fps: float
    mad_fps: float
    k_warn: float = DEFAULT_K_WARN
    k_block: float = DEFAULT_K_BLOCK
    sample_count: int = 0


@dataclass
class OracleResult:
    """Full verdict record produced by :func:`compare`"""

    verdict: OracleVerdict  # High-level verdict: PASS / WARN / BLOCK / HARD_FAILURE
    bisect_verdict: str  # GOOD / BAD / SKIP for bisect compatibility
    # Phase from build_bench_result: import/init/runtime/oom/hang/driver/config_mismatch, or None
    failure_phase: str | None
    measured_fps: float | None  # Mean FPS after excluded-frame filtering, or None on hard failure; blocking metric
    baseline_fps: float | None  # baseline.median_fps (or pinned center), or None
    regression_pct: float | None  # ((measured_fps - baseline_fps) / baseline_fps) * 100, or None
    fps_median: float | None  # Median FPS of the filtered series [informational]
    fps_p5: float | None  # 5th-percentile FPS of the filtered series [informational]
    fps_p95: float | None  # 95th-percentile FPS of the filtered series [informational]
    gpu_mem_used_mb: float | None  # GPU memory used at benchmark time [MiB], or None
    startup_time_s: float | None  # Startup time reported by the benchmark process [s]
    wall_time_s: float | None  # Wall-clock time of the benchmark run [s]
    was_retried: bool  # True when the benchmark succeeded only after a retry, False otherwise
    task_id: str  # Benchmark task identifier
    backend: str  # Physics/Render backend name
    spread_fps: float | None = None  # Robust spread used for the bands [FPS], or None on a seed run
    threshold_source: str = "seed"  # How the bands were derived: seed / window(n=..) / override_pin / override_skip
    note: str | None = None  # Optional annotation (e.g. "skipped_by_override", "tail")


# ---------------------------------------------------------------------------
# Bisect verdict mapping (spec section "Bisect verdict mapping")
# ---------------------------------------------------------------------------

# failure_phase values that map HARD_FAILURE -> "BAD"
_BISECT_BAD_PHASES: frozenset[str] = frozenset({"init", "runtime"})

# failure_phase values (and None) that map HARD_FAILURE -> "SKIP"
_BISECT_SKIP_PHASES: frozenset[str | None] = frozenset({"import", "driver", "oom", "hang", None})


def _bisect_verdict(verdict: OracleVerdict, was_retried: bool, failure_phase: str | None) -> str:
    """Compute the bisect-friendly label for a given verdict"""
    if verdict == OracleVerdict.PASS:
        return "SKIP" if was_retried else "GOOD"
    if verdict == OracleVerdict.WARN:
        return "SKIP"
    if verdict == OracleVerdict.BLOCK:
        return "BAD"
    # HARD_FAILURE
    if failure_phase in _BISECT_BAD_PHASES:
        return "BAD"
    return "SKIP"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _percentile(sorted_data: list[float], p: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list

    Args:
        sorted_data: Ascending-sorted FPS values (must be non-empty)
        p: Percentile in [0, 100]

    Returns:
        Interpolated value at the requested percentile
    """
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
    note: str | None = None,
) -> OracleResult:
    """Build a HARD_FAILURE result (missing perf data or config drift); no usable FPS to report."""
    return OracleResult(
        verdict=OracleVerdict.HARD_FAILURE,
        bisect_verdict=_bisect_verdict(OracleVerdict.HARD_FAILURE, was_retried, failure_phase),
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
        backend=bench_result["backend"],
        threshold_source="n/a",
        note=note,
    )


def compare(
    bench_result: dict,
    baseline: "Baseline | None",
    fps_mean_floor: float,
    excluded_frames: "frozenset[int]",
    artifact_dir: "Path",
    overrides: "dict | None" = None,
) -> OracleResult:
    """Compare a benchmark result against its baseline and return an :class:`OracleResult`

    Oracle logic (in order):

    1. If perf data is missing, return ``HARD_FAILURE`` without reading files.
    2. Load ``artifact_dir/perf_regression_gate_info.json`` and extract the FPS series.
    3. Filter the series with :func:`apply_excluded_frames` and compute the mean FPS.
    4. Resolve threshold knobs (overrides > baseline-stored > module defaults).
    5. Apply the hard floor; then the floored median+MAD bands when the window is
       trusted (``sample_count >= MIN_WINDOW``) or a pin is set; otherwise PASS
       (seed run).
    6. Downgrade PASS to WARN when ``was_retried`` is True.
    7. Apply the advisory tail-p99 check (opt-in) and compute the bisect verdict.

    Args:
        bench_result: Dict matching the ``perf_regression_gate_result.json`` schema
        baseline: Rolling baseline statistics, or None for a seed run
        fps_mean_floor: Absolute minimum acceptable FPS (hard floor; 0 disables it)
        excluded_frames: 0-based frame indices to drop before computing mean FPS
        artifact_dir: Directory that contains ``perf_regression_gate_info.json``
        overrides: Merged per-task/gpu override block (k_warn/k_block/min_spread_pct/
            pin_center_fps/pin_spread_fps/skip/tail_p99_warn), or None

    Returns:
        Fully populated :class:`OracleResult`
    """
    overrides = overrides or {}
    task_id: str = bench_result["task_id"]
    backend: str = bench_result["backend"]
    failure_phase: str | None = bench_result["failure_phase"]
    was_retried: bool = bench_result["was_retried"]
    startup_time_s: float | None = bench_result.get("startup_time_s")
    wall_time_s: float | None = bench_result.get("wall_time_s")
    gpu_mem_used_mb: float | None = (bench_result.get("gpu_diag") or {}).get("gpu_mem_used_mb")

    # Treat missing perf data as HARD_FAILURE regardless of failure_phase
    if not bench_result["perf_regression_gate_info_present"]:
        return _hard_failure(bench_result, failure_phase, was_retried, gpu_mem_used_mb)

    # Config drift: the run succeeded but used a different config than the gate
    # launched it with, so the FPS is no longer comparable to the calibrated
    # window. Treat as a structural failure, not a regression.
    if failure_phase == "config_mismatch":
        return _hard_failure(
            bench_result, failure_phase, was_retried, gpu_mem_used_mb, note=bench_result.get("config_mismatch")
        )

    # Load perf_regression_gate_info.json
    perf_regression_gate_info_path = Path(artifact_dir) / "perf_regression_gate_info.json"
    with perf_regression_gate_info_path.open() as fh:
        perf_regression_gate_info = json.load(fh)

    # Extract FPS series
    fps_series: list[float] = []
    for phase in perf_regression_gate_info:
        if phase.get("phase_name") == "runtime":
            for measurement in phase.get("measurements", []):
                if measurement.get("name", "").endswith("Step Frametimes"):
                    fps_series = measurement["value"]["Environment step effective FPS"]
                    break
            break

    # Compute filtered data, mean, and informational statistics
    filtered = apply_excluded_frames(fps_series, excluded_frames)
    if not filtered:
        return _hard_failure(bench_result, failure_phase, was_retried, gpu_mem_used_mb)
    mean_fps = statistics.mean(filtered)
    sorted_filtered = sorted(filtered)
    fps_median = _percentile(sorted_filtered, 50.0)
    fps_p5 = _percentile(sorted_filtered, 5.0)
    fps_p95 = _percentile(sorted_filtered, 95.0)

    # ----- Resolve threshold knobs: overrides > baseline-stored > module defaults
    base_k_warn = baseline.k_warn if baseline is not None else DEFAULT_K_WARN
    base_k_block = baseline.k_block if baseline is not None else DEFAULT_K_BLOCK
    k_warn = float(overrides.get("k_warn", base_k_warn))
    k_block = float(overrides.get("k_block", base_k_block))
    min_spread_pct = float(overrides.get("min_spread_pct", DEFAULT_MIN_SPREAD_PCT))

    # ----- Resolve the comparison center + robust spread (with anti-flap floor)
    center: float | None = None
    spread: float | None = None
    threshold_source = "seed"
    if baseline is not None and baseline.sample_count >= MIN_WINDOW:
        center = baseline.median_fps
        spread = max(MAD_TO_STD * baseline.mad_fps, min_spread_pct / 100.0 * center)
        threshold_source = f"window(n={baseline.sample_count})"
    elif baseline is not None:
        # Window exists but is too small to trust its MAD -> seed run (rubber-stamp).
        center = baseline.median_fps

    # Manual pins win and enable gating even before the window fills.
    if "pin_center_fps" in overrides:
        center = float(overrides["pin_center_fps"])
        if spread is None:
            spread = min_spread_pct / 100.0 * center
        threshold_source = "override_pin"
    if "pin_spread_fps" in overrides:
        spread = float(overrides["pin_spread_fps"])
        if threshold_source == "seed":
            threshold_source = "override_pin"

    # ----- Verdict
    note: str | None = None
    if overrides.get("skip"):
        verdict = OracleVerdict.PASS
        threshold_source = "override_skip"
        note = "skipped_by_override"
    elif mean_fps < fps_mean_floor:
        # Catastrophic absolute floor (relative-to-center value computed by aggregate).
        verdict = OracleVerdict.BLOCK
        note = "below_hard_floor"
    elif center is None or spread is None:
        # Seed run: no trusted reference yet.
        verdict = OracleVerdict.PASS
    else:
        block_thresh = center - k_block * spread
        warn_thresh = center - k_warn * spread
        if mean_fps < block_thresh:
            verdict = OracleVerdict.BLOCK
        elif mean_fps < warn_thresh:
            verdict = OracleVerdict.WARN
        else:
            verdict = OracleVerdict.PASS

    # A result that only succeeded after a retry is inherently suspect.
    if verdict == OracleVerdict.PASS and was_retried:
        verdict = OracleVerdict.WARN
        note = note or "was_retried"

    # Advisory tail check (opt-in): WARN (never BLOCK) when the post-warm-up
    # p99/median step-time ratio exceeds the per-task ceiling. Surfaces tail/spike
    # regressions the FPS mean hides.
    tail_ceiling = overrides.get("tail_p99_warn")
    p99_over_median = bench_result.get("p99_over_median")
    if (
        verdict == OracleVerdict.PASS
        and tail_ceiling is not None
        and isinstance(p99_over_median, (int, float))
        and float(p99_over_median) > float(tail_ceiling)
    ):
        verdict = OracleVerdict.WARN
        note = f"tail(p99_over_median={float(p99_over_median):g}>{float(tail_ceiling):g})"

    baseline_fps: float | None = (
        center if center is not None else (baseline.median_fps if baseline is not None else None)
    )
    regression_pct: float | None = None
    if baseline_fps:
        regression_pct = ((mean_fps - baseline_fps) / baseline_fps) * 100.0

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
        spread_fps=spread,
        threshold_source=threshold_source,
        note=note,
    )
