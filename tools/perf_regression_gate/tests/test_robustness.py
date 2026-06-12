# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adversarial / property robustness tests for the unified gate.

Where ``test_oracle.py`` / ``test_baseline_manager.py`` exercise the documented
happy paths, this file attacks the edges that broke the *previous* perf-smoke
comparator (degenerate baselines, GPU-string conflation, poisoned series, pinned
centers, config drift) and asserts the hardened unified gate either handles them
correctly or fails *structurally* (HARD_FAILURE) -- never with a Python traceback
and never by silently disabling the gate. Several tests are randomized property
checks run over many trials to catch order/scale-dependent regressions.
"""

import random
import statistics

import baseline_manager as bm
from _helpers import make_bench_result, write_info
from oracle import MIN_WINDOW, Baseline, OracleVerdict, compare

_SEVERITY = {OracleVerdict.PASS: 0, OracleVerdict.WARN: 1, OracleVerdict.BLOCK: 2}


def _compare(tmp_path, fps_series, baseline=None, floor=0.0, overrides=None, excluded=frozenset(), **br_kwargs):
    write_info(tmp_path, fps_series)
    br = make_bench_result(**br_kwargs)
    return compare(
        bench_result=br,
        baseline=baseline,
        fps_mean_floor=floor,
        excluded_frames=excluded,
        artifact_dir=tmp_path,
        overrides=overrides or {},
    )


def _trusted(median=1000.0, mad=10.0, n=10):
    return Baseline(median_fps=median, mad_fps=mad, sample_count=n)


# --------------------------------------------------------------------------- A
# Degenerate baselines must degrade gracefully, never raise (old BUG A: /0 crash).


def test_zero_center_baseline_does_not_crash(tmp_path):
    r = _compare(tmp_path, [50000.0] * 20, baseline=_trusted(median=0.0, mad=0.0))
    assert r.verdict in (OracleVerdict.PASS, OracleVerdict.WARN, OracleVerdict.BLOCK)
    # No usable center -> regression_pct must be None rather than a ZeroDivisionError.
    assert r.regression_pct is None


def test_huge_mad_widens_band_and_passes(tmp_path):
    # A pathologically large spread should make the band permissive, not error.
    r = _compare(tmp_path, [1.0] * 20, baseline=_trusted(median=1000.0, mad=1e9))
    assert r.verdict is OracleVerdict.PASS


def test_all_frames_excluded_is_hard_failure(tmp_path):
    # A truncated run whose every frame is excluded yields no comparable sample.
    r = _compare(tmp_path, [10.0, 10.0, 10.0], baseline=_trusted(), excluded=frozenset({0, 1, 2}))
    assert r.verdict is OracleVerdict.HARD_FAILURE


def test_negative_frames_do_not_crash(tmp_path):
    # Impossible per-frame FPS must not raise; with a 0 floor a negative mean blocks.
    r = _compare(tmp_path, [1000.0] * 100 + [-1000.0] * 200, baseline=_trusted())
    assert r.verdict in (OracleVerdict.BLOCK, OracleVerdict.WARN, OracleVerdict.PASS)


# --------------------------------------------------------------------------- B
# GPU bucketing is exact: 'L40' must not be judged against an 'L40S' window
# (old BUG B: substring conflation of distinct GPUs).


def test_gpu_buckets_are_exact_not_substring(tmp_path):
    for _ in range(MIN_WINDOW + 1):
        bm.update_baseline(tmp_path, "NVIDIA L40S", "T", "physx", 1000.0, fingerprint="warp1.12/h/c")
    bl, matched = bm.load_baseline_resolved(tmp_path, "NVIDIA L40", "T", "physx", "warp1.12/h/c")
    assert bl is None and matched is None


# --------------------------------------------------------------------------- C
# Structural failures are not silenceable by perf-only knobs.


def test_config_mismatch_hard_fails_regardless_of_fps(tmp_path):
    rng = random.Random(7)
    for _ in range(50):
        fps = rng.uniform(1.0, 1e6)
        r = _compare(
            tmp_path,
            [fps] * 16,
            baseline=_trusted(),
            failure_phase="config_mismatch",
            config_mismatch="num_envs(ran=256,want=512)",
        )
        assert r.verdict is OracleVerdict.HARD_FAILURE


def test_skip_does_not_rescue_a_crashed_task(tmp_path):
    # `skip` quarantines perf flakiness, not crashes: a missing-perf-data run is a
    # real (structural) failure and stays HARD_FAILURE even with skip set.
    write_info(tmp_path, [1000.0] * 16)
    br = make_bench_result(present=False, failure_phase="init")
    r = compare(br, _trusted(), 0.0, frozenset(), tmp_path, overrides={"skip": True})
    assert r.verdict is OracleVerdict.HARD_FAILURE


def test_pinned_center_scales_its_own_spread(tmp_path):
    # Pinning a much higher center must derive the floor from the NEW center, so a
    # run within the floor of the pinned center passes (old BUG I: collapsed band).
    r = _compare(tmp_path, [285000.0] * 20, baseline=_trusted(median=1000.0), overrides={"pin_center_fps": 300000.0})
    # 285000 is -5% of 300000; min_spread 1.5% * 300000 = 4500 -> warn floor 3*4500 below.
    assert r.verdict is OracleVerdict.WARN  # not BLOCK, not crash
    assert r.spread_fps == 4500.0


# --------------------------------------------------------------------------- D
# Randomized property checks.


def test_verdict_is_monotonic_in_measured_fps(tmp_path):
    # Lower measured FPS must never produce a *less* severe verdict than higher FPS
    # against the same trusted baseline (no retry / tail / floor interference).
    bl = _trusted(median=1000.0, mad=20.0, n=12)
    rng = random.Random(1234)
    for _ in range(300):
        hi = rng.uniform(1.0, 2000.0)
        lo = rng.uniform(1.0, hi)
        v_hi = _compare(tmp_path, [hi] * 16, baseline=bl).verdict
        v_lo = _compare(tmp_path, [lo] * 16, baseline=bl).verdict
        assert _SEVERITY[v_lo] >= _SEVERITY[v_hi], (lo, hi, v_lo, v_hi)


def test_spread_floor_bounds_block_when_mad_zero(tmp_path):
    # With mad=0 the spread is exactly min_spread_pct% of center; a drop just inside
    # k_block*floor passes, just outside blocks. Verifies the anti-flap floor math.
    bl = _trusted(median=1000.0, mad=0.0, n=10)  # spread = 15 (1.5% of 1000); block at 1000-6*15=910
    assert _compare(tmp_path, [911.0] * 16, baseline=bl).verdict is not OracleVerdict.BLOCK
    assert _compare(tmp_path, [909.0] * 16, baseline=bl).verdict is OracleVerdict.BLOCK


def test_window_cap_and_recency_invariant(tmp_path):
    # After arbitrary appends the stored window holds at most WINDOW_MAX samples and
    # its median equals the median of exactly the most-recent WINDOW_MAX values.
    rng = random.Random(99)
    appended: list[float] = []
    n = bm.WINDOW_MAX * 3 + 7
    for _ in range(n):
        v = rng.uniform(10.0, 10000.0)
        appended.append(v)
        bm.update_baseline(tmp_path, "L40S", "T", "physx", v, fingerprint="warp1.12/h/c")
    bl = bm.load_baseline(tmp_path, "L40S", "T", "physx", fingerprint="warp1.12/h/c")
    assert bl.sample_count == bm.WINDOW_MAX
    assert bl.median_fps == statistics.median(appended[-bm.WINDOW_MAX :])


def test_fingerprint_fallback_never_raises_and_prefers_specific(tmp_path):
    rng = random.Random(2024)
    # Seed a random-depth bucket; querying a deeper fingerprint must resolve to the
    # deepest *populated* prefix and never raise.
    depths = ["warp1.12", "warp1.12/rt", "warp1.12/rt/code"]
    populated = rng.choice(depths)
    for _ in range(MIN_WINDOW + 1):
        bm.update_baseline(tmp_path, "L40S", "T", "physx", 777.0, fingerprint=populated)
    bl, matched = bm.load_baseline_resolved(tmp_path, "L40S", "T", "physx", "warp1.12/rt/code/extra")
    assert matched == populated
    assert bl is not None and bl.median_fps == 777.0


def test_seed_run_never_blocks_below_min_window(tmp_path):
    # No matter how bad the measurement, an untrusted (small) window cannot BLOCK.
    rng = random.Random(55)
    for n in range(0, MIN_WINDOW):
        bl = Baseline(median_fps=1000.0, mad_fps=10.0, sample_count=n)
        r = _compare(tmp_path, [rng.uniform(1.0, 10.0)] * 16, baseline=bl)
        assert r.verdict is OracleVerdict.PASS
        assert r.threshold_source == "seed"
