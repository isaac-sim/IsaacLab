# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the hardened oracle verdict math."""

from _helpers import make_bench_result, write_info
from oracle import MIN_WINDOW, Baseline, OracleVerdict, compare


def _compare(tmp_path, fps_series, baseline=None, floor=0.0, overrides=None, step_times=None, **br_kwargs):
    write_info(tmp_path, fps_series, step_times=step_times)
    br = make_bench_result(**br_kwargs)
    return compare(
        bench_result=br,
        baseline=baseline,
        fps_mean_floor=floor,
        excluded_frames=frozenset(),
        artifact_dir=tmp_path,
        overrides=overrides or {},
    )


def _trusted_baseline(median=1000.0, mad=10.0, n=10):
    return Baseline(median_fps=median, mad_fps=mad, sample_count=n)


def test_within_band_passes(tmp_path):
    r = _compare(tmp_path, [1000.0] * 20, baseline=_trusted_baseline())
    assert r.verdict is OracleVerdict.PASS
    assert r.spread_fps == 15.0  # max(1.4826*10, 1.5% of 1000)


def test_below_warn_warns(tmp_path):
    # spread=15, warn_thresh = 1000 - 3*15 = 955
    r = _compare(tmp_path, [940.0] * 20, baseline=_trusted_baseline())
    assert r.verdict is OracleVerdict.WARN


def test_below_block_blocks(tmp_path):
    # spread=15, block_thresh = 1000 - 6*15 = 910
    r = _compare(tmp_path, [900.0] * 20, baseline=_trusted_baseline())
    assert r.verdict is OracleVerdict.BLOCK


def test_spread_floor_prevents_flap_when_mad_zero(tmp_path):
    # mad=0 would collapse the band onto the median without the floor; a 1% dip must NOT block.
    r = _compare(tmp_path, [990.0] * 20, baseline=_trusted_baseline(mad=0.0))
    assert r.spread_fps == 15.0
    assert r.verdict is OracleVerdict.PASS


def test_seed_run_passes_below_min_window(tmp_path):
    # A window with < MIN_WINDOW samples is not trusted: PASS even far below center.
    bl = Baseline(median_fps=1000.0, mad_fps=10.0, sample_count=MIN_WINDOW - 1)
    r = _compare(tmp_path, [500.0] * 20, baseline=bl)
    assert r.verdict is OracleVerdict.PASS
    assert r.threshold_source == "seed"


def test_none_baseline_seed_passes(tmp_path):
    r = _compare(tmp_path, [500.0] * 20, baseline=None)
    assert r.verdict is OracleVerdict.PASS
    assert r.threshold_source == "seed"


def test_hard_floor_blocks(tmp_path):
    r = _compare(tmp_path, [940.0] * 20, baseline=_trusted_baseline(), floor=950.0)
    assert r.verdict is OracleVerdict.BLOCK
    assert r.note == "below_hard_floor"


def test_retry_downgrades_pass_to_warn(tmp_path):
    r = _compare(tmp_path, [1000.0] * 20, baseline=_trusted_baseline(), was_retried=True)
    assert r.verdict is OracleVerdict.WARN


def test_override_skip_forces_pass(tmp_path):
    r = _compare(tmp_path, [100.0] * 20, baseline=_trusted_baseline(), overrides={"skip": True})
    assert r.verdict is OracleVerdict.PASS
    assert r.note == "skipped_by_override"


def test_pin_center_enables_gating_on_seed(tmp_path):
    # No window, but a pinned center lets the gate act. min_spread 1.5% of 500 = 7.5.
    r_warn = _compare(tmp_path, [470.0] * 20, baseline=None, overrides={"pin_center_fps": 500.0})
    assert r_warn.verdict is OracleVerdict.WARN
    assert r_warn.threshold_source == "override_pin"
    r_pass = _compare(tmp_path, [480.0] * 20, baseline=None, overrides={"pin_center_fps": 500.0})
    assert r_pass.verdict is OracleVerdict.PASS


def test_tail_p99_warns_when_opted_in(tmp_path):
    r = _compare(
        tmp_path,
        [1000.0] * 20,
        baseline=_trusted_baseline(),
        overrides={"tail_p99_warn": 1.8},
        p99_over_median=2.0,
    )
    assert r.verdict is OracleVerdict.WARN
    assert r.note is not None and r.note.startswith("tail(")


def test_tail_p99_ignored_when_not_opted_in(tmp_path):
    r = _compare(tmp_path, [1000.0] * 20, baseline=_trusted_baseline(), p99_over_median=5.0)
    assert r.verdict is OracleVerdict.PASS


def test_config_mismatch_is_hard_failure(tmp_path):
    r = _compare(
        tmp_path,
        [1000.0] * 20,
        baseline=_trusted_baseline(),
        failure_phase="config_mismatch",
        config_mismatch="num_envs(ran=256,want=512)",
    )
    assert r.verdict is OracleVerdict.HARD_FAILURE
    assert r.bisect_verdict == "SKIP"
    assert r.note == "num_envs(ran=256,want=512)"


def test_missing_perf_data_is_hard_failure(tmp_path):
    write_info(tmp_path, [1000.0] * 20)
    br = make_bench_result(present=False, failure_phase="init")
    r = compare(br, _trusted_baseline(), 0.0, frozenset(), tmp_path)
    assert r.verdict is OracleVerdict.HARD_FAILURE
    assert r.bisect_verdict == "BAD"  # init -> BAD


def test_regression_pct_against_center(tmp_path):
    r = _compare(tmp_path, [950.0] * 20, baseline=_trusted_baseline())
    assert r.regression_pct is not None
    assert round(r.regression_pct, 1) == -5.0
