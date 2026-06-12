# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for baseline storage: capped rolling window + fingerprint fallback chain."""

import baseline_manager as bm


def test_window_capped_at_max(tmp_path):
    for i in range(WANT := bm.WINDOW_MAX + 12):
        bm.update_baseline(tmp_path, "L40S", "T", "physx", 1000.0 + i, fingerprint="warp1.12/h/c")
    assert WANT > bm.WINDOW_MAX
    bl = bm.load_baseline(tmp_path, "L40S", "T", "physx", fingerprint="warp1.12/h/c")
    assert bl.sample_count == bm.WINDOW_MAX


def test_window_evicts_oldest(tmp_path):
    # Old low values must roll off so the median tracks the recent (high) regime.
    for v in [1.0] * bm.WINDOW_MAX:
        bm.update_baseline(tmp_path, "L40S", "T", "physx", v)
    for v in [1000.0] * bm.WINDOW_MAX:
        bm.update_baseline(tmp_path, "L40S", "T", "physx", v)
    bl = bm.load_baseline(tmp_path, "L40S", "T", "physx")
    assert bl.median_fps == 1000.0


def test_fingerprint_candidates_order():
    assert bm.fingerprint_candidates("a/b/c") == ["a/b/c", "a/b", "a", None]
    assert bm.fingerprint_candidates(None) == [None]
    assert bm.fingerprint_candidates("") == [None]


def test_fallback_resolves_to_looser_bucket(tmp_path):
    # Only a looser bucket has data; an exact-fingerprint query must fall back to it.
    for _ in range(6):
        bm.update_baseline(tmp_path, "L40S", "T", "physx", 500.0, fingerprint="warp1.12")
    bl, matched = bm.load_baseline_resolved(tmp_path, "L40S", "T", "physx", "warp1.12/runtimeX/codeY")
    assert matched == "warp1.12"
    assert bl is not None and bl.sample_count == 6


def test_fallback_prefers_most_specific(tmp_path):
    bm.update_baseline(tmp_path, "L40S", "T", "physx", 100.0, fingerprint="warp1.12")
    bm.update_baseline(tmp_path, "L40S", "T", "physx", 900.0, fingerprint="warp1.12/r/c")
    bl, matched = bm.load_baseline_resolved(tmp_path, "L40S", "T", "physx", "warp1.12/r/c")
    assert matched == "warp1.12/r/c"
    assert bl.median_fps == 900.0


def test_resolve_returns_none_when_unseen(tmp_path):
    bl, matched = bm.load_baseline_resolved(tmp_path, "L40S", "Unseen", "physx", "warp1.12/r/c")
    assert bl is None and matched is None


def test_seed_with_spread_populates_window(tmp_path):
    bm.seed_baseline_with_spread(tmp_path, "L40S", "T", "physx", center_fps=1000.0, n_samples=8, seed=1)
    bl = bm.load_baseline(tmp_path, "L40S", "T", "physx")
    assert bl.sample_count == 8
    assert 900.0 < bl.median_fps < 1100.0
