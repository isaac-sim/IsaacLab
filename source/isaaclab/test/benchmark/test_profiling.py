# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for benchmark cProfile parsing (Isaac-Sim-free)."""

import cProfile
import os

from isaaclab.test.benchmark.profiling import parse_cprofile_stats


def _profiled():
    def inner(n):
        return sum(range(n))

    p = cProfile.Profile()
    p.enable()
    for _ in range(3):
        inner(1000)
    p.disable()
    return p


def test_parse_cprofile_returns_4_tuples():
    p = _profiled()
    # Treat this test file's directory as an "isaaclab prefix" so functions
    # defined here are captured by the filter.
    prefixes = [os.path.dirname(os.path.abspath(__file__))]
    rows = parse_cprofile_stats(p, prefixes, top_n=10)
    assert rows, "expected at least one profiled function"
    for row in rows:
        assert len(row) == 4
        label, own_ms, cum_ms, calls = row
        assert isinstance(label, str)
        assert isinstance(own_ms, float) and isinstance(cum_ms, float)
        assert isinstance(calls, int) and calls >= 1


def test_whitelist_placeholder_is_4_tuple():
    p = _profiled()
    prefixes = [os.path.dirname(os.path.abspath(__file__))]
    rows = parse_cprofile_stats(p, prefixes, whitelist=["nonexistent.module:does_not_exist"])
    # unmatched pattern yields a placeholder row, still a 4-tuple
    assert any(r[0] == "nonexistent.module:does_not_exist" and r[3] == 0 for r in rows)


def test_parse_cprofile_prefers_most_specific_source_root():
    namespace = {}
    exec(
        compile(
            "def profile_target():\n    return sum(range(10))",
            "/tmp/repro/source/isaaclab_tasks/isaaclab_tasks/example.py",
            "exec",
        ),
        namespace,
    )
    profile = cProfile.Profile()
    profile.runcall(namespace["profile_target"])

    rows = parse_cprofile_stats(
        profile,
        ["/tmp/repro/source/isaaclab", "/tmp/repro/source/isaaclab_tasks"],
    )

    assert any(label == "isaaclab_tasks.example:profile_target" for label, *_ in rows)
