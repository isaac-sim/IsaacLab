# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`scripts.benchmarks.utils.parse_cprofile_stats`.

The function is expected to return 4-tuples
``(label, tottime_ms, cumtime_ms, ncalls)`` after the T2.2 reliability fix.
Before the fix, the function returned 3-tuples and CProfileFunction.calls was
always 0 in the downstream startup bundle.
"""

from __future__ import annotations

import cProfile
import os
import sys

# scripts/benchmarks/utils.py is not an installable package; add the repo
# root to sys.path so the import works.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.benchmarks.utils import parse_cprofile_stats  # noqa: E402


def _profiled_call(n_outer: int, n_inner: int) -> cProfile.Profile:
    """Run a couple of nested helpers a known number of times under cProfile."""

    def inner():
        return sum(range(10))

    def outer():
        for _ in range(n_inner):
            inner()

    prof = cProfile.Profile()
    prof.enable()
    for _ in range(n_outer):
        outer()
    prof.disable()
    return prof


def test_top_n_returns_ncalls():
    # The synthetic functions live in THIS test file, so _is_isaaclab will
    # not match them — they come through the "first-level external call from
    # an IsaacLab caller" path only if we pass this file's directory as an
    # isaaclab prefix. Do so to include them.
    test_dir = os.path.abspath(os.path.dirname(__file__))
    prof = _profiled_call(n_outer=3, n_inner=5)

    results = parse_cprofile_stats(prof, isaaclab_prefixes=[test_dir], top_n=30)

    # Each row must be a 4-tuple now.
    assert results, "parse_cprofile_stats should return at least one row"
    for row in results:
        assert len(row) == 4, f"expected (label, tot, cum, ncalls) 4-tuple, got {row!r}"
        label, tot, cum, ncalls = row
        assert isinstance(label, str)
        assert isinstance(tot, float)
        assert isinstance(cum, float)
        assert isinstance(ncalls, int)
        assert ncalls >= 0

    # Locate our two functions by suffix and check their call counts.
    outer_rows = [r for r in results if r[0].endswith(":outer")]
    inner_rows = [r for r in results if r[0].endswith(":inner")]
    assert outer_rows, f"outer() should be in results, got labels: {[r[0] for r in results]}"
    assert inner_rows, f"inner() should be in results, got labels: {[r[0] for r in results]}"
    assert outer_rows[0][3] == 3, f"outer ncalls should be 3, got {outer_rows[0][3]}"
    assert inner_rows[0][3] == 15, f"inner ncalls should be 3*5=15, got {inner_rows[0][3]}"


def test_whitelist_path_returns_ncalls():
    test_dir = os.path.abspath(os.path.dirname(__file__))
    prof = _profiled_call(n_outer=2, n_inner=4)

    results = parse_cprofile_stats(
        prof,
        isaaclab_prefixes=[test_dir],
        whitelist=["*:inner", "*:definitely_not_a_real_function"],
    )

    # Matched row carries the real ncalls; placeholder row carries 0.
    labels = {r[0]: r for r in results}
    inner_label = next((lbl for lbl in labels if lbl.endswith(":inner")), None)
    assert inner_label is not None, f"inner() should match wildcard whitelist, labels: {list(labels)}"
    assert labels[inner_label][3] == 8, f"inner ncalls should be 2*4=8, got {labels[inner_label][3]}"

    placeholder = labels.get("*:definitely_not_a_real_function")
    assert placeholder is not None, "placeholder row should be emitted for unmatched pattern"
    assert placeholder == ("*:definitely_not_a_real_function", 0.0, 0.0, 0)
