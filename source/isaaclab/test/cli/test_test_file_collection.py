# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for CI file selection across combined package jobs."""

from pathlib import Path

import pytest
from test_test_orchestrator_result_handling import _load_orchestrator_module


@pytest.fixture
def collection_tree(tmp_path, monkeypatch):
    """Keep file discovery real while isolating it from the checkout and CI shard settings."""
    for path in ("alpha/test_a.py", "alpha/test_b.py", "beta/test_c.py", "gamma/test_d.py"):
        target = tmp_path / path
        target.parent.mkdir(exist_ok=True)
        target.touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TEST_SHARD_INDEX", raising=False)
    monkeypatch.delenv("TEST_SHARD_COUNT", raising=False)
    return _load_orchestrator_module()


@pytest.mark.parametrize(
    "include,exclude,expected",
    [
        ("alpha", "", {"alpha/test_a.py", "alpha/test_b.py"}),
        ("alpha,beta", "", {"alpha/test_a.py", "alpha/test_b.py", "beta/test_c.py"}),
        (" , alpha, ,beta,alpha, ", "", {"alpha/test_a.py", "alpha/test_b.py", "beta/test_c.py"}),
        ("alpha,beta", "test_b, test_c", {"alpha/test_a.py"}),
        ("", "", {"alpha/test_a.py", "alpha/test_b.py", "beta/test_c.py", "gamma/test_d.py"}),
    ],
)
def test_file_selection(collection_tree, include, exclude, expected):
    selected = collection_tree._collect_test_files(["."], include, exclude, set(), False, False)
    paths = [Path(path).as_posix().removeprefix("./") for path in selected]
    assert set(paths) == expected
    assert len(paths) == len(expected), "overlapping include patterns must not duplicate files"


def test_combined_packages_are_partitioned_without_lost_or_duplicate_files(collection_tree, monkeypatch):
    shards = []
    monkeypatch.setenv("TEST_SHARD_COUNT", "2")
    for index in range(2):
        monkeypatch.setenv("TEST_SHARD_INDEX", str(index))
        selected = collection_tree._collect_test_files(["."], "alpha,beta", "", set(), False, False)
        shards.append({Path(path).as_posix().removeprefix("./") for path in selected})

    assert all(shards), "each shard should receive work when there are enough files"
    assert shards[0].isdisjoint(shards[1])
    assert shards[0] | shards[1] == {"alpha/test_a.py", "alpha/test_b.py", "beta/test_c.py"}
