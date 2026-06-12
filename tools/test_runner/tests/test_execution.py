# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for test_runner.execution: the unit command contract + status merge."""

from __future__ import annotations

import os

from test_runner.execution import ExecContext, UnitRunner, merge_statuses
from test_runner.planning import RunnerConfig, Unit


def _runner(isaacsim_ci=False):
    return UnitRunner(RunnerConfig(workspace_root="/ws", isaacsim_ci=isaacsim_ci))


def _ctx():
    return ExecContext(file_name="test_x.py", timeout=60, startup_deadline=30, env={"BASE": "1"})


def _unit(mask, split=False):
    return Unit("source/pkg/test_x.py", mask, split=split)


def test_mix_ok_unit_sets_mask_no_selector_no_k():
    cmd, env, report, label, suffix = _runner()._build_cmd(_unit("110"), _ctx())
    assert env["ISAACLAB_TEST_DEVICES"] == "110"
    assert env["BASE"] == "1"  # base env preserved
    assert "test_runner.selector" not in cmd
    assert "-k" not in cmd
    assert report == "tests/test-reports-source__pkg__test_x.py.xml"
    assert label == "test_x.py"
    assert suffix == ""
    assert cmd[-1] == "source/pkg/test_x.py"


def test_cpu_split_unit_injects_selector_and_is_suffixed():
    cmd, _, report, _, suffix = _runner()._build_cmd(_unit("100", split=True), _ctx())
    assert "test_runner.selector" in cmd  # cpu half still drops out-of-scope cuda variants
    assert suffix == "-100"
    assert report.endswith("test_x.py-100.xml")


def test_cuda_split_unit_injects_selector_and_tools_pythonpath():
    cmd, env, _, _, _ = _runner()._build_cmd(_unit("010", split=True), _ctx())
    assert cmd.count("-p") == 1 and "test_runner.selector" in cmd
    assert env["ISAACLAB_TEST_DEVICES"] == "010"
    assert env["PYTHONPATH"].split(os.pathsep)[0] == os.path.join("/ws", "tools")


def test_shard_unit_injects_selector_without_a_suffix():
    cmd, env, report, _, suffix = _runner()._build_cmd(_unit("0001"), _ctx())
    assert "test_runner.selector" in cmd  # cpu-less shard
    assert suffix == ""
    assert report == "tests/test-reports-source__pkg__test_x.py.xml"


def test_isaacsim_ci_adds_marker_selector():
    cmd, *_ = _runner(isaacsim_ci=True)._build_cmd(_unit("110"), _ctx())
    assert any(cmd[i] == "-m" and cmd[i + 1] == "isaacsim_ci" for i in range(len(cmd) - 1))


def test_no_k_selector_for_any_mask():
    for mask, split in [("110", False), ("100", True), ("010", True), ("0001", False)]:
        cmd, *_ = _runner()._build_cmd(_unit(mask, split=split), _ctx())
        assert "-k" not in cmd


def test_merge_statuses_sums_counters_and_takes_worst_result():
    a = {
        "errors": 0,
        "failures": 0,
        "skipped": 1,
        "tests": 3,
        "time_elapsed": 1.0,
        "wall_time": 2.0,
        "result": "passed",
    }
    b = {
        "errors": 1,
        "failures": 0,
        "skipped": 0,
        "tests": 2,
        "time_elapsed": 0.5,
        "wall_time": 1.0,
        "result": "FAILED",
    }
    merged = merge_statuses(a, b)
    assert merged["tests"] == 5 and merged["skipped"] == 1 and merged["errors"] == 1
    assert merged["result"] == "FAILED"  # more severe wins
    assert merge_statuses(None, b) == b  # first unit seeds the entry
