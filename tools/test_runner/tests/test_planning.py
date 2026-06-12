# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for test_runner.planning (Planner + RunnerConfig)."""

from __future__ import annotations

import pytest

from test_runner.planning import Planner, RunnerConfig, Unit

_ISOLATED = "import pytest\npytestmark = pytest.mark.device_isolated\n"


# ---- Planner.plan ----


def test_mix_ok_file_is_a_single_unit():
    assert Planner("110").plan("a.py", "def test_x(): pass\n") == [Unit("a.py", "110")]


def test_isolated_file_splits_on_a_multi_device_run():
    assert Planner("110").plan("a.py", _ISOLATED) == [Unit("a.py", "100", split=True), Unit("a.py", "010", split=True)]


def test_isolated_file_does_not_split_on_a_single_device_run():
    assert Planner("0001").plan("a.py", _ISOLATED) == [Unit("a.py", "0001")]


def test_split_uses_each_set_bit_position():
    assert Planner("1010").plan("a.py", _ISOLATED) == [
        Unit("a.py", "1000", split=True),
        Unit("a.py", "0010", split=True),
    ]


def test_runtime_mask_with_trailing_x_is_rejected():
    with pytest.raises(ValueError):
        Planner("11X")


# ---- Planner.is_isolated ----


def test_is_isolated_detects_single_and_list_forms():
    p = Planner("110")
    assert p.is_isolated("pytestmark = pytest.mark.device_isolated") is True
    assert p.is_isolated("pytestmark = [pytest.mark.device_isolated, pytest.mark.slow]") is True


def test_is_isolated_ignores_comments_and_other_marks():
    p = Planner("110")
    assert p.is_isolated("# device_isolated explains the lock") is False
    assert p.is_isolated("pytestmark = pytest.mark.slow") is False


# ---- RunnerConfig ----


def test_config_defaults_and_derived_paths():
    c = RunnerConfig(workspace_root="/ws")
    assert c.config_file == "/ws/pyproject.toml"
    assert c.tools_dir == "/ws/tools"
    assert c.runtime_mask == "110"  # single-GPU default
    assert c.process_failure_retries["test_visualizer_integration_physx.py"] == 4


def test_from_env_reads_mask_flags_and_includes(monkeypatch):
    monkeypatch.setenv("ISAACLAB_TEST_DEVICES", "0001")
    monkeypatch.setenv("ISAACSIM_CI_SHORT", "true")
    monkeypatch.setenv("TEST_INCLUDE_FILES", "a/b/test_x.py, test_y.py")
    c = RunnerConfig.from_env("/ws")
    assert c.runtime_mask == "0001"
    assert c.isaacsim_ci is True
    assert c.include_files == frozenset({"test_x.py", "test_y.py"})


def test_from_env_defaults_mask_when_unset(monkeypatch):
    monkeypatch.delenv("ISAACLAB_TEST_DEVICES", raising=False)
    assert RunnerConfig.from_env("/ws").runtime_mask == "110"
