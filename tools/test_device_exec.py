# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior guard for the executor's per-unit command construction.

These assert the device-selection contract without launching a subprocess: the
unit's mask becomes ``ISAACLAB_TEST_DEVICES`` (so ``test_devices()`` selects the
variants), the ``_agnostic_select`` plugin is injected only for a cpu-less mask
(shards and the cuda unit of a split), ``-k`` is never used, and split units get
a suffixed report name while single units keep the mgpu-aggregator-friendly one.
"""

from __future__ import annotations

import os

from _device_exec import _build_unit_cmd, _UnitContext
from _device_plan import Unit


def _ctx(test_file="source/pkg/test_x.py", isaacsim_ci=False):
    return _UnitContext(
        test_file=test_file,
        file_name="test_x.py",
        workspace_root="/ws",
        isaacsim_ci=isaacsim_ci,
        timeout=60,
        startup_deadline=30,
        env={"BASE": "1"},
    )


def test_mix_ok_unit_sets_mask_with_no_plugin_and_no_k():
    cmd, env, report_file, label = _build_unit_cmd(_ctx(), Unit("source/pkg/test_x.py", "110"))
    assert env["ISAACLAB_TEST_DEVICES"] == "110"
    assert env["BASE"] == "1"  # base env preserved
    assert "_agnostic_select" not in cmd
    assert "-k" not in cmd
    assert report_file == "tests/test-reports-source__pkg__test_x.py.xml"  # no suffix for a single unit
    assert label == "test_x.py"
    assert cmd[-1] == "source/pkg/test_x.py"


def test_cpu_split_unit_is_suffixed_and_keeps_agnostic_tests():
    cmd, env, report_file, label = _build_unit_cmd(_ctx(), Unit("source/pkg/test_x.py", "100", split=True))
    assert env["ISAACLAB_TEST_DEVICES"] == "100"
    assert "_agnostic_select" not in cmd  # cpu in mask -> agnostic tests run here
    assert report_file.endswith("test_x.py-100.xml")
    assert label == "test_x.py-100"


def test_cuda_split_unit_injects_agnostic_plugin_and_tools_pythonpath():
    cmd, env, report_file, _ = _build_unit_cmd(_ctx(), Unit("source/pkg/test_x.py", "010", split=True))
    assert env["ISAACLAB_TEST_DEVICES"] == "010"
    assert cmd.count("-p") == 1 and "_agnostic_select" in cmd
    assert env["PYTHONPATH"].split(os.pathsep)[0] == os.path.join("/ws", "tools")
    assert report_file.endswith("test_x.py-010.xml")


def test_shard_unit_injects_agnostic_plugin_without_a_suffix():
    cmd, env, report_file, label = _build_unit_cmd(_ctx(), Unit("source/pkg/test_x.py", "0001"))
    assert env["ISAACLAB_TEST_DEVICES"] == "0001"
    assert "_agnostic_select" in cmd  # cpu-less mask -> drop device-agnostic tests
    # one unit per file on a shard -> suffix-free name the aggregator unslugs to a path
    assert report_file == "tests/test-reports-source__pkg__test_x.py.xml"
    assert label == "test_x.py"


def test_isaacsim_ci_adds_the_marker_selector():
    # ``-m pytest`` appears first, so look for the ``-m isaacsim_ci`` pair specifically.
    cmd, _, _, _ = _build_unit_cmd(_ctx(isaacsim_ci=True), Unit("source/pkg/test_x.py", "110"))
    assert any(cmd[i] == "-m" and cmd[i + 1] == "isaacsim_ci" for i in range(len(cmd) - 1))


def test_no_isaacsim_ci_marker_when_disabled():
    cmd, _, _, _ = _build_unit_cmd(_ctx(isaacsim_ci=False), Unit("source/pkg/test_x.py", "110"))
    assert "isaacsim_ci" not in cmd


def test_no_k_selector_for_any_mask():
    for mask, split in [("110", False), ("100", True), ("010", True), ("0001", False)]:
        cmd, *_ = _build_unit_cmd(_ctx(), Unit("source/pkg/test_x.py", mask, split=split))
        assert "-k" not in cmd
