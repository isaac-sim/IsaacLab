# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for aggregate's override merge precedence."""

import aggregate


def test_overrides_defaults_only():
    doc = {"_defaults": {"k_warn": 3.0, "k_block": 6.0}}
    ov = aggregate._overrides_for(doc, "Isaac-Cartpole", "NVIDIA L40S")
    assert ov == {"k_warn": 3.0, "k_block": 6.0}


def test_overrides_task_scalar_overrides_defaults():
    doc = {"_defaults": {"k_block": 6.0}, "Isaac-Cartpole": {"k_block": 8.0}}
    ov = aggregate._overrides_for(doc, "Isaac-Cartpole", "NVIDIA L40S")
    assert ov["k_block"] == 8.0


def test_overrides_gpu_block_wins():
    doc = {
        "_defaults": {"k_block": 6.0},
        "Isaac-Cartpole": {"k_block": 8.0, "NVIDIA L40S": {"k_block": 10.0, "skip": True}},
    }
    ov = aggregate._overrides_for(doc, "Isaac-Cartpole", "NVIDIA L40S")
    assert ov["k_block"] == 10.0
    assert ov["skip"] is True


def test_overrides_gpu_substring_match():
    doc = {"Isaac-Cartpole": {"NVIDIA L40S": {"pin_center_fps": 123.0}}}
    ov = aggregate._overrides_for(doc, "Isaac-Cartpole", "L40S")
    assert ov["pin_center_fps"] == 123.0


def test_overrides_unknown_task_empty():
    doc = {"_defaults": {"k_warn": 3.0}, "Isaac-Cartpole": {"k_block": 8.0}}
    ov = aggregate._overrides_for(doc, "Other-Task", "NVIDIA L40S")
    assert ov == {"k_warn": 3.0}
