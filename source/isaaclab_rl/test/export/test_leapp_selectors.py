# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for selector representations at LEAPP export boundaries."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp

pytest.importorskip("leapp")

from isaaclab.utils.leapp.export_annotator import ExportPatcher
from isaaclab.utils.leapp.leapp_semantics import select_element_names


def _make_warp_selector():
    """Create a CPU Warp selector for a fresh parameterized test case."""
    return wp.array([0, 2], dtype=wp.int32, device="cpu")


@pytest.mark.parametrize(
    ("selector_factory", "expected_indices"),
    [
        pytest.param(lambda: torch.tensor([0, 2], dtype=torch.int32), [0, 2], id="torch"),
        pytest.param(_make_warp_selector, [0, 2], id="warp"),
        pytest.param(lambda: [0, 2], [0, 2], id="list"),
        pytest.param(list, [], id="empty-list"),
        pytest.param(lambda: slice(1, None), [1, 2], id="slice"),
        pytest.param(lambda: None, [0, 1, 2], id="none"),
    ],
)
def test_static_action_gains_accept_selector_representations(selector_factory, expected_indices):
    """Test gain tensors and element names use the same normalized selector."""
    selector = selector_factory()
    kp_gains = torch.tensor([[10.0, 20.0, 30.0]])
    kd_gains = torch.tensor([[1.0, 2.0, 3.0]])
    asset = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_stiffness=SimpleNamespace(torch=kp_gains),
            default_joint_damping=SimpleNamespace(torch=kd_gains),
        ),
        actuators={},
        joint_names=["joint_0", "joint_1", "joint_2"],
    )
    term = SimpleNamespace(_asset=asset, _joint_ids=selector)
    action_manager = SimpleNamespace(_terms={"binary_gripper": term})

    outputs = ExportPatcher("onnx-dynamo")._collect_action_static_outputs(action_manager)
    outputs_by_name = {output.name: output for output in outputs}

    assert term._joint_ids is selector
    assert torch.equal(outputs_by_name["binary_gripper_kp_gains"].ref, kp_gains[:, expected_indices])
    assert torch.equal(outputs_by_name["binary_gripper_kd_gains"].ref, kd_gains[:, expected_indices])
    expected_names = [[asset.joint_names[index] for index in expected_indices]]
    assert outputs_by_name["binary_gripper_kp_gains"].element_names == expected_names
    assert outputs_by_name["binary_gripper_kd_gains"].element_names == expected_names


def test_element_names_accept_warp_selector():
    """Test LEAPP write semantics resolve names from a Warp selector."""
    assert select_element_names(["joint_0", "joint_1", "joint_2"], _make_warp_selector()) == [
        "joint_0",
        "joint_2",
    ]
