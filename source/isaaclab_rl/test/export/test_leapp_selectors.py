# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for selector representations at LEAPP export boundaries."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import warp as wp

leapp = pytest.importorskip("leapp")

from isaaclab.envs.leapp_deployment_env import LeappDeploymentEnv
from isaaclab.utils.leapp.export_annotator import ExportPatcher
from isaaclab.utils.leapp.leapp_semantics import select_element_names


def _make_warp_selector():
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
def test_static_action_gains_accept_selector_representations(
    selector_factory: Callable[[], Any], expected_indices: list[int]
) -> None:
    """Gain tensors and element names use the same normalized selector."""
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
    assert select_element_names(asset.joint_names, selector) == expected_names[0]
    assert outputs_by_name["binary_gripper_kp_gains"].axes == [
        leapp.Axis(leapp.AxisKind.SYSTEM),
        leapp.Axis(leapp.AxisKind.ELEMENT, names=expected_names[0]),
    ]
    assert outputs_by_name["binary_gripper_kd_gains"].axes == [
        leapp.Axis(leapp.AxisKind.SYSTEM),
        leapp.Axis(leapp.AxisKind.ELEMENT, names=expected_names[0]),
    ]


@pytest.mark.parametrize("legacy", [False, True])
def test_deployment_preserves_joint_axis_order(legacy: bool):
    """New and saved legacy bundles read and write joint subsets in their declared order."""
    positions = torch.tensor([[10.0, 20.0, 30.0]])
    written = torch.zeros_like(positions)
    names = ["shoulder", "elbow", "wrist"]

    def find_joints(joint_names, preserve_order):
        assert preserve_order
        return [names.index(name) for name in joint_names], joint_names

    def write_target(target, joint_ids=None):
        written[:, joint_ids] = target

    entity = SimpleNamespace(
        joint_names=names,
        find_joints=find_joints,
        data=SimpleNamespace(joint_pos=SimpleNamespace(torch=positions)),
        write_target=write_target,
    )
    labels = (
        {"element_names": [["wrist", "shoulder"]]}
        if legacy
        else {"axes": [None, {"kind": "element", "names": ["wrist", "shoulder"]}]}
    )
    env = LeappDeploymentEnv.__new__(LeappDeploymentEnv)
    env.scene = {"robot": entity}
    env._input_mapping = {}
    env._output_mapping = {}
    env._leapp_desc = {"pipeline": {"inputs": {"policy": ["q"]}, "outputs": {"policy": ["target"]}}}
    env.inference = SimpleNamespace(
        nodes={
            "policy": SimpleNamespace(
                input_descriptions=[{"name": "q", "isaaclab_connection": "state:robot:joint_pos", **labels}],
                output_descriptions=[{"name": "target", "isaaclab_connection": "write:robot:write_target", **labels}],
            )
        }
    )
    env._resolve_io()
    torch.testing.assert_close(env._read_inputs()["policy/q"], torch.tensor([[30.0, 10.0]]))
    env._write_outputs({"policy/target": torch.tensor([[3.0, 1.0]])})
    torch.testing.assert_close(written, torch.tensor([[1.0, 0.0, 3.0]]))


def test_variable_osc_exports_only_actuator_gains(monkeypatch):
    """Seven joint gains are published while six Cartesian OSC gains remain internal."""
    from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

    kp, kd = torch.full((1, 7), 10.0), torch.full((1, 7), 2.0)
    joint_names = [f"joint_{index}" for index in range(7)]
    asset = SimpleNamespace(
        joint_names=joint_names,
        actuators={},
        data=SimpleNamespace(
            default_joint_stiffness=SimpleNamespace(torch=kp),
            default_joint_damping=SimpleNamespace(torch=kd),
        ),
    )
    osc = OperationalSpaceController(
        OperationalSpaceControllerCfg(target_types=["pose_abs"], impedance_mode="variable_kp"),
        num_envs=1,
        device="cpu",
    )
    manager = SimpleNamespace(
        _terms={"arm": SimpleNamespace(_asset=asset, _joint_ids=list(range(7)), _osc=osc)},
        process_action=lambda actions: None,
        apply_action=lambda: None,
    )
    outputs = []
    monkeypatch.setattr(
        leapp.annotate,
        "output_tensors",
        lambda name, tensors, static_outputs, **kwargs: outputs.extend([*tensors, *(static_outputs or [])]),
    )
    patcher = ExportPatcher("onnx")
    patcher._patch_action_manager_methods(manager)
    manager.process_action(torch.zeros(1, 7))
    manager.apply_action()

    expected = {"arm_kp_gains": kp, "arm_kd_gains": kd}
    assert {output.name for output in outputs} == expected.keys()
    for output in outputs:
        torch.testing.assert_close(output.ref, expected[output.name])
        assert output.axes == [leapp.Axis(leapp.AxisKind.SYSTEM), leapp.Axis(leapp.AxisKind.ELEMENT, names=joint_names)]
