# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("leapp")

from isaaclab.envs import mdp
from isaaclab.test.mock_interfaces.assets.mock_articulation import MockArticulationData
from isaaclab.utils import math as math_utils
from isaaclab.utils.leapp import leapp_input_tensor
from isaaclab.utils.leapp import utils as leapp_utils
from isaaclab.utils.leapp.export_annotator import ExportPatcher
from isaaclab.utils.leapp.leapp_semantics import InputKindEnum
from isaaclab.utils.leapp.proxy import _DataProxy, _EnvProxy


class _TestScene(dict):
    """Minimal scene mapping for LEAPP proxy tests."""

    sensors = {}


def _make_articulation_data() -> tuple[MockArticulationData, torch.Tensor]:
    """Create mock articulation data with a non-identity root orientation."""
    data = MockArticulationData(num_instances=2, num_joints=0, num_bodies=1, device="cpu")
    root_pose_w = torch.zeros(2, 7, dtype=torch.float32)
    root_pose_w[:, 6] = 1.0
    root_pose_w[1, 3] = math.sin(math.pi / 4.0)
    root_pose_w[1, 6] = math.cos(math.pi / 4.0)
    data.set_root_link_pose_w(root_pose_w)
    return data, root_pose_w


def _capture_leapp_inputs(monkeypatch: pytest.MonkeyPatch) -> list:
    """Capture LEAPP input annotations while returning their tensor references."""
    annotated_inputs = []

    def _record_input_tensor(task_name, semantics):
        annotated_inputs.append((task_name, semantics))
        if isinstance(semantics, dict):
            assert len(semantics) == 1
            return next(iter(semantics.values()))
        return semantics.ref

    monkeypatch.setattr(leapp_utils.annotate, "input_tensors", _record_input_tensor)
    return annotated_inputs


def test_direct_projected_gravity_b_read_preserves_vector3d_input(monkeypatch: pytest.MonkeyPatch):
    """Test direct data proxy reads keep projected gravity as its own semantic input."""
    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    data, _ = _make_articulation_data()

    proxy = _DataProxy(
        data,
        entity_name="robot",
        task_name="Isaac-Velocity-Flat-G1-v0",
        property_resolution_cache={},
        cache={},
        input_name_resolver=lambda property_name: f"robot_{property_name}",
    )

    assert proxy.projected_gravity_b.torch.shape == (2, 3)

    assert len(annotated_inputs) == 1
    task_name, semantics = annotated_inputs[0]
    assert task_name == "Isaac-Velocity-Flat-G1-v0"
    assert semantics.name == "robot_projected_gravity_b"
    assert semantics.kind == InputKindEnum.VECTOR3D
    assert semantics.extra == {"isaaclab_connection": "state:robot:projected_gravity_b"}


def test_projected_gravity_observation_exports_root_quat_w_input(monkeypatch: pytest.MonkeyPatch):
    """Test the projected-gravity observation is export-lowered through root quaternion."""
    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    data, root_pose_w = _make_articulation_data()
    scene = _TestScene({"robot": SimpleNamespace(data=data)})
    env = SimpleNamespace(scene=scene)
    proxy_env = _EnvProxy(env, "Isaac-Velocity-Flat-G1-v0", {}, {})

    term_cfg = SimpleNamespace(func=mdp.projected_gravity, noise="noise")
    obs_manager = SimpleNamespace(_group_obs_term_cfgs={"policy": [term_cfg]}, compute=lambda *args, **kwargs: None)
    patcher = ExportPatcher(export_method="onnx-dynamo", required_obs_groups={"policy"})
    patcher.task_name = "Isaac-Velocity-Flat-G1-v0"
    patcher._patch_observation_manager(obs_manager, proxy_env)

    projected_gravity_b = term_cfg.func(env)

    expected = math_utils.quat_apply_inverse(
        root_pose_w[:, 3:7],
        torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32).expand(2, 3),
    )
    assert torch.allclose(projected_gravity_b, expected)
    assert term_cfg.noise is None

    assert len(annotated_inputs) == 1
    task_name, semantics = annotated_inputs[0]
    assert task_name == "Isaac-Velocity-Flat-G1-v0"
    assert semantics.name == "robot_root_quat_w"
    assert semantics.kind == InputKindEnum.BODY_ROTATION
    assert semantics.extra == {"isaaclab_connection": "state:robot:root_quat_w"}


def test_manual_input_tensor_alias_survives_trace_cache_clear(monkeypatch: pytest.MonkeyPatch):
    """Test manual inputs can satisfy later proxy reads after the trace cache is cleared."""
    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    data = MockArticulationData(num_instances=1, num_joints=7, num_bodies=0, device="cpu")
    joint_pos = torch.arange(7, dtype=torch.float32).reshape(1, 7)
    data.set_joint_pos(joint_pos)
    scene = _TestScene({"robot": SimpleNamespace(data=data)})
    env = SimpleNamespace(scene=scene, unwrapped=SimpleNamespace(spec=SimpleNamespace(id="Task-v0")))
    cache = {}
    proxy_env = _EnvProxy(env, "Task-v0", {}, cache)
    proxy_env.scene["robot"]

    selected_joint_pos = data.joint_pos.torch[:, :7]
    annotated_joint_pos = leapp_input_tensor(
        proxy_env,
        "robot_joint_pos",
        selected_joint_pos,
        kind=InputKindEnum.JOINT_POSITION,
        element_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
        connection="state:robot:joint_pos",
        cache=("robot", "joint_pos"),
    )
    cache.clear()

    action_side_joint_pos = proxy_env.scene["robot"].data.joint_pos.torch[:, list(range(7))]

    assert torch.allclose(action_side_joint_pos, annotated_joint_pos)
    assert len(annotated_inputs) == 1
    semantics = annotated_inputs[0][1]
    assert semantics.name == "robot_joint_pos"
    assert semantics.ref is selected_joint_pos
    assert semantics.kind == InputKindEnum.JOINT_POSITION
    assert semantics.element_names == [["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]]
    assert semantics.extra == {"isaaclab_connection": "state:robot:joint_pos"}
