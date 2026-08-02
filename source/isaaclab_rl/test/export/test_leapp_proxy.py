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
        return semantics.ref

    monkeypatch.setattr(leapp_utils.annotate, "input_tensors", _record_input_tensor)
    return annotated_inputs


def test_static_action_outputs_select_tensor_joint_ids():
    """Test static action gains can be selected with tensor joint ids."""

    class _Buffer:
        def __init__(self, tensor: torch.Tensor):
            self.torch = tensor

    real_asset = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_stiffness=_Buffer(torch.arange(6.0).reshape(1, 6)),
            default_joint_damping=_Buffer(torch.arange(10.0, 16.0).reshape(1, 6)),
        ),
        actuators={},
        joint_names=[f"j{index}" for index in range(6)],
    )
    term = SimpleNamespace(_asset=real_asset, _joint_ids=torch.tensor([0, 2, 4], dtype=torch.long))
    action_manager = SimpleNamespace(_terms={"joint_pos": term})

    patcher = ExportPatcher(export_method="onnx-dynamo")
    values = patcher._collect_action_static_outputs(action_manager)

    assert [value.name for value in values] == ["joint_pos_kp_gains", "joint_pos_kd_gains"]
    assert torch.equal(values[0].ref, torch.tensor([[0.0, 2.0, 4.0]]))
    assert torch.equal(values[1].ref, torch.tensor([[10.0, 12.0, 14.0]]))
    assert values[0].element_names == [["j0", "j2", "j4"]]
    assert values[1].element_names == [["j0", "j2", "j4"]]


def test_direct_projected_gravity_b_read_preserves_vector3d_input(monkeypatch: pytest.MonkeyPatch):
    """Test direct data proxy reads keep projected gravity as its own semantic input."""
    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    data, _ = _make_articulation_data()

    proxy = _DataProxy(
        data,
        entity_name="robot",
        task_name="Isaac-Velocity-Flat-G1",
        property_resolution_cache={},
        cache={},
        input_name_resolver=lambda property_name: f"robot_{property_name}",
    )

    assert proxy.projected_gravity_b.torch.shape == (2, 3)

    assert len(annotated_inputs) == 1
    task_name, semantics = annotated_inputs[0]
    assert task_name == "Isaac-Velocity-Flat-G1"
    assert semantics.name == "robot_projected_gravity_b"
    assert semantics.kind == InputKindEnum.VECTOR3D
    assert semantics.extra == {"isaaclab_connection": "state:robot:projected_gravity_b"}


def test_projected_gravity_observation_exports_root_quat_w_input(monkeypatch: pytest.MonkeyPatch):
    """Test the projected-gravity observation is export-lowered through root quaternion."""
    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    data, root_pose_w = _make_articulation_data()
    scene = _TestScene({"robot": SimpleNamespace(data=data)})
    env = SimpleNamespace(scene=scene)
    proxy_env = _EnvProxy(env, "Isaac-Velocity-Flat-G1", {}, {})

    term_cfg = SimpleNamespace(func=mdp.projected_gravity, noise="noise")
    obs_manager = SimpleNamespace(_group_obs_term_cfgs={"policy": [term_cfg]}, compute=lambda *args, **kwargs: None)
    patcher = ExportPatcher(export_method="onnx-dynamo", required_obs_groups={"policy"})
    patcher.task_name = "Isaac-Velocity-Flat-G1"
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
    assert task_name == "Isaac-Velocity-Flat-G1"
    assert semantics.name == "robot_root_quat_w"
    assert semantics.kind == InputKindEnum.BODY_ROTATION
    assert semantics.extra == {"isaaclab_connection": "state:robot:root_quat_w"}
