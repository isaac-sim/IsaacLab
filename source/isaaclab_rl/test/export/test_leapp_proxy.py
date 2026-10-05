# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp

pytest.importorskip("leapp")

from isaaclab.assets.articulation import BaseArticulation, BaseArticulationData
from isaaclab.envs import leapp_deployment_env, mdp
from isaaclab.envs.leapp_deployment_env import LeappDeploymentEnv
from isaaclab.utils import math as math_utils
from isaaclab.utils.leapp import utils as leapp_utils
from isaaclab.utils.leapp.export_annotator import ExportPatcher
from isaaclab.utils.leapp.leapp_semantics import InputKindEnum
from isaaclab.utils.leapp.proxy import _ArticulationWriteProxy, _DataProxy, _EnvProxy
from isaaclab.utils.warp import ProxyArray


class _TestScene(dict):
    """Minimal scene mapping for LEAPP proxy tests."""

    sensors = {}


def test_deployment_env_reuses_and_delegates_manager_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test deployment reuses its registered environment and delegates task state."""
    pipeline_path = tmp_path / "pipeline.yaml"
    pipeline_path.write_text("pipeline:\n  inputs: {}\n  outputs: {}\n")
    closed = []
    base_env = SimpleNamespace(
        cfg=SimpleNamespace(),
        extras={},
        sim=SimpleNamespace(),
        scene={},
        event_manager=SimpleNamespace(),
        command_manager=SimpleNamespace(),
        has_rtx_sensors=False,
        num_envs=1,
        reset_buf=torch.zeros(1, dtype=torch.bool),
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(leapp_deployment_env, "InferenceManager", lambda _: SimpleNamespace(nodes={}))

    env = LeappDeploymentEnv(base_env, str(pipeline_path))

    assert env.scene is base_env.scene
    assert env.command_manager is base_env.command_manager
    assert env.reset_buf is base_env.reset_buf
    env.close()
    env.close()
    assert closed == [True]


def test_fixed_tendon_output_mapping_round_trips_through_deployment():
    """Test fixed-tendon targets retain their names and deployment selector."""

    class _TendonWriteSemantics:
        set_fixed_tendon_position_target_index = BaseArticulation.set_fixed_tendon_position_target_index

    class _TendonArticulation(_TendonWriteSemantics):
        joint_names = ["joint_0", "joint_1"]
        fixed_tendon_names = ["tendon_0", "tendon_1", "tendon_2"]
        data = SimpleNamespace()

        def __init__(self):
            self.writes = []

        def find_fixed_tendons(self, names, preserve_order=False):
            assert preserve_order
            return [self.fixed_tendon_names.index(name) for name in names], names

        def set_fixed_tendon_position_target_index(self, *, target, fixed_tendon_ids=None, env_ids=None):
            self.writes.append((target.clone(), fixed_tendon_ids, env_ids))

    asset = _TendonArticulation()
    outputs = []
    proxy = _ArticulationWriteProxy(
        real_asset=asset,
        entity_name="robot",
        term_name="tendon_pos",
        output_cache=outputs,
        method_resolution_cache={},
        captured_write_term_names=set(),
        data_proxy=_DataProxy(asset.data, "robot", "tendon-task", {}, {}, lambda name: name),
    )
    target = torch.tensor([[0.2, 0.4]])
    tendon_ids = torch.tensor([2, 0], dtype=torch.int32)

    proxy.set_fixed_tendon_position_target_index(target=target, fixed_tendon_ids=tendon_ids)

    assert len(outputs) == 1
    semantics = outputs[0]
    assert semantics.name == "tendon_pos"
    assert semantics.kind == "target/tendon/position"
    assert semantics.element_names == [["tendon_2", "tendon_0"]]
    assert semantics.extra == {"isaaclab_connection": "write:robot:set_fixed_tendon_position_target_index"}

    asset.writes.clear()
    env = object.__new__(LeappDeploymentEnv)
    env.scene = {"robot": asset}
    env.inference = SimpleNamespace(
        nodes={
            "tendon-task": SimpleNamespace(
                input_descriptions=[],
                output_descriptions=[{"name": semantics.name, **semantics.to_dict()}],
            )
        }
    )
    env._leapp_desc = {"pipeline": {"inputs": {}, "outputs": {"tendon-task": [semantics.name]}}}
    env._input_mapping = {}
    env._output_mapping = {}
    env._resolve_io()
    env._write_outputs({"tendon-task/tendon_pos": target})

    assert len(asset.writes) == 1
    written_target, written_ids, written_env_ids = asset.writes[0]
    assert torch.equal(written_target, target)
    assert written_ids == [2, 0]
    assert written_env_ids is None

    tendon_term = SimpleNamespace(_asset=asset, _tendon_ids=tendon_ids)
    assert (
        ExportPatcher("onnx-dynamo")._collect_action_static_outputs(SimpleNamespace(_terms={"tendon_pos": tendon_term}))
        == []
    )


class _ArticulationDataSemantics:
    """Production semantic properties used by the minimal test data object."""

    projected_gravity_b = BaseArticulationData.projected_gravity_b
    root_quat_w = BaseArticulationData.root_quat_w


class _ArticulationData(_ArticulationDataSemantics):
    """Minimal articulation data required by LEAPP proxy tests."""

    def __init__(self, root_quat_w: ProxyArray, projected_gravity_b: ProxyArray):
        self._root_quat_w = root_quat_w
        self._projected_gravity_b = projected_gravity_b

    @property
    def root_quat_w(self) -> ProxyArray:
        return self._root_quat_w

    @property
    def projected_gravity_b(self) -> ProxyArray:
        return self._projected_gravity_b


def _make_articulation_data() -> tuple[_ArticulationData, torch.Tensor]:
    """Create the minimal articulation data required by LEAPP proxy tests."""

    root_pose_w = torch.zeros(2, 7, dtype=torch.float32)
    root_pose_w[:, 6] = 1.0
    root_pose_w[1, 3] = math.sin(math.pi / 4.0)
    root_pose_w[1, 6] = math.cos(math.pi / 4.0)
    gravity_w = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32).expand(2, 3)
    data = _ArticulationData(
        root_quat_w=ProxyArray(wp.from_torch(root_pose_w[:, 3:7])),
        projected_gravity_b=ProxyArray(wp.from_torch(math_utils.quat_apply_inverse(root_pose_w[:, 3:7], gravity_w))),
    )
    return data, root_pose_w


def _capture_leapp_inputs(monkeypatch: pytest.MonkeyPatch) -> list:
    """Capture LEAPP input annotations while returning their tensor references."""
    annotated_inputs = []

    def _record_input_tensor(task_name, semantics):
        annotated_inputs.append((task_name, semantics))
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


def test_named_last_action_observations_use_independent_feedback_states(monkeypatch: pytest.MonkeyPatch):
    """Test named action terms are registered and updated as independent feedback states."""
    full_action = torch.arange(12, dtype=torch.float32).reshape(2, 6)
    terms = {
        "arm": SimpleNamespace(raw_actions=full_action[:, :2]),
        "hand": SimpleNamespace(raw_actions=full_action[:, 2:]),
    }
    action_manager = SimpleNamespace(
        action=full_action,
        _action=full_action,
        get_term=terms.__getitem__,
        process_action=lambda action: None,
        apply_action=lambda: None,
    )
    env = SimpleNamespace(action_manager=action_manager)
    state_payloads = {}
    state_updates = []

    def _record_state_tensors(task_name, tensors):
        assert task_name == "multi-term-task"
        state_payloads.update(tensors)
        return next(iter(tensors.values()))

    def _record_update_state(task_name, tensors):
        assert task_name == "multi-term-task"
        state_updates.append(tensors)
        return tuple(tensors.values())

    monkeypatch.setattr(leapp_utils.annotate, "state_tensors", _record_state_tensors)
    monkeypatch.setattr(leapp_utils.annotate, "update_state", _record_update_state)
    monkeypatch.setattr(leapp_utils.annotate, "output_tensors", lambda *args, **kwargs: None)
    patcher = ExportPatcher(export_method="onnx-dynamo")
    patcher.task_name = "multi-term-task"
    monkeypatch.setattr(patcher, "_collect_action_outputs", lambda action_manager: [])
    monkeypatch.setattr(patcher, "_collect_processed_action_fallbacks", lambda action_manager: [])
    monkeypatch.setattr(patcher, "_collect_action_static_outputs", lambda action_manager, fallback_terms: [])
    wrapped = patcher._wrap_last_action(mdp.last_action)

    assert torch.equal(wrapped(env, "arm"), full_action[:, :2])
    assert torch.equal(wrapped(env, "hand"), full_action[:, 2:])
    patcher._patch_action_manager_methods(action_manager)
    patcher._pending_action_output_export = True
    action_manager.apply_action()

    assert set(state_payloads) == {"last_action_arm", "last_action_hand"}
    assert len(state_updates) == 1
    assert set(state_updates[0]) == set(state_payloads)
    assert torch.equal(state_updates[0]["last_action_arm"], state_payloads["last_action_arm"])
    assert torch.equal(state_updates[0]["last_action_hand"], state_payloads["last_action_hand"])
