# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from types import SimpleNamespace

import pytest
import torch
import warp as wp

pytest.importorskip("leapp")

from isaaclab.assets.articulation import BaseArticulationData
from isaaclab.envs import mdp
from isaaclab.envs.leapp_deployment_env import LeappDeploymentEnv
from isaaclab.managers import ManagerTermBase
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.utils import math as math_utils
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.leapp import utils as leapp_utils
from isaaclab.utils.leapp.export_annotator import ExportPatcher
from isaaclab.utils.leapp.leapp_semantics import InputKindEnum
from isaaclab.utils.leapp.proxy import _DataProxy, _EnvProxy
from isaaclab.utils.warp import ProxyArray


class _TestScene(dict):
    """Minimal scene mapping for LEAPP proxy tests."""

    sensors = {}


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


@pytest.mark.parametrize("as_proxy", [True, False])
def test_camera_output_is_a_live_export_and_deployment_input(monkeypatch, as_proxy):
    """Camera dictionary entries retain their connection and refresh between steps."""
    annotated = _capture_leapp_inputs(monkeypatch)
    frame = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4, 1)
    data = CameraData()
    data._output = {"distance_to_image_plane": ProxyArray(wp.from_torch(frame)) if as_proxy else frame}
    scene = _TestScene({"camera": SimpleNamespace(data=data)})
    cache = {}
    proxy = _EnvProxy(SimpleNamespace(scene=scene), "depth_policy", {}, cache)
    for offset in (0, 10):
        frame.add_(offset)
        cache.clear()
        output = proxy.scene["camera"].data.output
        assert list(output) == ["distance_to_image_plane"]
        value = output["distance_to_image_plane"]
        actual = value.torch if as_proxy else value
        torch.testing.assert_close(actual, frame)
        assert output["distance_to_image_plane"] is value
    assert len(annotated) == 2
    semantics = annotated[-1][1]
    assert semantics.extra == {"isaaclab_connection": "state:camera:output:distance_to_image_plane"}

    env = object.__new__(LeappDeploymentEnv)
    env.scene = scene
    env._input_mapping, env._output_mapping = {}, {}
    env._leapp_desc = {"pipeline": {"inputs": {"depth_policy": [semantics.name]}, "outputs": {}}}
    env.inference = SimpleNamespace(
        nodes={
            "depth_policy": SimpleNamespace(
                input_descriptions=[
                    {
                        "name": semantics.name,
                        **semantics.extra,
                    }
                ]
            )
        }
    )
    env._resolve_io()
    key = f"depth_policy/{semantics.name}"
    torch.testing.assert_close(env._read_inputs()[key], frame)
    frame.add_(3)
    torch.testing.assert_close(env._read_inputs()[key], frame)


@pytest.mark.parametrize("stack_dim", [None, 1, -1])
@pytest.mark.parametrize("length", [1, 3])
def test_exported_history_matches_circular_buffer_after_partial_reset(monkeypatch, stack_dim, length):
    """Exported history preserves stacking and first-observation backfill after reset."""
    monkeypatch.setattr(leapp_utils.annotate, "state_tensors", lambda _, tensors: next(iter(tensors.values())))
    monkeypatch.setattr(leapp_utils.annotate, "update_state", lambda _, tensors: next(iter(tensors.values())))
    source = CircularBuffer(length, 2, "cpu", stack_dim=stack_dim)
    exported = CircularBuffer(length, 2, "cpu", stack_dim=stack_dim)
    patcher = ExportPatcher("onnx-dynamo")
    patcher.task_name = "history_policy"
    patcher._patch_history_buffer_append(exported, "history")
    for step in range(8):
        if step in (2, 5):
            ids = [1] if step == 2 else None
            source.reset(ids)
            exported.reset(ids)
        data = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) + 20 * step
        source.append(data)
        exported.append(data)
        torch.testing.assert_close(exported.buffer, source.buffer)
        if stack_dim is not None:
            torch.testing.assert_close(exported.stacked, source.stacked)


def test_observation_term_owned_buffer_is_exported(monkeypatch):
    """Buffers owned by manager terms participate even without manager-level history."""
    term = object.__new__(ManagerTermBase)
    term._stack = CircularBuffer(3, 1, "cpu", stack_dim=1)
    manager = SimpleNamespace(
        _group_obs_term_cfgs={"depth": [SimpleNamespace(func=term, history_length=0)]},
        _group_obs_term_names={"depth": ["image"]},
    )
    captured = []
    patcher = ExportPatcher("onnx-dynamo", required_obs_groups={"depth"})
    monkeypatch.setattr(patcher, "_patch_history_buffer_append", lambda buffer, name: captured.append(buffer))
    patcher._patch_history_buffers(manager)
    assert captured == [term._stack]


def test_compiled_history_refills_from_new_inputs_after_reset(tmp_path):
    """Serialized feedback uses deployment frames, including after a subsequent reset."""
    import leapp

    buffer = CircularBuffer(3, 1, "cpu", stack_dim=1)
    patcher = ExportPatcher("onnx-dynamo")
    patcher.task_name = "policy"
    patcher._patch_history_buffer_append(buffer, "history")
    leapp.start("history", save_path=str(tmp_path), max_cached_io=4)
    try:
        for step in range(4):
            frame = leapp.annotate.input_tensors("policy", {"frame": torch.full((1, 1, 2, 3), float(step))})
            buffer.append(frame)
            leapp.annotate.output_tensors("policy", {"stack": buffer.stacked.clone()}, export_with="onnx-dynamo")
    finally:
        leapp.stop()
    leapp.compile_graph(validate=True, visualize=False)
    inference = leapp.InferenceManager(str(tmp_path / "history/history.yaml"))
    source = CircularBuffer(3, 1, "cpu", stack_dim=1)
    with torch.inference_mode():
        for step in range(9):
            if step in (0, 5):
                inference.reset()
                source.reset()
            frame = torch.full((1, 1, 2, 3), float(step + 20))
            source.append(frame)
            actual = inference.run_policy({"policy/frame": frame.clone()})["policy/stack"]
            torch.testing.assert_close(actual.cpu(), source.stacked)


def test_deployment_reset_does_not_zero_fresh_scene_state():
    """LEAPP can retain aliases to scene inputs and inference-mode output tensors."""
    pose = torch.ones(1, 4)
    with torch.inference_mode():
        output = torch.ones(1)

    def reset_inference():
        pose.zero_()
        output.zero_()

    env = object.__new__(LeappDeploymentEnv)
    env.cfg = SimpleNamespace(sim=SimpleNamespace(dt=0.01))
    env.sim = SimpleNamespace(device="cpu", forward=lambda: None)
    env.scene = SimpleNamespace(
        reset=lambda _: pose.fill_(2),
        write_data_to_sim=lambda: None,
        update=lambda **_: None,
    )
    env.inference = SimpleNamespace(reset=reset_inference)
    env.event_manager = env.command_manager = None
    env.has_rtx_sensors = False
    env._read_inputs = lambda: {"pose": pose}
    torch.testing.assert_close(env.reset()["pose"], torch.full_like(pose, 2))
