# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp
import yaml

leapp = pytest.importorskip("leapp")

from isaaclab.assets.articulation import BaseArticulation, BaseArticulationData
from isaaclab.envs import mdp
from isaaclab.envs.leapp_deployment_env import LeappDeploymentEnv, StateInputSpec
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors.camera import CameraData
from isaaclab.utils import images as image_utils
from isaaclab.utils import math as math_utils
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.leapp import utils as leapp_utils
from isaaclab.utils.leapp.export_annotator import ExportPatcher, export_function_patches
from isaaclab.utils.leapp.leapp_semantics import Kind
from isaaclab.utils.leapp.proxy import _ArticulationWriteProxy, _DataProxy, _EnvProxy
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
    assert semantics.kind == Kind.VECTOR3D
    assert semantics.axes == [None, leapp.Axis(leapp.AxisKind.COMPONENT, names=["x", "y", "z"])]
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
    assert semantics.kind == Kind.FRAME_ORIENTATION
    assert semantics.extra == {"isaaclab_connection": "state:robot:root_quat_w"}


@pytest.mark.parametrize(
    ("property_name", "source_name", "components"),
    [
        ("root_lin_vel_w", "root_com_lin_vel_w", ["x", "y", "z"]),
        ("root_ang_vel_w", "root_com_ang_vel_w", ["angular_x", "angular_y", "angular_z"]),
        ("root_vel_w", "root_com_vel_w", ["x", "y", "z", "angular_x", "angular_y", "angular_z"]),
    ],
)
def test_root_velocity_components_follow_leapp_conventions(monkeypatch, property_name, source_name, components):
    """Root velocity aliases retain their values and use the standard component vocabulary."""

    class RootData(SimpleNamespace):
        root_lin_vel_w = BaseArticulationData.root_lin_vel_w
        root_ang_vel_w = BaseArticulationData.root_ang_vel_w
        root_vel_w = BaseArticulationData.root_vel_w

    values = torch.arange(2 * len(components), dtype=torch.float32).reshape(2, -1)
    data = RootData(**{source_name: ProxyArray(wp.from_torch(values))})
    captured = _capture_leapp_inputs(monkeypatch)
    proxy = _DataProxy(data, "robot", "velocity", {}, {}, lambda name: name)

    torch.testing.assert_close(getattr(proxy, property_name).torch, values)
    assert len(captured) == 1
    semantics = captured[0][1]
    assert semantics.axes == [None, leapp.Axis(leapp.AxisKind.COMPONENT, names=components)]


def test_ray_caster_coordinates_use_component_axes(monkeypatch: pytest.MonkeyPatch):
    """World-frame sensor poses use built-in quantities and coordinate component axes."""
    from isaaclab.sensors.ray_caster import RayCasterData

    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    data = RayCasterData()
    data.create_buffers(2, 4, "cpu")
    proxy = _DataProxy(data, "ray_caster", "ray-task", {}, {}, lambda name: f"ray_{name}")

    assert proxy.pos_w.torch.shape == (2, 3)
    assert proxy.quat_w.torch.shape == (2, 4)
    position, orientation = [semantics for _, semantics in annotated_inputs]
    assert position.kind == Kind.FRAME_POSITION
    assert orientation.kind == Kind.FRAME_ORIENTATION
    assert position.axes == [None, leapp.Axis(leapp.AxisKind.COMPONENT, names=["x", "y", "z"])]
    assert orientation.axes == [None, leapp.Axis(leapp.AxisKind.COMPONENT, names=["qx", "qy", "qz", "qw"])]
    assert position.extra == {"isaaclab_connection": "state:ray_caster:pos_w"}
    assert orientation.extra == {"isaaclab_connection": "state:ray_caster:quat_w"}


@pytest.mark.parametrize("layout", ["physx-aggregate", "physx-filtered", "ovphysx", "newton"])
def test_contact_axes_follow_backend_tensor_layout(monkeypatch: pytest.MonkeyPatch, layout):
    """Contact poses retain body order and backend layout; force aliases keep their component axes."""
    common = dict(
        num_envs=2,
        num_sensors=2,
        history_length=1,
        track_pose=True,
        track_air_time=False,
        track_contact_points=False,
        track_friction_forces=True,
        device="cpu",
    )
    if layout == "newton":
        contact_data = pytest.importorskip("isaaclab_newton.sensors.contact_sensor.contact_sensor_data")
        data = contact_data.ContactSensorData()
        data.create_buffers(**common, num_filter_objects=1, generate_force_matrix=True)
        pose_shape, friction_shape = (2,), (2, 2)
    else:
        backend = "isaaclab_ov" if layout == "ovphysx" else "isaaclab_physx"
        contact_data = pytest.importorskip(f"{backend}.sensors.contact_sensor.contact_sensor_data")
        data = contact_data.ContactSensorData()
        if layout == "ovphysx":
            data.create_buffers(**common, num_filter_shapes=1)
        else:
            data.create_buffers(
                **common, num_filter_shapes=1, supports_total_contact_forces=layout == "physx-aggregate"
            )
        data._body_names = ["right_foot", "left_foot"]
        pose_shape = (2, 2)
        friction_shape = (2, 2, 1) if layout == "physx-filtered" else (2, 2)

    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    proxy = _DataProxy(data, "contact", "contact-task", {}, {}, lambda name: f"contact_{name}")
    warning = (
        pytest.warns(UserWarning, match="only provides filtered friction")
        if layout == "physx-filtered"
        else nullcontext()
    )
    with warning:
        assert proxy.pos_w.torch.shape == (*pose_shape, 3)
        assert proxy.quat_w.torch.shape == (*pose_shape, 4)
        assert proxy.friction_forces_w.torch.shape == (*friction_shape, 3)
    position, orientation, friction = [semantics for _, semantics in annotated_inputs]
    pose_axes = [leapp.Axis(leapp.AxisKind.SYSTEM)]
    if layout != "newton":
        pose_axes.append(leapp.Axis(leapp.AxisKind.ELEMENT, names=["right_foot", "left_foot"]))
    assert position.axes == pose_axes + [leapp.Axis(leapp.AxisKind.COMPONENT, names=["x", "y", "z"])]
    assert orientation.axes == pose_axes + [leapp.Axis(leapp.AxisKind.COMPONENT, names=["qx", "qy", "qz", "qw"])]
    if layout == "newton":
        with pytest.raises(NotImplementedError, match="pose_w is not supported"):
            _ = proxy.pose_w
    else:
        pose = proxy.pose_w.torch
        torch.testing.assert_close(pose[..., :3], data.pos_w.torch)
        torch.testing.assert_close(pose[..., 3:], data.quat_w.torch)
        assert annotated_inputs[-1][1].axes == pose_axes + [
            leapp.Axis(leapp.AxisKind.COMPONENT, names=["x", "y", "z", "qx", "qy", "qz", "qw"])
        ]
    assert friction.axes == [None] * len(friction_shape) + [leapp.Axis(leapp.AxisKind.COMPONENT, names=["x", "y", "z"])]


def test_camera_output_mapping_round_trips_through_deployment(monkeypatch: pytest.MonkeyPatch):
    """Test camera buffers are independently wired and resolved during deployment."""
    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    camera_data = CameraData()
    camera_data._output = {
        "rgb": ProxyArray(wp.from_torch(torch.zeros(1, 4, 4, 4, dtype=torch.uint8))),
        "depth": ProxyArray(wp.from_torch(torch.ones(1, 4, 4, 1, dtype=torch.float32))),
    }
    proxy = _DataProxy(
        camera_data,
        entity_name="camera",
        task_name="camera-task",
        property_resolution_cache={},
        cache={},
        input_name_resolver=lambda property_name: f"camera_{property_name}",
    )

    assert proxy.output["rgb"].torch.shape == (1, 4, 4, 4)
    assert proxy.output["depth"].torch.shape == (1, 4, 4, 1)
    _ = proxy.output["rgb"].torch  # Re-reading a buffer must not add another graph input.
    assert len(annotated_inputs) == 2
    assert set(semantics.name for _, semantics in annotated_inputs) == {"camera_output_rgb", "camera_output_depth"}
    assert {semantics.extra["isaaclab_connection"] for _, semantics in annotated_inputs} == {
        "state:camera:output.rgb",
        "state:camera:output.depth",
    }

    rgb_semantics = next(semantics for _, semantics in annotated_inputs if semantics.name == "camera_output_rgb")
    _, entity_name, property_name = rgb_semantics.extra["isaaclab_connection"].split(":", 2)
    env = object.__new__(LeappDeploymentEnv)
    env.scene = {entity_name: SimpleNamespace(data=camera_data)}
    env._input_mapping = {
        "camera-task/camera_output_rgb": StateInputSpec(entity_name=entity_name, property_name=property_name)
    }
    assert env._read_inputs() == {"camera-task/camera_output_rgb": camera_data.output["rgb"].torch}


def test_camera_pixels_drive_joint_target_through_export_and_deployment(tmp_path: Path):
    """Random camera pixels drive a joint target through the compiled graph and deployment mapping."""
    from leapp import InferenceManager, annotate

    writes = []

    class TargetAsset:
        set_joint_position_target_index = BaseArticulation.set_joint_position_target_index
        joint_names = ["cart_to_pole", "slider_to_cart"]
        actuators = SimpleNamespace(
            target_command=SimpleNamespace(set_position_index=lambda **kwargs: writes.append(kwargs))
        )

        def find_joints(self, names, preserve_order=False):
            assert preserve_order
            return [self.joint_names.index(name) for name in names], names

    asset = TargetAsset()
    pixels = torch.randint(0, 256, (1, 4, 4, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(13))
    camera_data = CameraData()
    camera_data._output = {"rgb": ProxyArray(wp.from_torch(pixels))}
    cache = {}
    camera_proxy = _DataProxy(camera_data, "camera", "camera-action", {}, cache, lambda name: f"camera_{name}")
    outputs = []
    action_proxy = _ArticulationWriteProxy(asset, "robot", "position", outputs, {}, set(), None)
    leapp.start("camera-action", save_path=str(tmp_path))
    try:
        for shift in (0, 17):
            pixels.add_(shift)
            cache.clear()
            outputs.clear()
            rgb = camera_proxy.output["rgb"].torch
            target = (rgb.float().mean(dim=(1, 2, 3)) / 255.0).unsqueeze(-1)
            action_proxy.set_joint_position_target_index(target=target, joint_ids=[1])
            annotate.output_tensors("camera-action", outputs, export_with="onnx-dynamo")
    finally:
        leapp.stop()
    leapp.compile_graph(visualize=False, validate=True)

    assert len(writes) == 2
    assert writes[-1]["joint_ids"] == [1]
    output = outputs[0]
    assert output.kind == Kind.JOINT_POSITION
    assert output.extra == {"isaaclab_connection": "write:robot:set_joint_position_target_index"}
    assert output.axes == [
        leapp.Axis(leapp.AxisKind.SYSTEM),
        leapp.Axis(leapp.AxisKind.ELEMENT, names=["slider_to_cart"]),
    ]

    pipeline = tmp_path / "camera-action" / "camera-action.yaml"
    description = yaml.safe_load(pipeline.read_text())
    manager = InferenceManager(str(pipeline))
    input_name = "camera-action/camera_output_rgb"
    output_name = "camera-action/position"
    assert manager.inputs == [input_name]
    manager.reset()
    dark_action = manager.run_policy({input_name: torch.zeros_like(pixels)})[output_name].clone()
    manager.reset()
    bright_action = manager.run_policy({input_name: torch.full_like(pixels, 255)})[output_name].clone()
    torch.testing.assert_close(dark_action, torch.zeros(1, 1))
    torch.testing.assert_close(bright_action, torch.ones(1, 1))

    env = object.__new__(LeappDeploymentEnv)
    env.scene = {"camera": SimpleNamespace(data=camera_data), "robot": asset}
    env._leapp_desc = description
    env.inference = manager
    env._input_mapping = {}
    env._output_mapping = {}
    env._resolve_io()
    pixels.fill_(255)
    manager.reset()
    env._write_outputs(manager.run_policy(env._read_inputs()))
    assert len(writes) == 3
    assert writes[-1]["joint_ids"] == [1]
    torch.testing.assert_close(writes[-1]["value"], torch.ones(1, 1))


@pytest.mark.parametrize(
    "frame_stack,channel_first,mean,dtype",
    [
        (1, True, None, torch.uint8),
        (2, True, 0.5, torch.uint8),
        (2, False, None, torch.uint8),
        (1, False, 0.5, torch.float32),
    ],
)
def test_rgb_observation_export_keeps_live_camera_input(
    tmp_path: Path, frame_stack: int, channel_first: bool, mean: float | None, dtype: torch.dtype
):
    """Test exported RGB observations retain live inputs and frame history across inference calls."""
    from leapp import InferenceManager, annotate

    rgba = torch.arange(64, dtype=dtype).reshape(1, 4, 4, 4)
    rgb = rgba[..., :3]
    camera_buffer = rgba.clone()[..., :3]
    changed_rgb = rgb.clone()
    changed_rgb[:, :2] = 0
    frames = [rgb, changed_rgb, rgb.flip(1)]
    camera_data = CameraData()
    camera_data._output = {"rgb": ProxyArray(wp.from_torch(camera_buffer))}
    camera = SimpleNamespace(data=camera_data)
    scene = _TestScene()
    scene.sensors = {"base_camera": camera}
    env = SimpleNamespace(num_envs=1, device="cpu", scene=scene)
    params = {
        "sensor_cfg": SceneEntityCfg("base_camera"),
        "normalize": True,
        "channel_first": channel_first,
        "frame_stack": frame_stack,
        "mean": mean,
    }
    cfg = ObservationTermCfg(func=mdp.image_rgb, params=params)
    term_cfg = SimpleNamespace(func=mdp.image_rgb(cfg, env), params=params, noise=None)
    obs_manager = SimpleNamespace(
        _group_obs_term_cfgs={"policy": [term_cfg]},
        _group_obs_term_names={"policy": ["image"]},
        compute=lambda *args, **kwargs: None,
    )
    cache = {}
    proxy_env = _EnvProxy(env, "camera-task", {}, cache)
    patcher = ExportPatcher(export_method="onnx-dynamo", required_obs_groups={"policy"})
    patcher.task_name = "camera-task"
    patcher._patch_observation_manager(obs_manager, proxy_env)
    patcher._patch_history_buffers(obs_manager)

    shape = (1, 3 * frame_stack, 4, 4) if channel_first else (1, 4, 4, 3 * frame_stack)
    destination = torch.empty(shape, dtype=torch.float32)
    leapp.start("camera-task", save_path=str(tmp_path), patching=export_function_patches())
    try:
        for frame in frames:
            camera_buffer.copy_(frame)
            cache.clear()
            observation = term_cfg.func(env, **params, out=destination)
            assert observation is destination
            annotate.output_tensors("camera-task", {"observation": observation}, export_with="onnx-dynamo")
    finally:
        leapp.stop()
    leapp.compile_graph(visualize=False, validate=True, strict=True)

    pipeline = tmp_path / "camera-task" / "camera-task.yaml"
    manager = InferenceManager(str(pipeline))
    input_name = "camera-task/base_camera_output_rgb"
    assert manager.inputs == [input_name]
    # Replay a different sequence and restart it to check feedback initialization as well as updates.
    fresh_frames = [rgb.flip(2), changed_rgb.flip(2), torch.full_like(rgb, 127)]
    for sequence in (fresh_frames, fresh_frames[::-1]):
        manager.reset()
        history = [sequence[0]] * frame_stack
        for frame in sequence:
            history = history[1:] + [frame]
            expected = torch.cat(history, dim=-1).float() / 255.0
            expected = expected - (expected.mean(dim=(1, 2), keepdim=True) if mean is None else mean)
            if channel_first:
                expected = expected.movedim(-1, 1)
            actual = manager.run_policy({input_name: frame.clone()})["camera-task/observation"]
            torch.testing.assert_close(actual, expected)


def test_rgb_function_patch_preserves_destination(tmp_path: Path):
    """Direct normalization writes into and returns the supplied output during export and replay."""
    from leapp import InferenceManager, annotate

    destination = torch.empty(1, 3, 2, 2)
    frames = [torch.full((1, 2, 2, 3), value, dtype=torch.uint8) for value in (0, 255)]
    leapp.start("normalize", save_path=str(tmp_path), patching=export_function_patches())
    try:
        for frame, expected in zip(frames, (-0.5, 0.5)):
            image = annotate.input_tensors("rgb", {"image": frame})
            result = image_utils.normalize_rgb(image, output_channel_dim=1, mean=0.5, out=destination)
            assert result is destination
            torch.testing.assert_close(destination, torch.full_like(destination, expected))
            annotate.output_tensors("rgb", {"normalized": destination}, export_with="onnx")
    finally:
        leapp.stop()
    leapp.compile_graph(visualize=False, validate=True, strict=True)

    manager = InferenceManager(str(tmp_path / "normalize" / "normalize.yaml"))
    image = torch.full((1, 2, 2, 3), 102, dtype=torch.uint8)
    actual = manager.run_policy({"rgb/image": image})["rgb/normalized"]
    torch.testing.assert_close(actual, torch.full((1, 3, 2, 2), -0.1))


def test_broadcast_function_patch_preserves_rotations_distributions_and_aliases(tmp_path: Path):
    """Original rotation/distribution helpers and broadcast view writes export and replay fresh inputs."""
    from leapp import InferenceManager, annotate

    def compute(quat, vec, loc, scale):
        forward = math_utils.quat_apply(quat, vec)
        distribution = torch.distributions.Normal(loc, scale, validate_args=False)
        scalar_scale = torch.distributions.Normal(loc, 0.7, validate_args=False)
        scalar_loc = torch.distributions.Normal(0.2, scale, validate_args=False)
        root = loc.clone()
        view, _ = torch.broadcast_tensors(root, scale)
        view.add_(0.125)
        return {
            "forward": forward,
            "inverse": math_utils.quat_apply_inverse(quat, vec),
            "log_prob": distribution.log_prob(forward),
            "entropy": distribution.entropy(),
            "scalar_scale": scalar_scale.entropy() + scalar_scale.mean,
            "scalar_loc": scalar_loc.entropy() + scalar_loc.mean,
            "mutated_root": root,
            "mutated_view": view,
        }

    original_broadcast = torch.broadcast_tensors
    generator = torch.Generator().manual_seed(42)
    samples = []
    for index in range(13):
        quat = torch.randn(2, 1, 4, generator=generator)
        quat /= quat.norm(dim=-1, keepdim=True)
        inputs = {
            "quat": quat,
            "vec": torch.randn(1, 3, 3, generator=generator) * (index + 1) * 0.13,
            "loc": torch.randn(2, 3, 3, generator=generator) * 0.3 + index * 0.05,
            "scale": torch.rand(3, generator=generator) * 0.5 + 0.1 + index * 0.02,
        }
        samples.append((inputs, compute(**inputs)))

    leapp.start("broadcast", save_path=str(tmp_path), max_cached_io=5, patching=export_function_patches())
    try:
        # Plain tensors continue to use the original functions during capture.
        inputs, expected = samples[0]
        for name, actual in compute(**inputs).items():
            torch.testing.assert_close(actual, expected[name])
        for inputs, expected in samples[:5]:
            values = annotate.input_tensors("compute", {name: value.clone() for name, value in inputs.items()})
            actual = compute(*values)
            for name, value in actual.items():
                torch.testing.assert_close(value, expected[name])
            annotate.output_tensors("compute", actual, export_with="onnx")
    finally:
        leapp.stop()
    assert torch.broadcast_tensors is original_broadcast
    leapp.compile_graph(visualize=False, validate=True, strict=True)

    manager = InferenceManager(str(tmp_path / "broadcast" / "broadcast.yaml"))
    for inputs, expected in samples[5:]:
        actual = manager.run_policy({f"compute/{name}": value.clone() for name, value in inputs.items()})
        for name, value in expected.items():
            torch.testing.assert_close(actual[f"compute/{name}"], value)


@pytest.mark.parametrize("stack_dim", [None, 1])
def test_observation_history_export_preserves_buffer_behavior(tmp_path: Path, stack_dim: int | None):
    """Test both history layouts, partial backfill, and compiled feedback against the runtime buffer."""
    from leapp import InferenceManager, annotate

    buffer = CircularBuffer(3, 2, "cpu", stack_dim=stack_dim)
    reference = CircularBuffer(3, 2, "cpu", stack_dim=stack_dim)
    patcher = ExportPatcher(export_method="onnx-dynamo", required_obs_groups={"policy"})
    patcher.task_name = "history"
    obs_manager = SimpleNamespace(
        _group_obs_term_cfgs={"policy": [SimpleNamespace(func=lambda env: None, history_length=3)]},
        _group_obs_term_names={"policy": ["positions"]},
        _group_obs_term_history_buffer={"policy": {"positions": buffer}},
    )
    patcher._patch_history_buffers(obs_manager)
    frames = [torch.arange(6, dtype=torch.float32).reshape(2, 3) + value for value in (0, 10, 30)]
    for frame in frames:
        buffer.append(frame)
        reference.append(frame)
        torch.testing.assert_close(buffer.buffer, reference.buffer)
        buffer.reset([0])
        reference.reset([0])
    # Capture with one reset environment and one populated history.

    leapp.start("history", save_path=str(tmp_path))
    try:
        for frame in frames:
            frame = annotate.input_tensors("history", {"frame": frame})
            buffer.append(frame)
            annotate.output_tensors("history", {"observation": buffer.buffer.clone()}, export_with="onnx-dynamo")
    finally:
        leapp.stop()
    leapp.compile_graph(visualize=False, validate=True)

    manager = InferenceManager(str(tmp_path / "history" / "history.yaml"))
    assert manager.inputs == ["history/frame"]
    for frame in frames[::-1]:
        reference.append(frame)
        actual = manager.run_policy({"history/frame": frame})["history/observation"]
        torch.testing.assert_close(actual, reference.buffer)


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


def test_deformable_nodal_positions_use_inherited_semantics(monkeypatch: pytest.MonkeyPatch):
    """Test PhysX nodal positions are registered with inherited input semantics."""
    from isaaclab_physx.assets.deformable_object.deformable_object_data import DeformableObjectData

    annotated_inputs = _capture_leapp_inputs(monkeypatch)
    nodal_pos_w = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    class DeformableView:
        count, max_simulation_nodes_per_body = nodal_pos_w.shape[:2]
        max_simulation_elements_per_body = 1
        max_collision_elements_per_body = 1

        def get_simulation_nodal_positions(self):
            return wp.from_torch(nodal_pos_w)

    root_view = DeformableView()
    data = DeformableObjectData(root_view, "cpu")
    proxy = _DataProxy(
        data,
        entity_name="deformable",
        task_name="Isaac-Lift-Soft-Franka",
        property_resolution_cache={},
        cache={},
        input_name_resolver=lambda property_name: f"deformable_{property_name}",
    )

    assert torch.equal(proxy.nodal_pos_w.torch, nodal_pos_w)

    assert len(annotated_inputs) == 1
    task_name, semantics = annotated_inputs[0]
    assert task_name == "Isaac-Lift-Soft-Franka"
    assert semantics.name == "deformable_nodal_pos_w"
    assert semantics.kind == Kind.FRAME_POSITION
    assert semantics.axes == [None, None, leapp.Axis(leapp.AxisKind.COMPONENT, names=["x", "y", "z"])]
    assert semantics.extra == {"isaaclab_connection": "state:deformable:nodal_pos_w"}


@pytest.mark.parametrize(
    ("command_type", "custom_axes"), [("velocity", False), ("pose", False), ("planar", False), ("velocity", True)]
)
def test_command_metadata_uses_the_actual_reference_frame(monkeypatch, command_type, custom_axes):
    """Command annotations use the backend root and leave yaw-only references unspecified."""
    from isaaclab.envs.mdp.commands import (
        UniformPose2dCommand,
        UniformPose2dCommandCfg,
        UniformPoseCommand,
        UniformPoseCommandCfg,
        UniformVelocityCommand,
        UniformVelocityCommandCfg,
    )

    robot = SimpleNamespace(
        body_names=["leg", "base"],
        backend_body_names=["base", "leg"],
        find_bodies=lambda name: ([0], [name]),
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        num_envs=1,
        device="cpu",
        sim=SimpleNamespace(vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda term: None)),
    )
    common = {"asset_name": "robot", "resampling_time_range": (1.0, 1.0)}
    if custom_axes:
        common["axes"] = [None, leapp.Axis("operator_controls", names=["forward", "sideways", "turn"])]
    if command_type == "velocity":
        cfg = UniformVelocityCommandCfg(
            **common,
            ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
        )
        command = UniformVelocityCommand(cfg, env)
    elif command_type == "pose":
        cfg = UniformPoseCommandCfg(
            **common,
            body_name="leg",
            ranges=UniformPoseCommandCfg.Ranges(
                pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), pos_z=(0.0, 0.0), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
            ),
        )
        command = UniformPoseCommand(cfg, env)
    else:
        cfg = UniformPose2dCommandCfg(
            **common,
            simple_heading=False,
            ranges=UniformPose2dCommandCfg.Ranges(pos_x=(0.0, 0.0), pos_y=(0.0, 0.0), heading=(0.0, 0.0)),
        )
        command = UniformPose2dCommand(cfg, env)
    env.command_manager = SimpleNamespace(get_term=lambda name: command, get_command=lambda name: command.command)
    term_cfg = SimpleNamespace(params={"command_name": "target"})
    captured = _capture_leapp_inputs(monkeypatch)
    patcher = ExportPatcher("onnx")
    patcher.task_name = "command"
    patcher._wrap_generated_commands(mdp.generated_commands, term_cfg)(env, "target")
    semantics = captured[0][1]
    assert semantics.is_setpoint is True
    assert (
        semantics.kind
        == {
            "velocity": leapp.Kind.FRAME_TWIST,
            "pose": leapp.Kind.FRAME_POSE,
            "planar": "command/body/pose",
        }[command_type]
    )
    if custom_axes:
        assert semantics.axes == common["axes"]
    else:
        names = {
            "velocity": ["x", "y", "angular_z"],
            "pose": ["x", "y", "z", "qx", "qy", "qz", "qw"],
            "planar": ["x", "y", "z", "heading"],
        }[command_type]
        assert semantics.axes == [leapp.Axis(leapp.AxisKind.SYSTEM), leapp.Axis(leapp.AxisKind.COMPONENT, names=names)]
    assert semantics.expressed_in == ("base" if command_type == "velocity" else None)
    assert semantics.reference == ("base" if command_type == "pose" else None)
