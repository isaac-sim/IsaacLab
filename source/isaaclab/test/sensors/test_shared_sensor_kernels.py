# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Literal sensor math oracles independent of solver integration fixtures."""

import importlib
from itertools import combinations

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.sensors.pva.pva_data import PvaData as NewtonPvaData

from isaaclab.sensors.frame_transformer.kernels import frame_transformer_update_kernel
from isaaclab.sensors.imu.kernels import imu_update_kernel
from isaaclab.sensors.pva.kernels import make_pva_update, pva_reset_kernel
from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.unit


def test_imu_and_pva_transport_preserve_masks_and_distinct_gravity_contracts():
    with wp.ScopedDevice("cpu"):
        mask = wp.array([True, False], dtype=wp.bool)
        transforms = wp.array([[1, 2, 3, 0, 0, 0, 1]] * 2, dtype=wp.transformf)
        velocity = wp.array([[1, 2, 3, 0, 0, 2]] * 2, dtype=wp.spatial_vectorf)
        acceleration = wp.array([[4, 5, 6, 0, 0, 3]] * 2, dtype=wp.spatial_vectorf)
        com = wp.array([[0.25, 0, 0, 0, 0, 0, 1]] * 2, dtype=wp.transformf)
        offset = wp.array([[1.25, 0, 0]] * 2, dtype=wp.vec3f)
        orientation = wp.array([[0, 0, 2.0**-0.5, 2.0**-0.5]] * 2, dtype=wp.quatf)
        inputs = [mask, transforms, velocity, acceleration, com, offset, orientation]
        imu = [wp.full(2, wp.vec3f(99.0)) for _ in range(2)]
        wp.launch(imu_update_kernel, 2, inputs + [wp.vec3f(0, 0, 9.81)] + imu)
        for actual, expected in zip(imu, [[0, 0, 2], [8, 0, 15.81]], strict=True):
            np.testing.assert_allclose(actual.numpy()[0], expected, atol=3e-6)
            np.testing.assert_array_equal(actual.numpy()[1], 99.0)
        expected = {
            "pose": {"pose_w": [2.25, 2, 3, 0, 0, 2.0**-0.5, 2.0**-0.5]},
            "velocity": {"lin_vel_b": [4, -1, 3], "ang_vel_b": [0, 0, 2]},
            "acceleration": {"lin_acc_b": [8, 0, 6], "ang_acc_b": [0, 0, 3]},
            "projected_gravity": {"projected_gravity_b": [0, 0, -1]},
        }
        for subset in (s for n in range(4) for s in combinations(("velocity", "acceleration", "projected_gravity"), n)):
            data_types = frozenset(("pose", *subset))
            kernel = make_pva_update(data_types)
            data = NewtonPvaData(data_types)
            data.create_buffers(2, "cpu")
            for group in data_types:
                for field in expected[group]:
                    array = getattr(data, field).warp
                    array.fill_(array.dtype(99.0))
            selected_inputs = inputs.copy()
            if not data_types & {"velocity", "acceleration"}:
                selected_inputs[2] = selected_inputs[4] = None
            if "acceleration" not in data_types:
                selected_inputs[3] = None
            wp.launch(
                kernel,
                2,
                selected_inputs
                + [wp.vec3f(0, 0, -1), data.pose_w.warp, data._velocity_b, data._acceleration_b, data._gravity_b],
            )
            for group in data_types:
                for field, value in expected[group].items():
                    actual = getattr(data, field).warp.numpy()
                    np.testing.assert_allclose(actual[0], value, atol=3e-6)
                    np.testing.assert_array_equal(actual[1], 99.0)
            wp.launch(
                pva_reset_kernel,
                2,
                [mask, data.pose_w.warp, data._velocity_b, data._acceleration_b, data._gravity_b],
            )
            np.testing.assert_array_equal(data.pose_w.warp.numpy()[0], [0, 0, 0, 0, 0, 0, 1])
            for group in data_types - {"pose"}:
                for field in expected[group]:
                    value = [0, 0, -1] if group == "projected_gravity" else [0, 0, 0]
                    np.testing.assert_array_equal(getattr(data, field).warp.numpy()[0], value)


def test_pva_data_owns_views_and_reports_unavailable_outputs(monkeypatch):
    data = NewtonPvaData(["pose", "velocity"])
    assert data.pose_w is None
    assert data.lin_vel_b is None
    with pytest.raises(NotImplementedError, match="acceleration"):
        _ = data.lin_acc_b
    data.create_buffers(2, "cpu")
    pose = data.pose_w
    data.pos_w.warp.fill_(wp.vec3f(1, 2, 3))
    data.ang_vel_b.warp.fill_(wp.vec3f(4, 5, 6))
    np.testing.assert_array_equal(pose.warp.numpy(), [[1, 2, 3, 0, 0, 0, 1]] * 2)
    np.testing.assert_array_equal(data._velocity_b.numpy(), [[0, 0, 0, 4, 5, 6]] * 2)
    # Existing cached Torch views must observe mutations without a packing read.
    held = pose.torch
    data.quat_w.warp.fill_(wp.quatf(0, 0, 1, 0))
    torch.testing.assert_close(held[:, 3:], torch.tensor([[0.0, 0.0, 1.0, 0.0]] * 2))

    def forbid_launch(*args, **kwargs):
        raise AssertionError("Reading PVA data must not launch a kernel")

    with monkeypatch.context() as m:
        m.setattr(wp, "launch", forbid_launch)
        assert data.pose_w is pose
        np.testing.assert_array_equal(data.pos_w.warp.numpy(), [[1, 2, 3]] * 2)
    data.create_buffers(1, "cpu")
    assert data.pose_w is not pose
    np.testing.assert_array_equal(data.pose_w.warp.numpy(), [[0, 0, 0, 0, 0, 0, 1]])
    np.testing.assert_array_equal(held.numpy()[:, :3], [[1, 2, 3]] * 2)


def test_pva_data_hierarchy_remains_importable(monkeypatch):
    from isaaclab.sensors.pva.base_pva_data import BasePvaData
    from isaaclab.sensors.pva.pva_data import PvaData

    core = importlib.import_module("isaaclab.sensors.pva")
    assert core.BasePvaData is BasePvaData
    data_classes = {}
    for backend in ("newton", "ov", "physx"):
        package = importlib.import_module(f"isaaclab_{backend}.sensors.pva")
        data_class = package.PvaData
        data_classes[backend] = data_class
        assert issubclass(data_class, BasePvaData)
        assert data_class(["pose"]).data_types == frozenset({"pose"})
    assert issubclass(PvaData, BasePvaData)
    monkeypatch.setattr(PvaData, "_registry", {})
    monkeypatch.setattr(PvaData, "_get_backend", classmethod(lambda cls, *args, **kwargs: "newton"))
    assert isinstance(PvaData(["pose"]), data_classes["newton"])


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_pva_partial_outputs_replay_and_reset_aliases(device):
    data = NewtonPvaData(["pose", "acceleration"])
    data.create_buffers(2, device)
    pose = wp.array([[1, 2, 3, 0, 0, 0, 1]] * 2, dtype=wp.transformf, device=device)
    velocity = wp.array([[0, 0, 0, 0, 0, 2]] * 2, dtype=wp.spatial_vectorf, device=device)
    acceleration = wp.zeros(2, dtype=wp.spatial_vectorf, device=device)
    offset = wp.full(2, wp.vec3f(1, 0, 0), device=device)
    quat = wp.full(2, wp.quatf(0, 0, 0, 1), device=device)
    mask = wp.ones(2, dtype=wp.bool, device=device)
    args = [
        mask,
        pose,
        velocity,
        acceleration,
        pose,
        offset,
        quat,
        wp.vec3f(),
        data.pose_w.warp,
        data._velocity_b,
        data._acceleration_b,
        data._gravity_b,
    ]
    kernel = make_pva_update(data.data_types)
    wp.launch(kernel, 2, args, device=device)
    with wp.ScopedCapture(device=device) as capture:
        wp.launch(kernel, 2, args, device=device)
    pose.fill_(wp.transformf(wp.vec3f(0), wp.quatf(0, 0, 0, 1)))
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(data.lin_acc_b.warp.numpy(), [[-4, 0, 0]] * 2)
    np.testing.assert_allclose(data.pose_w.warp.numpy(), [[1, 0, 0, 0, 0, 0, 1]] * 2)
    held = data.pose_w.torch
    mask.assign(np.array([True, False]))
    wp.launch(
        pva_reset_kernel,
        2,
        [mask, data.pose_w.warp, data._velocity_b, data._acceleration_b, data._gravity_b],
        device=device,
    )
    torch.testing.assert_close(held, torch.tensor([[0.0, 0, 0, 0, 0, 0, 1], [1.0, 0, 0, 0, 0, 0, 1]], device=device))


def test_pva_rejects_unsupported_features_before_backend_initialization():
    from isaaclab.sensors.pva import BasePva, PvaCfg

    class PoseOnlyPva(BasePva):
        supported_data_types = frozenset({"pose"})

        def _initialize_impl(self):
            raise AssertionError("Invalid configuration must fail before initialization")

        def _update_buffers_impl(self, env_mask):
            raise AssertionError("Invalid configuration must fail before sampling")

    with pytest.raises(ValueError, match="does not support outputs.*acceleration"):
        PoseOnlyPva(PvaCfg(prim_path="/NoStage", data_types=["pose", "acceleration"]))
    with pytest.raises(ValueError, match="visualization requires"):
        PoseOnlyPva(PvaCfg(prim_path="/NoStage", data_types=["pose"], debug_vis=True))


def test_frame_outputs_preserve_coordinates_and_masked_rows():
    from isaaclab_physx.sensors.frame_transformer.frame_transformer_data import FrameTransformerData

    device = "cpu"
    with wp.ScopedDevice(device):
        mask = wp.array([True, False], dtype=wp.bool)
        s = 2.0**-0.5
        transforms = wp.array(
            [[1, 2, 3, 0, 0, s, s], [1, 4, 3, 0, 0, 0, 1], [2, 2, 3, 0, 0, 0, 1]], dtype=wp.transformf
        )
        source_ids = wp.array([0, 0], dtype=wp.int32)
        target_ids = wp.array([[1, 2], [1, 2]], dtype=wp.int32)
        offsets = wp.zeros(2, dtype=wp.vec3f)
        rotations = wp.array([[0, 0, 0, 1]] * 2, dtype=wp.quatf)
        data = FrameTransformerData()
        data.create_buffers(2, 2, ["first", "second"], device)
        outputs = [
            data._source_pos_w,
            data._source_quat_w,
            data._target_pos_w,
            data._target_quat_w,
            data._target_pos_source,
            data._target_quat_source,
        ]
        for output in outputs:
            output.fill_(output.dtype(99.0))
        wp.launch(
            frame_transformer_update_kernel,
            (2, 2),
            [mask, transforms, source_ids, target_ids, offsets, rotations, offsets, rotations] + outputs,
        )
        np.testing.assert_allclose(outputs[0].numpy()[0], [1, 2, 3], atol=1e-6)
        np.testing.assert_allclose(outputs[4].numpy()[0], [[2, 0, 0], [0, -1, 0]], atol=1e-6)
        np.testing.assert_allclose(outputs[5].numpy()[0], [[0, 0, -s, s]] * 2, atol=1e-6)
        for output in outputs:
            np.testing.assert_array_equal(output.numpy()[1], 99.0)
        np.testing.assert_allclose(data.source_pose_w.warp.numpy()[0], [1, 2, 3, 0, 0, s, s], atol=1e-6)
        np.testing.assert_allclose(
            data.target_pose_source.warp.numpy()[0], [[2, 0, 0, 0, 0, -s, s], [0, -1, 0, 0, 0, -s, s]], atol=1e-6
        )
        np.testing.assert_allclose(data.target_pose_w.warp.numpy()[0], [[1, 4, 3, 0, 0, 0, 1], [2, 2, 3, 0, 0, 0, 1]])


@pytest.mark.parametrize("backend", ["physx", "ov"])
@pytest.mark.parametrize("family", ["frame_transformer", "contact_sensor"])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_packed_sensor_readers_follow_buffers_through_capture_and_recreation(backend, family, device):
    """Exercise each real native-memory caller, including its own optional buffers and launch cache."""
    module = importlib.import_module(f"isaaclab_{backend}.sensors.{family}.{family}_data")
    if family == "frame_transformer":
        data = module.FrameTransformerData()
        fields = ("source_pose_w", "target_pose_w", "target_pose_source")
        allocation = dict(num_target_frames=2, target_frame_names=["first", "second"])
    else:
        data = module.ContactSensorData()
        fields = ("pose_w",)
        allocation = dict(
            num_sensors=2,
            num_filter_shapes=0,
            history_length=0,
            track_pose=True,
            track_air_time=False,
            track_contact_points=False,
            track_friction_forces=False,
        )
        if backend == "physx":
            allocation["supports_total_contact_forces"] = False

    data.create_buffers(num_envs=2, device=device, **allocation)
    for field in fields:
        pos = getattr(data, field.replace("pose", "pos")).warp
        quat = getattr(data, field.replace("pose", "quat")).warp
        pos.fill_(wp.vec3f(2.0))
        quat.fill_(wp.quat_identity())
        held = getattr(data, field)
        np.testing.assert_array_equal(
            held.warp.numpy(), np.broadcast_to([2, 2, 2, 0, 0, 0, 1], held.warp.numpy().shape)
        )
        with wp.ScopedCapture(device=device) as capture:
            assert getattr(data, field) is held
        pos.fill_(wp.vec3f(3.0))
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(held.warp.numpy()[..., :3], 3.0)
        # Ordinary cached reads must also refresh the held allocation.
        pos.fill_(wp.vec3f(5.0))
        assert getattr(data, field) is held
        np.testing.assert_array_equal(held.warp.numpy()[..., :3], 5.0)

    # Recreated buffers must replace every recorded command's bound pointers.
    previous = {field: getattr(data, field) for field in fields}
    data.create_buffers(num_envs=1, device=device, **allocation)
    for field in fields:
        getattr(data, field.replace("pose", "pos")).warp.fill_(wp.vec3f(9.0))
        getattr(data, field.replace("pose", "quat")).warp.fill_(wp.quat_identity())
        actual = getattr(data, field)
        assert actual is not previous[field]
        np.testing.assert_array_equal(actual.warp.numpy()[..., :3], 9.0)
        np.testing.assert_array_equal(previous[field].warp.numpy()[..., :3], 5.0)
    if family == "contact_sensor":
        data.create_buffers(num_envs=1, device=device, **(allocation | {"track_pose": False}))
        assert data.pose_w is None and data.pos_w is None and data.quat_w is None


def test_pva_storage_has_one_owner(source_checkout_root):
    """Keep PVA output storage in core while preserving the public backend data types."""
    import ast

    source = source_checkout_root / "source"
    core = source / "isaaclab/isaaclab/sensors/pva"
    assert (core / "base_pva_data.py").is_file()
    for path in (core / "pva_data.py", core / "kernels.py"):
        tree = ast.parse(path.read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(
            name and name.startswith(("isaaclab_newton", "isaaclab_ov", "isaaclab_physx")) for name in imports
        )
    for backend in ("newton", "ov", "physx"):
        package = source / f"isaaclab_{backend}/isaaclab_{backend}/sensors/pva"
        tree = ast.parse((package / "pva_data.py").read_text())
        data_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PvaData")
        assert [base.id for base in data_class.bases] == ["_PvaDataStorage"]
        assert not any(isinstance(node, ast.FunctionDef) for node in data_class.body)
        if backend != "newton":
            assert not (package / "kernels.py").exists()
