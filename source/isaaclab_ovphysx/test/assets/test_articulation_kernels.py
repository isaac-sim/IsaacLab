# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVPhysX articulation Warp kernels."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_ovphysx.assets import kernels as shared_kernels
from isaaclab_ovphysx.assets.articulation import kernels as articulation_kernels


def _selector(values: list[int], dtype: type) -> wp.array:
    """Create a CPU Warp selector with the requested integer width."""
    return wp.array(values, dtype=dtype, device="cpu")


def _articulation_class():
    """Import the OVPhysX articulation class without unrelated config deprecations."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "^'RigidBodyMaterialCfg' is deprecated and will be removed in 5\\.0\\. Use "
                "'isaaclab_physx\\.sim\\.spawners\\.materials\\.PhysxRigidBodyMaterialCfg' for PhysX properties, "
                "or 'isaaclab\\.sim\\.spawners\\.materials\\.RigidBodyMaterialBaseCfg' for solver-common properties "
                "only\\.$"
            ),
            category=DeprecationWarning,
        )
        from isaaclab_ovphysx.assets.articulation.articulation import Articulation

    return Articulation


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize(
    "kernel_name",
    [
        "set_root_link_pose_to_sim_index",
        "set_root_com_pose_to_sim_index",
        "set_root_com_velocity_to_sim_index",
        "set_root_link_velocity_to_sim_index",
    ],
)
def test_root_index_workers_accept_selector_widths(kernel_name: str, env_dtype: type) -> None:
    """Launch every root-index worker with either supported environment selector width."""
    env_ids = _selector([1, 0], env_dtype)
    if "pose" in kernel_name:
        data = wp.array(
            np.asarray([[11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            dtype=wp.transformf,
            device="cpu",
        )
        root_link_pose = wp.zeros(2, dtype=wp.transformf, device="cpu")
        if kernel_name == "set_root_link_pose_to_sim_index":
            wp.launch(
                getattr(shared_kernels, kernel_name),
                dim=2,
                inputs=[data, env_ids],
                outputs=[root_link_pose],
                device="cpu",
            )
            output = root_link_pose
        else:
            body_com_pose = wp.array(
                np.asarray([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]] * 2),
                dtype=wp.transformf,
                device="cpu",
            )
            root_com_pose = wp.zeros(2, dtype=wp.transformf, device="cpu")
            wp.launch(
                getattr(shared_kernels, kernel_name),
                dim=2,
                inputs=[data, body_com_pose, env_ids],
                outputs=[root_com_pose, root_link_pose],
                device="cpu",
            )
            output = root_com_pose
    else:
        data = wp.array(
            np.asarray([[11.0, 12.0, 13.0, 14.0, 15.0, 16.0], [21.0, 22.0, 23.0, 24.0, 25.0, 26.0]]),
            dtype=wp.spatial_vectorf,
            device="cpu",
        )
        root_com_velocity = wp.zeros(2, dtype=wp.spatial_vectorf, device="cpu")
        body_acceleration = wp.zeros((2, 1), dtype=wp.spatial_vectorf, device="cpu")
        if kernel_name == "set_root_com_velocity_to_sim_index":
            wp.launch(
                getattr(shared_kernels, kernel_name),
                dim=2,
                inputs=[data, env_ids, 1],
                outputs=[root_com_velocity, body_acceleration],
                device="cpu",
            )
            output = root_com_velocity
        else:
            body_com_pose = wp.array(
                np.asarray([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]] * 2),
                dtype=wp.transformf,
                device="cpu",
            )
            root_link_pose = wp.array(
                np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2),
                dtype=wp.transformf,
                device="cpu",
            )
            root_link_velocity = wp.zeros(2, dtype=wp.spatial_vectorf, device="cpu")
            wp.launch(
                getattr(shared_kernels, kernel_name),
                dim=2,
                inputs=[data, body_com_pose, root_link_pose, env_ids, 1],
                outputs=[root_link_velocity, root_com_velocity, body_acceleration],
                device="cpu",
            )
            output = root_link_velocity

    np.testing.assert_array_equal(output.numpy(), data.numpy()[[1, 0]])


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize("item_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize(
    "kernel_name",
    [
        "write_2d_data_to_buffer_with_indices",
        "write_joint_state_to_buffer_with_indices",
        "write_joint_position_limit_to_buffer_index",
        "clamp_default_joint_pos_and_update_soft_limits_index",
        "write_joint_friction_data_to_buffer_index",
    ],
)
def test_item_index_workers_accept_selector_widths(kernel_name: str, env_dtype: type, item_dtype: type) -> None:
    """Launch every item-index worker with all supported selector-width combinations."""
    env_ids = _selector([1, 0], env_dtype)
    item_ids = _selector([2, 0], item_dtype)
    if kernel_name == "write_2d_data_to_buffer_with_indices":
        data = wp.array(np.asarray([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32), device="cpu")
        output = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
        wp.launch(
            shared_kernels.write_2d_data_to_buffer_with_indices,
            dim=(2, 2),
            inputs=[data, env_ids, item_ids],
            outputs=[output],
            device="cpu",
        )
        expected = np.asarray([[22.0, -1.0, 21.0], [12.0, -1.0, 11.0]], dtype=np.float32)
        np.testing.assert_array_equal(output.numpy(), expected)
    elif kernel_name == "write_joint_state_to_buffer_with_indices":
        position = wp.array(np.asarray([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32), device="cpu")
        velocity = wp.array(np.asarray([[111.0, 112.0], [121.0, 122.0]], dtype=np.float32), device="cpu")
        outputs = [wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu") for _ in range(4)]
        wp.launch(
            shared_kernels.write_joint_state_to_buffer_with_indices,
            dim=(2, 2),
            inputs=[position, velocity, env_ids, item_ids],
            outputs=outputs,
            device="cpu",
        )
        expected_position = np.asarray([[22.0, -1.0, 21.0], [12.0, -1.0, 11.0]], dtype=np.float32)
        expected_velocity = np.asarray([[122.0, -1.0, 121.0], [112.0, -1.0, 111.0]], dtype=np.float32)
        np.testing.assert_array_equal(outputs[0].numpy(), expected_position)
        np.testing.assert_array_equal(outputs[1].numpy(), expected_velocity)
        np.testing.assert_array_equal(outputs[2].numpy(), expected_velocity)
        np.testing.assert_array_equal(outputs[3].numpy(), (expected_position != -1.0).astype(np.float32) - 1.0)
    elif kernel_name == "write_joint_position_limit_to_buffer_index":
        data = wp.array(
            np.asarray([[[11.0, 12.0], [13.0, 14.0]], [[21.0, 22.0], [23.0, 24.0]]], dtype=np.float32),
            device="cpu",
        )
        output = wp.zeros((2, 3), dtype=wp.vec2f, device="cpu")
        wp.launch(
            shared_kernels.write_joint_position_limit_to_buffer_index,
            dim=(2, 2),
            inputs=[data, env_ids, item_ids],
            outputs=[output],
            device="cpu",
        )
        assert output.numpy()[0, 0].tolist() == [23.0, 24.0]
        assert output.numpy()[1, 2].tolist() == [11.0, 12.0]
    elif kernel_name == "clamp_default_joint_pos_and_update_soft_limits_index":
        limits = wp.array(
            np.asarray([[[-1.0, 1.0]] * 3, [[-2.0, 2.0]] * 3], dtype=np.float32),
            dtype=wp.vec2f,
            device="cpu",
        )
        defaults = wp.array(np.asarray([[3.0, 0.0, -3.0], [4.0, 0.0, -4.0]], dtype=np.float32), device="cpu")
        soft_limits = wp.zeros((2, 3), dtype=wp.vec2f, device="cpu")
        clamped_count = wp.zeros(1, dtype=wp.int32, device="cpu")
        wp.launch(
            articulation_kernels.clamp_default_joint_pos_and_update_soft_limits_index,
            dim=(2, 2),
            inputs=[limits, env_ids, item_ids, 0.5],
            outputs=[defaults, soft_limits, clamped_count],
            device="cpu",
        )
        assert clamped_count.numpy()[0] == 4
        assert defaults.numpy()[0, 0] == pytest.approx(1.0)
        assert defaults.numpy()[1, 2] == pytest.approx(-2.0)
    else:
        static = wp.array(np.asarray([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32), device="cpu")
        dynamic = wp.array(np.asarray([[111.0, 112.0], [121.0, 122.0]], dtype=np.float32), device="cpu")
        viscous = wp.array(np.asarray([[211.0, 212.0], [221.0, 222.0]], dtype=np.float32), device="cpu")
        output = wp.full((2, 3, 3), value=-1.0, dtype=wp.float32, device="cpu")
        wp.launch(
            articulation_kernels.write_joint_friction_data_to_buffer_index,
            dim=(2, 2),
            inputs=[static, dynamic, viscous, env_ids, item_ids],
            outputs=[output],
            device="cpu",
        )
        assert output.numpy()[0, 0].tolist() == [22.0, 122.0, 222.0]
        assert output.numpy()[1, 2].tolist() == [11.0, 111.0, 211.0]


@pytest.mark.parametrize(("torch_dtype", "warp_dtype"), [(torch.int32, wp.int32), (torch.int64, wp.int64)])
def test_selector_resolvers_preserve_item_width_and_alias_source(torch_dtype: torch.dtype, warp_dtype: type) -> None:
    """Alias int32/int64 item selectors while keeping environment IDs int32 for OVPhysX bindings."""
    Articulation = _articulation_class()
    articulation = type("ResolverStub", (), {"_device": "cpu"})()
    selector = torch.tensor([1, 0], dtype=torch_dtype)

    assert Articulation._resolve_env_ids(articulation, selector).dtype == wp.int32
    for resolver_name in (
        "_resolve_joint_ids",
        "_resolve_body_ids",
        "_resolve_fixed_tendon_ids",
        "_resolve_spatial_tendon_ids",
    ):
        resolved = getattr(Articulation, resolver_name)(articulation, selector)
        assert resolved.dtype == warp_dtype
        assert resolved.ptr == selector.data_ptr()


@pytest.mark.parametrize("state_write", [False, True], ids=["velocity", "state"])
def test_ordered_joint_writes_reject_unsupported_selector_before_launch(monkeypatch, state_write: bool) -> None:
    """Reject an unsupported item selector through the shared factory before calling Warp launch."""
    Articulation = _articulation_class()
    articulation = Articulation.__new__(Articulation)
    articulation._initialize_handle = None
    articulation._invalidate_initialize_handle = None
    articulation._prim_deletion_handle = None
    articulation._device = "cpu"
    articulation.assert_shape_and_dtype = lambda *args, **kwargs: None
    articulation._joint_user_to_backend_map = lambda: wp.zeros(1, dtype=wp.int32, device="cpu")

    def _buffer() -> SimpleNamespace:
        return SimpleNamespace(data=wp.zeros((1, 1), dtype=wp.float32, device="cpu"))

    joint_pos_backend = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    joint_vel_backend = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    articulation._data = SimpleNamespace(
        joint_ordering=None,
        _joint_pos_buf=_buffer(),
        _joint_vel_buf=_buffer(),
        _previous_joint_vel=wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
        _joint_acc=_buffer(),
        _get_joint_pos_write_buffer=lambda partial: joint_pos_backend,
        _get_joint_vel_write_buffer=lambda partial: joint_vel_backend,
    )
    monkeypatch.setattr(wp, "launch", lambda *args, **kwargs: pytest.fail("wp.launch must not be called"))

    env_ids = wp.array([0], dtype=wp.int32, device="cpu")
    joint_ids = wp.array([0], dtype=wp.float32, device="cpu")
    position = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    velocity = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    with pytest.raises(TypeError, match="signed 32-bit or signed 64-bit integers"):
        if state_write:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                articulation.write_joint_state_to_sim(position, velocity, joint_ids=joint_ids, env_ids=env_ids)
        else:
            articulation.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=joint_ids, env_ids=env_ids)
