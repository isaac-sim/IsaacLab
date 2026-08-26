# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused OVPhysX articulation ordering and derived-cache tests."""

from unittest.mock import Mock

import pytest
import torch
import warp as wp
from isaaclab_ov import tensor_types as TT
from isaaclab_ov.assets import Articulation
from isaaclab_ov.assets.articulation.articulation_data import ArticulationData

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.assets.articulation import ordering_kernels
from isaaclab.utils.warp.launch_cache import _WarpLaunchCache

pytestmark = pytest.mark.unit


def test_joint_dof_sign_resolution_traverses_instance_proxies() -> None:
    """Joint direction resolution must inspect joints below instance proxies."""
    source_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(source_stage, "/Robot")
    base = UsdGeom.Xform.Define(source_stage, "/Robot/base").GetPrim()
    link = UsdGeom.Xform.Define(source_stage, "/Robot/link").GetPrim()
    joint = UsdPhysics.RevoluteJoint.Define(source_stage, "/Robot/joint")
    joint.CreateBody0Rel().SetTargets([link.GetPath()])
    joint.CreateBody1Rel().SetTargets([base.GetPath()])

    stage = Usd.Stage.CreateInMemory()
    instance = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    instance.GetReferences().AddReference(source_stage.GetRootLayer().identifier, "/Robot")
    instance.SetInstanceable(True)
    articulation = Mock(
        cfg=Mock(prim_path="/World/Robot"),
        _joint_names=["joint"],
        _body_names=["base", "link"],
    )

    assert Articulation._resolve_joint_dof_signs(articulation, stage) == (-1,)


def test_ordering_install_and_invalidation_clear_recorded_reads() -> None:
    """Ordering replacement and scene invalidation must discard stale read launches."""

    class MinimalData(ArticulationData):
        def __dir__(self):
            return []

    class Buffer:
        timestamp = 1.0

    data = MinimalData.__new__(MinimalData)
    read_launch_cache = Mock()
    data._read_launch_cache = read_launch_cache
    data._configure_ordering_buffers = lambda: None
    data._make_jacobian_body_user_to_backend = lambda: object()
    data.joint_ordering = None
    data._body_com_jacobian_w = Buffer()
    data._mass_matrix = Buffer()
    data._gravity_compensation_forces = Buffer()

    data._apply_ordering_maps_after_resolve()

    read_launch_cache.clear.assert_called_once_with()
    assert data._body_com_jacobian_w.timestamp == -1.0
    assert data._mass_matrix.timestamp == -1.0
    assert data._gravity_compensation_forces.timestamp == -1.0

    data._is_primed = True
    data._sim_timestamp = 1.0
    data._invalidate_initialize_callback(None)

    assert read_launch_cache.clear.call_count == 2
    assert data._is_primed is False
    assert data._sim_timestamp == 0.0


def test_generalized_dynamics_gathers_both_axes_into_public_joint_order() -> None:
    """A nonidentity joint map must reorder both axes of the mass matrix."""

    class Buffer:
        def __init__(self):
            self.data = wp.zeros((1, 2, 2), dtype=wp.float32, device="cpu")
            self.timestamp = -1.0

    data = ArticulationData.__new__(ArticulationData)
    data.device = "cpu"
    data._sim_timestamp = 1.0
    data._read_launch_cache = _WarpLaunchCache("cpu")
    data.joint_ordering = object()
    data._jacobian_joint_user_to_backend = wp.array([1, 0], dtype=wp.int32, device="cpu")
    data._joint_dof_signs = wp.ones(2, dtype=wp.int32, device="cpu")
    data._has_reversed_joints = False
    data._num_base_dofs = 0
    backend_values = wp.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=wp.float32, device="cpu")
    backend_buffer = wp.zeros_like(backend_values)
    buffer = Buffer()
    data._binding_read = lambda tensor_type, destination: destination.assign(backend_values)

    data._refresh_generalized_dynamics_buffer(
        buffer,
        backend_buffer,
        TT.MASS_MATRIX,
        ordering_kernels.reorder_mass_matrix_backend_to_user,
    )

    torch.testing.assert_close(wp.to_torch(buffer.data), torch.tensor([[[4.0, 3.0], [2.0, 1.0]]]))
    assert buffer.timestamp == 1.0
