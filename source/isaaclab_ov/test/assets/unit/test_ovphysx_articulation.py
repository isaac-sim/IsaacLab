# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""OVPhysX articulation data-cache and joint-direction unit tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import warp as wp

from pxr import Usd, UsdGeom, UsdPhysics

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets import Articulation  # noqa: E402
from isaaclab_ov.assets.articulation.articulation_data import ArticulationData  # noqa: E402

from isaaclab.utils.warp.launch_cache import _WarpLaunchCache  # noqa: E402

pytestmark = pytest.mark.unit


def test_cached_read_launches_reset_on_ordering_and_invalidation():
    """Ordering installation and simulation invalidation should discard recorded reads."""

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


def test_static_property_reads_are_not_invalidated_by_simulation_steps():
    """Joint properties and body mass/inertia should be read once per invalidation, not once per step.

    On OVPhysX these are blocking CPU-only binding reads whose cost scales with the number of
    environments, so re-reading them every physics step made per-step consumers (such as the
    native actuator telemetry sync) host-bound. State buffers must still refresh every step.
    """

    class Buffer:
        def __init__(self, shape):
            self.data = wp.zeros(shape, dtype=wp.float32, device="cpu")
            self.timestamp = -1.0

    data = ArticulationData.__new__(ArticulationData)
    data.device = "cpu"
    data.num_instances = 1
    data.num_joints = 2
    data._sim_timestamp = 1.0
    data.body_ordering = None
    data._get_binding = lambda tensor_type: object()
    reads: list[int] = []
    data._binding_read = lambda tensor_type, dst: reads.append(tensor_type)

    # Joint properties: one read across several steps, one more after explicit invalidation
    # (simulation reinitialization).
    data.joint_ordering = None
    stiffness = Buffer((1, 2))
    for _ in range(3):
        data._sim_timestamp += 1.0
        data._read_joint_property_binding(TT.DOF_STIFFNESS, stiffness, None)
    assert reads.count(TT.DOF_STIFFNESS) == 1
    stiffness.timestamp = -1.0
    data._read_joint_property_binding(TT.DOF_STIFFNESS, stiffness, None)
    assert reads.count(TT.DOF_STIFFNESS) == 2

    # Body properties behave the same; body state buffers still refresh every step.
    mass, link_pose = Buffer((1, 2)), Buffer((1, 2))
    for _ in range(3):
        data._sim_timestamp += 1.0
        data._refresh_reordered_body_buffer(mass, None, TT.BODY_MASS, static=True)
        data._refresh_reordered_body_buffer(link_pose, None, TT.LINK_POSE)
    assert reads.count(TT.BODY_MASS) == 1
    assert reads.count(TT.LINK_POSE) == 3
    mass.timestamp = -1.0
    data._refresh_reordered_body_buffer(mass, None, TT.BODY_MASS, static=True)
    assert reads.count(TT.BODY_MASS) == 2

    # Under a non-identity joint ordering the property is gathered once, then served from cache.
    data.joint_ordering = SimpleNamespace(user_to_backend=wp.array([1, 0], dtype=wp.int32, device="cpu"))
    data._read_launch_cache = _WarpLaunchCache("cpu")
    user_buffer, backend_buffer = Buffer((1, 2)), Buffer((1, 2))
    data._binding_read = lambda tensor_type, dst: (reads.append(tensor_type), dst.assign([[1.0, 2.0]]))
    for _ in range(3):
        data._sim_timestamp += 1.0
        data._read_joint_property_binding(TT.DOF_DAMPING, user_buffer, backend_buffer)
    assert reads.count(TT.DOF_DAMPING) == 1
    torch.testing.assert_close(wp.to_torch(user_buffer.data), torch.tensor([[2.0, 1.0]]))


def test_joint_dof_sign_resolution_traverses_instance_proxies():
    """Resolve reversed joints inside an instanceable articulation."""
    source_stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(source_stage, "/Robot")
    UsdGeom.Xform.Define(source_stage, "/Robot/base")
    UsdGeom.Xform.Define(source_stage, "/Robot/link")
    joint = UsdPhysics.RevoluteJoint.Define(source_stage, "/Robot/joint")
    joint.GetBody0Rel().SetTargets(["/Robot/link"])
    joint.GetBody1Rel().SetTargets(["/Robot/base"])
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
