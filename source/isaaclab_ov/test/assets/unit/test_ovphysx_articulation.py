# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""OVPhysX articulation unit tests: data caches, joint directions, tendon scoping, kernels, and actuator control."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import warp as wp

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets import Articulation, kernels  # noqa: E402
from isaaclab_ov.assets.articulation import actuator_control  # noqa: E402
from isaaclab_ov.assets.articulation.actuator_control import OvPhysxActuatorControl  # noqa: E402
from isaaclab_ov.assets.articulation.articulation_data import ArticulationData  # noqa: E402
from isaaclab_ov.physics import OvPhysxManager  # noqa: E402
from isaaclab_ov.test.fixtures.views import MockOvPhysxBindingSet  # noqa: E402

from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
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


def _define_tendon_joint(stage: Usd.Stage, path: str, schema_name: str) -> None:
    """Define a revolute joint prim with a tendon schema marker."""
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    schemas = Sdf.TokenListOp()
    schemas.explicitItems = [schema_name]
    joint.GetPrim().SetMetadata("apiSchemas", schemas)


def _make_articulation_root_stage_usda() -> str:
    """Serialize one relevant articulation subtree and unrelated joints in memory."""
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/envs", "Xform")
    stage.DefinePrim("/World/envs/env_0", "Xform")
    stage.DefinePrim("/World/envs/env_0/Robot", "Xform")
    stage.DefinePrim("/World/envs/env_0/Robot/root", "Xform")
    stage.DefinePrim("/World/unrelated", "Xform")

    _define_tendon_joint(
        stage,
        "/World/envs/env_0/Robot/root/fixed_joint",
        "PhysxTendonAxisRootAPI:inst0",
    )
    _define_tendon_joint(
        stage,
        "/World/envs/env_0/Robot/root/spatial_joint",
        "PhysxTendonAttachmentRootAPI:inst0",
    )
    _define_tendon_joint(
        stage,
        "/World/unrelated/unrelated_fixed_joint",
        "PhysxTendonAxisRootAPI:inst0",
    )
    _define_tendon_joint(
        stage,
        "/World/unrelated/unrelated_spatial_joint",
        "PhysxTendonAttachmentLeafAPI:inst0",
    )

    return stage.Flatten().ExportToString()


def _make_articulation_shell() -> Articulation:
    """Create a minimal ovphysx articulation shell for tendon processing tests."""
    articulation = object.__new__(Articulation)
    bindings = MockOvPhysxBindingSet(
        num_instances=1,
        num_joints=2,
        num_bodies=2,
        num_fixed_tendons=1,
        num_spatial_tendons=1,
    )
    # The migrated Articulation reads tendon counts off its OvPhysxView; inject the mock
    # view over these bindings so the metadata passthrough resolves without a real view.
    object.__setattr__(articulation, "_root_view", bindings.view)
    object.__setattr__(articulation, "_articulation_root_path", "/World/envs/env_0/Robot/root")
    object.__setattr__(articulation, "_initialize_handle", None)
    object.__setattr__(articulation, "_invalidate_initialize_handle", None)
    object.__setattr__(articulation, "_prim_deletion_handle", None)
    object.__setattr__(articulation, "_debug_vis_handle", None)
    object.__setattr__(
        articulation,
        "_data",
        SimpleNamespace(
            _num_fixed_tendons=0,
            _num_spatial_tendons=0,
            fixed_tendon_names=[],
            spatial_tendon_names=[],
        ),
    )
    return articulation


def test_process_tendons_scopes_to_articulation_root():
    """Tendon discovery should ignore joints that live outside the current articulation subtree."""
    articulation = _make_articulation_shell()
    stage_usda = _make_articulation_root_stage_usda()
    old_stage_usda = OvPhysxManager._stage_usda
    OvPhysxManager._stage_usda = stage_usda
    try:
        articulation._process_tendons()
    finally:
        OvPhysxManager._stage_usda = old_stage_usda

    # the tendon is reported by its schema INSTANCE name, matching PhysX; scope leakage would
    # add the identically-named instance from /World/unrelated, giving two entries
    assert articulation.fixed_tendon_names == ["inst0"]
    assert articulation.spatial_tendon_names == ["spatial_joint"]


def _selector(values: list[int], dtype: type) -> wp.array:
    return wp.array(values, dtype=dtype, device="cpu")


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
def test_root_worker_accepts_selector_widths(env_dtype: type) -> None:
    env_ids = _selector([1, 0], env_dtype)
    data = wp.array(
        [[11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        dtype=wp.transformf,
        device="cpu",
    )
    output = wp.zeros(2, dtype=wp.transformf, device="cpu")
    sim_env_ids = wp.empty(2, dtype=wp.int32, device="cpu")
    kernel = kernels.set_root_link_pose_to_sim_index
    if env_dtype == wp.int64:
        kernel = kernels.set_root_link_pose_to_sim_index_kernel(env_ids)

    wp.launch(kernel, dim=2, inputs=[data, env_ids], outputs=[output, sim_env_ids], device="cpu")

    np.testing.assert_array_equal(output.numpy(), data.numpy()[[1, 0]])
    np.testing.assert_array_equal(sim_env_ids.numpy(), [1, 0])


@pytest.mark.parametrize(("env_dtype", "item_dtype"), [(wp.int32, wp.int32), (wp.int64, wp.int64)])
def test_item_worker_accepts_selector_widths(env_dtype: type, item_dtype: type) -> None:
    env_ids = _selector([1, 0], env_dtype)
    item_ids = _selector([2, 0], item_dtype)
    data = wp.array([[11.0, 12.0], [21.0, 22.0]], dtype=wp.float32, device="cpu")
    output = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    kernel = kernels.write_2d_data_to_buffer_with_indices
    if env_dtype != wp.int32 or item_dtype != wp.int32:
        kernel = kernels.write_2d_data_to_buffer_with_indices_kernel(env_ids, item_ids)

    wp.launch(kernel, dim=(2, 2), inputs=[data, env_ids, item_ids], outputs=[output], device="cpu")

    np.testing.assert_array_equal(output.numpy(), [[22.0, -1.0, 21.0], [12.0, -1.0, 11.0]])


def test_prepare_native_actuators_leaves_implicit_only_articulation_on_standard_path(monkeypatch):
    """Keep implicit-only articulations on the unchanged solver-drive path."""
    runtime_prepare_calls = []
    runtime = SimpleNamespace(
        prepare=lambda *args, **kwargs: runtime_prepare_calls.append(True), wrapper=None, adapter=None
    )
    articulation = SimpleNamespace(
        _sim_cfg=SimpleNamespace(use_newton_actuators=True),
        cfg=SimpleNamespace(prim_path="/World/Robot"),
    )
    monkeypatch.setattr(actuator_control, "PhysxActuatorRuntime", lambda *args, **kwargs: runtime)
    monkeypatch.setattr(actuator_control, "find_first_matching_prim", lambda _: None)

    control = OvPhysxActuatorControl(articulation)
    native_groups = control.prepare_native_actuators(
        collection=None,
        actuator_cfgs={"implicit": ImplicitActuatorCfg(joint_names_expr=["joint"], stiffness=10.0, damping=1.0)},
    )

    assert native_groups == set()
    assert not control.native_actuator_path_active
    assert not articulation._has_newton_actuators
    assert runtime_prepare_calls == []


@pytest.mark.parametrize(
    "module_name",
    [
        "isaaclab_physx.assets.articulation.actuator_control",
        "isaaclab_ov.assets.articulation.actuator_control",
    ],
)
def test_host_actuator_control_import_does_not_probe_optional_newton_runtime(monkeypatch, module_name):
    """Import host controls without probing an unrequested Newton optional dependency."""
    original_find_spec = importlib.util.find_spec

    def reject_newton_probe(name, *args, **kwargs):
        if name.startswith("isaaclab_newton"):
            raise AssertionError("host actuator-control import eagerly probed Newton")
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", reject_newton_probe)
    importlib.reload(importlib.import_module(module_name))
