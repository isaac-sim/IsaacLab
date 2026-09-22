# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Mocked backend asset factories and shared helpers for the asset interface tests.

The factories build articulation, rigid-object, and rigid-object-collection shells on top of the
backend mock views so the interface contract can be exercised without Isaac Sim or a physics scene.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import warp as wp
from _iface_test_boot import simulation_app  # noqa: F401

from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.assets.rigid_object_collection.rigid_object_collection_cfg import RigidObjectCollectionCfg
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.wrench_composer import WrenchComposer

BACKENDS: list[str] = []
BACKEND_UNAVAILABLE_REASONS: dict[str, str] = {}

try:
    from isaaclab_physx.assets.articulation.articulation import Articulation as PhysXArticulation
    from isaaclab_physx.assets.articulation.articulation_data import ArticulationData as PhysXArticulationData
    from isaaclab_physx.assets.rigid_object.rigid_object import RigidObject as PhysXRigidObject
    from isaaclab_physx.assets.rigid_object.rigid_object_data import RigidObjectData as PhysXRigidObjectData
    from isaaclab_physx.assets.rigid_object_collection.rigid_object_collection import (
        RigidObjectCollection as PhysXRigidObjectCollection,
    )
    from isaaclab_physx.assets.rigid_object_collection.rigid_object_collection_data import (
        RigidObjectCollectionData as PhysXRigidObjectCollectionData,
    )
    from isaaclab_physx.physics import PhysxManager
    from isaaclab_physx.test.fixtures.views import MockArticulationViewWarp as PhysXMockArticulationViewWarp
    from isaaclab_physx.test.fixtures.views import MockRigidBodyViewWarp as PhysXMockRigidBodyViewWarp
except ImportError as error:
    BACKEND_UNAVAILABLE_REASONS["physx"] = f"{type(error).__name__}: {error}"
else:
    # PhysX data classes read gravity from the physics scene, which these tests never create.
    _mock_physics_sim_view = MagicMock()
    _mock_physics_sim_view.get_gravity.return_value = (0.0, 0.0, -9.81)
    PhysxManager.get_physics_sim_view = MagicMock(return_value=_mock_physics_sim_view)
    BACKENDS.append("physx")

try:
    import isaaclab_newton.assets.articulation.articulation_data as newton_articulation_data_module
    import isaaclab_newton.assets.rigid_object.rigid_object_data as newton_rigid_object_data_module
    import isaaclab_newton.assets.rigid_object_collection.rigid_object_collection as newton_collection_module
    import isaaclab_newton.assets.rigid_object_collection.rigid_object_collection_data as newton_collection_data_module
    from isaaclab_newton.assets.articulation.articulation import Articulation as NewtonArticulation
    from isaaclab_newton.assets.articulation.articulation_data import ArticulationData as NewtonArticulationData
    from isaaclab_newton.assets.rigid_object.rigid_object import RigidObject as NewtonRigidObject
    from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData as NewtonRigidObjectData
    from isaaclab_newton.assets.rigid_object_collection.rigid_object_collection import (
        RigidObjectCollection as NewtonRigidObjectCollection,
    )
    from isaaclab_newton.assets.rigid_object_collection.rigid_object_collection_data import (
        RigidObjectCollectionData as NewtonRigidObjectCollectionData,
    )
    from isaaclab_newton.test.fixtures.views import MockNewtonArticulationView, MockNewtonCollectionView
except ImportError as error:
    BACKEND_UNAVAILABLE_REASONS["newton"] = f"{type(error).__name__}: {error}"
else:
    BACKENDS.append("newton")

try:
    import ovphysx  # noqa: F401
    from isaaclab_ov.assets.articulation.articulation import Articulation as OvPhysxArticulation
    from isaaclab_ov.assets.articulation.articulation_data import ArticulationData as OvPhysxArticulationData
    from isaaclab_ov.assets.rigid_object.rigid_object import RigidObject as OvPhysxRigidObject
    from isaaclab_ov.assets.rigid_object.rigid_object_data import RigidObjectData as OvPhysxRigidObjectData
    from isaaclab_ov.assets.rigid_object_collection.rigid_object_collection import (
        RigidObjectCollection as OvPhysxRigidObjectCollection,
    )
    from isaaclab_ov.assets.rigid_object_collection.rigid_object_collection_data import (
        RigidObjectCollectionData as OvPhysxRigidObjectCollectionData,
    )
    from isaaclab_ov.test.fixtures.views import MockOvPhysxBindingSet
except ImportError as error:
    BACKEND_UNAVAILABLE_REASONS["ovphysx"] = f"{type(error).__name__}: {error}"
else:
    BACKENDS.append("ovphysx")


def backends_parametrize(*names: str):
    """Parametrize ``backend`` over the available backends, restricted to ``names`` when given."""
    selected = [backend for backend in (names or BACKENDS) if backend in BACKENDS]
    return pytest.mark.parametrize("backend", selected)


def requires_backend(name: str):
    """Skip marker for tests that need one specific backend."""
    return pytest.mark.skipif(
        name not in BACKENDS, reason=BACKEND_UNAVAILABLE_REASONS.get(name, f"{name} backend unavailable")
    )


"""
Shell construction helpers.
"""


def _index_array(count: int, device: str) -> wp.array:
    return wp.array(np.arange(count, dtype=np.int32), device=device)


def _set_attrs(obj, **attrs) -> None:
    for name, value in attrs.items():
        object.__setattr__(obj, name, value)


def _finalize_shell(asset) -> None:
    """Attach production wrench composers and null the lifecycle handles a shell never registers."""
    _set_attrs(
        asset,
        _instantaneous_wrench_composer=WrenchComposer(asset),
        _permanent_wrench_composer=WrenchComposer(asset),
        _initialize_handle=None,
        _invalidate_initialize_handle=None,
        _prim_deletion_handle=None,
        _debug_vis_handle=None,
    )


def _newton_mock_manager(num_instances: int, device: str, **model_attrs) -> MagicMock:
    """Mock ``NewtonManager`` exposing a model with gravity and any extra model-wide counts."""
    model = MagicMock()
    model.world_count = num_instances
    model.gravity = wp.array(
        np.tile(np.array([[0.0, 0.0, -9.81]], dtype=np.float32), (num_instances + 1, 1)), dtype=wp.vec3f, device=device
    )
    for name, value in model_attrs.items():
        setattr(model, name, value)
    state = MagicMock()
    manager = MagicMock()
    manager.get_model.return_value = model
    manager.get_state_0.return_value = state
    manager.get_state_1.return_value = state
    manager.get_control.return_value = MagicMock()
    return manager


@contextmanager
def _patched_simulation_manager(manager, *modules):
    """Temporarily replace ``SimulationManager`` in the given Newton modules."""
    originals = [module.SimulationManager for module in modules]
    for module in modules:
        module.SimulationManager = manager
    try:
        yield
    finally:
        for module, original in zip(modules, originals):
            module.SimulationManager = original


def _physx_pinned_staging(asset, num_instances: int, device: str, **buffers: tuple[int, ...]) -> None:
    """Allocate the pinned CPU staging buffers PhysX TensorAPI writes expect."""
    _set_attrs(
        asset,
        _sim_env_ids=wp.empty(num_instances, dtype=wp.int32, device=device),
        _sim_env_ids_views={},
        _cpu_env_ids_all=_index_array(num_instances, "cpu"),
        _cpu_env_ids=wp.empty(num_instances, dtype=wp.int32, device="cpu", pinned=True),
        _cpu_env_ids_views={},
        **{name: wp.zeros(shape, dtype=wp.float32, device="cpu") for name, shape in buffers.items()},
    )


"""
Articulation factories.
"""


def _articulation_cfg(joint_ordering, body_ordering) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/Robot",
        soft_joint_pos_limit_factor=1.0,
        actuators={},
        joint_ordering=joint_ordering,
        body_ordering=body_ordering,
    )


def create_physx_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    num_fixed_tendons: int = 0,
    num_spatial_tendons: int = 0,
    device: str = "cuda:0",
    is_fixed_base: bool = False,
    joint_ordering: tuple[str, ...] | None = None,
    body_ordering: tuple[str, ...] | None = None,
):
    """Create a PhysX articulation shell over a mock articulation view."""
    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]
    fixed_tendon_names = [f"fixed_tendon_{i}" for i in range(num_fixed_tendons)]
    spatial_tendon_names = [f"spatial_tendon_{i}" for i in range(num_spatial_tendons)]

    mock_view = PhysXMockArticulationViewWarp(
        count=num_instances,
        num_links=num_bodies,
        num_dofs=num_joints,
        device=device,
        max_fixed_tendons=num_fixed_tendons,
        max_spatial_tendons=num_spatial_tendons,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True
    metatype = MagicMock()
    metatype.fixed_base = is_fixed_base
    metatype.dof_count = num_joints
    metatype.link_count = num_bodies
    metatype.dof_names = joint_names
    metatype.link_names = body_names
    _set_attrs(mock_view, _shared_metatype=metatype)

    articulation = object.__new__(PhysXArticulation)
    articulation.cfg = _articulation_cfg(joint_ordering, body_ordering)
    data = PhysXArticulationData(mock_view, device)
    data.fixed_tendon_names = fixed_tendon_names
    data.spatial_tendon_names = spatial_tendon_names
    _set_attrs(
        articulation,
        _sim_cfg=None,
        _fixed_tendon_target_dirty=False,
        _root_view=mock_view,
        _device=device,
        _data=data,
        _fixed_tendon_names=fixed_tendon_names,
        _spatial_tendon_names=spatial_tendon_names,
        _ALL_INDICES=_index_array(num_instances, device),
        _ALL_BODY_INDICES=_index_array(num_bodies, device),
        _ALL_JOINT_INDICES=_index_array(num_joints, device),
        _ALL_FIXED_TENDON_INDICES=_index_array(num_fixed_tendons, device),
        _ALL_SPATIAL_TENDON_INDICES=_index_array(num_spatial_tendons, device),
        _ALL_INDICES_WP=_index_array(num_instances, device),
        _ALL_BODY_INDICES_WP=_index_array(num_bodies, device),
        _joint_pos_target_backend=None,
        _joint_vel_target_backend=None,
        _joint_effort_target_backend=None,
        _body_wrench_force_backend=None,
        _body_wrench_torque_backend=None,
    )
    _finalize_shell(articulation)
    articulation._resolve_and_install_ordering_maps()
    articulation._ordering_configure_backend_staging()

    n, j, b = num_instances, num_joints, num_bodies
    _set_attrs(
        articulation,
        _root_link_pose_w_f32=None,
        _root_com_vel_w_f32=None,
        _root_link_vel_w_f32=None,
        _inst_wrench_force_f32=None,
        _inst_wrench_torque_f32=None,
        _perm_wrench_force_f32=None,
        _perm_wrench_torque_f32=None,
    )
    _physx_pinned_staging(
        articulation,
        n,
        device,
        _cpu_joint_stiffness=(n, j),
        _cpu_joint_damping=(n, j),
        _cpu_joint_pos_limits=(n, j, 2),
        _cpu_joint_vel_limits=(n, j),
        _cpu_joint_effort_limits=(n, j),
        _cpu_joint_armature=(n, j),
        _cpu_joint_friction_props=(n, j, 3),
        _cpu_body_mass=(n, b),
        _cpu_body_coms=(n, b, 7),
        _cpu_body_inertia=(n, b, 9),
    )
    articulation._process_actuators_cfg()
    return articulation, mock_view


def create_ovphysx_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    num_fixed_tendons: int = 0,
    num_spatial_tendons: int = 0,
    device: str = "cuda:0",
    is_fixed_base: bool = False,
    joint_ordering: tuple[str, ...] | None = None,
    body_ordering: tuple[str, ...] | None = None,
):
    """Create an OvPhysX articulation shell over mocked tensor bindings."""
    from isaaclab_ov import tensor_types as TT

    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]
    fixed_tendon_names = [f"fixed_tendon_{i}" for i in range(num_fixed_tendons)]
    spatial_tendon_names = [f"spatial_tendon_{i}" for i in range(num_spatial_tendons)]

    mock_bindings = MockOvPhysxBindingSet(
        num_instances=num_instances,
        num_joints=num_joints,
        num_bodies=num_bodies,
        is_fixed_base=is_fixed_base,
        joint_names=joint_names,
        body_names=body_names,
        num_fixed_tendons=num_fixed_tendons,
        num_spatial_tendons=num_spatial_tendons,
    )
    mock_bindings.set_random_data()

    articulation = object.__new__(OvPhysxArticulation)
    articulation.cfg = _articulation_cfg(joint_ordering, body_ordering)
    # Counts come from the view, names are set afterwards.
    data = OvPhysxArticulationData(mock_bindings.view, device)
    data.body_names = body_names
    data.joint_names = joint_names
    data.fixed_tendon_names = fixed_tendon_names
    data.spatial_tendon_names = spatial_tendon_names
    data._is_fixed_base = is_fixed_base
    _set_attrs(
        articulation,
        _sim_cfg=None,
        _fixed_tendon_target_dirty=False,
        _device=device,
        _ovphysx=MagicMock(),
        _root_view=mock_bindings.view,
        _bindings=mock_bindings.bindings,
        _num_instances=num_instances,
        _num_joints=num_joints,
        _num_bodies=num_bodies,
        _is_fixed_base=is_fixed_base,
        _joint_names=joint_names,
        _body_names=body_names,
        _fixed_tendon_names=fixed_tendon_names,
        _spatial_tendon_names=spatial_tendon_names,
        _num_fixed_tendons=num_fixed_tendons,
        _num_spatial_tendons=num_spatial_tendons,
        _data=data,
    )
    # Allocate the index/mask caches and wrench buffers that ``_initialize_impl`` normally populates.
    articulation._resolve_and_install_ordering_maps()
    articulation._create_buffers()
    _finalize_shell(articulation)
    articulation._process_actuators_cfg()
    _set_attrs(
        articulation,
        _can_write_effort=articulation._get_binding(TT.DOF_ACTUATION_FORCE) is not None,
        _can_write_pos_target=articulation._get_binding(TT.DOF_POSITION_TARGET) is not None,
        _can_write_vel_target=articulation._get_binding(TT.DOF_VELOCITY_TARGET) is not None,
    )
    return articulation, mock_bindings


def create_newton_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    num_fixed_tendons: int = 0,
    num_spatial_tendons: int = 0,
    device: str = "cuda:0",
    is_fixed_base: bool = False,
    joint_ordering: tuple[str, ...] | None = None,
    body_ordering: tuple[str, ...] | None = None,
):
    """Create a Newton articulation shell over a mock articulation view.

    Newton supports fixed tendons but not spatial tendons, so ``num_spatial_tendons`` is ignored.
    """
    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]
    fixed_tendon_names = [f"fixed_tendon_{i}" for i in range(num_fixed_tendons)]

    mock_view = MockNewtonArticulationView(
        num_instances=num_instances,
        num_bodies=num_bodies,
        num_joints=num_joints,
        device=device,
        is_fixed_base=is_fixed_base,
        joint_names=joint_names,
        body_names=body_names,
        tendon_names=fixed_tendon_names,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    # Model-wide counts equal the per-articulation counts because the mock holds one homogeneous batch;
    # the task-space scratch buffers in ``NewtonArticulationData`` size themselves from them.
    total_dofs = num_joints + (0 if is_fixed_base else 6)
    manager = _newton_mock_manager(
        num_instances,
        device,
        articulation_count=num_instances,
        max_joints_per_articulation=num_bodies,
        max_dofs_per_articulation=total_dofs,
        joint_dof_count=num_instances * total_dofs,
        body_count=num_instances * num_bodies,
    )
    with _patched_simulation_manager(manager, newton_articulation_data_module):
        data = NewtonArticulationData(mock_view, device)
    mock_view._tendon_count = num_fixed_tendons
    data.fixed_tendon_names = fixed_tendon_names
    data.spatial_tendon_names = []

    articulation = object.__new__(NewtonArticulation)
    articulation.cfg = _articulation_cfg(joint_ordering, body_ordering)
    _set_attrs(
        articulation,
        _sim_cfg=None,
        _fixed_tendon_target_dirty=False,
        _root_view=mock_view,
        _device=device,
        _data=data,
        _test_simulation_manager=manager,
        # the solver builds this adapter; the shell has no model, so it stays absent
        _fixed_tendon_control=None,
        _fixed_tendon_names=fixed_tendon_names,
        _spatial_tendon_names=[],
        _ALL_INDICES=_index_array(num_instances, device),
        _ALL_BODY_INDICES=_index_array(num_bodies, device),
        _ALL_JOINT_INDICES=_index_array(num_joints, device),
        _ALL_ENV_MASK=wp.ones((num_instances,), dtype=wp.bool, device=device),
        _ALL_BODY_MASK=wp.ones((num_bodies,), dtype=wp.bool, device=device),
        _ALL_JOINT_MASK=wp.ones((num_joints,), dtype=wp.bool, device=device),
        _ALL_FIXED_TENDON_INDICES=_index_array(num_fixed_tendons, device),
        _ALL_FIXED_TENDON_MASK=wp.ones((num_fixed_tendons,), dtype=wp.bool, device=device),
        _ALL_SPATIAL_TENDON_INDICES=_index_array(0, device),
        _ALL_SPATIAL_TENDON_MASK=wp.ones((0,), dtype=wp.bool, device=device),
    )
    _finalize_shell(articulation)
    articulation._resolve_and_install_ordering_maps()
    articulation._ordering_configure_backend_staging()
    articulation._process_actuators_cfg()
    return articulation, mock_view


_ARTICULATION_FACTORIES = {
    "physx": create_physx_articulation,
    "ovphysx": create_ovphysx_articulation,
    "newton": create_newton_articulation,
}


def get_articulation(
    backend: str,
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    num_fixed_tendons: int = 0,
    num_spatial_tendons: int = 0,
    device: str = "cuda:0",
    is_fixed_base: bool = False,
    joint_ordering: tuple[str, ...] | None = None,
    body_ordering: tuple[str, ...] | None = None,
):
    """Create an articulation shell and its mock view for the given backend."""
    if backend not in _ARTICULATION_FACTORIES:
        raise ValueError(f"Invalid backend: {backend}")
    return _ARTICULATION_FACTORIES[backend](
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        is_fixed_base=is_fixed_base,
        joint_ordering=joint_ordering,
        body_ordering=body_ordering,
    )


"""
Rigid object factories.
"""


def create_physx_rigid_object(num_instances: int = 2, device: str = "cuda:0"):
    """Create a PhysX rigid object shell over a mock rigid body view."""
    mock_view = PhysXMockRigidBodyViewWarp(count=num_instances, device=device)
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    rigid_object = object.__new__(PhysXRigidObject)
    rigid_object.cfg = RigidObjectCfg(prim_path="/World/Object")
    data = PhysXRigidObjectData(mock_view, device)
    data.body_names = ["body_0"]
    _set_attrs(
        rigid_object,
        _root_view=mock_view,
        _device=device,
        _data=data,
        _ALL_INDICES=_index_array(num_instances, device),
        _ALL_BODY_INDICES=_index_array(1, device),
        _root_link_pose_w_f32=None,
        _root_com_vel_w_f32=None,
        _inst_wrench_force_f32=None,
        _inst_wrench_torque_f32=None,
        _perm_wrench_force_f32=None,
        _perm_wrench_torque_f32=None,
    )
    _finalize_shell(rigid_object)
    n = num_instances
    _physx_pinned_staging(
        rigid_object, n, device, _cpu_body_mass=(n, 1), _cpu_body_coms=(n, 1, 7), _cpu_body_inertia=(n, 1, 9)
    )
    return rigid_object, mock_view


def create_newton_rigid_object(num_instances: int = 2, device: str = "cuda:0"):
    """Create a Newton rigid object shell over a single-body mock articulation view."""
    mock_view = MockNewtonArticulationView(
        num_instances=num_instances,
        num_bodies=1,
        num_joints=0,
        device=device,
        is_fixed_base=False,
        joint_names=[],
        body_names=["body_0"],
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True
    with _patched_simulation_manager(_newton_mock_manager(num_instances, device), newton_rigid_object_data_module):
        data = NewtonRigidObjectData(mock_view, device)

    rigid_object = object.__new__(NewtonRigidObject)
    rigid_object.cfg = RigidObjectCfg(prim_path="/World/Object")
    _set_attrs(
        rigid_object,
        _root_view=mock_view,
        _device=device,
        _data=data,
        _ALL_INDICES=_index_array(num_instances, device),
        _ALL_BODY_INDICES=_index_array(1, device),
        _ALL_ENV_MASK=wp.ones((num_instances,), dtype=wp.bool, device=device),
        _ALL_BODY_MASK=wp.ones((1,), dtype=wp.bool, device=device),
    )
    _finalize_shell(rigid_object)
    return rigid_object, mock_view


def create_ovphysx_rigid_object(num_instances: int = 2, device: str = "cuda:0"):
    """Create an OvPhysX rigid object shell over mocked tensor bindings."""
    body_names = ["base_link"]
    mock_bindings = MockOvPhysxBindingSet(
        num_instances=num_instances, num_joints=0, num_bodies=1, body_names=body_names, asset_kind="rigid_object"
    )
    mock_bindings.set_random_data()

    rigid_object = object.__new__(OvPhysxRigidObject)
    rigid_object.cfg = RigidObjectCfg(prim_path="/World/object")
    data = OvPhysxRigidObjectData(mock_bindings.view, device)
    data.num_instances = num_instances
    data.num_bodies = 1
    data._is_primed = True
    _set_attrs(
        rigid_object,
        _device=device,
        _ovphysx=MagicMock(),
        _root_view=mock_bindings.view,
        _bindings=mock_bindings.bindings,
        _num_instances=num_instances,
        _num_bodies=1,
        _body_names=body_names,
        _data=data,
    )
    # Allocate the index/mask caches and staging buffers that ``_initialize_impl`` normally populates.
    rigid_object._create_buffers()
    _finalize_shell(rigid_object)
    return rigid_object, mock_bindings


_RIGID_OBJECT_FACTORIES = {
    "physx": create_physx_rigid_object,
    "ovphysx": create_ovphysx_rigid_object,
    "newton": create_newton_rigid_object,
}


def get_rigid_object(backend: str, num_instances: int = 2, device: str = "cuda:0"):
    """Create a rigid object shell and its mock view for the given backend."""
    if backend not in _RIGID_OBJECT_FACTORIES:
        raise ValueError(f"Invalid backend: {backend}")
    return _RIGID_OBJECT_FACTORIES[backend](num_instances, device)


"""
Rigid object collection factories.
"""


def _collection_cfg(num_bodies: int) -> RigidObjectCollectionCfg:
    return RigidObjectCollectionCfg(
        rigid_objects={f"object_{i}": RigidObjectCfg(prim_path=f"/World/Object_{i}") for i in range(num_bodies)}
    )


def create_physx_rigid_object_collection(num_instances: int = 2, num_bodies: int = 3, device: str = "cuda:0"):
    """Create a PhysX rigid object collection shell over a body-major mock rigid body view."""
    body_names = [f"object_{i}" for i in range(num_bodies)]
    mock_view = PhysXMockRigidBodyViewWarp(count=num_instances * num_bodies, device=device)
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    collection = object.__new__(PhysXRigidObjectCollection)
    collection.cfg = _collection_cfg(num_bodies)
    data = PhysXRigidObjectCollectionData(mock_view, num_bodies, device)
    data.body_names = body_names
    num_view_ids = num_instances * num_bodies
    all_view_ids = _index_array(num_view_ids, device)
    cpu_all_view_ids = wp.empty(num_view_ids, dtype=wp.int32, device="cpu", pinned=True)
    wp.copy(cpu_all_view_ids, all_view_ids)
    _set_attrs(
        collection,
        _root_view=mock_view,
        _device=device,
        _num_bodies=num_bodies,
        _num_instances=num_instances,
        _body_names_list=body_names,
        _data=data,
        _ALL_ENV_INDICES=_index_array(num_instances, device),
        _ALL_BODY_INDICES=_index_array(num_bodies, device),
        _ALL_VIEW_INDICES=all_view_ids,
        _sim_view_ids=wp.empty(num_view_ids, dtype=wp.int32, device=device),
        _sim_view_ids_views={},
        _cpu_all_view_ids=cpu_all_view_ids,
        _cpu_view_ids=wp.empty(num_view_ids, dtype=wp.int32, device="cpu", pinned=True),
        _cpu_view_ids_views={},
    )
    _finalize_shell(collection)
    return collection, mock_view


def create_newton_rigid_object_collection(num_instances: int = 2, num_bodies: int = 3, device: str = "cuda:0"):
    """Create a Newton rigid object collection shell over a mock collection view."""
    body_names = [f"object_{i}" for i in range(num_bodies)]
    mock_view = MockNewtonCollectionView(
        num_envs=num_instances, num_bodies=num_bodies, device=device, body_names=body_names
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True
    manager = _newton_mock_manager(num_instances, device)
    with _patched_simulation_manager(manager, newton_collection_data_module, newton_collection_module):
        data = NewtonRigidObjectCollectionData(mock_view, num_bodies, device)
    data.body_names = body_names

    collection = object.__new__(NewtonRigidObjectCollection)
    collection.cfg = _collection_cfg(num_bodies)
    _set_attrs(
        collection,
        _root_view=mock_view,
        _device=device,
        _num_bodies=num_bodies,
        _num_instances=num_instances,
        _body_names_list=body_names,
        _data=data,
        _ALL_ENV_INDICES=_index_array(num_instances, device),
        _ALL_BODY_INDICES=_index_array(num_bodies, device),
        _ALL_ENV_MASK=wp.ones((num_instances,), dtype=wp.bool, device=device),
        _ALL_BODY_MASK=wp.ones((num_bodies,), dtype=wp.bool, device=device),
    )
    _finalize_shell(collection)
    return collection, mock_view


def create_ovphysx_rigid_object_collection(num_instances: int = 2, num_bodies: int = 3, device: str = "cuda:0"):
    """Create an OvPhysX rigid object collection shell over mocked tensor bindings."""
    body_names = [f"object_{i}" for i in range(num_bodies)]
    # Articulation-mode bindings with no joints yield the (N, B, ...) tensor layout of a collection.
    mock_bindings = MockOvPhysxBindingSet(
        num_instances=num_instances,
        num_joints=0,
        num_bodies=num_bodies,
        body_names=body_names,
        asset_kind="articulation",
    )
    mock_bindings.set_random_data()

    collection = object.__new__(OvPhysxRigidObjectCollection)
    collection.cfg = _collection_cfg(num_bodies)
    data = OvPhysxRigidObjectCollectionData(mock_bindings.view, num_bodies, device)
    data.num_instances = num_instances
    data.num_bodies = num_bodies
    data._is_primed = True
    _set_attrs(
        collection,
        _device=device,
        _ovphysx=MagicMock(),
        _root_view=mock_bindings.view,
        _bindings=mock_bindings.bindings,
        _num_instances=num_instances,
        _num_bodies=num_bodies,
        _body_names_list=body_names,
        _data=data,
    )
    collection._create_buffers()
    _finalize_shell(collection)
    return collection, mock_bindings


_COLLECTION_FACTORIES = {
    "physx": create_physx_rigid_object_collection,
    "ovphysx": create_ovphysx_rigid_object_collection,
    "newton": create_newton_rigid_object_collection,
}


def get_rigid_object_collection(backend: str, num_instances: int = 2, num_bodies: int = 3, device: str = "cuda:0"):
    """Create a rigid object collection shell and its mock view for the given backend."""
    if backend not in _COLLECTION_FACTORIES:
        raise ValueError(f"Invalid backend: {backend}")
    return _COLLECTION_FACTORIES[backend](num_instances, num_bodies, device)


"""
Test data helpers.
"""

WP_DTYPE_TRAILING = {wp.transformf: 7, wp.spatial_vectorf: 6, wp.vec2f: 2, wp.vec3f: 3, wp.quatf: 4, wp.float32: 0}
"""Trailing torch dimension for each warp dtype used by the writers."""

SHAPE_ERRORS = (AssertionError, RuntimeError)
"""Exceptions the writers raise on a shape mismatch."""


def make_data_torch(shape: tuple[int, ...], device: str, wp_dtype=wp.float32, trailing: int | None = None):
    """Create valid torch writer data: identity transforms, ``[-1, 1]`` limits, ones otherwise."""
    if trailing is None:
        trailing = WP_DTYPE_TRAILING[wp_dtype]
    data = torch.ones((*shape, trailing) if trailing else shape, device=device, dtype=torch.float32)
    if wp_dtype == wp.transformf:
        data[..., :6] = 0.0
    elif wp_dtype == wp.spatial_vectorf:
        data.zero_()
    elif wp_dtype == wp.vec2f:
        data[..., 0] = -1.0
    return data


def make_data_warp(shape: tuple[int, ...], device: str, wp_dtype=wp.float32, trailing: int | None = None):
    """Create valid warp writer data; structured dtypes absorb the trailing torch dimension."""
    return wp.from_torch(make_data_torch(shape, device, wp_dtype, trailing).contiguous(), dtype=wp_dtype)


def make_com_data(backend: str, shape: tuple[int, ...], device: str) -> wp.array:
    """Center-of-mass writer data in the layout the backend expects (Newton stores positions only)."""
    if backend == "newton":
        return wp.zeros(shape, dtype=wp.vec3f, device=device)
    return make_data_warp(shape, device, wp.transformf)


def make_mask(total: int, selected: list[int], device: str) -> wp.array:
    """Bool warp mask with ``True`` at the selected indices."""
    mask = np.zeros(total, dtype=bool)
    mask[selected] = True
    return wp.array(mask, dtype=wp.bool, device=device)


def make_env_ids(device: str) -> torch.Tensor:
    """Environment selector for the first instance."""
    return torch.tensor([0], dtype=torch.int32, device=device)


def check_proxy_array(arr, *, expected_shape: tuple[int, ...], expected_dtype: type, name: str) -> None:
    """Assert ``arr`` is a :class:`ProxyArray` with the expected shape and dtype."""
    assert isinstance(arr, ProxyArray), f"{name}: expected ProxyArray, got {type(arr)}"
    assert arr.shape == expected_shape, f"{name}: expected shape {expected_shape}, got {arr.shape}"
    assert arr.dtype == expected_dtype, f"{name}: expected dtype {expected_dtype}, got {arr.dtype}"


def check_data_properties(data, expected: dict[str, tuple[tuple[int, ...], type]]) -> None:
    """Assert every listed data property is a proxy array with the expected shape and dtype."""
    failures = []
    for name, (shape, dtype) in expected.items():
        arr = getattr(data, name)
        if not isinstance(arr, ProxyArray) or arr.shape != shape or arr.dtype != dtype:
            actual = f"{type(arr).__name__} {getattr(arr, 'shape', None)} {getattr(arr, 'dtype', None)}"
            failures.append(f"{name}: expected ProxyArray {shape} {dtype}, got {actual}")
    assert not failures, "\n".join(failures)


def check_aliases(data, aliases: dict[str, str]) -> None:
    """Assert alias properties match the shape and dtype of their canonical counterparts."""
    for alias, canonical in aliases.items():
        alias_arr, canonical_arr = getattr(data, alias), getattr(data, canonical)
        assert alias_arr.shape == canonical_arr.shape, alias
        assert alias_arr.dtype == canonical_arr.dtype, alias


def prime_timestamped_properties(data, property_buffer_pairs: list[tuple[str, str]]) -> list[tuple[str, object]]:
    """Read lazy public properties, then stamp their private buffers as fresh for staleness checks."""
    buffers = []
    for property_name, buffer_name in property_buffer_pairs:
        getattr(data, property_name)
        buffer = getattr(data, buffer_name)
        assert buffer is not None, buffer_name
        buffer.timestamp = data._sim_timestamp
        buffers.append((buffer_name, buffer))
    return buffers


def assert_buffers_stale(data, buffers: list[tuple[str, object]]) -> None:
    """Assert every primed buffer was invalidated relative to the current simulation timestamp."""
    for name, buffer in buffers:
        assert buffer.timestamp < data._sim_timestamp, name


def exercise_writer(
    asset,
    name: str,
    kwarg: str,
    wp_dtype,
    *,
    shape: tuple[int, ...],
    device: str,
    item_axis: tuple[str, str] | None = None,
    mask_item_axis: bool = True,
    trailing: int | None = None,
    scalar: bool | tuple[type[Exception], ...] | None = None,
) -> None:
    """Drive the ``_index`` and ``_mask`` variants of a writer through the full input matrix.

    Both variants receive torch and warp data for all instances and for a partial selection, and must
    reject data whose leading dimension does not match.

    Args:
        asset: Asset exposing ``{name}_index`` and ``{name}_mask``.
        name: Writer name without the ``_index``/``_mask`` suffix.
        kwarg: Keyword the data is passed as.
        wp_dtype: Warp dtype of the data.
        shape: Full leading shape, ``(num_instances,)`` or ``(num_instances, num_items)``.
        device: Device the data lives on.
        item_axis: ``(ids_kwarg, mask_kwarg)`` for the item dimension, or ``None`` for root writers.
        mask_item_axis: Whether the mask variant accepts the item mask keyword.
        trailing: Explicit trailing torch dimension for unstructured dtypes (e.g. ``9`` for inertias).
        scalar: ``True`` if a float scalar is accepted, the exception types raised if it must be rejected.
    """
    index_writer = getattr(asset, f"{name}_index")
    mask_writer = getattr(asset, f"{name}_mask")
    bad_shape = (shape[0] + 1, *shape[1:])
    sub_shape = (1, *shape[1:])
    selection = {"env_ids": make_env_ids(device)}
    mask_selection = {"env_mask": make_mask(shape[0], [0], device)}
    if item_axis is not None:
        num_sub = min(2, shape[1])
        sub_shape = (1, num_sub)
        selection[item_axis[0]] = list(range(num_sub))
        if mask_item_axis:
            mask_selection[item_axis[1]] = make_mask(shape[1], list(range(num_sub)), device)

    for make in (make_data_torch, make_data_warp):
        index_writer(**{kwarg: make(shape, device, wp_dtype, trailing)})
        index_writer(**{kwarg: make(sub_shape, device, wp_dtype, trailing)}, **selection)
        mask_writer(**{kwarg: make(shape, device, wp_dtype, trailing)})
        mask_writer(**{kwarg: make(shape, device, wp_dtype, trailing)}, **mask_selection)
        for writer in (index_writer, mask_writer):
            with pytest.raises(SHAPE_ERRORS):
                writer(**{kwarg: make(bad_shape, device, wp_dtype, trailing)})
    if scalar is True:
        index_writer(**{kwarg: 1.0})
        mask_writer(**{kwarg: 1.0})
    elif scalar is not None:
        with pytest.raises(scalar):
            index_writer(**{kwarg: 1.0})
