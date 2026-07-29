# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy PhysX target construction for asset micro-benchmarks."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from isaaclab.benchmark.asset_suites.types import AssetBenchmarkTargets

args = SimpleNamespace(no_shape_checks=False)


def _load_runtime_symbols() -> None:
    global Articulation, ArticulationCfg, ArticulationData, MockArticulationViewWarp
    global MockRigidBodyViewWarp, MockWrenchComposer, PhysxManager, RigidObject, RigidObjectCfg
    global RigidObjectCollection, RigidObjectCollectionCfg, RigidObjectCollectionData, RigidObjectData
    global np, torch, wp

    import numpy as np
    import torch
    import warp as wp

    from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
    from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
    from isaaclab.assets.rigid_object_collection.rigid_object_collection_cfg import RigidObjectCollectionCfg
    from isaaclab.test.mock_interfaces.utils import MockWrenchComposer

    from isaaclab_physx.assets.articulation.articulation import Articulation
    from isaaclab_physx.assets.articulation.articulation_data import ArticulationData
    from isaaclab_physx.assets.rigid_object.rigid_object import RigidObject
    from isaaclab_physx.assets.rigid_object.rigid_object_data import RigidObjectData
    from isaaclab_physx.assets.rigid_object_collection.rigid_object_collection import RigidObjectCollection
    from isaaclab_physx.assets.rigid_object_collection.rigid_object_collection_data import RigidObjectCollectionData
    from isaaclab_physx.physics import PhysxManager
    from isaaclab_physx.test.mock_interfaces.views import MockArticulationViewWarp, MockRigidBodyViewWarp


def create_test_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    device: str = "cuda:0",
) -> tuple[Articulation, MockArticulationViewWarp, MagicMock]:
    """Create a test Articulation instance with mocked dependencies."""
    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]

    articulation = object.__new__(Articulation)

    articulation.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        soft_joint_pos_limit_factor=1.0,
        actuators={},
    )

    # Create PhysX mock view
    mock_view = MockArticulationViewWarp(
        count=num_instances,
        num_links=num_bodies,
        num_dofs=num_joints,
        device=device,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    # Set up the mock view's metatype for accessing names/counts
    mock_metatype = MagicMock()
    mock_metatype.fixed_base = False
    mock_metatype.dof_count = num_joints
    mock_metatype.link_count = num_bodies
    mock_metatype.dof_names = joint_names
    mock_metatype.link_names = body_names
    object.__setattr__(mock_view, "_shared_metatype", mock_metatype)

    object.__setattr__(articulation, "_root_view", mock_view)
    object.__setattr__(articulation, "_device", device)
    object.__setattr__(articulation, "_check_shapes", not args.no_shape_checks)

    # Create ArticulationData instance (SimulationManager already mocked at module level)
    data = ArticulationData(mock_view, device)
    object.__setattr__(articulation, "_data", data)

    # Create mock wrench composers (pass articulation which has num_instances, num_bodies, device properties)
    mock_inst_wrench = MockWrenchComposer(articulation)
    mock_perm_wrench = MockWrenchComposer(articulation)
    object.__setattr__(articulation, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(articulation, "_permanent_wrench_composer", mock_perm_wrench)

    # Set up other required attributes
    object.__setattr__(articulation, "actuators", {})
    object.__setattr__(articulation, "_has_implicit_actuators", False)

    # Use warp arrays for _ALL_* indices (matching real _create_buffers)
    import numpy as np

    all_indices_wp = wp.array(np.arange(num_instances, dtype=np.int32), device=device)
    all_joint_indices_wp = wp.array(np.arange(num_joints, dtype=np.int32), device=device)
    all_body_indices_wp = wp.array(np.arange(num_bodies, dtype=np.int32), device=device)
    object.__setattr__(articulation, "_ALL_INDICES", all_indices_wp)
    object.__setattr__(articulation, "_ALL_JOINT_INDICES", all_joint_indices_wp)
    object.__setattr__(articulation, "_ALL_BODY_INDICES", all_body_indices_wp)

    # Warp arrays for set_external_force_and_torque
    object.__setattr__(articulation, "_ALL_INDICES_WP", all_indices_wp)
    object.__setattr__(articulation, "_ALL_BODY_INDICES_WP", all_body_indices_wp)

    # Initialize joint targets
    object.__setattr__(articulation, "_joint_pos_target_sim", torch.zeros(num_instances, num_joints, device=device))
    object.__setattr__(articulation, "_joint_vel_target_sim", torch.zeros(num_instances, num_joints, device=device))
    object.__setattr__(articulation, "_joint_effort_target_sim", torch.zeros(num_instances, num_joints, device=device))

    # Cached .view() wrappers
    object.__setattr__(articulation, "_root_link_pose_w_f32", None)
    object.__setattr__(articulation, "_root_com_vel_w_f32", None)
    object.__setattr__(articulation, "_root_link_vel_w_f32", None)

    # Pre-allocated pinned CPU buffers for PhysX TensorAPI writes
    N, J, B = num_instances, num_joints, num_bodies
    object.__setattr__(articulation, "_cpu_env_ids_all", wp.zeros(N, dtype=wp.int32, device="cpu", pinned=True))
    wp.copy(articulation._cpu_env_ids_all, all_indices_wp)
    object.__setattr__(
        articulation, "_cpu_joint_stiffness", wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
    )
    object.__setattr__(
        articulation, "_cpu_joint_damping", wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
    )
    object.__setattr__(
        articulation, "_cpu_joint_pos_limits", wp.zeros((N, J, 2), dtype=wp.float32, device="cpu", pinned=True)
    )
    object.__setattr__(
        articulation, "_cpu_joint_vel_limits", wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
    )
    object.__setattr__(
        articulation, "_cpu_joint_effort_limits", wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
    )
    object.__setattr__(
        articulation, "_cpu_joint_armature", wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
    )
    object.__setattr__(
        articulation, "_cpu_joint_friction_props", wp.zeros((N, J, 3), dtype=wp.float32, device="cpu", pinned=True)
    )
    object.__setattr__(articulation, "_cpu_body_mass", wp.zeros((N, B), dtype=wp.float32, device="cpu", pinned=True))
    object.__setattr__(articulation, "_cpu_body_coms", wp.zeros((N, B, 7), dtype=wp.float32, device="cpu", pinned=True))
    object.__setattr__(
        articulation, "_cpu_body_inertia", wp.zeros((N, B, 9), dtype=wp.float32, device="cpu", pinned=True)
    )

    return articulation, mock_view, None


# =============================================================================
# Input Generators (Torch-only for PhysX backend)
def create_test_rigid_object(
    num_instances: int = 2,
    num_bodies: int = 1,
    device: str = "cuda:0",
) -> tuple[RigidObject, MockRigidBodyViewWarp, MagicMock]:
    """Create a test RigidObject instance with mocked dependencies."""
    rigid_object = object.__new__(RigidObject)

    rigid_object.cfg = RigidObjectCfg(
        prim_path="/World/Object",
    )

    # Create PhysX mock view
    mock_view = MockRigidBodyViewWarp(
        count=num_instances,
        device=device,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    # Set up attributes required before _create_buffers
    object.__setattr__(rigid_object, "_root_view", mock_view)
    object.__setattr__(rigid_object, "_device", device)

    # Create RigidObjectData instance (mocks already set up at module level)
    data = RigidObjectData(mock_view, device)
    object.__setattr__(rigid_object, "_data", data)

    # Call _create_buffers to set up all internal buffers and wrench composers
    rigid_object._create_buffers()

    return rigid_object, mock_view, None


# =============================================================================
# Input Generators (Torch-only for PhysX backend)
def create_test_collection(
    num_instances: int = 2,
    num_bodies: int = 4,
    device: str = "cuda:0",
) -> tuple[RigidObjectCollection, MockRigidBodyViewWarp]:
    """Create a test RigidObjectCollection instance with mocked dependencies."""
    object_names = [f"object_{i}" for i in range(num_bodies)]

    collection = object.__new__(RigidObjectCollection)

    # Create a minimal config with dummy rigid objects
    from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg

    rigid_objects = {name: RigidObjectCfg(prim_path=f"/World/{name}") for name in object_names}
    collection.cfg = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

    # Create PhysX mock view (total count = num_instances * num_bodies)
    total_count = num_instances * num_bodies
    mock_view = MockRigidBodyViewWarp(
        count=total_count,
        device=device,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    # Set up attributes required before _create_buffers
    object.__setattr__(collection, "_root_view", mock_view)
    object.__setattr__(collection, "_device", device)
    object.__setattr__(collection, "_body_names_list", object_names)

    # Create RigidObjectCollectionData instance
    data = RigidObjectCollectionData(mock_view, num_bodies, device)
    data.object_names = object_names
    object.__setattr__(collection, "_data", data)

    # Call _create_buffers to set up all internal buffers and wrench composers
    collection._create_buffers()

    return collection, mock_view


# =============================================================================
# Input Generators


def _refresh_articulation_data(data, _config) -> None:
    data._sim_timestamp += 1.0
    data._fk_timestamp = data._sim_timestamp


def _refresh_rigid_object_data(mock_view, data, config) -> None:
    mock_view.set_mock_transforms(
        wp.from_torch(torch.randn(config.num_instances, 7, device=config.device), dtype=wp.float32)
    )
    mock_view.set_mock_velocities(
        wp.from_torch(torch.randn(config.num_instances, 6, device=config.device), dtype=wp.float32)
    )
    mock_view.set_mock_accelerations(
        wp.from_torch(torch.randn(config.num_instances, 6, device=config.device), dtype=wp.float32)
    )
    mock_view.set_mock_coms(wp.from_torch(torch.randn(config.num_instances, 7, device=config.device), dtype=wp.float32))
    data._sim_timestamp += 1.0


def _refresh_collection_data(mock_view, data, config) -> None:
    total_count = config.num_instances * config.num_bodies
    mock_view.set_mock_transforms(wp.from_torch(torch.randn(total_count, 7, device=config.device), dtype=wp.float32))
    mock_view.set_mock_velocities(wp.from_torch(torch.randn(total_count, 6, device=config.device), dtype=wp.float32))
    mock_view.set_mock_accelerations(wp.from_torch(torch.randn(total_count, 6, device=config.device), dtype=wp.float32))
    mock_view.set_mock_coms(wp.from_torch(torch.randn(total_count, 7, device=config.device), dtype=wp.float32))
    data._sim_timestamp += 1.0


def _configure_articulation_view(mock_view, config) -> None:
    mock_view.get_root_transforms = lambda: mock_view._as_structured(
        mock_view._root_transforms, wp.transformf, (mock_view._count,)
    )
    mock_view.get_root_velocities = lambda: mock_view._as_structured(
        mock_view._root_velocities, wp.spatial_vectorf, (mock_view._count,)
    )
    mock_view.get_link_transforms = lambda: mock_view._as_structured(
        mock_view._link_transforms, wp.transformf, (mock_view._count, mock_view._num_links)
    )
    mock_view.get_link_velocities = lambda: mock_view._as_structured(
        mock_view._link_velocities, wp.spatial_vectorf, (mock_view._count, mock_view._num_links)
    )
    mock_view.get_link_accelerations = lambda: mock_view._as_structured(
        mock_view._link_accelerations, wp.spatial_vectorf, (mock_view._count, mock_view._num_links)
    )
    mock_view.get_dof_positions = lambda: mock_view._dof_positions
    mock_view.get_dof_velocities = lambda: mock_view._dof_velocities
    num_dofs = config.num_joints + 6
    jacobians = wp.zeros((config.num_instances, config.num_bodies, 6, num_dofs), dtype=wp.float32, device=config.device)
    mass_matrices = wp.zeros((config.num_instances, num_dofs, num_dofs), dtype=wp.float32, device=config.device)
    gravity_forces = wp.zeros((config.num_instances, num_dofs), dtype=wp.float32, device=config.device)
    mock_view.get_jacobians = lambda: jacobians
    mock_view.get_generalized_mass_matrices = lambda: mass_matrices
    mock_view.get_gravity_compensation_forces = lambda: gravity_forces


def _create_data_target(component, config):
    if component == "articulation":
        mock_view = MockArticulationViewWarp(
            count=config.num_instances,
            num_links=config.num_bodies,
            num_dofs=config.num_joints,
            device=config.device,
        )
        mock_view.set_random_mock_data()
        _configure_articulation_view(mock_view, config)
        data = ArticulationData(mock_view, config.device)
        data._apply_ordering_maps_after_resolve()
        return data, lambda cfg, _mock_view=mock_view: _refresh_articulation_data(data, cfg)
    if component == "rigid_object":
        mock_view = MockRigidBodyViewWarp(count=config.num_instances, device=config.device)
        mock_view.set_random_mock_data()
        data = RigidObjectData(mock_view, config.device)
        return data, lambda cfg: _refresh_rigid_object_data(mock_view, data, cfg)
    mock_view = MockRigidBodyViewWarp(count=config.num_instances * config.num_bodies, device=config.device)
    mock_view.set_random_mock_data()
    data = RigidObjectCollectionData(mock_view, config.num_bodies, config.device)
    return data, lambda cfg: _refresh_collection_data(mock_view, data, cfg)


@contextmanager
def open_asset_targets(adapter, request, method_config, data_config):
    """Open a simulator app and the selected mocked PhysX asset target."""
    from isaaclab.app import AppLauncher

    app_launcher = (
        AppLauncher(headless=True, args=request.launcher_args) if request.launcher_args else AppLauncher(headless=True)
    )
    try:
        _load_runtime_symbols()
        args.no_shape_checks = not request.check_shapes
        physics_view = MagicMock()
        physics_view.get_gravity.return_value = (0.0, 0.0, -9.81)
        with patch.object(PhysxManager, "get_physics_sim_view", return_value=physics_view):
            if adapter.component == "articulation":
                target, _, _ = create_test_articulation(
                    num_instances=method_config.num_instances,
                    num_bodies=method_config.num_bodies,
                    num_joints=method_config.num_joints,
                    device=method_config.device,
                )
            elif adapter.component == "rigid_object":
                target, _, _ = create_test_rigid_object(
                    num_instances=method_config.num_instances,
                    num_bodies=method_config.num_bodies,
                    device=method_config.device,
                )
            else:
                target, _ = create_test_collection(
                    num_instances=method_config.num_instances,
                    num_bodies=method_config.num_bodies,
                    device=method_config.device,
                )
            data, refresh_data = _create_data_target(adapter.component, data_config)
            yield AssetBenchmarkTargets(
                method_target=target,
                data_target=data,
                refresh_data=refresh_data,
            )
    finally:
        app_launcher.app.close()
