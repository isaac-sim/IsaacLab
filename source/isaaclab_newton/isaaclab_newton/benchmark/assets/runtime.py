# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy Newton target construction for asset micro-benchmarks."""

from __future__ import annotations

import importlib
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

from isaaclab.benchmark.asset_suites.types import AssetBenchmarkTargets

args = SimpleNamespace(no_shape_checks=False)


def _initialize_mock_asset(asset) -> None:
    for name in ("_initialize_handle", "_invalidate_initialize_handle", "_prim_deletion_handle", "_debug_vis_handle"):
        object.__setattr__(asset, name, None)


def _configure_articulation_model(model, num_instances: int, num_bodies: int, num_joints: int) -> None:
    """Configure mock model dimensions required by Newton articulation data buffers."""
    model.articulation_count = num_instances
    model.max_joints_per_articulation = num_bodies
    model.max_dofs_per_articulation = num_joints + 6
    model.joint_dof_count = num_instances * (num_joints + 6)
    model.body_count = num_instances * num_bodies


def _load_runtime_symbols() -> None:
    global ArticulationCfg, MockNewtonArticulationView, MockNewtonCollectionView, MockWrenchComposer
    global RigidObjectCfg, RigidObjectCollectionCfg, create_mock_newton_manager, np, wp

    import numpy as np
    import warp as wp

    from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
    from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
    from isaaclab.assets.rigid_object_collection.rigid_object_collection_cfg import RigidObjectCollectionCfg

    from isaaclab_newton.test.mock_interfaces import (
        MockNewtonArticulationView,
        MockWrenchComposer,
        create_mock_newton_manager,
    )
    from isaaclab_newton.test.mock_interfaces.views import MockNewtonCollectionView


def create_test_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    device: str = "cuda:0",
):
    """Create a test Articulation instance with mocked dependencies."""
    from isaaclab_newton.assets.articulation.articulation import Articulation

    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]

    articulation = object.__new__(Articulation)
    _initialize_mock_asset(articulation)

    articulation.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        soft_joint_pos_limit_factor=1.0,
        actuators={},
    )
    mock_view = MockNewtonArticulationView(
        num_instances=num_instances,
        num_bodies=num_bodies,
        num_joints=num_joints,
        device=device,
        joint_names=joint_names,
        body_names=body_names,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    object.__setattr__(articulation, "_root_view", mock_view)
    object.__setattr__(articulation, "_device", device)
    object.__setattr__(articulation, "_check_shapes", not args.no_shape_checks)

    from isaaclab_newton.assets.articulation import articulation_data as data_module

    # ArticulationData allocates buffers from the model dimensions at construction.
    _configure_articulation_model(data_module.SimulationManager.get_model(), num_instances, num_bodies, num_joints)
    data = data_module.ArticulationData(mock_view, device)
    object.__setattr__(articulation, "_data", data)

    mock_inst_wrench = MockWrenchComposer(articulation)
    mock_perm_wrench = MockWrenchComposer(articulation)
    object.__setattr__(articulation, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(articulation, "_permanent_wrench_composer", mock_perm_wrench)

    object.__setattr__(articulation, "actuators", {})
    object.__setattr__(articulation, "_has_implicit_actuators", False)
    object.__setattr__(articulation, "_ALL_INDICES", wp.array(np.arange(num_instances, dtype=np.int32), device=device))
    object.__setattr__(
        articulation, "_ALL_BODY_INDICES", wp.array(np.arange(num_bodies, dtype=np.int32), device=device)
    )
    object.__setattr__(
        articulation, "_ALL_JOINT_INDICES", wp.array(np.arange(num_joints, dtype=np.int32), device=device)
    )
    object.__setattr__(articulation, "_ALL_ENV_MASK", wp.ones((num_instances,), dtype=wp.bool, device=device))
    object.__setattr__(articulation, "_ALL_JOINT_MASK", wp.ones((num_joints,), dtype=wp.bool, device=device))
    object.__setattr__(articulation, "_ALL_BODY_MASK", wp.ones((num_bodies,), dtype=wp.bool, device=device))
    object.__setattr__(articulation, "_ALL_FIXED_TENDON_INDICES", wp.array([], dtype=wp.int32, device=device))
    object.__setattr__(articulation, "_ALL_FIXED_TENDON_MASK", wp.zeros((0,), dtype=wp.bool, device=device))
    object.__setattr__(articulation, "_ALL_SPATIAL_TENDON_INDICES", wp.array([], dtype=wp.int32, device=device))
    object.__setattr__(articulation, "_ALL_SPATIAL_TENDON_MASK", wp.zeros((0,), dtype=wp.bool, device=device))

    object.__setattr__(
        articulation, "_joint_pos_target_sim", wp.zeros((num_instances, num_joints), dtype=wp.float32, device=device)
    )
    object.__setattr__(
        articulation, "_joint_vel_target_sim", wp.zeros((num_instances, num_joints), dtype=wp.float32, device=device)
    )
    object.__setattr__(
        articulation,
        "_joint_effort_target_sim",
        wp.zeros((num_instances, num_joints), dtype=wp.float32, device=device),
    )

    return articulation, mock_view


# =============================================================================
# Input Generators (Torch-only for Newton backend)
def create_test_rigid_object(
    num_instances: int = 2,
    num_bodies: int = 1,
    device: str = "cuda:0",
):
    """Create a test RigidObject instance with mocked dependencies."""
    from isaaclab_newton.assets.rigid_object.rigid_object import RigidObject

    body_names = [f"body_{i}" for i in range(num_bodies)]

    rigid_object = object.__new__(RigidObject)
    _initialize_mock_asset(rigid_object)

    rigid_object.cfg = RigidObjectCfg(
        prim_path="/World/Object",
    )

    mock_view = MockNewtonArticulationView(
        num_instances=num_instances,
        num_bodies=num_bodies,
        num_joints=0,
        device=device,
        joint_names=[],
        body_names=body_names,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    object.__setattr__(rigid_object, "_root_view", mock_view)
    object.__setattr__(rigid_object, "_device", device)
    object.__setattr__(rigid_object, "_check_shapes", not args.no_shape_checks)

    from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData

    data = RigidObjectData(mock_view, device)
    object.__setattr__(rigid_object, "_data", data)

    mock_inst_wrench = MockWrenchComposer(rigid_object)
    mock_perm_wrench = MockWrenchComposer(rigid_object)
    object.__setattr__(rigid_object, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(rigid_object, "_permanent_wrench_composer", mock_perm_wrench)

    object.__setattr__(rigid_object, "actuators", {})
    object.__setattr__(rigid_object, "_ALL_INDICES", wp.array(np.arange(num_instances, dtype=np.int32), device=device))
    object.__setattr__(
        rigid_object, "_ALL_BODY_INDICES", wp.array(np.arange(num_bodies, dtype=np.int32), device=device)
    )
    object.__setattr__(rigid_object, "_ALL_ENV_MASK", wp.ones((num_instances,), dtype=wp.bool, device=device))
    object.__setattr__(rigid_object, "_ALL_BODY_MASK", wp.ones((num_bodies,), dtype=wp.bool, device=device))

    # set information about rigid body into data
    data.body_names = body_names

    return rigid_object, mock_view


# =============================================================================
# Input Generators (Torch-only for Newton backend)
def create_test_collection(
    num_instances: int = 2,
    num_bodies: int = 3,
    device: str = "cuda:0",
):
    """Create a test RigidObjectCollection instance with mocked dependencies."""
    from isaaclab_newton.assets.rigid_object_collection.rigid_object_collection import RigidObjectCollection

    object_names = [f"object_{i}" for i in range(num_bodies)]

    collection = object.__new__(RigidObjectCollection)
    _initialize_mock_asset(collection)

    from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg

    rigid_objects = {name: RigidObjectCfg(prim_path=f"/World/{name}") for name in object_names}
    collection.cfg = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

    mock_view = MockNewtonCollectionView(
        num_envs=num_instances,
        num_bodies=num_bodies,
        device=device,
        body_names=object_names,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    object.__setattr__(collection, "_root_view", mock_view)
    object.__setattr__(collection, "_device", device)
    object.__setattr__(collection, "_body_names_list", object_names)
    object.__setattr__(collection, "_check_shapes", not args.no_shape_checks)

    from isaaclab_newton.assets.rigid_object_collection.rigid_object_collection_data import RigidObjectCollectionData

    data = RigidObjectCollectionData(mock_view, num_bodies, device)
    object.__setattr__(collection, "_data", data)

    mock_inst_wrench = MockWrenchComposer(collection)
    mock_perm_wrench = MockWrenchComposer(collection)
    object.__setattr__(collection, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(collection, "_permanent_wrench_composer", mock_perm_wrench)

    object.__setattr__(
        collection, "_ALL_ENV_INDICES", wp.array(np.arange(num_instances, dtype=np.int32), device=device)
    )
    object.__setattr__(collection, "_ALL_BODY_INDICES", wp.array(np.arange(num_bodies, dtype=np.int32), device=device))
    object.__setattr__(collection, "_ALL_ENV_MASK", wp.ones((num_instances,), dtype=wp.bool, device=device))
    object.__setattr__(collection, "_ALL_BODY_MASK", wp.ones((num_bodies,), dtype=wp.bool, device=device))

    # Temporary 2D wrench buffer for write_data_to_sim
    object.__setattr__(
        collection,
        "_wrench_buffer",
        wp.zeros((num_instances, num_bodies), dtype=wp.spatial_vectorf, device=device),
    )

    return collection, mock_view


# =============================================================================
# Input Generators (Torch-only for Newton backend)


def _refresh_articulation_data(data, _config) -> None:
    data._sim_timestamp += 1.0
    data._fk_timestamp = data._sim_timestamp


def _refresh_rigid_object_data(mock_view, data, config) -> None:
    num_instances, num_bodies, device = config.num_instances, config.num_bodies, config.device
    root_transforms = np.random.randn(num_instances, 1, 7).astype(np.float32)
    root_transforms[..., 3:7] /= np.linalg.norm(root_transforms[..., 3:7], axis=-1, keepdims=True)
    mock_view.set_mock_root_transforms(wp.array(root_transforms, dtype=wp.transformf, device=device))
    mock_view.set_mock_root_velocities(
        wp.array(
            np.random.randn(num_instances, 1, 6).astype(np.float32),
            dtype=wp.spatial_vectorf,
            device=device,
        )
    )
    link_transforms = np.random.randn(num_instances, 1, num_bodies, 7).astype(np.float32)
    link_transforms[..., 3:7] /= np.linalg.norm(link_transforms[..., 3:7], axis=-1, keepdims=True)
    mock_view.set_mock_link_transforms(wp.array(link_transforms, dtype=wp.transformf, device=device))
    mock_view.set_mock_link_velocities(
        wp.array(
            np.random.randn(num_instances, 1, num_bodies, 6).astype(np.float32),
            dtype=wp.spatial_vectorf,
            device=device,
        )
    )
    mock_view.set_mock_coms(
        wp.array(
            np.random.randn(num_instances, 1, num_bodies, 3).astype(np.float32),
            dtype=wp.vec3f,
            device=device,
        )
    )
    mock_view.set_mock_inertias(
        wp.array(
            np.random.randn(num_instances, 1, num_bodies, 9).astype(np.float32),
            dtype=wp.mat33f,
            device=device,
        )
    )
    mock_view.set_mock_masses(
        wp.array(
            (np.random.rand(num_instances, 1, num_bodies) * 10 + 0.1).astype(np.float32),
            dtype=wp.float32,
            device=device,
        )
    )
    data._sim_timestamp += 1.0


def _refresh_collection_data(mock_view, data, config) -> None:
    num_instances, num_bodies, device = config.num_instances, config.num_bodies, config.device
    root_transforms = np.random.randn(num_instances, num_bodies, 7).astype(np.float32)
    root_transforms[..., 3:7] /= np.linalg.norm(root_transforms[..., 3:7], axis=-1, keepdims=True)
    mock_view.set_mock_root_transforms(wp.array(root_transforms, dtype=wp.transformf, device=device))
    mock_view.set_mock_root_velocities(
        wp.array(
            np.random.randn(num_instances, num_bodies, 6).astype(np.float32),
            dtype=wp.spatial_vectorf,
            device=device,
        )
    )
    mock_view.set_mock_coms(
        wp.array(
            np.random.randn(num_instances, num_bodies, 1, 3).astype(np.float32),
            dtype=wp.vec3f,
            device=device,
        )
    )
    mock_view.set_mock_masses(
        wp.array(
            (np.random.rand(num_instances, num_bodies, 1) * 10 + 0.1).astype(np.float32),
            dtype=wp.float32,
            device=device,
        )
    )
    mock_view.set_mock_inertias(
        wp.array(
            np.random.randn(num_instances, num_bodies, 1, 9).astype(np.float32),
            dtype=wp.mat33f,
            device=device,
        )
    )
    data._sim_timestamp += 1.0


def _create_articulation_data_target(config):
    from isaaclab_newton.assets.articulation import articulation_data as data_module

    simulation_manager = globals().get("NewtonSimulationManager", data_module.SimulationManager)
    data_type = globals().get("NewtonArticulationData", data_module.ArticulationData)
    model = simulation_manager.get_model()
    _configure_articulation_model(model, config.num_instances, config.num_bodies, config.num_joints)
    mock_view = MockNewtonArticulationView(
        num_instances=config.num_instances,
        num_bodies=config.num_bodies,
        num_joints=config.num_joints,
        device=config.device,
    )
    mock_view.set_random_mock_data()
    mock_view.eval_jacobian = lambda state, *, J, joint_S_s: None
    mock_view.eval_mass_matrix = lambda state, *, H, J, body_I_s, joint_S_s: None
    data = data_type(mock_view, config.device)
    data._apply_ordering_maps_after_resolve()
    return data, lambda cfg, _mock_view=mock_view: _refresh_articulation_data(data, cfg)


def _create_data_target(component, config):
    if component == "articulation":
        return _create_articulation_data_target(config)
    if component == "rigid_object":
        from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData

        mock_view = MockNewtonArticulationView(
            num_instances=config.num_instances,
            num_bodies=config.num_bodies,
            num_joints=0,
            device=config.device,
        )
        mock_view.set_random_mock_data()
        data = RigidObjectData(mock_view, config.device)
        return data, lambda cfg: _refresh_rigid_object_data(mock_view, data, cfg)
    from isaaclab_newton.assets.rigid_object_collection.rigid_object_collection_data import (
        RigidObjectCollectionData,
    )

    mock_view = MockNewtonCollectionView(
        num_envs=config.num_instances, num_bodies=config.num_bodies, device=config.device
    )
    mock_view.set_random_mock_data()
    data = RigidObjectCollectionData(mock_view, config.num_bodies, config.device)
    return data, lambda cfg: _refresh_collection_data(mock_view, data, cfg)


def _manager_modules(component: str) -> tuple[str, str]:
    base = f"isaaclab_newton.assets.{component}.{component}"
    return f"{base}_data.SimulationManager", f"{base}.SimulationManager"


@contextmanager
def open_asset_targets(adapter, request, method_config, data_config):
    """Open manager patches and the selected mocked Newton asset target."""
    _load_runtime_symbols()
    args.no_shape_checks = not request.check_shapes
    with ExitStack() as stack:
        for module in _manager_modules(adapter.component):
            importlib.import_module(module.rsplit(".", 1)[0])
            stack.enter_context(create_mock_newton_manager(module, gravity=(0.0, 0.0, -9.81)))
        if adapter.component == "articulation":
            target, _ = create_test_articulation(
                num_instances=method_config.num_instances,
                num_bodies=method_config.num_bodies,
                num_joints=method_config.num_joints,
                device=method_config.device,
            )
        elif adapter.component == "rigid_object":
            target, _ = create_test_rigid_object(
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
