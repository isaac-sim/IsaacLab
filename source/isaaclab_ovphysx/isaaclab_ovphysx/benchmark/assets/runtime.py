# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy Omniverse PhysX target construction for asset micro-benchmarks."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from isaaclab.benchmark.asset_suites.types import AssetBenchmarkTargets

args = SimpleNamespace(no_shape_checks=False)


def _load_runtime_symbols() -> None:
    global ArticulationCfg, MockOvPhysxBindingSet, RigidObjectCfg, RigidObjectCollectionCfg

    from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
    from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
    from isaaclab.assets.rigid_object_collection.rigid_object_collection_cfg import RigidObjectCollectionCfg

    from isaaclab_ovphysx.test.mock_interfaces import MockOvPhysxBindingSet


def create_test_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    device: str = "cuda:0",
):
    """Create a test Articulation instance with mocked dependencies."""
    from isaaclab_ovphysx.assets.articulation.articulation import Articulation
    from isaaclab_ovphysx.assets.articulation.articulation_data import ArticulationData

    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]
    binding_set = MockOvPhysxBindingSet(
        num_instances=num_instances,
        num_joints=num_joints,
        num_bodies=num_bodies,
        joint_names=joint_names,
        body_names=body_names,
    )
    binding_set.set_random_data()
    mock_view = binding_set.view

    articulation = object.__new__(Articulation)
    articulation.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        soft_joint_pos_limit_factor=1.0,
        actuators={},
    )
    object.__setattr__(articulation, "_initialize_handle", None)
    object.__setattr__(articulation, "_invalidate_initialize_handle", None)
    object.__setattr__(articulation, "_prim_deletion_handle", None)
    object.__setattr__(articulation, "_debug_vis_handle", None)
    object.__setattr__(articulation, "_root_view", mock_view)
    object.__setattr__(articulation, "_device", device)
    object.__setattr__(articulation, "_check_shapes", not args.no_shape_checks)
    object.__setattr__(articulation, "_num_instances", num_instances)
    object.__setattr__(articulation, "_num_bodies", num_bodies)
    object.__setattr__(articulation, "_num_joints", num_joints)
    object.__setattr__(articulation, "_num_fixed_tendons", 0)
    object.__setattr__(articulation, "_num_spatial_tendons", 0)
    object.__setattr__(articulation, "_is_fixed_base", False)
    object.__setattr__(articulation, "_joint_names", joint_names)
    object.__setattr__(articulation, "_body_names", body_names)
    object.__setattr__(articulation, "_fixed_tendon_names", [])
    object.__setattr__(articulation, "_spatial_tendon_names", [])

    data = ArticulationData(mock_view, device)
    data._apply_ordering_maps_after_resolve()
    object.__setattr__(articulation, "_data", data)
    object.__setattr__(articulation, "actuators", {})
    object.__setattr__(articulation, "_has_implicit_actuators", False)
    articulation._create_buffers()

    return articulation, mock_view


def create_test_rigid_object(
    num_instances: int = 2,
    num_bodies: int = 1,
    device: str = "cuda:0",
):
    """Create a test RigidObject instance with mocked dependencies."""
    from isaaclab_ovphysx.assets.rigid_object.rigid_object import RigidObject
    from isaaclab_ovphysx.assets.rigid_object.rigid_object_data import RigidObjectData

    if num_bodies != 1:
        raise ValueError(f"RigidObject requires exactly one body, got {num_bodies}.")

    binding_set = MockOvPhysxBindingSet(
        num_instances=num_instances,
        num_joints=0,
        num_bodies=1,
        body_names=["base_link"],
        asset_kind="rigid_object",
    )
    binding_set.set_random_data()
    mock_view = binding_set.view

    rigid_object = object.__new__(RigidObject)
    rigid_object.cfg = RigidObjectCfg(prim_path="/World/Object")
    object.__setattr__(rigid_object, "_initialize_handle", None)
    object.__setattr__(rigid_object, "_invalidate_initialize_handle", None)
    object.__setattr__(rigid_object, "_prim_deletion_handle", None)
    object.__setattr__(rigid_object, "_debug_vis_handle", None)
    object.__setattr__(rigid_object, "_root_view", mock_view)
    object.__setattr__(rigid_object, "_device", device)
    object.__setattr__(rigid_object, "_check_shapes", not args.no_shape_checks)
    object.__setattr__(rigid_object, "_num_instances", num_instances)
    object.__setattr__(rigid_object, "_num_bodies", 1)
    object.__setattr__(rigid_object, "_body_names", ["base_link"])

    data = RigidObjectData(mock_view, device, check_shapes=not args.no_shape_checks)
    object.__setattr__(rigid_object, "_data", data)
    rigid_object._create_buffers()

    return rigid_object, mock_view


def create_test_collection(
    num_instances: int = 2,
    num_bodies: int = 3,
    device: str = "cuda:0",
):
    """Create a test RigidObjectCollection instance with mocked dependencies."""
    from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg

    from isaaclab_ovphysx.assets.rigid_object_collection.rigid_object_collection import RigidObjectCollection
    from isaaclab_ovphysx.assets.rigid_object_collection.rigid_object_collection_data import RigidObjectCollectionData

    object_names = [f"object_{i}" for i in range(num_bodies)]
    binding_set = MockOvPhysxBindingSet(
        num_instances=num_instances,
        num_joints=0,
        num_bodies=num_bodies,
        body_names=object_names,
    )
    binding_set.set_random_data()
    mock_view = binding_set.view

    collection = object.__new__(RigidObjectCollection)
    rigid_objects = {name: RigidObjectCfg(prim_path=f"/World/{name}") for name in object_names}
    collection.cfg = RigidObjectCollectionCfg(rigid_objects=rigid_objects)
    object.__setattr__(collection, "_initialize_handle", None)
    object.__setattr__(collection, "_invalidate_initialize_handle", None)
    object.__setattr__(collection, "_prim_deletion_handle", None)
    object.__setattr__(collection, "_debug_vis_handle", None)
    object.__setattr__(collection, "_root_view", mock_view)
    object.__setattr__(collection, "_device", device)
    object.__setattr__(collection, "_check_shapes", not args.no_shape_checks)
    object.__setattr__(collection, "_num_instances", num_instances)
    object.__setattr__(collection, "_num_bodies", num_bodies)
    object.__setattr__(collection, "_body_names_list", object_names)
    object.__setattr__(collection, "_object_names", object_names)

    data = RigidObjectCollectionData(mock_view, num_bodies, device)
    object.__setattr__(collection, "_data", data)
    collection._create_buffers()

    return collection, mock_view


def _refresh_data(data, component: str) -> None:
    data._sim_timestamp += 1.0
    if component == "articulation":
        data._fk_timestamp = data._sim_timestamp


def _create_data_target(component, config):
    binding_kwargs = {
        "num_instances": config.num_instances,
        "num_joints": config.num_joints if component == "articulation" else 0,
        "num_bodies": config.num_bodies,
    }
    if component == "rigid_object":
        binding_kwargs["asset_kind"] = "rigid_object"
    binding_set = MockOvPhysxBindingSet(**binding_kwargs)
    binding_set.set_random_data()
    mock_view = binding_set.view
    if component == "articulation":
        from isaaclab_ovphysx.assets.articulation.articulation_data import ArticulationData

        data = ArticulationData(mock_view, config.device)
        data._apply_ordering_maps_after_resolve()
    elif component == "rigid_object":
        from isaaclab_ovphysx.assets.rigid_object.rigid_object_data import RigidObjectData

        data = RigidObjectData(mock_view, config.device, check_shapes=not args.no_shape_checks)
    else:
        from isaaclab_ovphysx.assets.rigid_object_collection.rigid_object_collection_data import (
            RigidObjectCollectionData,
        )

        data = RigidObjectCollectionData(mock_view, config.num_bodies, config.device)
    return data, lambda _config: _refresh_data(data, component)


@contextmanager
def open_asset_targets(adapter, request, method_config, data_config):
    """Open the selected kitless Omniverse PhysX mock target."""
    _load_runtime_symbols()
    args.no_shape_checks = not request.check_shapes
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
