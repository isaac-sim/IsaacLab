# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX declarations for shared asset micro-benchmark suites."""

from isaaclab.benchmark.asset_suites.adapters import PackageAssetBenchmarkAdapter
from isaaclab.benchmark.asset_suites.dependencies import (
    ARTICULATION_DEPENDENCIES,
    OBJECT_COLLECTION_DEPENDENCIES,
    RIGID_OBJECT_DEPENDENCIES,
)
from isaaclab.benchmark.asset_suites.generators import make_indexed_generators
from isaaclab.benchmark.asset_suites.suites import get_asset_benchmark_suite

_CAPABILITIES = frozenset({"physx_legacy_state", "tensor_fill"})
_COMPONENT_DEFAULTS = {
    "articulation": (12, 11),
    "rigid_object": (1, 0),
    "rigid_object_collection": (4, 0),
}
_ARTICULATION_EXCLUDED = {"joint_pos_limits_lower", "joint_pos_limits_upper"}
_COLLECTION_EXCLUDED = {
    "body_acc_w",
    "body_ang_acc_w",
    "body_ang_vel_w",
    "body_lin_acc_w",
    "body_lin_vel_w",
    "body_pos_w",
    "body_pose_w",
    "body_quat_w",
    "body_vel_w",
    "com_pos_b",
    "com_quat_b",
    "object_com_state_w",
    "object_link_state_w",
}


def _tensor_generator(shapes, indices):
    return make_indexed_generators(shapes, indices)["torch_tensor"]


def _generator_overrides(component: str):
    if component == "rigid_object":
        return {
            ("set_masses", "torch_tensor"): _tensor_generator(
                {"masses": ("instances", "bodies")}, {"env_ids": "instances"}
            ),
            ("set_coms", "torch_tensor"): _tensor_generator(
                {"coms": ("instances", "bodies", 3)}, {"env_ids": "instances"}
            ),
            ("set_inertias", "torch_tensor"): _tensor_generator(
                {"inertias": ("instances", "bodies", 9)}, {"env_ids": "instances"}
            ),
        }
    if component == "rigid_object_collection":
        return {
            ("set_inertias", "torch_tensor"): _tensor_generator(
                {"inertias": ("instances", "bodies", 3, 3)},
                {"env_ids": "instances", "body_ids": "bodies"},
            )
        }
    return {}


def get_asset_benchmark_adapter(component: str) -> PackageAssetBenchmarkAdapter:
    """Return the PhysX adapter for an asset component."""
    default_num_bodies, default_num_joints = _COMPONENT_DEFAULTS[component]
    if component == "articulation":
        excluded = _ARTICULATION_EXCLUDED
    elif component == "rigid_object_collection":
        excluded = _COLLECTION_EXCLUDED
    else:
        excluded = set()
    dependencies = {
        "articulation": ARTICULATION_DEPENDENCIES,
        "rigid_object": RIGID_OBJECT_DEPENDENCIES,
        "rigid_object_collection": OBJECT_COLLECTION_DEPENDENCIES,
    }[component]
    properties = frozenset(
        prop.name for prop in get_asset_benchmark_suite(component).properties if prop.name not in excluded
    )
    return PackageAssetBenchmarkAdapter(
        physics="physx",
        component=component,
        artifact_prefix="",
        runtime_module="isaaclab_physx.benchmark.assets.runtime",
        capabilities=_CAPABILITIES,
        supported_properties=properties,
        default_num_bodies=default_num_bodies,
        default_num_joints=default_num_joints,
        generator_overrides=_generator_overrides(component),
        property_dependency_overrides=dependencies,
    )
