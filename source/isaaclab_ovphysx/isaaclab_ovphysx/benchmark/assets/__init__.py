# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Omniverse PhysX declarations for shared asset micro-benchmark suites."""

from isaaclab.benchmark.asset_suites.adapters import PackageAssetBenchmarkAdapter
from isaaclab.benchmark.asset_suites.dependencies import (
    ARTICULATION_DEPENDENCIES,
    BODY_COLLECTION_DEPENDENCIES,
    OVPHYSX_RIGID_OBJECT_DEPENDENCIES,
)
from isaaclab.benchmark.asset_suites.generators import make_indexed_generators, make_mask_generator
from isaaclab.benchmark.asset_suites.suites import get_asset_benchmark_suite

_CAPABILITIES = frozenset({"warp_mask"})
_COMPONENT_DEFAULTS = {
    "articulation": (12, 11),
    "rigid_object": (1, 0),
    "rigid_object_collection": (3, 0),
}
_ARTICULATION_EXCLUDED = {
    "gravity_compensation_forces",
    "has_body_ordering",
    "has_joint_ordering",
    "joint_pos_limits_lower",
    "joint_pos_limits_upper",
}
_COLLECTION_EXCLUDED = {"default_body_state", "body_state_w", "body_link_state_w", "body_com_state_w"}


def _com_generator_overrides():
    shapes = {"coms": ("instances", "bodies", 7)}
    indexed = make_indexed_generators(shapes, {"env_ids": "instances", "body_ids": "bodies"})
    mask = make_mask_generator(shapes, {"env_mask": "instances", "body_mask": "bodies"})
    return {
        ("set_coms", "torch_list"): indexed["torch_list"],
        ("set_coms", "torch_tensor"): indexed["torch_tensor"],
        ("set_coms_mask", "warp_mask"): mask,
    }


def get_asset_benchmark_adapter(component: str) -> PackageAssetBenchmarkAdapter:
    """Return the Omniverse PhysX adapter for an asset component."""
    default_num_bodies, default_num_joints = _COMPONENT_DEFAULTS[component]
    if component == "articulation":
        excluded = _ARTICULATION_EXCLUDED
    elif component == "rigid_object_collection":
        excluded = _COLLECTION_EXCLUDED
    else:
        excluded = set()
    dependencies = {
        "articulation": ARTICULATION_DEPENDENCIES,
        "rigid_object": OVPHYSX_RIGID_OBJECT_DEPENDENCIES,
        "rigid_object_collection": BODY_COLLECTION_DEPENDENCIES,
    }[component]
    properties = frozenset(
        prop.name for prop in get_asset_benchmark_suite(component).properties if prop.name not in excluded
    )
    overrides = _com_generator_overrides() if component != "articulation" else {}
    return PackageAssetBenchmarkAdapter(
        physics="ovphysx",
        physics_variant="ovphysx",
        component=component,
        artifact_prefix="ovphysx_",
        runtime_module="isaaclab_ovphysx.benchmark.assets.runtime",
        capabilities=_CAPABILITIES,
        supported_properties=properties,
        default_num_bodies=default_num_bodies,
        default_num_joints=default_num_joints,
        generator_overrides=overrides,
        property_dependency_overrides=dependencies,
    )
