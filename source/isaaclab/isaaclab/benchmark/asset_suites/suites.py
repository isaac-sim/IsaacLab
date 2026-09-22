# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declarative asset method and property micro-benchmark definitions."""

from __future__ import annotations

from .generators import (
    make_indexed_generators,
    make_item_selector_generators,
    make_mask_generator,
    make_scaled_tensor_generator,
    make_signed_joint_limits_generator,
)
from .types import AssetBenchmarkSuite, AssetMethodSpec, AssetPropertySpec


def _indexed(
    method_name: str,
    tensor_shapes,
    index_dimensions,
    category: str,
    *,
    requires: str | None = None,
) -> AssetMethodSpec:
    item_keys = tuple(name for name in index_dimensions if name in {"body_ids", "joint_ids"})
    if item_keys:
        input_generators = make_item_selector_generators(tensor_shapes, index_dimensions)
    else:
        input_generators = make_indexed_generators(tensor_shapes, index_dimensions)
    return AssetMethodSpec(
        name=method_name,
        method_name=method_name,
        input_generators=input_generators,
        category=category,
        requires=requires,
    )


def _masked(method_name: str, tensor_shapes, mask_dimensions, category: str) -> AssetMethodSpec:
    return AssetMethodSpec(
        name=method_name,
        method_name=method_name,
        input_generators={"warp_mask": make_mask_generator(tensor_shapes, mask_dimensions)},
        category=category,
        requires="warp_mask",
    )


def _scaled_tensor_field(spec: AssetMethodSpec, field_name: str, scale: float) -> AssetMethodSpec:
    return AssetMethodSpec(
        name=spec.name,
        method_name=spec.method_name,
        input_generators={
            mode: make_scaled_tensor_generator(generator, field_name, scale)
            for mode, generator in spec.input_generators.items()
        },
        category=spec.category,
        requires=spec.requires,
        prepare_target=spec.prepare_target,
    )


_ARTICULATION_GENERATOR_SCALES = {
    "write_joint_velocity_limit_to_sim": ("limits", 10.0),
    "write_joint_effort_limit_to_sim": ("limits", 100.0),
    "write_joint_armature_to_sim": ("armature", 0.1),
    "write_joint_friction_coefficient_to_sim": ("joint_friction_coeff", 0.5),
}


def _signed_joint_limits(spec: AssetMethodSpec) -> AssetMethodSpec:
    return AssetMethodSpec(
        name=spec.name,
        method_name=spec.method_name,
        input_generators={
            mode: make_signed_joint_limits_generator(generator) for mode, generator in spec.input_generators.items()
        },
        category=spec.category,
        requires=spec.requires,
        prepare_target=spec.prepare_target,
    )


def _with_articulation_ranges(spec: AssetMethodSpec, writer: str) -> AssetMethodSpec:
    """Give the joint-parameter inputs of ``writer`` ordered limits or their expected magnitude."""
    if writer == "write_joint_position_limit_to_sim":
        return _signed_joint_limits(spec)
    if scale := _ARTICULATION_GENERATOR_SCALES.get(writer):
        return _scaled_tensor_field(spec, *scale)
    return spec


_JOINT_INDEXED = {"env_ids": "instances", "joint_ids": "joints"}
_BODY_INDEXED = {"env_ids": "instances", "body_ids": "bodies"}

# (method name, tensor shapes, index dimensions, category) of every indexed articulation writer.
_ARTICULATION_WRITERS = (
    ("write_root_state_to_sim", {"root_state": ("instances", 13)}, {"env_ids": "instances"}, "root_state"),
    ("write_root_com_state_to_sim", {"root_state": ("instances", 13)}, {"env_ids": "instances"}, "root_state"),
    ("write_root_link_state_to_sim", {"root_state": ("instances", 13)}, {"env_ids": "instances"}, "root_state"),
    ("write_root_link_pose_to_sim", {"root_pose": ("instances", 7)}, {"env_ids": "instances"}, "root_pose"),
    ("write_root_com_pose_to_sim", {"root_pose": ("instances", 7)}, {"env_ids": "instances"}, "root_pose"),
    ("write_root_link_velocity_to_sim", {"root_velocity": ("instances", 6)}, {"env_ids": "instances"}, "root_velocity"),
    ("write_root_com_velocity_to_sim", {"root_velocity": ("instances", 6)}, {"env_ids": "instances"}, "root_velocity"),
    (
        "write_joint_state_to_sim",
        {"position": ("instances", "joints"), "velocity": ("instances", "joints")},
        _JOINT_INDEXED,
        "joint_state",
    ),
    ("write_joint_position_to_sim", {"position": ("instances", "joints")}, _JOINT_INDEXED, "joint_state"),
    ("write_joint_velocity_to_sim", {"velocity": ("instances", "joints")}, _JOINT_INDEXED, "joint_state"),
    ("write_joint_stiffness_to_sim", {"stiffness": ("instances", "joints")}, _JOINT_INDEXED, "joint_params"),
    ("write_joint_damping_to_sim", {"damping": ("instances", "joints")}, _JOINT_INDEXED, "joint_params"),
    ("write_joint_position_limit_to_sim", {"limits": ("instances", "joints", 2)}, _JOINT_INDEXED, "joint_params"),
    ("write_joint_velocity_limit_to_sim", {"limits": ("instances", "joints")}, _JOINT_INDEXED, "joint_params"),
    ("write_joint_effort_limit_to_sim", {"limits": ("instances", "joints")}, _JOINT_INDEXED, "joint_params"),
    ("write_joint_armature_to_sim", {"armature": ("instances", "joints")}, _JOINT_INDEXED, "joint_params"),
    (
        "write_joint_friction_coefficient_to_sim",
        {"joint_friction_coeff": ("instances", "joints")},
        _JOINT_INDEXED,
        "joint_params",
    ),
    ("set_joint_position_target", {"target": ("instances", "joints")}, _JOINT_INDEXED, "joint_targets"),
    ("set_joint_velocity_target", {"target": ("instances", "joints")}, _JOINT_INDEXED, "joint_targets"),
    ("set_joint_effort_target", {"target": ("instances", "joints")}, _JOINT_INDEXED, "joint_targets"),
    ("set_masses", {"masses": ("instances", "bodies")}, _BODY_INDEXED, "body_props"),
    ("set_coms", {"coms": ("instances", "bodies", 7)}, _BODY_INDEXED, "body_props"),
    ("set_inertias", {"inertias": ("instances", "bodies", 9)}, _BODY_INDEXED, "body_props"),
    (
        "set_external_force_and_torque",
        {"forces": ("instances", "bodies", 3), "torques": ("instances", "bodies", 3)},
        {"env_ids": "instances"},
        "external_wrench",
    ),
)

_ARTICULATION_PLAIN = tuple(
    _with_articulation_ranges(_indexed(method_name, shapes, index_dimensions, category), method_name)
    for method_name, shapes, index_dimensions, category in _ARTICULATION_WRITERS
)

# Writers without a mask variant.
_UNMASKED_ARTICULATION_WRITERS = frozenset(
    {
        "write_root_state_to_sim",
        "write_root_com_state_to_sim",
        "write_root_link_state_to_sim",
        "set_external_force_and_torque",
    }
)

_ARTICULATION_MASKS = tuple(
    _with_articulation_ranges(
        _masked(
            f"{method_name}_mask",
            shapes,
            {
                "env_mask": "instances",
                **({"joint_mask": "joints"} if "joint_ids" in index_dimensions else {}),
                **({"body_mask": "bodies"} if "body_ids" in index_dimensions else {}),
            },
            category,
        ),
        method_name,
    )
    for method_name, shapes, index_dimensions, category in _ARTICULATION_WRITERS
    if method_name not in _UNMASKED_ARTICULATION_WRITERS
)


def _finder_specs(method_name: str) -> tuple[AssetMethodSpec, ...]:
    return (
        AssetMethodSpec(
            name=f"{method_name}_default",
            method_name=method_name,
            input_generators={"default": lambda _config: {"name_keys": ".*"}},
            category="selector_finder",
        ),
        AssetMethodSpec(
            name=f"{method_name}_proxy_cold",
            method_name=method_name,
            input_generators={"proxy_cold": lambda _config: {"name_keys": ".*", "as_proxy": True}},
            category="selector_finder",
            prepare_target=lambda target: target._clear_selector_cache(),
        ),
        AssetMethodSpec(
            name=f"{method_name}_proxy_cached",
            method_name=method_name,
            input_generators={"proxy_cached": lambda _config: {"name_keys": ".*", "as_proxy": True}},
            category="selector_finder",
        ),
    )


_ARTICULATION_FINDERS = _finder_specs("find_bodies") + _finder_specs("find_joints")


def _rigid_writers(prefix: str) -> tuple[tuple[str, dict, dict, str], ...]:
    """Return the rigid pose, velocity, and body-property writers as ``_ARTICULATION_WRITERS``-style rows.

    ``prefix`` is ``"root"`` for rigid objects and ``"body"`` for rigid object collections, whose
    writers address individual bodies.
    """
    per_body = prefix == "body"
    pose_key, velocity_key = (f"{prefix}_poses", f"{prefix}_velocities") if per_body else ("root_pose", "root_velocity")
    pose_shape = ("instances", "bodies", 7) if per_body else ("instances", 7)
    velocity_shape = ("instances", "bodies", 6) if per_body else ("instances", 6)
    kinematic_indices = _BODY_INDEXED if per_body else {"env_ids": "instances"}
    return (
        (f"write_{prefix}_link_pose_to_sim", {pose_key: pose_shape}, kinematic_indices, f"{prefix}_pose"),
        (f"write_{prefix}_com_pose_to_sim", {pose_key: pose_shape}, kinematic_indices, f"{prefix}_pose"),
        (
            f"write_{prefix}_link_velocity_to_sim",
            {velocity_key: velocity_shape},
            kinematic_indices,
            f"{prefix}_velocity",
        ),
        (
            f"write_{prefix}_com_velocity_to_sim",
            {velocity_key: velocity_shape},
            kinematic_indices,
            f"{prefix}_velocity",
        ),
        ("set_masses", {"masses": ("instances", "bodies")}, _BODY_INDEXED, "body_props"),
        ("set_coms", {"coms": ("instances", "bodies", 3)}, _BODY_INDEXED, "body_props"),
        ("set_inertias", {"inertias": ("instances", "bodies", 9)}, _BODY_INDEXED, "body_props"),
    )


def _rigid_common(prefix: str) -> tuple[AssetMethodSpec, ...]:
    return tuple(_indexed(*writer) for writer in _rigid_writers(prefix)) + (
        _indexed(
            "set_external_force_and_torque",
            {"forces": ("instances", "bodies", 3), "torques": ("instances", "bodies", 3)},
            {"env_ids": "instances"},
            "external_wrench",
        ),
    )


def _rigid_masks(prefix: str) -> tuple[AssetMethodSpec, ...]:
    return tuple(
        _masked(
            f"{method_name}_mask",
            shapes,
            {"env_mask": "instances", **({"body_mask": "bodies"} if "body_ids" in index_dimensions else {})},
            category,
        )
        for method_name, shapes, index_dimensions, category in _rigid_writers(prefix)
    )


_RIGID_LEGACY = (
    _indexed(
        "write_root_state_to_sim",
        {"root_state": ("instances", 13)},
        {"env_ids": "instances"},
        "root_state",
        requires="physx_legacy_state",
    ),
    _indexed(
        "write_root_com_state_to_sim",
        {"root_state": ("instances", 13)},
        {"env_ids": "instances"},
        "root_state",
        requires="physx_legacy_state",
    ),
    _indexed(
        "write_root_link_state_to_sim",
        {"root_state": ("instances", 13)},
        {"env_ids": "instances"},
        "root_state",
        requires="physx_legacy_state",
    ),
    _indexed(
        "write_root_pose_to_sim",
        {"root_pose": ("instances", 7)},
        {"env_ids": "instances"},
        "root_pose",
        requires="physx_legacy_state",
    ),
    _indexed(
        "write_root_velocity_to_sim",
        {"root_velocity": ("instances", 6)},
        {"env_ids": "instances"},
        "root_velocity",
        requires="physx_legacy_state",
    ),
)

_COLLECTION_LEGACY = (
    _indexed(
        "write_body_state_to_sim",
        {"body_states": ("instances", "bodies", 13)},
        {"env_ids": "instances", "body_ids": "bodies"},
        "body_state",
        requires="physx_legacy_state",
    ),
    _indexed(
        "write_body_link_state_to_sim",
        {"body_states": ("instances", "bodies", 13)},
        {"env_ids": "instances", "body_ids": "bodies"},
        "body_state",
        requires="physx_legacy_state",
    ),
    _indexed(
        "write_body_com_state_to_sim",
        {"body_states": ("instances", "bodies", 13)},
        {"env_ids": "instances", "body_ids": "bodies"},
        "body_state",
        requires="physx_legacy_state",
    ),
    _indexed(
        "write_body_pose_to_sim",
        {"body_poses": ("instances", "bodies", 7)},
        {"env_ids": "instances", "body_ids": "bodies"},
        "body_pose",
        requires="physx_legacy_state",
    ),
    _indexed(
        "write_body_velocity_to_sim",
        {"body_velocities": ("instances", "bodies", 6)},
        {"env_ids": "instances", "body_ids": "bodies"},
        "body_velocity",
        requires="physx_legacy_state",
    ),
)

# Property definitions are the shared union; backend adapters select their supported subset.
_RIGID_PROPERTIES = tuple(
    AssetPropertySpec(name)
    for name in sorted(
        {
            "body_com_acc_w",
            "body_com_ang_acc_w",
            "body_com_ang_vel_w",
            "body_com_lin_acc_w",
            "body_com_lin_vel_w",
            "body_com_pos_b",
            "body_com_pos_w",
            "body_com_pose_b",
            "body_com_pose_w",
            "body_com_quat_b",
            "body_com_quat_w",
            "body_com_vel_w",
            "body_inertia",
            "body_link_ang_vel_w",
            "body_link_lin_vel_w",
            "body_link_pos_w",
            "body_link_pose_w",
            "body_link_quat_w",
            "body_link_vel_w",
            "body_mass",
            "default_root_pose",
            "default_root_vel",
            "heading_w",
            "projected_gravity_b",
            "root_com_ang_vel_b",
            "root_com_ang_vel_w",
            "root_com_lin_vel_b",
            "root_com_lin_vel_w",
            "root_com_pos_w",
            "root_com_pose_w",
            "root_com_quat_w",
            "root_com_vel_w",
            "root_link_ang_vel_b",
            "root_link_ang_vel_w",
            "root_link_lin_vel_b",
            "root_link_lin_vel_w",
            "root_link_pos_w",
            "root_link_pose_w",
            "root_link_quat_w",
            "root_link_vel_w",
        }
    )
)

_ARTICULATION_PROPERTIES = tuple(
    AssetPropertySpec(name)
    for name in sorted(
        {prop.name for prop in _RIGID_PROPERTIES}
        | {
            "body_com_jacobian_w",
            "body_link_jacobian_w",
            "default_joint_pos",
            "default_joint_vel",
            "joint_acc",
            "joint_armature",
            "joint_damping",
            "joint_effort_limits",
            "joint_friction_coeff",
            "joint_pos",
            "joint_pos_limits",
            "joint_pos_target",
            "joint_stiffness",
            "joint_vel",
            "joint_vel_limits",
            "joint_vel_target",
            "mass_matrix",
            "soft_joint_pos_limits",
            "soft_joint_vel_limits",
            "gravity_compensation_forces",
            "has_body_ordering",
            "has_joint_ordering",
            "joint_dynamic_friction_coeff",
            "joint_viscous_friction_coeff",
            "joint_pos_limits_lower",
            "joint_pos_limits_upper",
        }
    )
)

_COLLECTION_PROPERTIES = tuple(
    AssetPropertySpec(name)
    for name in sorted(
        {
            "body_acc_w",
            "body_ang_acc_w",
            "body_ang_vel_w",
            "body_com_acc_w",
            "body_com_ang_acc_w",
            "body_com_ang_vel_b",
            "body_com_ang_vel_w",
            "body_com_lin_acc_w",
            "body_com_lin_vel_b",
            "body_com_lin_vel_w",
            "body_com_pos_b",
            "body_com_pos_w",
            "body_com_pose_b",
            "body_com_pose_w",
            "body_com_quat_b",
            "body_com_quat_w",
            "body_com_state_w",
            "body_com_vel_w",
            "body_inertia",
            "body_lin_acc_w",
            "body_lin_vel_w",
            "body_link_ang_vel_b",
            "body_link_ang_vel_w",
            "body_link_lin_vel_b",
            "body_link_lin_vel_w",
            "body_link_pos_w",
            "body_link_pose_w",
            "body_link_quat_w",
            "body_link_state_w",
            "body_link_vel_w",
            "body_mass",
            "body_pos_w",
            "body_pose_w",
            "body_quat_w",
            "body_state_w",
            "body_vel_w",
            "com_pos_b",
            "com_quat_b",
            "default_body_pose",
            "default_body_state",
            "default_body_vel",
            "default_object_pose",
            "default_object_state",
            "default_object_vel",
            "heading_w",
            "object_acc_w",
            "object_ang_acc_w",
            "object_ang_vel_b",
            "object_ang_vel_w",
            "object_com_acc_w",
            "object_com_ang_acc_w",
            "object_com_ang_vel_b",
            "object_com_ang_vel_w",
            "object_com_lin_acc_w",
            "object_com_lin_vel_b",
            "object_com_lin_vel_w",
            "object_com_pos_b",
            "object_com_pos_w",
            "object_com_pose_b",
            "object_com_pose_w",
            "object_com_quat_b",
            "object_com_quat_w",
            "object_com_state_w",
            "object_com_vel_w",
            "object_lin_acc_w",
            "object_lin_vel_b",
            "object_lin_vel_w",
            "object_link_ang_vel_b",
            "object_link_ang_vel_w",
            "object_link_lin_vel_b",
            "object_link_lin_vel_w",
            "object_link_pos_w",
            "object_link_pose_w",
            "object_link_quat_w",
            "object_link_state_w",
            "object_link_vel_w",
            "object_pos_w",
            "object_pose_w",
            "object_quat_w",
            "object_state_w",
            "object_vel_w",
            "projected_gravity_b",
        }
    )
)

_SUITES = {
    "articulation": AssetBenchmarkSuite(
        component="articulation",
        methods=_ARTICULATION_PLAIN + _ARTICULATION_FINDERS + _ARTICULATION_MASKS,
        properties=_ARTICULATION_PROPERTIES,
    ),
    "rigid_object": AssetBenchmarkSuite(
        component="rigid_object",
        methods=_RIGID_LEGACY + _rigid_common("root") + _rigid_masks("root"),
        properties=_RIGID_PROPERTIES,
    ),
    "rigid_object_collection": AssetBenchmarkSuite(
        component="rigid_object_collection",
        methods=_COLLECTION_LEGACY + _rigid_common("body") + _rigid_masks("body"),
        properties=_COLLECTION_PROPERTIES,
    ),
}


def get_asset_benchmark_suite(component: str) -> AssetBenchmarkSuite:
    """Return the declarative suite for an asset component."""
    try:
        return _SUITES[component]
    except KeyError as exc:
        raise ValueError(f"Unsupported asset component: {component!r}") from exc
