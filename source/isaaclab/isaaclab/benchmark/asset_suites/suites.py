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


_ARTICULATION_PLAIN = (
    _indexed("write_root_state_to_sim", {"root_state": ("instances", 13)}, {"env_ids": "instances"}, "root_state"),
    _indexed("write_root_com_state_to_sim", {"root_state": ("instances", 13)}, {"env_ids": "instances"}, "root_state"),
    _indexed("write_root_link_state_to_sim", {"root_state": ("instances", 13)}, {"env_ids": "instances"}, "root_state"),
    _indexed("write_root_link_pose_to_sim", {"root_pose": ("instances", 7)}, {"env_ids": "instances"}, "root_pose"),
    _indexed("write_root_com_pose_to_sim", {"root_pose": ("instances", 7)}, {"env_ids": "instances"}, "root_pose"),
    _indexed(
        "write_root_link_velocity_to_sim",
        {"root_velocity": ("instances", 6)},
        {"env_ids": "instances"},
        "root_velocity",
    ),
    _indexed(
        "write_root_com_velocity_to_sim",
        {"root_velocity": ("instances", 6)},
        {"env_ids": "instances"},
        "root_velocity",
    ),
    _indexed(
        "write_joint_state_to_sim",
        {"position": ("instances", "joints"), "velocity": ("instances", "joints")},
        {"env_ids": "instances", "joint_ids": "joints"},
        "joint_state",
    ),
    _indexed(
        "write_joint_position_to_sim",
        {"position": ("instances", "joints")},
        {"env_ids": "instances", "joint_ids": "joints"},
        "joint_state",
    ),
    _indexed(
        "write_joint_velocity_to_sim",
        {"velocity": ("instances", "joints")},
        {"env_ids": "instances", "joint_ids": "joints"},
        "joint_state",
    ),
    _indexed(
        "write_joint_stiffness_to_sim",
        {"stiffness": ("instances", "joints")},
        {"env_ids": "instances", "joint_ids": "joints"},
        "joint_params",
    ),
    _indexed(
        "write_joint_damping_to_sim",
        {"damping": ("instances", "joints")},
        {"env_ids": "instances", "joint_ids": "joints"},
        "joint_params",
    ),
    _signed_joint_limits(
        _indexed(
            "write_joint_position_limit_to_sim",
            {"limits": ("instances", "joints", 2)},
            {"env_ids": "instances", "joint_ids": "joints"},
            "joint_params",
        )
    ),
    _scaled_tensor_field(
        _indexed(
            "write_joint_velocity_limit_to_sim",
            {"limits": ("instances", "joints")},
            {"env_ids": "instances", "joint_ids": "joints"},
            "joint_params",
        ),
        "limits",
        10.0,
    ),
    _scaled_tensor_field(
        _indexed(
            "write_joint_effort_limit_to_sim",
            {"limits": ("instances", "joints")},
            {"env_ids": "instances", "joint_ids": "joints"},
            "joint_params",
        ),
        "limits",
        100.0,
    ),
    _scaled_tensor_field(
        _indexed(
            "write_joint_armature_to_sim",
            {"armature": ("instances", "joints")},
            {"env_ids": "instances", "joint_ids": "joints"},
            "joint_params",
        ),
        "armature",
        0.1,
    ),
    _scaled_tensor_field(
        _indexed(
            "write_joint_friction_coefficient_to_sim",
            {"joint_friction_coeff": ("instances", "joints")},
            {"env_ids": "instances", "joint_ids": "joints"},
            "joint_params",
        ),
        "joint_friction_coeff",
        0.5,
    ),
    _indexed(
        "set_joint_position_target",
        {"target": ("instances", "joints")},
        {"env_ids": "instances", "joint_ids": "joints"},
        "joint_targets",
    ),
    _indexed(
        "set_joint_velocity_target",
        {"target": ("instances", "joints")},
        {"env_ids": "instances", "joint_ids": "joints"},
        "joint_targets",
    ),
    _indexed(
        "set_joint_effort_target",
        {"target": ("instances", "joints")},
        {"env_ids": "instances", "joint_ids": "joints"},
        "joint_targets",
    ),
    _indexed(
        "set_masses",
        {"masses": ("instances", "bodies")},
        {"env_ids": "instances", "body_ids": "bodies"},
        "body_props",
    ),
    _indexed(
        "set_coms",
        {"coms": ("instances", "bodies", 7)},
        {"env_ids": "instances", "body_ids": "bodies"},
        "body_props",
    ),
    _indexed(
        "set_inertias",
        {"inertias": ("instances", "bodies", 9)},
        {"env_ids": "instances", "body_ids": "bodies"},
        "body_props",
    ),
    _indexed(
        "set_external_force_and_torque",
        {"forces": ("instances", "bodies", 3), "torques": ("instances", "bodies", 3)},
        {"env_ids": "instances"},
        "external_wrench",
    ),
)

_ARTICULATION_MASKS = tuple(
    _masked(
        f"{spec.method_name}_mask" if not spec.method_name.startswith("set_joint_") else f"{spec.method_name}_mask",
        {
            key: shape
            for key, shape in {
                "root_pose": ("instances", 7),
                "root_velocity": ("instances", 6),
            }.items()
            if key
            in (
                ("root_pose",)
                if "pose" in spec.method_name
                else ("root_velocity",)
                if "velocity_to_sim" in spec.method_name and "joint" not in spec.method_name
                else ()
            )
        },
        {"env_mask": "instances"},
        spec.category,
    )
    for spec in _ARTICULATION_PLAIN[3:7]
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


def _articulation_mask_specs() -> tuple[AssetMethodSpec, ...]:
    specs = list(_ARTICULATION_MASKS)
    for spec in _ARTICULATION_PLAIN[7:20]:
        name = f"{spec.method_name}_mask"
        if spec.method_name == "write_joint_state_to_sim":
            shapes = {"position": ("instances", "joints"), "velocity": ("instances", "joints")}
        elif spec.method_name in ("write_joint_position_to_sim",):
            shapes = {"position": ("instances", "joints")}
        elif spec.method_name in ("write_joint_velocity_to_sim",):
            shapes = {"velocity": ("instances", "joints")}
        elif spec.method_name == "write_joint_stiffness_to_sim":
            shapes = {"stiffness": ("instances", "joints")}
        elif spec.method_name == "write_joint_damping_to_sim":
            shapes = {"damping": ("instances", "joints")}
        elif "limit" in spec.method_name:
            shapes = {
                "limits": (
                    ("instances", "joints", 2)
                    if spec.method_name == "write_joint_position_limit_to_sim"
                    else ("instances", "joints")
                )
            }
        elif spec.method_name == "write_joint_armature_to_sim":
            shapes = {"armature": ("instances", "joints")}
        elif spec.method_name == "write_joint_friction_coefficient_to_sim":
            shapes = {"joint_friction_coeff": ("instances", "joints")}
        else:
            shapes = {"target": ("instances", "joints")}
        mask_spec = _masked(name, shapes, {"env_mask": "instances", "joint_mask": "joints"}, spec.category)
        if spec.method_name == "write_joint_position_limit_to_sim":
            mask_spec = _signed_joint_limits(mask_spec)
        elif scale := _ARTICULATION_GENERATOR_SCALES.get(spec.method_name):
            mask_spec = _scaled_tensor_field(mask_spec, *scale)
        specs.append(mask_spec)
    for spec in _ARTICULATION_PLAIN[20:23]:
        shape = {
            "set_masses": {"masses": ("instances", "bodies")},
            "set_coms": {"coms": ("instances", "bodies", 7)},
            "set_inertias": {"inertias": ("instances", "bodies", 9)},
        }[spec.method_name]
        specs.append(
            _masked(
                f"{spec.method_name}_mask",
                shape,
                {"env_mask": "instances", "body_mask": "bodies"},
                spec.category,
            )
        )
    return tuple(specs)


def _rigid_methods(prefix: str) -> tuple[AssetMethodSpec, ...]:
    return (
        _indexed(
            f"write_{prefix}_link_pose_to_sim",
            {
                f"{prefix}_poses" if prefix == "body" else "root_pose": ("instances", "bodies", 7)
                if prefix == "body"
                else ("instances", 7)
            },
            {"env_ids": "instances", **({"body_ids": "bodies"} if prefix == "body" else {})},
            f"{prefix}_pose" if prefix == "body" else "root_pose",
        ),
        _indexed(
            f"write_{prefix}_com_pose_to_sim",
            {
                f"{prefix}_poses" if prefix == "body" else "root_pose": ("instances", "bodies", 7)
                if prefix == "body"
                else ("instances", 7)
            },
            {"env_ids": "instances", **({"body_ids": "bodies"} if prefix == "body" else {})},
            f"{prefix}_pose" if prefix == "body" else "root_pose",
        ),
        _indexed(
            f"write_{prefix}_link_velocity_to_sim",
            {
                f"{prefix}_velocities" if prefix == "body" else "root_velocity": ("instances", "bodies", 6)
                if prefix == "body"
                else ("instances", 6)
            },
            {"env_ids": "instances", **({"body_ids": "bodies"} if prefix == "body" else {})},
            f"{prefix}_velocity" if prefix == "body" else "root_velocity",
        ),
        _indexed(
            f"write_{prefix}_com_velocity_to_sim",
            {
                f"{prefix}_velocities" if prefix == "body" else "root_velocity": ("instances", "bodies", 6)
                if prefix == "body"
                else ("instances", 6)
            },
            {"env_ids": "instances", **({"body_ids": "bodies"} if prefix == "body" else {})},
            f"{prefix}_velocity" if prefix == "body" else "root_velocity",
        ),
    )


def _rigid_common(prefix: str) -> tuple[AssetMethodSpec, ...]:
    return _rigid_methods(prefix) + (
        _indexed(
            "set_masses",
            {"masses": ("instances", "bodies")},
            {"env_ids": "instances", "body_ids": "bodies"},
            "body_props",
        ),
        _indexed(
            "set_coms",
            {"coms": ("instances", "bodies", 3)},
            {"env_ids": "instances", "body_ids": "bodies"},
            "body_props",
        ),
        _indexed(
            "set_inertias",
            {"inertias": ("instances", "bodies", 9)},
            {"env_ids": "instances", "body_ids": "bodies"},
            "body_props",
        ),
        _indexed(
            "set_external_force_and_torque",
            {"forces": ("instances", "bodies", 3), "torques": ("instances", "bodies", 3)},
            {"env_ids": "instances"},
            "external_wrench",
        ),
    )


def _rigid_masks(prefix: str) -> tuple[AssetMethodSpec, ...]:
    pose_key = f"{prefix}_poses" if prefix == "body" else "root_pose"
    velocity_key = f"{prefix}_velocities" if prefix == "body" else "root_velocity"
    pose_shape = ("instances", "bodies", 7) if prefix == "body" else ("instances", 7)
    velocity_shape = ("instances", "bodies", 6) if prefix == "body" else ("instances", 6)
    masks = {"env_mask": "instances", **({"body_mask": "bodies"} if prefix == "body" else {})}
    return (
        _masked(f"write_{prefix}_link_pose_to_sim_mask", {pose_key: pose_shape}, masks, f"{prefix}_pose"),
        _masked(f"write_{prefix}_com_pose_to_sim_mask", {pose_key: pose_shape}, masks, f"{prefix}_pose"),
        _masked(
            f"write_{prefix}_link_velocity_to_sim_mask", {velocity_key: velocity_shape}, masks, f"{prefix}_velocity"
        ),
        _masked(
            f"write_{prefix}_com_velocity_to_sim_mask", {velocity_key: velocity_shape}, masks, f"{prefix}_velocity"
        ),
        _masked(
            "set_masses_mask",
            {"masses": ("instances", "bodies")},
            {"env_mask": "instances", "body_mask": "bodies"},
            "body_props",
        ),
        _masked(
            "set_coms_mask",
            {"coms": ("instances", "bodies", 3)},
            {"env_mask": "instances", "body_mask": "bodies"},
            "body_props",
        ),
        _masked(
            "set_inertias_mask",
            {"inertias": ("instances", "bodies", 9)},
            {"env_mask": "instances", "body_mask": "bodies"},
            "body_props",
        ),
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
            "gear_ratio",
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
        methods=_ARTICULATION_PLAIN + _ARTICULATION_FINDERS + _articulation_mask_specs(),
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
