# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical property contracts shared by asset initialization and USD export.

Array rows follow the public asset body/joint order, never USD traversal order. Backends
must supply the corresponding prim identities separately. Geometry, materials, filtering,
tendons and other spawn schemas remain owned by the authored USD stage.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

    from isaaclab.assets import BaseArticulationData, BaseRigidObjectCollectionData, BaseRigidObjectData


# The payload consumed by ActuatorControl.write_resolved_joint_properties. Adding a solver
# property extends this contract; exporter coverage checks require its USD semantics too.
JOINT_PROPERTY_SOURCES = {
    "stiffness": "joint_stiffness",
    "damping": "joint_damping",
    "armature": "joint_armature",
    "friction": "joint_friction_coeff",
    "dynamic_friction": "joint_dynamic_friction_coeff",
    "viscous_friction": "joint_viscous_friction_coeff",
    "joint_effort_limit": "joint_effort_limits",
    "joint_velocity_limit": "joint_vel_limits",
}
"""Resolved solver properties and their public data sources.

Stiffness [N/m or N*m/rad], damping and viscous friction [N*s/m or N*m*s/rad],
armature [kg or kg*m^2], effort limit [N or N*m], and velocity limit [m/s or rad/s]
depend on joint type. Static/dynamic friction follow the backend's documented convention;
an adapter must not silently reinterpret a dimensionless coefficient as an effort.
Explicit actuator controller gains are separate from these solver drive gains.
"""

# Routes describe ownership, not a second cfg interpreter. Spawner/schema fields are
# consumed by their existing writers and preserved wholesale in USD. Keep the exceptions
# here so a newly added asset/actuator field cannot silently disappear during export.
ASSET_CONFIGURATION_SOURCES = {
    "construction": {"class_type", "cloning_contexts", "prim_path", "articulation_root_prim_path"},
    "usd": {"spawn", "collision_group"},
    "initial_state": {"init_state"},
    "joint_properties": {"actuators"},
    "identity": {"joint_ordering", "body_ordering"},
    "collection": {"rigid_objects"},
    "runtime_only": {
        "soft_joint_pos_limit_factor",
        "debug_vis",
        "disable_shape_checks",
        "actuator_value_resolution_debug_print",
    },
}
ACTUATOR_CONFIGURATION_SOURCES = {
    "construction": {"class_type", "joint_names_expr"},
    "solver": set(JOINT_PROPERTY_SOURCES),
    "controller": {"actuator_effort_limit", "actuator_velocity_limit"},
    "aliases": {"effort_limit_sim", "velocity_limit_sim", "effort_limit", "velocity_limit"},
}


def validate_configuration_coverage(cfg: object, *, actuator: bool = False) -> None:
    """Reject configuration fields with no declared export owner.

    Derived native actuator fields are covered by their existing schema authoring contract;
    this check covers the shared actuator base consumed by ActuatorControl.
    """
    if actuator:
        from isaaclab.actuators import ActuatorBaseCfg

        names = {field.name for field in fields(ActuatorBaseCfg)}
        routes = ACTUATOR_CONFIGURATION_SOURCES
    else:
        names = {field.name for field in fields(cfg)}
        routes = ASSET_CONFIGURATION_SOURCES
    missing = names - set().union(*routes.values())
    if missing:
        raise NotImplementedError(f"Undeclared physical configuration export fields on {type(cfg).__name__}: {missing}")


def read_joint_properties(data: BaseArticulationData) -> dict[str, torch.Tensor]:
    """Read resolved solver properties in public joint order, shape [N, J].

    Returns views of the public data tensors. Callers retaining a snapshot must copy them.
    """
    values = {}
    for name, source in JOINT_PROPERTY_SOURCES.items():
        value = getattr(data, source, None)
        if value is None and name in {"dynamic_friction", "viscous_friction"}:
            values[name] = data.joint_stiffness.torch.new_zeros(data.joint_stiffness.torch.shape)
        else:
            values[name] = value.torch
    return values


@dataclass(frozen=True)
class BodyPhysicsProperties:
    """One instance's rigid bodies, in public body order.

    Attributes:
        mass: Mass [kg], shape [B].
        inertia: Inertia about COM in the body frame [kg*m^2], shape [B, 3, 3].
        com_pose: Body-local COM pose [m, xyzw], shape [B, 7]. Inertia already
            uses the body frame and must not be rotated again by this quaternion.
        pose: Body-link world pose [m, xyzw], shape [B, 7].
        velocity: COM world linear/angular velocity [m/s, rad/s], shape [B, 6].
    """

    mass: np.ndarray
    inertia: np.ndarray
    com_pose: np.ndarray
    pose: np.ndarray
    velocity: np.ndarray

    @classmethod
    def from_data(
        cls, data: BaseArticulationData | BaseRigidObjectData | BaseRigidObjectCollectionData, row: int
    ) -> BodyPhysicsProperties:
        """Copy an instance from any supported rigid asset's public data interface."""
        sources = (
            ("body_mass", (-1,)),
            ("body_inertia", (-1, 3, 3)),
            ("body_com_pose_b", (-1, 7)),
            ("body_link_pose_w", (-1, 7)),
            ("body_com_vel_w", (-1, 6)),
        )
        arrays = [
            getattr(data, name).torch[row].detach().cpu().numpy().reshape(shape).copy() for name, shape in sources
        ]
        if len({len(array) for array in arrays}) != 1:
            raise RuntimeError("Rigid-body data properties disagree on body coverage.")
        return cls(*arrays)
