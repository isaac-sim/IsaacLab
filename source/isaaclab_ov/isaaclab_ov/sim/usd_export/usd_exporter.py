# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX USD export: every engine value that differs from what OVPhysX loaded from the stage."""

from __future__ import annotations

import re

import numpy as np

from pxr import Usd

from isaaclab.sim.usd_export.base_usd_exporter import UsdEdit
from isaaclab.sim.usd_export.physx_usd_exporter import PhysxUsdExporter
from isaaclab.sim.utils.queries import path_expr_to_glob

from isaaclab_ov import tensor_types as TT
from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView, _expand_env_pattern

ARTICULATION_TENSORS = {
    "stiffness": TT.DOF_STIFFNESS,
    "damping": TT.DOF_DAMPING,
    "max_force": TT.DOF_MAX_FORCE,
    "limits": TT.DOF_LIMIT,
    "armature": TT.DOF_ARMATURE,
    "max_velocity": TT.DOF_MAX_VELOCITY,
    "friction": TT.DOF_FRICTION_PROPERTIES,
    "mass": TT.BODY_MASS,
    "com_pose": TT.BODY_COM_POSE,
    "inertia": TT.BODY_INERTIA,
}
"""Articulation reading -> OVPhysX binding."""

RIGID_BODY_TENSORS = {
    "mass": TT.RIGID_BODY_MASS,
    "com_pose": TT.RIGID_BODY_COM_POSE,
    "inertia": TT.RIGID_BODY_INERTIA,
    "material": TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION,
    "contact_offset": TT.RIGID_BODY_CONTACT_OFFSET,
    "rest_offset": TT.RIGID_BODY_REST_OFFSET,
}
"""Rigid body reading -> OVPhysX binding."""


LEGACY_JOINT_ATTRIBUTES = {
    "armature": ("physxJoint:armature", "newton:armature"),
    "max_velocity": ("physxJoint:maxJointVelocity", "newton:velocityLimit"),
    "static_friction": ("physxJoint:jointFriction", "newton:friction"),
    "dynamic_friction": ("physxJoint:jointFriction", "newton:friction"),
    "viscous_friction": ("physxJoint:jointFriction", "newton:damping"),
}
"""Joint property -> the ``PhysxJointAPI`` attribute Isaac Lab migrates before OVPhysX 0.7 loads the stage, and the
Newton attribute the migration keeps instead."""

NEWTON_JOINT_ATTRIBUTES = {
    "armature": "newton:armature",
    "max_velocity": "newton:velocityLimit",
    "static_friction": "newton:friction",
    "viscous_friction": "newton:damping",
}
"""Joint property -> the Newton attribute OVPhysX 0.7 keeps when the per-axis value is not positive."""


class UsdExporter(PhysxUsdExporter):
    """Export an OVPhysX scene environment: its USD plus every engine value Isaac Lab changed."""

    def compute_diff(self, stage: Usd.Stage) -> list[UsdEdit]:
        edits = super().compute_diff(stage)
        if not self._uses_tasks():
            return edits
        # OVPhysX 0.7 applies a per-axis joint value only when it is positive and otherwise keeps an authored Newton
        # value, so a Newton value the live engine no longer holds is restated.
        readings, _ = self._read_assets(stage)
        for (path, axis), values in readings.items():
            if axis is None:
                continue
            prim, table = stage.GetPrimAtPath(path), self._joint_table(axis)
            for name, attribute in NEWTON_JOINT_ATTRIBUTES.items():
                live = float(values[name]) * (1.0 if axis == "linear" else table[name].angular_scale)
                stated = self._stated(prim, attribute)
                if stated is not None and not self._close(stated, live):
                    edits.append(UsdEdit(path, attribute, live))
        return edits

    def _not_imported(self, prim, axis):
        if not self._uses_tasks():
            return set()
        # OVPhysX 0.7 ignores PhysxJointAPI; Isaac Lab migrated its values into the stage the engine loaded.
        table = self._joint_table(axis)
        return {
            name
            for name, (legacy, newton) in LEGACY_JOINT_ATTRIBUTES.items()
            if prim.GetAttribute(legacy).HasAuthoredValue()
            and not prim.GetAttribute(newton).HasAuthoredValue()
            and not prim.GetAttribute(table[name].name.format(axis=axis)).HasAuthoredValue()
        }

    @staticmethod
    def _uses_tasks() -> bool:
        from isaaclab_ov.physics import ovphysx_compat

        return ovphysx_compat.OVPHYSX_USES_TASKS

    def _read_engine(self, path_expr: str, articulation: bool) -> dict[str, np.ndarray]:
        # Bound as the assets bind them: OVPhysX matches globs, one path per environment in environment order.
        pattern = path_expr_to_glob(re.sub(r"\{ENV_REGEX_NS\}", "*", path_expr))
        paths = _expand_env_pattern(pattern, self.scene.sim.get_clone_plan())
        physx = self.scene.sim.backend_manager.get_physx_instance()
        view = OvPhysxView(physx, prim_paths=paths, device=self.scene.sim.device)
        try:
            tensors = ARTICULATION_TENSORS if articulation else RIGID_BODY_TENSORS
            return {name: view.get_attribute(tensor).numpy().copy() for name, tensor in tensors.items()}
        finally:
            view.close()
