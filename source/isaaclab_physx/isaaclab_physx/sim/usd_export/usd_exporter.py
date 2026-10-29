# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim PhysX USD export: every engine value that differs from what PhysX loaded from the stage."""

from __future__ import annotations

import numpy as np

from isaaclab.sim.usd_export.physx_usd_exporter import PhysxUsdExporter
from isaaclab.sim.utils.queries import path_expr_to_glob

ARTICULATION_GETTERS = {
    "stiffness": "get_dof_stiffnesses",
    "damping": "get_dof_dampings",
    "max_force": "get_dof_max_forces",
    "limits": "get_dof_limits",
    "armature": "get_dof_armatures",
    "max_velocity": "get_dof_max_velocities",
    "friction": "get_dof_friction_properties",
    "mass": "get_masses",
    "com_pose": "get_coms",
    "inertia": "get_inertias",
}
"""Articulation reading -> PhysX articulation view getter."""

RIGID_BODY_GETTERS = {
    "mass": "get_masses",
    "com_pose": "get_coms",
    "inertia": "get_inertias",
    "material": "get_material_properties",
    "contact_offset": "get_contact_offsets",
    "rest_offset": "get_rest_offsets",
}
"""Rigid body reading -> PhysX rigid body view getter."""


class UsdExporter(PhysxUsdExporter):
    """Export an Isaac Sim PhysX scene environment: its USD plus every engine value Isaac Lab changed."""

    # PhysX loads deformables from their USD schemas; Isaac Lab writes only their nodal state.
    exports_deformables = True

    def _read_engine(self, path_expr: str, articulation: bool) -> dict[str, np.ndarray]:
        sim_view = self.scene.sim.backend_manager.get_physics_sim_view()
        glob = path_expr_to_glob(path_expr)
        view = sim_view.create_articulation_view(glob) if articulation else sim_view.create_rigid_body_view(glob)
        getters = ARTICULATION_GETTERS if articulation else RIGID_BODY_GETTERS
        values = {name: getattr(view, getter)() for name, getter in getters.items()}
        # The view returns NumPy or Warp arrays, depending on its tensor frontend.
        return {name: np.asarray(v.numpy() if hasattr(v, "numpy") else v).copy() for name, v in values.items()}
