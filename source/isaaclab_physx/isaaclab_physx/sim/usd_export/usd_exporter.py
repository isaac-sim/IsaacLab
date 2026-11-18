# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim PhysX USD export: every engine value that differs from what PhysX loaded from the stage."""

from __future__ import annotations

import contextlib

import numpy as np

from isaaclab.sim.usd_export.physx_usd_exporter import PhysxUsdExporter
from isaaclab.sim.utils.queries import path_expr_to_glob

COLLIDER_GETTERS = {
    "material": "get_material_properties",
    "contact_offset": "get_contact_offsets",
    "rest_offset": "get_rest_offsets",
}
"""Collider value -> PhysX rigid body view getter."""


class UsdExporter(PhysxUsdExporter):
    """Export an Isaac Sim PhysX scene environment: its USD plus every engine value Isaac Lab changed."""

    # PhysX loads deformables from their USD schemas.
    exports_deformables = True

    @contextlib.contextmanager
    def _open_engine_view(self, path_expr: str, articulation: bool):
        sim_view = self.scene.sim.backend_manager.get_physics_sim_view()
        glob = path_expr_to_glob(path_expr)
        yield sim_view.create_articulation_view(glob) if articulation else sim_view.create_rigid_body_view(glob)

    def _read_collider_values(self, view) -> dict[str, np.ndarray]:
        values = {name: getattr(view, getter)() for name, getter in COLLIDER_GETTERS.items()}
        # The view returns NumPy or Warp arrays, depending on its tensor frontend.
        return {name: np.asarray(v.numpy() if hasattr(v, "numpy") else v).copy() for name, v in values.items()}
