# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim PhysX USD export: every engine value the exported stage does not already state."""

from __future__ import annotations

import numpy as np

from isaaclab.sim.usd_export.physx_usd_exporter import PhysxUsdExporter

RIGID_BODY_GETTERS = {
    "material": "get_material_properties",
    "contact_offset": "get_contact_offsets",
    "rest_offset": "get_rest_offsets",
}
"""Collider reading -> PhysX rigid body view getter."""


class UsdExporter(PhysxUsdExporter):
    """Export an Isaac Sim PhysX scene environment: its USD plus every engine value the USD does not state."""

    def _read_rigid_body(self, path: str) -> dict[str, np.ndarray]:
        from isaaclab_physx.physics import PhysxManager

        view = PhysxManager.get_physics_sim_view().create_rigid_body_view(path)
        values = {name: getattr(view, getter)() for name, getter in RIGID_BODY_GETTERS.items()}
        # The view returns NumPy or Warp arrays, depending on its tensor frontend.
        return {name: np.asarray(v.numpy() if hasattr(v, "numpy") else v)[0].copy() for name, v in values.items()}
