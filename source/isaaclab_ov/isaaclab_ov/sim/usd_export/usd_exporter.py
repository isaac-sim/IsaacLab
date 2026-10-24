# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX USD export: every engine value the exported stage does not already state."""

from __future__ import annotations

import numpy as np

from isaaclab.sim.usd_export.physx_usd_exporter import PhysxUsdExporter

from isaaclab_ov import tensor_types as TT
from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView

RIGID_BODY_TENSORS = {
    "material": TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION,
    "contact_offset": TT.RIGID_BODY_CONTACT_OFFSET,
    "rest_offset": TT.RIGID_BODY_REST_OFFSET,
}
"""Collider reading -> OVPhysX rigid body binding."""


class UsdExporter(PhysxUsdExporter):
    """Export an OVPhysX scene environment: its USD plus every engine value the USD does not state."""

    def _read_rigid_body(self, path: str) -> dict[str, np.ndarray]:
        view = OvPhysxView(
            self.scene.sim.backend_manager.get_physx_instance(), prim_paths=[path], device=self.scene.sim.device
        )
        try:
            return {name: view.get_attribute(tensor).numpy()[0].copy() for name, tensor in RIGID_BODY_TENSORS.items()}
        finally:
            view.close()
