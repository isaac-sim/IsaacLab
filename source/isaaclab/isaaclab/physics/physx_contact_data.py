# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical contact data shared by the Isaac Sim and OVPhysX tensor owners."""

import numpy as np

from isaaclab.assets.physics_properties import UsdAttribute, usd_field


class PhysxContactData:
    """One native body's shape buffers with their PhysX column semantics.

    Public arrays use [environment, shape] layout. The owner supplies a single
    resolved body view; distinct per-shape values still require native collider IDs.
    """

    def __init__(self, disabled, materials=(), offsets=(), rest_offsets=()):
        self._disabled = np.asarray([[disabled]], dtype=bool)
        self._materials = np.asarray(materials).reshape(1, -1, 3)
        self._offsets = np.asarray(offsets).reshape(1, -1)
        self._rest_offsets = np.asarray(rest_offsets).reshape(1, -1)

    @property
    @usd_field(UsdAttribute("physxRigidBody:disableGravity", "PhysxRigidBodyAPI", type_name="bool"), scope="body")
    def disable_gravity(self):
        """Whether gravity is disabled, shape [1, 1]."""
        return self._disabled

    @property
    @usd_field(UsdAttribute("physics:staticFriction", "PhysicsMaterialAPI"), scope="material")
    def static_friction(self):
        """Static friction coefficient, shape [1, shape_count]."""
        return self._materials[..., 0]

    @property
    @usd_field(UsdAttribute("physics:dynamicFriction", "PhysicsMaterialAPI"), scope="material")
    def dynamic_friction(self):
        """Dynamic friction coefficient, shape [1, shape_count]."""
        return self._materials[..., 1]

    @property
    @usd_field(UsdAttribute("physics:restitution", "PhysicsMaterialAPI"), scope="material")
    def restitution(self):
        """Restitution coefficient, shape [1, shape_count]."""
        return self._materials[..., 2]

    @property
    @usd_field(UsdAttribute("physxCollision:contactOffset", "PhysxCollisionAPI", type_name="float"), scope="collision")
    def contact_offset(self):
        """Contact offset [m], shape [1, shape_count]."""
        return self._offsets

    @property
    @usd_field(UsdAttribute("physxCollision:restOffset", "PhysxCollisionAPI", type_name="float"), scope="collision")
    def rest_offset(self):
        """Rest offset [m], shape [1, shape_count]."""
        return self._rest_offsets

    def validate_uniform_contacts(self, body_path: str) -> None:
        """Reject ambiguous native shape rows without stable collider identities."""
        count = self._materials.shape[1]
        if not count or any(values.shape[1] != count for values in (self._offsets, self._rest_offsets)):
            raise NotImplementedError(f"Unmatched collider identities/count at {body_path}.")
        if any(not np.all(values == values[:, :1]) for values in (self._materials, self._offsets, self._rest_offsets)):
            raise NotImplementedError(
                f"Distinct per-shape contact values lack stable collider identities at {body_path}."
            )
