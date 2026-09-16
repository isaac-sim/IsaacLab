# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical contact data shared by the Isaac Sim and OVPhysX tensor owners."""

import numpy as np

from pxr import Usd, UsdPhysics

from .physics_properties import UsdAttribute, source_units, usd_field


class PhysxContactData:
    """One native body's shape buffers with their PhysX column semantics.

    Public arrays use [environment, shape] layout. The owner supplies a single
    resolved body view; distinct per-shape values still require native collider IDs.
    """

    def __init__(self, disabled, materials, offsets, rest_offsets):
        self._disabled = np.asarray([[disabled]], dtype=bool)
        self._materials = np.asarray(materials).reshape(1, -1, 3)
        self._offsets = np.asarray(offsets).reshape(1, -1)
        self._rest_offsets = np.asarray(rest_offsets).reshape(1, -1)

    @property
    @source_units("1")
    @usd_field(UsdAttribute("physxRigidBody:disableGravity", "PhysxRigidBodyAPI", type_name="bool"), scope="body")
    def disable_gravity(self):
        """Whether gravity is disabled, shape [1, 1]."""
        return self._disabled

    @property
    @source_units("1")
    @usd_field(UsdAttribute("physics:staticFriction", "PhysicsMaterialAPI"), scope="material")
    def static_friction(self):
        """Static friction coefficient, shape [1, shape_count]."""
        return self._materials[..., 0]

    @property
    @source_units("1")
    @usd_field(UsdAttribute("physics:dynamicFriction", "PhysicsMaterialAPI"), scope="material")
    def dynamic_friction(self):
        """Dynamic friction coefficient, shape [1, shape_count]."""
        return self._materials[..., 1]

    @property
    @source_units("1")
    @usd_field(UsdAttribute("physics:restitution", "PhysicsMaterialAPI"), scope="material")
    def restitution(self):
        """Restitution coefficient, shape [1, shape_count]."""
        return self._materials[..., 2]

    @property
    @source_units("m")
    @usd_field(UsdAttribute("physxCollision:contactOffset", "PhysxCollisionAPI", type_name="float"), scope="collision")
    def contact_offset(self):
        """Contact offset [m], shape [1, shape_count]."""
        return self._offsets

    @property
    @source_units("m")
    @usd_field(UsdAttribute("physxCollision:restOffset", "PhysxCollisionAPI", type_name="float"), scope="collision")
    def rest_offset(self):
        """Rest offset [m], shape [1, shape_count]."""
        return self._rest_offsets

    def author_configuration(self, writer, body_path: str) -> None:
        """Resolve unambiguous collider rows; retain source contacts before randomization."""
        writer.write_properties(body_path, None, self, row=0, scope="body", env_index=0)
        if writer.preserve_source_contacts:
            return
        body = writer.stage.GetPrimAtPath(body_path)
        descendants = iter(Usd.PrimRange(body))
        colliders = []
        for prim in descendants:
            if prim != body and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                descendants.PruneChildren()
            elif prim.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(str(prim.GetPath()))
        if not colliders:
            return
        count = self._materials.shape[1]
        if not count or any(values.shape[1] != count for values in (self._offsets, self._rest_offsets)):
            raise NotImplementedError(f"Unmatched collider identities/count at {body_path}.")
        if any(not np.all(values == values[:, :1]) for values in (self._materials, self._offsets, self._rest_offsets)):
            raise NotImplementedError(
                f"Distinct per-shape contact values lack stable collider identities at {body_path}."
            )
        for path in colliders:
            writer.write_properties(path, None, self, row=0, scope="collision", env_index=0)
            writer.write_material_override(path, self, 0, env_index=0)
