# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Newton collider physical data and native USD declarations."""

import numpy as np
from newton.usd import PrimType, SchemaResolverNewton

from isaaclab.assets.physics_properties import UsdAttribute, property_metadata, usd_field, usd_fields


class NewtonContactData:
    """Global native collider rows; the manager resolves their prim/world identities."""

    def __init__(self, model):
        self.model = model

    def _shape_values(self, name: str) -> np.ndarray:
        """Read native collider values with one leading world row."""
        return getattr(self.model, name).numpy()[None, :]

    @classmethod
    def restore_fixed_configuration(cls, prim, builder, row: int) -> None:
        """Restore declared contacts on geometry constructed outside native USD import."""
        from pxr import UsdShade

        from isaaclab.sim.usd_export import validate_stage_units

        validate_stage_units(prim.GetStage())
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        for name, declarations in usd_fields(cls).items():
            scope = property_metadata(cls, name, "_usd_field")[2]
            owner = material.GetPrim() if scope == "material" else prim
            if not owner:
                continue
            for declaration in declarations:
                value = owner.GetAttribute(declaration.attribute).Get()
                if value is None:
                    continue
                getattr(builder, name)[row] = float(value)

    @property
    @usd_field(
        UsdAttribute(
            SchemaResolverNewton.mapping[PrimType.SHAPE]["margin"].name, "NewtonCollisionAPI", type_name="float"
        ),
        scope="collision",
    )
    def shape_margin(self) -> np.ndarray:
        """Native margin [m], shape [1, shape_count]."""
        return self._shape_values("shape_margin")

    @property
    @usd_field(
        UsdAttribute(SchemaResolverNewton.mapping[PrimType.SHAPE]["gap"].name, "NewtonCollisionAPI", type_name="float"),
        scope="collision",
    )
    def shape_gap(self) -> np.ndarray:
        """Native gap [m], shape [1, shape_count]."""
        return self._shape_values("shape_gap")

    @property
    @usd_field(UsdAttribute("physics:dynamicFriction", "PhysicsMaterialAPI"), scope="material")
    def shape_material_mu(self) -> np.ndarray:
        """Native mu [1], shape [1, shape_count]."""
        return self._shape_values("shape_material_mu")

    @property
    @usd_field(UsdAttribute("physics:restitution", "PhysicsMaterialAPI"), scope="material")
    def shape_material_restitution(self) -> np.ndarray:
        """Native restitution [1], shape [1, shape_count]."""
        return self._shape_values("shape_material_restitution")

    @property
    @usd_field(
        UsdAttribute(
            SchemaResolverNewton.mapping[PrimType.MATERIAL]["ke"].name, "NewtonMaterialAPI", type_name="float"
        ),
        scope="material",
    )
    def shape_material_ke(self) -> np.ndarray:
        """Native ke [N/m], shape [1, shape_count]."""
        return self._shape_values("shape_material_ke")

    @property
    @usd_field(
        UsdAttribute(
            SchemaResolverNewton.mapping[PrimType.MATERIAL]["kd"].name, "NewtonMaterialAPI", type_name="float"
        ),
        scope="material",
    )
    def shape_material_kd(self) -> np.ndarray:
        """Native kd [N*s/m], shape [1, shape_count]."""
        return self._shape_values("shape_material_kd")

    @property
    @usd_field(
        UsdAttribute(
            SchemaResolverNewton.mapping[PrimType.MATERIAL]["kf"].name, "NewtonMaterialAPI", type_name="float"
        ),
        scope="material",
    )
    def shape_material_kf(self) -> np.ndarray:
        """Native kf [N*s/m], shape [1, shape_count]."""
        return self._shape_values("shape_material_kf")

    @property
    @usd_field(
        UsdAttribute(
            SchemaResolverNewton.mapping[PrimType.MATERIAL]["ka"].name, "NewtonMaterialAPI", type_name="float"
        ),
        scope="material",
    )
    def shape_material_ka(self) -> np.ndarray:
        """Native ka [m], shape [1, shape_count]."""
        return self._shape_values("shape_material_ka")

    @property
    @usd_field(
        UsdAttribute(
            SchemaResolverNewton.mapping[PrimType.MATERIAL]["mu_torsional"].name, "NewtonMaterialAPI", type_name="float"
        ),
        scope="material",
    )
    def shape_material_mu_torsional(self) -> np.ndarray:
        """Native mu_torsional [m], shape [1, shape_count]."""
        return self._shape_values("shape_material_mu_torsional")

    @property
    @usd_field(
        UsdAttribute(
            SchemaResolverNewton.mapping[PrimType.MATERIAL]["mu_rolling"].name, "NewtonMaterialAPI", type_name="float"
        ),
        scope="material",
    )
    def shape_material_mu_rolling(self) -> np.ndarray:
        """Native mu_rolling [m], shape [1, shape_count]."""
        return self._shape_values("shape_material_mu_rolling")
