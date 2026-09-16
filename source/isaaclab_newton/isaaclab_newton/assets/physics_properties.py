# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Newton collider physical data and native USD declarations."""

import numpy as np
from newton.usd import PrimType, SchemaResolverNewton

from isaaclab.assets.physics_properties import UsdAttribute, source_units, usd_field


class NewtonContactData:
    """Global native collider rows; the manager resolves their prim/world identities."""

    def __init__(self, model):
        self.model = model

    @property
    @source_units("m")
    @usd_field(
        UsdAttribute(
            SchemaResolverNewton.mapping[PrimType.SHAPE]["margin"].name, "NewtonCollisionAPI", type_name="float"
        ),
        scope="collision",
    )
    def shape_margin(self) -> np.ndarray:
        """Native margin [m], shape [1, shape_count]."""
        return self.model.shape_margin.numpy()[None, :]

    @property
    @source_units("m")
    @usd_field(
        UsdAttribute(SchemaResolverNewton.mapping[PrimType.SHAPE]["gap"].name, "NewtonCollisionAPI", type_name="float"),
        scope="collision",
    )
    def shape_gap(self) -> np.ndarray:
        """Native gap [m], shape [1, shape_count]."""
        return self.model.shape_gap.numpy()[None, :]

    @property
    @source_units("1")
    @usd_field(UsdAttribute("physics:dynamicFriction", "PhysicsMaterialAPI"), scope="material")
    def shape_material_mu(self) -> np.ndarray:
        """Native mu [1], shape [1, shape_count]."""
        return self.model.shape_material_mu.numpy()[None, :]

    @property
    @source_units("1")
    @usd_field(UsdAttribute("physics:restitution", "PhysicsMaterialAPI"), scope="material")
    def shape_material_restitution(self) -> np.ndarray:
        """Native restitution [1], shape [1, shape_count]."""
        return self.model.shape_material_restitution.numpy()[None, :]

    @property
    @source_units("N/m")
    @usd_field(
        UsdAttribute(
            SchemaResolverNewton.mapping[PrimType.MATERIAL]["ke"].name, "NewtonMaterialAPI", type_name="float"
        ),
        scope="material",
    )
    def shape_material_ke(self) -> np.ndarray:
        """Native ke [N/m], shape [1, shape_count]."""
        return self.model.shape_material_ke.numpy()[None, :]

    @property
    @source_units("N*s/m")
    @usd_field(
        UsdAttribute(
            SchemaResolverNewton.mapping[PrimType.MATERIAL]["kd"].name, "NewtonMaterialAPI", type_name="float"
        ),
        scope="material",
    )
    def shape_material_kd(self) -> np.ndarray:
        """Native kd [N*s/m], shape [1, shape_count]."""
        return self.model.shape_material_kd.numpy()[None, :]

    @property
    @source_units("N*s/m")
    @usd_field(
        UsdAttribute(
            SchemaResolverNewton.mapping[PrimType.MATERIAL]["kf"].name, "NewtonMaterialAPI", type_name="float"
        ),
        scope="material",
    )
    def shape_material_kf(self) -> np.ndarray:
        """Native kf [N*s/m], shape [1, shape_count]."""
        return self.model.shape_material_kf.numpy()[None, :]

    @property
    @source_units("m")
    @usd_field(
        UsdAttribute(
            SchemaResolverNewton.mapping[PrimType.MATERIAL]["ka"].name, "NewtonMaterialAPI", type_name="float"
        ),
        scope="material",
    )
    def shape_material_ka(self) -> np.ndarray:
        """Native ka [m], shape [1, shape_count]."""
        return self.model.shape_material_ka.numpy()[None, :]

    @property
    @source_units("m")
    @usd_field(
        UsdAttribute(
            SchemaResolverNewton.mapping[PrimType.MATERIAL]["mu_torsional"].name, "NewtonMaterialAPI", type_name="float"
        ),
        scope="material",
    )
    def shape_material_mu_torsional(self) -> np.ndarray:
        """Native mu_torsional [m], shape [1, shape_count]."""
        return self.model.shape_material_mu_torsional.numpy()[None, :]

    @property
    @source_units("m")
    @usd_field(
        UsdAttribute(
            SchemaResolverNewton.mapping[PrimType.MATERIAL]["mu_rolling"].name, "NewtonMaterialAPI", type_name="float"
        ),
        scope="material",
    )
    def shape_material_mu_rolling(self) -> np.ndarray:
        """Native mu_rolling [m], shape [1, shape_count]."""
        return self.model.shape_material_mu_rolling.numpy()[None, :]
