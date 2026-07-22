# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton deformable-body and deformable-material fragment cfg classes."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import dataclasses

from isaaclab.sim.schemas import DeformableBodyFragment
from isaaclab.sim.spawners.materials import DeformableMaterialFragment
from isaaclab.utils.string import to_camel_case
from isaaclab_newton.sim.schemas import NewtonDeformableBodyCfg
from isaaclab_newton.sim.spawners.materials import (
    NewtonSurfaceDeformableMaterialCfg,
    NewtonVolumeDeformableMaterialCfg,
)


def test_newton_deformable_fragments_metadata():
    body = NewtonDeformableBodyCfg()
    assert isinstance(body, DeformableBodyFragment)
    assert type(body)._usd_namespace == "newton"
    assert [f.name for f in dataclasses.fields(body)] == ["func"]  # placeholder: namespace reserved
    vol = NewtonVolumeDeformableMaterialCfg(k_mu=1e5)
    surf = NewtonSurfaceDeformableMaterialCfg(tri_ke=1e4)
    assert isinstance(vol, DeformableMaterialFragment) and isinstance(surf, DeformableMaterialFragment)
    assert type(vol)._usd_applied_schema is None and type(surf)._usd_applied_schema is None


def test_newton_material_fragments_author_exactly_what_contrib_hooks_read():
    """Parity guard: authored newton:* attr names == the names the contrib deformable builder
    hooks read off the bound material prim."""
    hook_reads = {
        "density",
        "particleRadius",
        "kMu",
        "kLambda",
        "kDamp",
        "triKe",
        "triKa",
        "triKd",
        "edgeKe",
        "edgeKd",
    }
    authored = set()
    for cls in (NewtonVolumeDeformableMaterialCfg, NewtonSurfaceDeformableMaterialCfg):
        authored |= {to_camel_case(f.name, "cC") for f in dataclasses.fields(cls) if f.name != "func"}
    assert authored == hook_reads
