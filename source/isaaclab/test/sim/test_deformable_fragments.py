# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest

from pxr import UsdGeom

pytestmark = pytest.mark.integration


def _make_xform(stage, path="/World/Body"):
    UsdGeom.Xform.Define(stage, path)
    return stage.GetPrimAtPath(path)


# -------------------------------------------------------------------------------------
# Fragment metadata -- DeformableBodyFragment marker, OmniPhysicsDeformableBodyCfg
# -------------------------------------------------------------------------------------


def test_deformable_body_fragment_metadata_defaults():
    from isaaclab.sim.schemas import DeformableBodyFragment, OmniPhysicsDeformableBodyCfg, SchemaFragment

    cfg = OmniPhysicsDeformableBodyCfg(mass=2.0)
    assert isinstance(cfg, DeformableBodyFragment) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_namespace == "omniphysics"
    assert type(cfg)._usd_applied_schema is None  # anchor applied by the backend manager
    assert type(cfg)._deformable_types == ("volume", "surface")
    assert cfg.func == "isaaclab.sim.schemas:apply_namespaced"
    assert cfg.mass == 2.0 and cfg.deformable_body_enabled is None and cfg.kinematic_enabled is None
