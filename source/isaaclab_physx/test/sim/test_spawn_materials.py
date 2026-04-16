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
from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import DeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


@pytest.fixture
def sim():
    """Create a simulation context."""
    sim_utils.create_new_stage()
    dt = 0.1
    sim = SimulationContext(SimulationCfg(dt=dt))
    sim_utils.update_stage()
    yield sim
    sim.stop()
    sim.clear_instance()


def test_spawn_deformable_body_material(sim):
    """Test spawning a deformable body material."""
    cfg = DeformableBodyMaterialCfg(
        density=1.0,
        dynamic_friction=0.25,
        youngs_modulus=50000000.0,
        poissons_ratio=0.5,
        elasticity_damping=0.005,
    )
    prim = cfg.func("/Looks/DeformableBodyMaterial", cfg)
    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/Looks/DeformableBodyMaterial").IsValid()
    # Check properties
    assert prim.GetAttribute("omniphysics:density").Get() == cfg.density
    assert prim.GetAttribute("omniphysics:dynamicFriction").Get() == cfg.dynamic_friction
    assert prim.GetAttribute("omniphysics:youngsModulus").Get() == cfg.youngs_modulus
    assert prim.GetAttribute("omniphysics:poissonsRatio").Get() == cfg.poissons_ratio
    assert prim.GetAttribute("physxDeformableBody:elasticityDamping").Get() == pytest.approx(cfg.elasticity_damping)
