# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real-backend tests for OvPhysX scene gravity randomization."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov.assets import RigidObject
from isaaclab_ov.physics import OvPhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs.mdp.events import randomize_physics_scene_gravity
from isaaclab.managers import EventTermCfg
from isaaclab.sim import SimulationCfg, build_simulation_context


def test_gravity_event_changes_rigid_body_motion():
    """The gravity event must dispatch through OvStage and change motion in OvPhysX."""
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device="cpu", dt=1.0 / 60.0)
    with build_simulation_context(device="cpu", sim_cfg=sim_cfg) as sim:
        cube = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Cube",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 10.0)),
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.5, 0.5),
                    rigid_props=sim_utils.RigidBodyBaseCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionBaseCfg(),
                ),
            )
        )
        sim.reset()
        dt = sim.get_physics_dt()
        cube.update(dt)

        env = SimpleNamespace(device=sim.device, sim=sim)
        event_cfg = EventTermCfg(
            func=randomize_physics_scene_gravity,
            params={"gravity_distribution_params": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), "operation": "abs"},
        )
        gravity_event = randomize_physics_scene_gravity(event_cfg, env)

        def step(count: int) -> None:
            for _ in range(count):
                cube.write_data_to_sim()
                sim.step()
                cube.update(dt)

        def set_gravity(gravity: tuple[float, float, float]) -> None:
            gravity_event(
                env,
                env_ids=None,
                gravity_distribution_params=(gravity, gravity),
                operation="abs",
            )

        set_gravity((0.0, 0.0, 0.0))
        initial_height = cube.data.root_pos_w.torch[0, 2].item()
        step(30)
        zero_gravity_height = cube.data.root_pos_w.torch[0, 2].item()

        set_gravity((0.0, 0.0, -9.81))
        # Changing scene gravity does not wake sleeping PhysX actors. Reset terms
        # write body state after gravity randomization, so mirror that wake-up here.
        root_velocity = torch.zeros((1, 6), device=sim.device)
        root_velocity[0, 2] = 0.001
        cube.write_root_velocity_to_sim_index(root_velocity=root_velocity)
        step(30)
        earth_gravity_height = cube.data.root_pos_w.torch[0, 2].item()

        assert zero_gravity_height == pytest.approx(initial_height, abs=1.0e-4)
        assert earth_gravity_height - zero_gravity_height < -0.5
