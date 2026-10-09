# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for masked resets of the Warp interactive scene on Newton."""

import torch
import warp as wp
from isaaclab_experimental.envs import InteractiveSceneWarp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.imu import ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class ImuSceneCfg(InteractiveSceneCfg):
    """Scene with a cube resting on the ground and an IMU on the cube."""

    env_spacing = 2.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    )

    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Cube")


def test_masked_reset_keeps_sensor_state_of_unselected_envs():
    """A masked scene reset zeroes the selected environment's IMU reading and keeps the other one."""
    sim_cfg = SimulationCfg(dt=1.0 / 200.0, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), num_substeps=1))
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveSceneWarp(ImuSceneCfg(num_envs=2))
        sim.reset()
        scene.reset()
        imu = scene["imu"]

        for _ in range(50):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt=sim.get_physics_dt())
        pre_reset_lin_acc = imu.data.lin_acc_b.torch.clone()
        assert (pre_reset_lin_acc[:, 2] > 5.0).all(), f"Expected a resting gravity reading, got {pre_reset_lin_acc}"

        scene.reset(env_mask=wp.array([True, False], dtype=wp.bool, device=imu.device))
        lin_acc = imu.data.lin_acc_b.torch
        torch.testing.assert_close(lin_acc[0], torch.zeros_like(lin_acc[0]))
        torch.testing.assert_close(lin_acc[1], pre_reset_lin_acc[1])
