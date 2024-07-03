# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import unittest

import omni.isaac.core.utils.stage as stage_utils

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sensors.imu import IMU, IMUCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # terrain - flat terrain plane
    terrain = TerrainImporterCfg(        
        prim_path="/World/ground",
        terrain_type="plane",
        max_init_terrain_level=None,
    )

    # rigid objects - balls
    balls = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.125)),
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),            
        )
    )

    # sensors - frame transformer (filled inside unit test)
    imu: IMUCfg = IMUCfg(
        prim_path="{ENV_REGEX_NS}/ball",
    )


class TestIMU(unittest.TestCase):
    """Test for IMU sensor."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Load kit helper
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
        # construct scene
        scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False)
        self.scene = InteractiveScene(scene_cfg)
        # Play the simulator
        self.sim.reset()

    def tearDown(self):
        """Stops simulator after each test."""
        # clear the stage
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    """
    Tests
    """

    def test_constant_velocity(self):
        """Test the IMU sensor with a constant velocity."""
        for _ in range(2):
            # set velocity
            self.scene.rigid_objects["balls"].write_root_velocity_to_sim(
                torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.scene.device).repeat(self.scene.num_envs, 1)
            )
            # write data to sim
            self.scene.write_data_to_sim()
            # perform step
            self.sim.step()
            # read data from sim
            self.scene.update(self.sim.get_physics_dt())

        # check the imu data
        torch.testing.assert_close(
            self.scene.sensors["imu"].data.lin_acc_b,
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=self.scene.device).repeat(self.scene.num_envs, 1),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_constant_acceleration(self):
        """Test the IMU sensor with a constant acceleration."""
        for idx in range(10):
            # set acceleration
            self.scene.rigid_objects["balls"].write_root_velocity_to_sim(
                torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.scene.device).repeat(self.scene.num_envs, 1) * (idx + 1)
            )
            # write data to sim
            self.scene.write_data_to_sim()
            # perform step
            self.sim.step()
            # read data from sim
            self.scene.update(self.sim.get_physics_dt())

            # skip first steo where initial velocity is zero
            if idx < 1:
                continue

            # check the imu data
            torch.testing.assert_close(
                self.scene.sensors["imu"].data.lin_acc_b,
                math_utils.quat_rotate_inverse(
                    self.scene.rigid_objects["balls"].data.root_quat_w,
                    torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=self.scene.device).repeat(self.scene.num_envs, 1) / self.sim.get_physics_dt(),
                ),
                rtol=1e-4,
                atol=1e-4,
            )

            # check the angular velocity
            torch.testing.assert_close(
                self.scene.sensors["imu"].data.ang_vel_b,
                self.scene.rigid_objects["balls"].data.root_ang_vel_b,
                rtol=1e-4,
                atol=1e-4,
            )


if __name__ == "__main__":
    run_tests()
