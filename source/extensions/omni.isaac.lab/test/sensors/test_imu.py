# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import unittest

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.sensor import IMUSensor

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sensors.imu import IMUCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip


POS_OFFSET = (-0.25565, 0.00255, 0.07672)
ROT_OFFSET = (0.0, 0.0, 1.0, 0.0)


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
        ),
    )

    # articulations - robot
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    # sensors - imu (filled inside unit test)
    imu_ball: IMUCfg = IMUCfg(
        prim_path="{ENV_REGEX_NS}/ball",
    )
    imu_robot: IMUCfg = IMUCfg(
        prim_path="{ENV_REGEX_NS}/robot/base",
        offset=IMUCfg.OffsetCfg(
            pos=POS_OFFSET,
            rot=ROT_OFFSET,
        ),
    )

    def __post_init__(self):
        """Post initialization."""
        # change position of the robot
        self.robot.init_state.pos = (0.0, 2.0, 0.5)


class TestIMU(unittest.TestCase):
    """Test for IMU sensor."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Load kit helper
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, use_fabric=False))
        # construct scene
        scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False)
        self.scene = InteractiveScene(scene_cfg)
        # create the isaac sim IMU sensor with same translation as our IMU sensor
        self.imu_sensor = IMUSensor(
            prim_path="/World/envs/env_0/robot/base/imu",
            name="imu",
            dt=self.sim.get_physics_dt(),
            translation=np.array(POS_OFFSET),
            orientation=np.array(ROT_OFFSET),
            linear_acceleration_filter_size=1,
            angular_velocity_filter_size=1,
            orientation_filter_size=1,
        )
        self.imu_sensor.initialize()
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
                torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.scene.device).repeat(
                    self.scene.num_envs, 1
                )
            )
            # write data to sim
            self.scene.write_data_to_sim()
            # perform step
            self.sim.step()
            # read data from sim
            self.scene.update(self.sim.get_physics_dt())

        # check the imu data
        torch.testing.assert_close(
            self.scene.sensors["imu_ball"].data.lin_acc_b,
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=self.scene.device).repeat(
                self.scene.num_envs, 1
            ),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_constant_acceleration(self):
        """Test the IMU sensor with a constant acceleration."""
        for idx in range(10):
            # set acceleration
            self.scene.rigid_objects["balls"].write_root_velocity_to_sim(
                torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.scene.device).repeat(
                    self.scene.num_envs, 1
                )
                * (idx + 1)
            )
            # write data to sim
            self.scene.write_data_to_sim()
            # perform step
            self.sim.step()
            # read data from sim
            self.scene.update(self.sim.get_physics_dt())

            # skip first step where initial velocity is zero
            if idx < 1:
                continue

            # check the imu data
            torch.testing.assert_close(
                self.scene.sensors["imu_ball"].data.lin_acc_b,
                math_utils.quat_rotate_inverse(
                    self.scene.rigid_objects["balls"].data.root_quat_w,
                    torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32, device=self.scene.device).repeat(
                        self.scene.num_envs, 1
                    )
                    / self.sim.get_physics_dt(),
                ),
                rtol=1e-4,
                atol=1e-4,
            )

            # check the angular velocity
            torch.testing.assert_close(
                self.scene.sensors["imu_ball"].data.ang_vel_b,
                self.scene.rigid_objects["balls"].data.root_ang_vel_b,
                rtol=1e-4,
                atol=1e-4,
            )

    def test_offset_calculation(self):
        # should achieve same results between the two imu sensors on the robot
        for idx in range(10):
            # set acceleration
            self.scene.articulations["robot"].write_root_velocity_to_sim(
                torch.tensor([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.scene.device).repeat(
                    self.scene.num_envs, 1
                )
                * (idx + 1)
            )
            # write data to sim
            self.scene.write_data_to_sim()
            # perform step
            self.sim.step()
            # read data from sim
            self.scene.update(self.sim.get_physics_dt())

            # get the imu readings from the Isaac Sim sensor
            isaac_sim_imu_data = self.imu_sensor.get_current_frame(read_gravity=True)

            # skip first step where initial velocity is zero
            if idx < 1:
                continue

            # check the imu data
            torch.testing.assert_close(
                self.scene.sensors["imu_robot"].data.lin_acc_b[0],
                isaac_sim_imu_data["lin_acc"],
                rtol=1e-4,
                atol=1e-4,
            )

            # check the angular velocity
            torch.testing.assert_close(
                self.scene.sensors["imu_robot"].data.ang_vel_b[0],
                isaac_sim_imu_data["ang_vel"],
                rtol=1e-4,
                atol=1e-4,
            )
            # check the orientation
            torch.testing.assert_close(
                self.scene.sensors["imu_robot"].data.quat_w[0],
                isaac_sim_imu_data["orientation"],
                rtol=1e-4,
                atol=1e-4,
            )


if __name__ == "__main__":
    run_tests()
