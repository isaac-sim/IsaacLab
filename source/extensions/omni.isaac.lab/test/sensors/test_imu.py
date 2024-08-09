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
from omni.isaac.lab.sensors.imu import ImuCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip
from omni.isaac.lab.utils.assets import NUCLEUS_ASSET_ROOT_DIR  # isort: skip


POS_OFFSET = (0.2488, 0.00835, 0.04628)
ROT_OFFSET = (0.7071068, 0, 0, 0.7071068)


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
    imu_ball: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/ball",
    )
    imu_robot_imu_link: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/robot/imu_link",
    )
    imu_robot_base: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/robot/base",
        offset=ImuCfg.OffsetCfg(
            pos=POS_OFFSET,
            rot=ROT_OFFSET,
        ),
    )

    def __post_init__(self):
        """Post initialization."""
        # change position of the robot
        self.robot.init_state.pos = (0.0, 2.0, 0.5)
        # change asset
        self.robot.spawn.usd_path = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac/Robots/ANYbotics/anymal_c.usd"


class TestImu(unittest.TestCase):
    """Test for Imu sensor."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Load kit helper
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, use_fabric=False))
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
        """Test the Imu sensor with a constant velocity."""
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
        """Test the Imu sensor with a constant acceleration."""
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

            # skip first step where initial velocity is zero
            if idx < 1:
                continue

            # check the imu data
            torch.testing.assert_close(
                self.scene.sensors["imu_robot_base"].data.lin_acc_b,
                self.scene.sensors["imu_robot_imu_link"].data.lin_acc_b,
                rtol=1e-4,
                atol=1e-4,
            )
            torch.testing.assert_close(
                self.scene.sensors["imu_robot_base"]._last_lin_vel_w,
                self.scene.sensors["imu_robot_imu_link"]._last_lin_vel_w,
                rtol=1e-4,
                atol=1e-4,
            )

            # check the angular velocity
            torch.testing.assert_close(
                self.scene.sensors["imu_robot_base"].data.ang_vel_b,
                self.scene.sensors["imu_robot_imu_link"].data.ang_vel_b,
                rtol=1e-4,
                atol=1e-4,
            )
            # check the orientation
            torch.testing.assert_close(
                self.scene.sensors["imu_robot_base"].data.quat_w,
                self.scene.sensors["imu_robot_imu_link"].data.quat_w,
                rtol=1e-4,
                atol=1e-4,
            )
            # check the position
            torch.testing.assert_close(
                self.scene.sensors["imu_robot_base"].data.pos_w,
                self.scene.sensors["imu_robot_imu_link"].data.pos_w,
                rtol=1e-4,
                atol=1e-4,
            )


if __name__ == "__main__":
    run_tests()
