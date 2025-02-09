# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests to verify contact sensor functionality on rigid object prims."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import unittest
from dataclasses import MISSING
from enum import Enum

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import build_simulation_context
from isaaclab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

##
# Custom helper classes.
##


class ContactTestMode(Enum):
    """Enum to declare the type of contact sensor test to execute."""

    IN_CONTACT = 0
    """Enum to test the condition where the test object is in contact with the ground plane."""
    NON_CONTACT = 1
    """Enum to test the condition where the test object is not in contact with the ground plane (air time)."""


@configclass
class TestContactSensorRigidObjectCfg(RigidObjectCfg):
    """Configuration for rigid objects used for the contact sensor test.

    This contains the expected values in the configuration to simplify test fixtures.
    """

    contact_pose: torch.Tensor = MISSING
    """6D pose of the rigid object under test when it is in contact with the ground surface."""
    non_contact_pose: torch.Tensor = MISSING
    """6D pose of the rigid object under test when it is not in contact."""


@configclass
class ContactSensorSceneCfg(InteractiveSceneCfg):
    """Configuration of the scene used by the contact sensor test."""

    terrain: TerrainImporterCfg = MISSING
    """Terrain configuration within the scene."""

    shape: TestContactSensorRigidObjectCfg = MISSING
    """RigidObject contact prim configuration."""

    contact_sensor: ContactSensorCfg = MISSING
    """Contact sensor configuration."""

    shape_2: TestContactSensorRigidObjectCfg = None
    """RigidObject contact prim configuration. Defaults to None, i.e. not included in the scene.

    This is a second prim used for testing contact filtering.
    """

    contact_sensor_2: ContactSensorCfg = None
    """Contact sensor configuration. Defaults to None, i.e. not included in the scene.

    This is a second contact sensor used for testing contact filtering.
    """


##
# Scene entity configurations.
##


CUBE_CFG = TestContactSensorRigidObjectCfg(
    prim_path="/World/Objects/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.5, 0.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 0.4)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, -1.0, 1.0)),
    contact_pose=torch.tensor([0, -1.0, 0, 1, 0, 0, 0]),
    non_contact_pose=torch.tensor([0, -1.0, 1.0, 1, 0, 0, 0]),
)
"""Configuration of the cube prim."""

SPHERE_CFG = TestContactSensorRigidObjectCfg(
    prim_path="/World/Objects/Sphere",
    spawn=sim_utils.SphereCfg(
        radius=0.25,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.6)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 1.0, 1.0)),
    contact_pose=torch.tensor([0, 1.0, 0.0, 1, 0, 0, 0]),
    non_contact_pose=torch.tensor([0, 1.0, 1.0, 1, 0, 0, 0]),
)
"""Configuration of the sphere prim."""

CYLINDER_CFG = TestContactSensorRigidObjectCfg(
    prim_path="/World/Objects/Cylinder",
    spawn=sim_utils.CylinderCfg(
        radius=0.5,
        height=0.01,
        axis="Y",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.4)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0.0, 1.0)),
    contact_pose=torch.tensor([0, 0, 0.0, 1, 0, 0, 0]),
    non_contact_pose=torch.tensor([0, 0, 1.0, 1, 0, 0, 0]),
)
"""Configuration of the cylinder prim."""

CAPSULE_CFG = TestContactSensorRigidObjectCfg(
    prim_path="/World/Objects/Capsule",
    spawn=sim_utils.CapsuleCfg(
        radius=0.25,
        height=0.5,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.4)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 1.5)),
    contact_pose=torch.tensor([1.0, 0.0, 0.0, 1, 0, 0, 0]),
    non_contact_pose=torch.tensor([1.0, 0.0, 1.5, 1, 0, 0, 0]),
)
"""Configuration of the capsule prim."""

CONE_CFG = TestContactSensorRigidObjectCfg(
    prim_path="/World/Objects/Cone",
    spawn=sim_utils.ConeCfg(
        radius=0.5,
        height=0.5,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.2, 0.4)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0, 0.0, 1.0)),
    contact_pose=torch.tensor([-1.0, 0.0, 0.0, 1, 0, 0, 0]),
    non_contact_pose=torch.tensor([-1.0, 0.0, 1.0, 1, 0, 0, 0]),
)
"""Configuration of the cone prim."""

FLAT_TERRAIN_CFG = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
"""Configuration of the flat ground plane."""

COBBLESTONE_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        seed=0,
        size=(3.0, 3.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        sub_terrains={
            "random_rough": HfRandomUniformTerrainCfg(
                proportion=1.0, noise_range=(0.0, 0.05), noise_step=0.01, border_width=0.25
            ),
        },
    ),
)
"""Configuration of the generated mesh terrain."""


class TestContactSensor(unittest.TestCase):
    """Unittest class for testing the contact sensor.

    This class includes test cases for the available rigid object primitives, and tests that the
    the contact sensor is reporting correct results for various contact durations, terrain types, and
    evaluation devices.
    """

    @classmethod
    def setUpClass(cls):
        """Contact sensor test suite init."""
        cls.sim_dt = 0.0025
        cls.durations = [cls.sim_dt, cls.sim_dt * 2, cls.sim_dt * 32, cls.sim_dt * 128]
        cls.terrains = [FLAT_TERRAIN_CFG, COBBLESTONE_TERRAIN_CFG]
        cls.devices = ["cuda:0", "cpu"]

    def test_cube_contact_time(self):
        """Checks contact sensor values for contact time and air time for a cube collision primitive."""
        self._run_contact_sensor_test(shape_cfg=CUBE_CFG)

    def test_sphere_contact_time(self):
        """Checks contact sensor values for contact time and air time for a sphere collision primitive."""
        self._run_contact_sensor_test(shape_cfg=SPHERE_CFG)

    def test_cube_stack_contact_filtering(self):
        """Checks contact sensor reporting for filtering stacked cube prims."""
        for device in self.devices:
            for num_envs in [1, 6, 24]:
                with self.subTest(device=device, num_envs=num_envs):
                    with build_simulation_context(device=device, dt=self.sim_dt, add_lighting=True) as sim:
                        sim._app_control_on_stop_handle = None
                        # Instance new scene for the current terrain and contact prim.
                        scene_cfg = ContactSensorSceneCfg(num_envs=num_envs, env_spacing=1.0, lazy_sensor_update=False)
                        scene_cfg.terrain = FLAT_TERRAIN_CFG.replace(prim_path="/World/ground")
                        # -- cube 1
                        scene_cfg.shape = CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cube_1")
                        scene_cfg.shape.init_state.pos = (0, -1.0, 1.0)
                        # -- cube 2 (on top of cube 1)
                        scene_cfg.shape_2 = CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cube_2")
                        scene_cfg.shape_2.init_state.pos = (0, -1.0, 1.525)
                        # -- contact sensor 1
                        scene_cfg.contact_sensor = ContactSensorCfg(
                            prim_path="{ENV_REGEX_NS}/Cube_1",
                            track_pose=True,
                            debug_vis=False,
                            update_period=0.0,
                            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_2"],
                        )
                        # -- contact sensor 2
                        scene_cfg.contact_sensor_2 = ContactSensorCfg(
                            prim_path="{ENV_REGEX_NS}/Cube_2",
                            track_pose=True,
                            debug_vis=False,
                            update_period=0.0,
                            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_1"],
                        )
                        scene = InteractiveScene(scene_cfg)

                        # Set variables internally for reference
                        self.sim = sim
                        self.scene = scene

                        # Play the simulation
                        self.sim.reset()

                        # Extract from scene for type hinting
                        contact_sensor: ContactSensor = self.scene["contact_sensor"]
                        contact_sensor_2: ContactSensor = self.scene["contact_sensor_2"]
                        # Check buffers have the right size
                        self.assertEqual(contact_sensor.contact_physx_view.filter_count, 1)
                        self.assertEqual(contact_sensor_2.contact_physx_view.filter_count, 1)

                        # Reset the contact sensors
                        self.scene.reset()
                        # Let the scene come to a rest
                        for _ in range(20):
                            self._perform_sim_step()

                        # Check values for cube 2
                        torch.testing.assert_close(
                            contact_sensor_2.data.force_matrix_w[:, :, 0], contact_sensor_2.data.net_forces_w
                        )
                        torch.testing.assert_close(
                            contact_sensor_2.data.force_matrix_w[:, :, 0], contact_sensor.data.force_matrix_w[:, :, 0]
                        )

    def test_sensor_print(self):
        """Test sensor print is working correctly."""
        with build_simulation_context(device="cuda:0", dt=self.sim_dt, add_lighting=False) as sim:
            sim._app_control_on_stop_handle = None
            # Spawn things into stage
            scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=1.0, lazy_sensor_update=False)
            scene_cfg.terrain = FLAT_TERRAIN_CFG.replace(prim_path="/World/ground")
            scene_cfg.shape = CUBE_CFG
            scene_cfg.contact_sensor = ContactSensorCfg(
                prim_path=scene_cfg.shape.prim_path,
                track_pose=True,
                debug_vis=False,
                update_period=0.0,
                track_air_time=True,
                history_length=3,
            )
            scene = InteractiveScene(scene_cfg)
            # Play the simulator
            sim.reset()
            # print info
            print(scene.sensors["contact_sensor"])

    """
    Internal helpers.
    """

    def _run_contact_sensor_test(self, shape_cfg: TestContactSensorRigidObjectCfg):
        """Runs a rigid body test for a given contact primitive configuration.

        This method iterates through each device and terrain combination in the simulation environment,
        running tests for contact sensors.

        Args:
            shape_cfg: The configuration parameters for the shape to be tested.
        """
        for device in self.devices:
            for terrain in self.terrains:
                with self.subTest(device=device, terrain=terrain):
                    with build_simulation_context(device=device, dt=self.sim_dt, add_lighting=True) as sim:
                        sim._app_control_on_stop_handle = None
                        # Instance new scene for the current terrain and contact prim.
                        scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=1.0, lazy_sensor_update=False)
                        scene_cfg.terrain = terrain
                        scene_cfg.shape = shape_cfg
                        scene_cfg.contact_sensor = ContactSensorCfg(
                            prim_path=shape_cfg.prim_path,
                            track_pose=True,
                            debug_vis=False,
                            update_period=0.0,
                            track_air_time=True,
                            history_length=3,
                        )
                        scene = InteractiveScene(scene_cfg)

                        # Set variables internally for reference
                        self.sim = sim
                        self.scene = scene

                        # Play the simulation
                        self.sim.reset()

                        # Run contact time and air time tests.
                        self._test_sensor_contact(
                            shape=self.scene["shape"],
                            sensor=self.scene["contact_sensor"],
                            mode=ContactTestMode.IN_CONTACT,
                        )
                        self._test_sensor_contact(
                            shape=self.scene["shape"],
                            sensor=self.scene["contact_sensor"],
                            mode=ContactTestMode.NON_CONTACT,
                        )

    def _test_sensor_contact(self, shape: RigidObject, sensor: ContactSensor, mode: ContactTestMode):
        """Test for the contact sensor.

        This test sets the contact prim to a pose either in contact or out of contact with the ground plane for
        a known duration. Once the contact duration has elapsed, the data stored inside the contact sensor
        associated with the contact prim is checked against the expected values.

        This process is repeated for all elements in :attr:`TestContactSensor.durations`, where each successive
        contact timing test is punctuated by setting the contact prim to the complement of the desired contact mode
        for 1 sim time-step.

        Args:
            shape: The contact prim used for the contact sensor test.
            sensor: The sensor reporting data to be verified by the contact sensor test.
            mode: The contact test mode: either contact with ground plane or air time.
        """
        # reset the test state
        sensor.reset()
        expected_last_test_contact_time = 0
        expected_last_reset_contact_time = 0

        # set poses for shape for a given contact sensor test mode.
        # desired contact mode to set for a given duration.
        test_pose = None
        # complement of the desired contact mode used to reset the contact sensor.
        reset_pose = None
        if mode == ContactTestMode.IN_CONTACT:
            test_pose = shape.cfg.contact_pose
            reset_pose = shape.cfg.non_contact_pose
        elif mode == ContactTestMode.NON_CONTACT:
            test_pose = shape.cfg.non_contact_pose
            reset_pose = shape.cfg.contact_pose
        else:
            raise ValueError("Received incompatible contact sensor test mode")

        for idx in range(len(self.durations)):
            current_test_time = 0
            duration = self.durations[idx]
            while current_test_time < duration:
                # set object states to contact the ground plane
                shape.write_root_pose_to_sim(root_pose=test_pose)
                # perform simulation step
                self._perform_sim_step()
                # increment contact time
                current_test_time += self.sim_dt
            # set last contact time to the previous desired contact duration plus the extra dt allowance.
            expected_last_test_contact_time = self.durations[idx - 1] + self.sim_dt if idx > 0 else 0
            # Check the data inside the contact sensor
            if mode == ContactTestMode.IN_CONTACT:
                self._check_prim_contact_state_times(
                    sensor=sensor,
                    expected_air_time=0.0,
                    expected_contact_time=self.durations[idx],
                    expected_last_contact_time=expected_last_test_contact_time,
                    expected_last_air_time=expected_last_reset_contact_time,
                    dt=duration + self.sim_dt,
                )
            elif mode == ContactTestMode.NON_CONTACT:
                self._check_prim_contact_state_times(
                    sensor=sensor,
                    expected_air_time=self.durations[idx],
                    expected_contact_time=0.0,
                    expected_last_contact_time=expected_last_reset_contact_time,
                    expected_last_air_time=expected_last_test_contact_time,
                    dt=duration + self.sim_dt,
                )
            # switch the contact mode for 1 dt step before the next contact test begins.
            shape.write_root_pose_to_sim(root_pose=reset_pose)
            # perform simulation step
            self._perform_sim_step()
            # set the last air time to 2 sim_dt steps, because last_air_time and last_contact_time
            # adds an additional sim_dt to the total time spent in the previous contact mode for uncertainty in
            # when the contact switch happened in between a dt step.
            expected_last_reset_contact_time = 2 * self.sim_dt

    def _check_prim_contact_state_times(
        self,
        sensor: ContactSensor,
        expected_air_time: float,
        expected_contact_time: float,
        expected_last_air_time: float,
        expected_last_contact_time: float,
        dt: float,
    ) -> None:
        """Checks contact sensor data matches expected values.

        Args:
            sensor: Instance of ContactSensor containing data to be tested.
            expected_air_time: Air time ground truth.
            expected_contact_time: Contact time ground truth.
            expected_last_air_time: Last air time ground truth.
            expected_last_contact_time: Last contact time ground truth.
            dt: Time since previous contact mode switch. If the contact prim left contact 0.1 seconds ago,
                dt should be 0.1 + simulation dt seconds.
        """
        # store current state of the contact prim
        in_air = False
        in_contact = False
        if expected_air_time > 0.0:
            in_air = True
        if expected_contact_time > 0.0:
            in_contact = True
        measured_contact_time = sensor.data.current_contact_time
        measured_air_time = sensor.data.current_air_time
        measured_last_contact_time = sensor.data.last_contact_time
        measured_last_air_time = sensor.data.last_air_time
        # check current contact state
        self.assertAlmostEqual(measured_contact_time.item(), expected_contact_time, places=2)
        self.assertAlmostEqual(measured_air_time.item(), expected_air_time, places=2)
        # check last contact state
        self.assertAlmostEqual(measured_last_contact_time.item(), expected_last_contact_time, places=2)
        self.assertAlmostEqual(measured_last_air_time.item(), expected_last_air_time, places=2)
        # check current contact mode
        self.assertEqual(sensor.compute_first_contact(dt=dt).item(), in_contact)
        self.assertEqual(sensor.compute_first_air(dt=dt).item(), in_air)

    def _perform_sim_step(self) -> None:
        """Updates sensors and steps the contact sensor test scene."""
        # write data to simulation
        self.scene.write_data_to_sim()
        # simulate
        self.sim.step(render=False)
        # update buffers at sim dt
        self.scene.update(dt=self.sim_dt)


if __name__ == "__main__":
    run_tests()
