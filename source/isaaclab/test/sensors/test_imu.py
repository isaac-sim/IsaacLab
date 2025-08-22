# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import pathlib
import torch

import isaacsim.core.utils.stage as stage_utils
import pytest

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.imu import ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # isort: skip

# offset of imu_link from base_link on anymal_c
POS_OFFSET = (0.2488, 0.00835, 0.04628)
ROT_OFFSET = (0.7071068, 0, 0, 0.7071068)

# offset of imu_link from link_1 on simple_2_link
PEND_POS_OFFSET = (0.4, 0.0, 0.1)
PEND_ROT_OFFSET = (0.5, 0.5, 0.5, 0.5)


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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -2.0, 0.5)),
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    )

    # articulations - robot
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
    pendulum = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/pendulum",
        spawn=sim_utils.UrdfFileCfg(
            fix_base=True,
            merge_fixed_joints=False,
            make_instanceable=False,
            asset_path=f"{pathlib.Path(__file__).parent.resolve()}/urdfs/simple_2_link.urdf",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "joint_1_act": ImplicitActuatorCfg(joint_names_expr=["joint_.*"], stiffness=0.0, damping=0.3),
        },
    )
    # sensors - imu (filled inside unit test)
    imu_ball: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        gravity_bias=(0.0, 0.0, 0.0),
    )
    imu_cube: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        gravity_bias=(0.0, 0.0, 0.0),
    )
    imu_robot_imu_link: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/robot/imu_link",
        gravity_bias=(0.0, 0.0, 0.0),
    )
    imu_robot_base: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/robot/base",
        offset=ImuCfg.OffsetCfg(
            pos=POS_OFFSET,
            rot=ROT_OFFSET,
        ),
        gravity_bias=(0.0, 0.0, 0.0),
    )

    imu_pendulum_imu_link: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/pendulum/imu_link",
        debug_vis=not app_launcher._headless,
        visualizer_cfg=RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Acceleration/imu_link"),
        gravity_bias=(0.0, 0.0, 9.81),
    )
    imu_pendulum_base: ImuCfg = ImuCfg(
        prim_path="{ENV_REGEX_NS}/pendulum/link_1",
        offset=ImuCfg.OffsetCfg(
            pos=PEND_POS_OFFSET,
            rot=PEND_ROT_OFFSET,
        ),
        debug_vis=not app_launcher._headless,
        visualizer_cfg=GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Acceleration/base"),
        gravity_bias=(0.0, 0.0, 9.81),
    )

    def __post_init__(self):
        """Post initialization."""
        # change position of the robot
        self.robot.init_state.pos = (0.0, 2.0, 1.0)
        self.pendulum.init_state.pos = (-1.0, 1.0, 0.5)

        # change asset
        self.robot.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_c/anymal_c.usd"
        # change iterations
        self.robot.spawn.articulation_props.solver_position_iteration_count = 32
        self.robot.spawn.articulation_props.solver_velocity_iteration_count = 32


@pytest.fixture
def setup_sim():
    """Create a simulation context and scene."""
    # Create a new stage
    stage_utils.create_new_stage()
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.001)
    sim_cfg.physx.solver_type = 0  # 0: PGS, 1: TGS --> use PGS for more accurate results
    sim = sim_utils.SimulationContext(sim_cfg)
    # construct scene
    scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    yield sim, scene
    # Cleanup
    sim.clear_all_callbacks()
    sim.clear_instance()


@pytest.mark.isaacsim_ci
def test_constant_velocity(setup_sim):
    """Test the Imu sensor with a constant velocity.

    Expected behavior is that the linear and angular are approx the same at every time step as in each step we set
    the same velocity and therefore reset the physx buffers.
    """
    sim, scene = setup_sim
    prev_lin_acc_ball = torch.zeros((scene.num_envs, 3), dtype=torch.float32, device=scene.device)
    prev_ang_acc_ball = torch.zeros((scene.num_envs, 3), dtype=torch.float32, device=scene.device)
    prev_lin_acc_cube = torch.zeros((scene.num_envs, 3), dtype=torch.float32, device=scene.device)
    prev_ang_acc_cube = torch.zeros((scene.num_envs, 3), dtype=torch.float32, device=scene.device)

    for idx in range(200):
        # set velocity
        scene.rigid_objects["balls"].write_root_velocity_to_sim(
            torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=scene.device).repeat(
                scene.num_envs, 1
            )
        )
        scene.rigid_objects["cube"].write_root_velocity_to_sim(
            torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=scene.device).repeat(
                scene.num_envs, 1
            )
        )
        # write data to sim
        scene.write_data_to_sim()

        # perform step
        sim.step()
        # read data from sim
        scene.update(sim.get_physics_dt())

        if idx > 1:
            # check the imu accelerations
            torch.testing.assert_close(
                scene.sensors["imu_ball"].data.lin_acc_b,
                prev_lin_acc_ball,
                rtol=1e-3,
                atol=1e-3,
            )
            torch.testing.assert_close(
                scene.sensors["imu_ball"].data.ang_acc_b,
                prev_ang_acc_ball,
                rtol=1e-3,
                atol=1e-3,
            )

            torch.testing.assert_close(
                scene.sensors["imu_cube"].data.lin_acc_b,
                prev_lin_acc_cube,
                rtol=1e-3,
                atol=1e-3,
            )
            torch.testing.assert_close(
                scene.sensors["imu_cube"].data.ang_acc_b,
                prev_ang_acc_cube,
                rtol=1e-3,
                atol=1e-3,
            )

            # check the imu velocities
            # NOTE: the expected lin_vel_b is the same as the set velocity, as write_root_velocity_to_sim is
            #       setting v_0 (initial velocity) and then a calculation step of v_i = v_0 + a*dt. Consequently,
            #       the data.lin_vel_b is returning approx. v_i.
            torch.testing.assert_close(
                scene.sensors["imu_ball"].data.lin_vel_b,
                torch.tensor([[1.0, 0.0, -scene.physics_dt * 9.81]], dtype=torch.float32, device=scene.device).repeat(
                    scene.num_envs, 1
                ),
                rtol=1e-4,
                atol=1e-4,
            )
            torch.testing.assert_close(
                scene.sensors["imu_cube"].data.lin_vel_b,
                torch.tensor([[1.0, 0.0, -scene.physics_dt * 9.81]], dtype=torch.float32, device=scene.device).repeat(
                    scene.num_envs, 1
                ),
                rtol=1e-4,
                atol=1e-4,
            )

        # update previous values
        prev_lin_acc_ball = scene.sensors["imu_ball"].data.lin_acc_b.clone()
        prev_ang_acc_ball = scene.sensors["imu_ball"].data.ang_acc_b.clone()
        prev_lin_acc_cube = scene.sensors["imu_cube"].data.lin_acc_b.clone()
        prev_ang_acc_cube = scene.sensors["imu_cube"].data.ang_acc_b.clone()


@pytest.mark.isaacsim_ci
def test_constant_acceleration(setup_sim):
    """Test the Imu sensor with a constant acceleration."""
    sim, scene = setup_sim
    for idx in range(100):
        # set acceleration
        scene.rigid_objects["balls"].write_root_velocity_to_sim(
            torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=scene.device).repeat(
                scene.num_envs, 1
            )
            * (idx + 1)
        )
        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # read data from sim
        scene.update(sim.get_physics_dt())

        # skip first step where initial velocity is zero
        if idx < 1:
            continue

        # check the imu data
        torch.testing.assert_close(
            scene.sensors["imu_ball"].data.lin_acc_b,
            math_utils.quat_apply_inverse(
                scene.rigid_objects["balls"].data.root_quat_w,
                torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32, device=scene.device).repeat(scene.num_envs, 1)
                / sim.get_physics_dt(),
            ),
            rtol=1e-4,
            atol=1e-4,
        )

        # check the angular velocity
        torch.testing.assert_close(
            scene.sensors["imu_ball"].data.ang_vel_b,
            scene.rigid_objects["balls"].data.root_ang_vel_b,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.isaacsim_ci
def test_single_dof_pendulum(setup_sim):
    """Test imu against analytical pendulum problem."""
    sim, scene = setup_sim
    # pendulum length
    pend_length = PEND_POS_OFFSET[0]

    # should achieve same results between the two imu sensors on the robot
    for idx in range(500):

        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # read data from sim
        scene.update(sim.get_physics_dt())

        # get pendulum joint state
        joint_pos = scene.articulations["pendulum"].data.joint_pos
        joint_vel = scene.articulations["pendulum"].data.joint_vel
        joint_acc = scene.articulations["pendulum"].data.joint_acc

        # IMU and base data
        imu_data = scene.sensors["imu_pendulum_imu_link"].data
        base_data = scene.sensors["imu_pendulum_base"].data

        # extract imu_link imu_sensor dynamics
        lin_vel_w_imu_link = math_utils.quat_apply(imu_data.quat_w, imu_data.lin_vel_b)
        lin_acc_w_imu_link = math_utils.quat_apply(imu_data.quat_w, imu_data.lin_acc_b)

        # calculate the joint dynamics from the imu_sensor (y axis of imu_link is parallel to joint axis of pendulum)
        joint_vel_imu = math_utils.quat_apply(imu_data.quat_w, imu_data.ang_vel_b)[..., 1].unsqueeze(-1)
        joint_acc_imu = math_utils.quat_apply(imu_data.quat_w, imu_data.ang_acc_b)[..., 1].unsqueeze(-1)

        # calculate analytical solution
        vx = -joint_vel * pend_length * torch.sin(joint_pos)
        vy = torch.zeros(2, 1, device=scene.device)
        vz = -joint_vel * pend_length * torch.cos(joint_pos)
        gt_linear_vel_w = torch.cat([vx, vy, vz], dim=-1)

        ax = -joint_acc * pend_length * torch.sin(joint_pos) - joint_vel**2 * pend_length * torch.cos(joint_pos)
        ay = torch.zeros(2, 1, device=scene.device)
        az = -joint_acc * pend_length * torch.cos(joint_pos) + joint_vel**2 * pend_length * torch.sin(joint_pos) + 9.81
        gt_linear_acc_w = torch.cat([ax, ay, az], dim=-1)

        # skip first step where initial velocity is zero
        if idx < 2:
            continue

        # compare imu projected gravity
        gravity_dir_w = torch.tensor((0.0, 0.0, -1.0), device=scene.device).repeat(2, 1)
        gravity_dir_b = math_utils.quat_apply_inverse(imu_data.quat_w, gravity_dir_w)
        torch.testing.assert_close(
            imu_data.projected_gravity_b,
            gravity_dir_b,
        )

        # compare imu angular velocity with joint velocity
        torch.testing.assert_close(
            joint_vel,
            joint_vel_imu,
            rtol=1e-1,
            atol=1e-3,
        )
        # compare imu angular acceleration with joint acceleration
        torch.testing.assert_close(
            joint_acc,
            joint_acc_imu,
            rtol=1e-1,
            atol=1e-3,
        )
        # compare imu linear velocity with simple pendulum calculation
        torch.testing.assert_close(
            gt_linear_vel_w,
            lin_vel_w_imu_link,
            rtol=1e-1,
            atol=1e-3,
        )
        # compare imu linear acceleration with simple pendulum calculation
        torch.testing.assert_close(
            gt_linear_acc_w,
            lin_acc_w_imu_link,
            rtol=1e-1,
            atol=1e0,
        )

        # check the position between offset and imu definition
        torch.testing.assert_close(
            base_data.pos_w,
            imu_data.pos_w,
            rtol=1e-5,
            atol=1e-5,
        )

        # check the orientation between offset and imu definition
        torch.testing.assert_close(
            base_data.quat_w,
            imu_data.quat_w,
            rtol=1e-4,
            atol=1e-4,
        )

        # check the angular velocities of the imus between offset and imu definition
        torch.testing.assert_close(
            base_data.ang_vel_b,
            imu_data.ang_vel_b,
            rtol=1e-4,
            atol=1e-4,
        )
        # check the angular acceleration of the imus between offset and imu definition
        torch.testing.assert_close(
            base_data.ang_acc_b,
            imu_data.ang_acc_b,
            rtol=1e-4,
            atol=1e-4,
        )

        # check the linear velocity of the imus between offset and imu definition
        torch.testing.assert_close(
            base_data.lin_vel_b,
            imu_data.lin_vel_b,
            rtol=1e-2,
            atol=5e-3,
        )

        # check the linear acceleration of the imus between offset and imu definition
        torch.testing.assert_close(
            base_data.lin_acc_b,
            imu_data.lin_acc_b,
            rtol=1e-1,
            atol=1e-1,
        )


@pytest.mark.isaacsim_ci
def test_offset_calculation(setup_sim):
    """Test offset configuration argument."""
    sim, scene = setup_sim
    # should achieve same results between the two imu sensors on the robot
    for idx in range(500):
        # set acceleration
        scene.articulations["robot"].write_root_velocity_to_sim(
            torch.tensor([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=scene.device).repeat(
                scene.num_envs, 1
            )
            * (idx + 1)
        )
        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # read data from sim
        scene.update(sim.get_physics_dt())

        # skip first step where initial velocity is zero
        if idx < 1:
            continue

        # check the accelerations
        torch.testing.assert_close(
            scene.sensors["imu_robot_base"].data.lin_acc_b,
            scene.sensors["imu_robot_imu_link"].data.lin_acc_b,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            scene.sensors["imu_robot_base"].data.ang_acc_b,
            scene.sensors["imu_robot_imu_link"].data.ang_acc_b,
            rtol=1e-4,
            atol=1e-4,
        )

        # check the velocities
        torch.testing.assert_close(
            scene.sensors["imu_robot_base"].data.ang_vel_b,
            scene.sensors["imu_robot_imu_link"].data.ang_vel_b,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            scene.sensors["imu_robot_base"].data.lin_vel_b,
            scene.sensors["imu_robot_imu_link"].data.lin_vel_b,
            rtol=1e-4,
            atol=1e-4,
        )

        # check the orientation
        torch.testing.assert_close(
            scene.sensors["imu_robot_base"].data.quat_w,
            scene.sensors["imu_robot_imu_link"].data.quat_w,
            rtol=1e-4,
            atol=1e-4,
        )
        # check the position
        torch.testing.assert_close(
            scene.sensors["imu_robot_base"].data.pos_w,
            scene.sensors["imu_robot_imu_link"].data.pos_w,
            rtol=1e-4,
            atol=1e-4,
        )
        # check the projected gravity
        torch.testing.assert_close(
            scene.sensors["imu_robot_base"].data.projected_gravity_b,
            scene.sensors["imu_robot_imu_link"].data.projected_gravity_b,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.isaacsim_ci
def test_env_ids_propagation(setup_sim):
    """Test that env_ids argument propagates through update and reset methods"""
    sim, scene = setup_sim
    scene.reset()

    for idx in range(10):
        # set acceleration
        scene.articulations["robot"].write_root_velocity_to_sim(
            torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=scene.device).repeat(
                scene.num_envs, 1
            )
            * (idx + 1)
        )
        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # read data from sim
        scene.update(sim.get_physics_dt())

    # reset scene for env 1
    scene.reset(env_ids=[1])
    # read data from sim
    scene.update(sim.get_physics_dt())
    # perform step
    sim.step()
    # read data from sim
    scene.update(sim.get_physics_dt())


@pytest.mark.isaacsim_ci
def test_sensor_print(setup_sim):
    """Test sensor print is working correctly."""
    sim, scene = setup_sim
    # Create sensor
    sensor = scene.sensors["imu_ball"]
    # print info
    print(sensor)
