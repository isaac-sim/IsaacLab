# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import pytest
import torch
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.pva import Pva, PvaCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # isort: skip

# offset of imu_link from base_link on anymal_c
POS_OFFSET = (0.2488, 0.00835, 0.04628)
ROT_OFFSET = (0, 0, 0.7071068, 0.7071068)

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
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=0.5),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -2.0, 0.5)),
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=0.5),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    )

    # articulations - robot
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")
    # pendulum - uses merge_fixed_joints=True so that fixed-joint
    # child links (base, imu_link) are merged into their parents during URDF XML
    # pre-processing. This avoids fixed-joint constraint violations at velocity level
    # (the solver uses velocity_iteration_count=0). A non-physics imu_link Xform is
    # created programmatically in the test fixture (see setup_sim).
    pendulum = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/pendulum",
        spawn=sim_utils.UrdfFileCfg(
            fix_base=True,
            merge_fixed_joints=True,
            make_instanceable=False,
            asset_path=f"{pathlib.Path(__file__).parent.resolve()}/urdfs/simple_2_link.urdf",
            articulation_props=[
                PhysxArticulationCfg(
                    enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
                ),
                NewtonArticulationCfg(self_collision_enabled=True),
            ],
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "joint_1_act": ImplicitActuatorCfg(joint_names_expr=["joint_.*"], stiffness=0.0, damping=0.3),
        },
    )

    # sensors - pva (filled inside unit test)
    pva_ball: PvaCfg = PvaCfg(prim_path="{ENV_REGEX_NS}/ball")
    pva_cube: PvaCfg = PvaCfg(prim_path="{ENV_REGEX_NS}/cube")
    pva_robot_imu_link: PvaCfg = PvaCfg(prim_path="{ENV_REGEX_NS}/robot/imu_link")
    pva_robot_base: PvaCfg = PvaCfg(
        prim_path="{ENV_REGEX_NS}/robot/base",
        offset=PvaCfg.OffsetCfg(
            pos=POS_OFFSET,
            rot=ROT_OFFSET,
        ),
    )
    pva_robot_norb: PvaCfg = PvaCfg(
        prim_path="{ENV_REGEX_NS}/robot/LF_HIP/LF_hip_fixed",
        offset=PvaCfg.OffsetCfg(
            pos=POS_OFFSET,
            rot=ROT_OFFSET,
        ),
    )
    # The new URDF converter (urdf-usd-converter) places links under Geometry/ in a nested
    # kinematic tree.  With merge_fixed_joints=True the hierarchy for simple_2_link.urdf is:
    #   Geometry/world/link_1  (base merged into world, imu_link merged into link_1)
    # A non-physics imu_link Xform is recreated in the test fixture (see setup_sim).
    pva_pendulum_imu_link: PvaCfg = PvaCfg(
        prim_path="{ENV_REGEX_NS}/pendulum/Geometry/world/link_1/imu_link",
        debug_vis=not app_launcher._headless,
        visualizer_cfg=RED_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Acceleration/imu_link"),
    )
    pva_pendulum_base: PvaCfg = PvaCfg(
        prim_path="{ENV_REGEX_NS}/pendulum/Geometry/world/link_1",
        offset=PvaCfg.OffsetCfg(
            pos=PEND_POS_OFFSET,
            rot=PEND_ROT_OFFSET,
        ),
        debug_vis=not app_launcher._headless,
        visualizer_cfg=GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Acceleration/base"),
    )

    def __post_init__(self):
        """Post initialization."""
        # change position of the robot
        self.robot.init_state.pos = (0.0, 2.0, 1.0)
        self.pendulum.init_state.pos = (-2.0, 1.0, 0.5)

        # change asset
        self.robot.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_c/anymal_c.usd"
        # change iterations -- the solver counts live on the PhysX articulation fragment
        physx_articulation = next(
            frag for frag in self.robot.spawn.articulation_props if isinstance(frag, PhysxArticulationCfg)
        )
        physx_articulation.solver_position_iteration_count = 32
        physx_articulation.solver_velocity_iteration_count = 32


@pytest.fixture
def setup_sim():
    """Create a simulation context and scene."""
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.001, physics=PhysxCfg(solver_type=0)
    )  # 0: PGS, 1: TGS --> use PGS for more accurate results
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        # construct scene
        scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False, replicate_physics=False)
        scene = InteractiveScene(scene_cfg)
        # The pendulum uses merge_fixed_joints=True, so the fixed-joint child link
        # imu_link is removed from the URDF before USD conversion.  Recreate it as a
        # plain Xform (no RigidBodyAPI) under link_1 for every environment.  The PVA
        # sensor must then resolve the rigid-body ancestor (link_1) and cache the
        # fixed offset — exercising the "indirect attachment" code path.
        for i in range(scene_cfg.num_envs):
            prim_path = f"/World/envs/env_{i}/pendulum/Geometry/world/link_1/imu_link"
            sim_utils.create_prim(prim_path, "Xform", translation=PEND_POS_OFFSET, orientation=PEND_ROT_OFFSET)
        # Play the simulator
        sim.reset()
        yield sim, scene
    # Cleanup is handled by build_simulation_context


@pytest.mark.isaacsim_ci
def test_constant_acceleration(setup_sim):
    """A constant applied force yields the solver acceleration F/m on top of free fall."""
    sim, scene = setup_sim
    balls = scene.rigid_objects["balls"]
    force = 0.25  # [N] on a 0.5 kg ball -> 0.5 m/s^2
    expected_acc = force / 0.5
    forces = torch.zeros((scene.num_envs, 1, 3), dtype=torch.float32, device=scene.device)
    forces[..., 0] = force
    # keep the window short so the ball stays airborne: PVA reports kinematic acceleration,
    # so the vertical component is the free-fall -g
    for idx in range(10):
        balls.set_external_force_and_torque(forces, torch.zeros_like(forces))
        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # read data from sim
        scene.update(sim.get_physics_dt())

        # skip first step where the solver has not integrated the force yet
        if idx < 1:
            continue

        # check the pva data
        torch.testing.assert_close(
            scene.sensors["pva_ball"].data.lin_acc_b.torch,
            math_utils.quat_apply_inverse(
                scene.rigid_objects["balls"].data.root_quat_w.torch,
                torch.tensor([[expected_acc, 0.0, -9.81]], dtype=torch.float32, device=scene.device).repeat(
                    scene.num_envs, 1
                ),
            ),
            rtol=1e-4,
            atol=1e-4,
        )

        # check the angular velocity
        torch.testing.assert_close(
            scene.sensors["pva_ball"].data.ang_vel_b.torch,
            scene.rigid_objects["balls"].data.root_ang_vel_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.isaacsim_ci
def test_offset_calculation(setup_sim):
    """Test the PVA sensor on independently driven bodies that share one scene.

    * Robot: a configured offset on the base matches a sensor on the ``imu_link`` child prim.
    * Pendulum: the sensor on a non-rigid ``imu_link`` Xform child (indirect attachment through the
      rigid-body ancestor) matches the analytic pendulum and a configured offset on ``link_1``.
    * Ball and cube: writing the same velocity every step yields constant accelerations and the
      written velocity.
    """
    sim, scene = setup_sim
    prev_lin_acc_ball = torch.zeros((scene.num_envs, 3), dtype=torch.float32, device=scene.device)
    prev_ang_acc_ball = torch.zeros((scene.num_envs, 3), dtype=torch.float32, device=scene.device)
    prev_lin_acc_cube = torch.zeros((scene.num_envs, 3), dtype=torch.float32, device=scene.device)
    prev_ang_acc_cube = torch.zeros((scene.num_envs, 3), dtype=torch.float32, device=scene.device)
    constant_velocity = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=scene.device)
    # pendulum length
    pend_length = PEND_POS_OFFSET[0]
    pendulum = scene.articulations["pendulum"]
    pendulum_link_id = pendulum.find_bodies("link_1")[0][0]

    for idx in range(500):
        # set acceleration
        scene.articulations["robot"].write_root_velocity_to_sim(
            torch.tensor([[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=scene.device).repeat(
                scene.num_envs, 1
            )
            * (idx + 1)
        )
        # set constant velocity
        scene.rigid_objects["balls"].write_root_velocity_to_sim(constant_velocity.repeat(scene.num_envs, 1))
        scene.rigid_objects["cube"].write_root_velocity_to_sim(constant_velocity.repeat(scene.num_envs, 1))
        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # read data from sim
        scene.update(sim.get_physics_dt())

        # constant velocity: the accelerations are the same at every step
        if idx > 1:
            torch.testing.assert_close(
                scene.sensors["pva_ball"].data.lin_acc_b.torch,
                prev_lin_acc_ball,
                rtol=1e-3,
                atol=1e-3,
            )
            torch.testing.assert_close(
                scene.sensors["pva_ball"].data.ang_acc_b.torch,
                prev_ang_acc_ball,
                rtol=1e-3,
                atol=1e-3,
            )
            torch.testing.assert_close(
                scene.sensors["pva_cube"].data.lin_acc_b.torch,
                prev_lin_acc_cube,
                rtol=1e-3,
                atol=1e-3,
            )
            torch.testing.assert_close(
                scene.sensors["pva_cube"].data.ang_acc_b.torch,
                prev_ang_acc_cube,
                rtol=1e-3,
                atol=1e-3,
            )

            # NOTE: the expected lin_vel_b is the same as the set velocity, as write_root_velocity_to_sim is
            #       setting v_0 (initial velocity) and then a calculation step of v_i = v_0 + a*dt. Consequently,
            #       the data.lin_vel_b is returning approx. v_i.
            torch.testing.assert_close(
                scene.sensors["pva_ball"].data.lin_vel_b.torch,
                torch.tensor([[1.0, 0.0, -scene.physics_dt * 9.81]], dtype=torch.float32, device=scene.device).repeat(
                    scene.num_envs, 1
                ),
                rtol=1e-4,
                atol=1e-4,
            )
            torch.testing.assert_close(
                scene.sensors["pva_cube"].data.lin_vel_b.torch,
                torch.tensor([[1.0, 0.0, -scene.physics_dt * 9.81]], dtype=torch.float32, device=scene.device).repeat(
                    scene.num_envs, 1
                ),
                rtol=1e-4,
                atol=1e-4,
            )
        prev_lin_acc_ball = scene.sensors["pva_ball"].data.lin_acc_b.torch.clone()
        prev_ang_acc_ball = scene.sensors["pva_ball"].data.ang_acc_b.torch.clone()
        prev_lin_acc_cube = scene.sensors["pva_cube"].data.lin_acc_b.torch.clone()
        prev_ang_acc_cube = scene.sensors["pva_cube"].data.ang_acc_b.torch.clone()

        # skip first step where initial velocity is zero
        if idx < 1:
            continue

        # robot: offset vs imu_link definition
        torch.testing.assert_close(
            scene.sensors["pva_robot_base"].data.lin_acc_b.torch,
            scene.sensors["pva_robot_imu_link"].data.lin_acc_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            scene.sensors["pva_robot_base"].data.ang_acc_b.torch,
            scene.sensors["pva_robot_imu_link"].data.ang_acc_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            scene.sensors["pva_robot_base"].data.ang_vel_b.torch,
            scene.sensors["pva_robot_imu_link"].data.ang_vel_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            scene.sensors["pva_robot_base"].data.lin_vel_b.torch,
            scene.sensors["pva_robot_imu_link"].data.lin_vel_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            scene.sensors["pva_robot_base"].data.quat_w.torch,
            scene.sensors["pva_robot_imu_link"].data.quat_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            scene.sensors["pva_robot_base"].data.pos_w.torch,
            scene.sensors["pva_robot_imu_link"].data.pos_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            scene.sensors["pva_robot_base"].data.projected_gravity_b.torch,
            scene.sensors["pva_robot_imu_link"].data.projected_gravity_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )

        # skip the second pendulum step where the initial velocity is still zero
        if idx < 2:
            continue

        # get pendulum joint state
        joint_pos = pendulum.data.joint_pos.torch
        joint_vel = pendulum.data.joint_vel.torch
        # Use the solver-reported link acceleration as the reference. The public joint
        # acceleration is finite-differenced and intentionally differs from PVA semantics.
        joint_ang_acc_w_y = pendulum.data.body_com_acc_w.torch[:, pendulum_link_id, 4].unsqueeze(-1)

        # PVA and base data
        pva_data = scene.sensors["pva_pendulum_imu_link"].data
        base_data = scene.sensors["pva_pendulum_base"].data

        # extract imu_link pva_sensor dynamics
        lin_vel_w_imu_link = math_utils.quat_apply(pva_data.quat_w.torch, pva_data.lin_vel_b.torch)
        lin_acc_w_imu_link = math_utils.quat_apply(pva_data.quat_w.torch, pva_data.lin_acc_b.torch)

        # calculate the joint dynamics from the pva_sensor (y axis of imu_link is parallel to joint axis of pendulum)
        joint_vel_pva = math_utils.quat_apply(pva_data.quat_w.torch, pva_data.ang_vel_b.torch)[..., 1].unsqueeze(-1)
        joint_acc_pva = math_utils.quat_apply(pva_data.quat_w.torch, pva_data.ang_acc_b.torch)[..., 1].unsqueeze(-1)

        # calculate analytical solution
        vx = -joint_vel * pend_length * torch.sin(joint_pos)
        vy = torch.zeros(2, 1, device=scene.device)
        vz = -joint_vel * pend_length * torch.cos(joint_pos)
        gt_linear_vel_w = torch.cat([vx, vy, vz], dim=-1)

        ax = -joint_ang_acc_w_y * pend_length * torch.sin(joint_pos) - joint_vel**2 * pend_length * torch.cos(joint_pos)
        ay = torch.zeros(2, 1, device=scene.device)
        az = -joint_ang_acc_w_y * pend_length * torch.cos(joint_pos) + joint_vel**2 * pend_length * torch.sin(joint_pos)
        gt_linear_acc_w = torch.cat([ax, ay, az], dim=-1)

        # compare pva projected gravity
        gravity_dir_w = torch.tensor((0.0, 0.0, -1.0), device=scene.device).repeat(2, 1)
        gravity_dir_b = math_utils.quat_apply_inverse(pva_data.quat_w.torch, gravity_dir_w)
        torch.testing.assert_close(
            pva_data.projected_gravity_b.torch,
            gravity_dir_b,
        )

        # compare pva angular velocity with joint velocity
        torch.testing.assert_close(
            joint_vel,
            joint_vel_pva,
            rtol=1e-1,
            atol=1e-3,
        )
        # compare pva angular acceleration with solver-reported link acceleration
        torch.testing.assert_close(
            joint_ang_acc_w_y,
            joint_acc_pva,
            rtol=1e-1,
            atol=1e-3,
        )
        # compare pva linear velocity with simple pendulum calculation
        torch.testing.assert_close(
            gt_linear_vel_w,
            lin_vel_w_imu_link,
            rtol=1e-1,
            atol=1e-3,
        )
        # compare pva linear acceleration with simple pendulum calculation
        torch.testing.assert_close(
            gt_linear_acc_w,
            lin_acc_w_imu_link,
            rtol=1e-1,
            atol=1e0,
        )

        # check the position between offset and pva definition
        torch.testing.assert_close(
            base_data.pos_w.torch,
            pva_data.pos_w.torch,
            rtol=1e-5,
            atol=1e-5,
        )

        # check the orientation between offset and pva definition
        torch.testing.assert_close(
            base_data.quat_w.torch,
            pva_data.quat_w.torch,
            rtol=1e-4,
            atol=1e-4,
        )

        # check the angular velocities of the pvas between offset and pva definition
        torch.testing.assert_close(
            base_data.ang_vel_b.torch,
            pva_data.ang_vel_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )
        # check the angular acceleration of the pvas between offset and pva definition
        torch.testing.assert_close(
            base_data.ang_acc_b.torch,
            pva_data.ang_acc_b.torch,
            rtol=1e-4,
            atol=1e-4,
        )

        # check the linear velocity of the pvas between offset and pva definition
        torch.testing.assert_close(
            base_data.lin_vel_b.torch,
            pva_data.lin_vel_b.torch,
            rtol=1e-2,
            atol=5e-3,
        )

        # check the linear acceleration of the pvas between offset and pva definition
        torch.testing.assert_close(
            base_data.lin_acc_b.torch,
            pva_data.lin_acc_b.torch,
            rtol=1e-1,
            atol=1e-1,
        )

    # the recorded-launch optimization must be active on CUDA; a recording failure would only
    # warn and silently fall back to eager launches, defeating the optimization.
    if "cuda" in str(scene.device):
        assert scene.sensors["pva_ball"]._update_cmd is not None
        assert scene.sensors["pva_cube"]._update_cmd is not None

    assert "number of sensors : 2" in str(scene.sensors["pva_ball"])

    # A PVA sensor cannot be attached directly to the world: it must have a rigid-body ancestor.
    with pytest.raises(RuntimeError, match="find a rigid body ancestor prim"):
        Pva(PvaCfg(prim_path="/World/envs/env_0"))._initialize_impl()


@configclass
class _StaleResetSceneCfg(InteractiveSceneCfg):
    """Minimal scene for the post-reset staleness regression test."""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.25, 0.25, 0.25),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=0.5),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
    )
    pva_cube: PvaCfg = PvaCfg(prim_path="{ENV_REGEX_NS}/cube")


def test_no_stale_data_after_scene_reset():
    """Regression for #4970: ``scene.reset(env_ids)`` must not surface pre-reset PVA values.

    Mirrors the ``ManagerBasedRLEnv._reset_idx`` flow where reset runs inside a step
    without a subsequent physics step. The PVA sensor's lazy ``data`` accessor must not
    refetch from the PhysX rigid-body view here (its buffers still reflect the previous
    physics step and would surface pre-reset velocities and accelerations).
    """
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, physics=PhysxCfg(solver_type=0))
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = _StaleResetSceneCfg(num_envs=2, env_spacing=2.0, lazy_sensor_update=False)
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()

        sensor: Pva = scene["pva_cube"]

        # Let the cube fall so the PVA sensor has a non-zero velocity to read.
        for _ in range(30):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt=sim.get_physics_dt())

        # Sanity: the cubes have gained downward velocity.
        pre_reset_lin_vel = sensor.data.lin_vel_b.torch.clone()
        assert (torch.linalg.norm(pre_reset_lin_vel, dim=-1) > 0.05).all(), (
            f"Expected non-zero velocity before reset; got {pre_reset_lin_vel}"
        )

        # Reset env 0 without writing fresh velocity/transform. The PhysX velocity
        # buffer therefore still holds the pre-reset (falling) value.
        scene.reset(env_ids=torch.tensor([0], device=sensor.device))

        # The public ``data`` accessor must not refetch a stale PhysX buffer for the reset env,
        # while env 1 keeps its measurement.
        post_reset_vel = sensor.data.lin_vel_b.torch
        post_reset_acc = sensor.data.lin_acc_b.torch
        torch.testing.assert_close(post_reset_vel[0], torch.zeros_like(post_reset_vel[0]))
        torch.testing.assert_close(post_reset_acc[0], torch.zeros_like(post_reset_acc[0]))
        torch.testing.assert_close(post_reset_vel[1], pre_reset_lin_vel[1])


@pytest.mark.parametrize("access_mode", ("lazy_read", "update_period"))
def test_velocity_writes_do_not_produce_spurious_acceleration(setup_sim, access_mode):
    """Directly written (teleported) velocities do not show up as fake accelerations.

    The PVA sensor reports the solver acceleration, so a velocity write — which involves no
    force — must not spike the sensor. This was a known artifact of the previous
    finite-difference implementation (e.g. on environment resets). The airborne ball is in
    free fall throughout, so the only acceleration left is ``-g`` along the world z axis.
    """
    sim, scene = setup_sim
    dt = sim.get_physics_dt()
    body = scene.rigid_objects["balls"]
    sensor = scene.sensors["pva_ball"]
    velocity = torch.zeros((scene.num_envs, 6), dtype=torch.float32, device=scene.device)

    body.write_root_velocity_to_sim_index(root_velocity=velocity)
    scene.write_data_to_sim()
    sim.step()
    scene.update(dt)
    _ = sensor.data

    scene.cfg.lazy_sensor_update = True
    if access_mode == "update_period":
        sensor.cfg.update_period = 4 * dt

    for step in range(4):
        velocity[:, 0] = 0.1 * (step + 1)
        velocity[:, 5] = 0.2 * (step + 1)
        body.write_root_velocity_to_sim_index(root_velocity=velocity)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        if access_mode == "update_period":
            _ = sensor.data

    expected_lin_acc = torch.tensor([[0.0, 0.0, -9.81]], device=scene.device).repeat(scene.num_envs, 1)
    torch.testing.assert_close(sensor.data.lin_acc_b.torch, expected_lin_acc, rtol=0.0, atol=1e-2)
    # The angular acceleration is not exactly zero because PhysX's default angular damping
    # opposes the written spin, so bound it well below the 0.2 / dt spike finite differencing
    # would report for the same velocity writes.
    assert torch.all(sensor.data.ang_acc_b.torch.abs() < 0.01 * 0.2 / dt)
