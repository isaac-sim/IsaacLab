# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests to verify frame transformer sensor functionality using Newton physics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import pytest
import scipy.spatial.transform as tf
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort:skip


def quat_from_euler_rpy(roll, pitch, yaw, degrees=False):
    """Converts Euler XYZ to Quaternion (x, y, z, w)."""
    quat = tf.Rotation.from_euler("xyz", (roll, pitch, yaw), degrees=degrees).as_quat()
    return tuple(quat.tolist())  # scipy already returns xyzw


def euler_rpy_apply(rpy, xyz, degrees=False):
    """Applies rotation from Euler XYZ on position vector."""
    rot = tf.Rotation.from_euler("xyz", rpy, degrees=degrees)
    return tuple(rot.apply(xyz).tolist())


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # terrain - flat terrain plane
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    # articulation - robot
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # block
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=PhysxRigidBodyCfg(max_depenetration_velocity=1.0),
            mass_props=sim_utils.MassCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 5)),
    )


@pytest.fixture
def sim():
    """Create a simulation context with Newton physics."""
    sim_cfg = SimulationCfg(
        dt=1 / 120,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=70,
                nconmax=70,
                ls_iterations=40,
                cone="elliptic",
                impratio=100,
                integrator="implicitfast",
            ),
            num_substeps=2,
            debug_mode=True,
        ),
    )
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        # Set main camera
        sim.set_camera_view(eye=(5.0, 5.0, 5.0), target=(0.0, 0.0, 0.0))
        yield sim
    # Cleanup is handled by build_simulation_context


def _feet_frames(prefixes: list[str]) -> list[FrameTransformerCfg.FrameCfg]:
    """Foot frames offset from the ANYmal shanks, named ``<prefix>_FOOT_USER``."""
    offsets = {
        "LF": (-1, (0.08795, 0.01305, -0.33797)),
        "RF": (1, (0.08795, -0.01305, -0.33797)),
        "LH": (-1, (-0.08795, 0.01305, -0.33797)),
        "RH": (1, (-0.08795, -0.01305, -0.33797)),
    }
    frames = []
    for prefix in prefixes:
        sign, xyz = offsets[prefix]
        frames.append(
            FrameTransformerCfg.FrameCfg(
                name=f"{prefix}_FOOT_USER",
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{prefix}_SHANK",
                offset=OffsetCfg(
                    pos=euler_rpy_apply(rpy=(0, 0, sign * math.pi / 2), xyz=xyz),
                    rot=quat_from_euler_rpy(0, 0, sign * math.pi / 2),
                ),
            )
        )
    return frames


def _assert_relative_poses(source_pos_w, source_quat_w, target_pos_w, target_quat_w, pos_source, quat_source):
    """Check the source-relative target poses against the world poses, target by target."""
    for index in range(target_pos_w.shape[1]):
        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            source_pos_w, source_quat_w, target_pos_w[:, index], target_quat_w[:, index]
        )
        torch.testing.assert_close(pos_source[:, index], target_pos_b)
        torch.testing.assert_close(quat_source[:, index], target_quat_b)


def test_frame_transformer_sources_and_targets(sim):
    """Frame transformers with different sources and targets track ground truth across scene resets.

    One scene hosts five sensors, each checked against asset ground truth every step:

    * ``ft_base``: offset foot frames on the shanks w.r.t. the robot base (root source).
    * ``ft_thigh``: foot frames w.r.t. a non-root source; target names follow ``find_bodies`` order.
    * ``ft_cube``: a separate rigid object as target of a robot body.
    * ``ft_offsets``: +-0.1 m offset frames on the cube w.r.t. the cube itself.
    * ``ft_all``: every robot body through a ``[^/]*`` wildcard, named after the bodies.
    """
    scene_cfg = MySceneCfg(num_envs=2, env_spacing=5.0, lazy_sensor_update=False)
    scene_cfg.ft_base = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base", target_frames=_feet_frames(["LF", "RF", "LH", "RH"])
    )
    scene_cfg.ft_thigh = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/LF_THIGH", target_frames=_feet_frames(["LF", "RF"])
    )
    scene_cfg.ft_cube = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[FrameTransformerCfg.FrameCfg(name="CUBE_USER", prim_path="{ENV_REGEX_NS}/cube")],
    )
    scene_cfg.ft_offsets = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        target_frames=[
            FrameTransformerCfg.FrameCfg(name="CUBE_CENTER", prim_path="{ENV_REGEX_NS}/cube"),
            FrameTransformerCfg.FrameCfg(
                name="CUBE_TOP",
                prim_path="{ENV_REGEX_NS}/cube",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1), rot=(0.0, 0.0, 0.0, 1.0)),
            ),
            FrameTransformerCfg.FrameCfg(
                name="CUBE_BOTTOM",
                prim_path="{ENV_REGEX_NS}/cube",
                offset=OffsetCfg(pos=(0.0, 0.0, -0.1), rot=(0.0, 0.0, 0.0, 1.0)),
            ),
        ],
    )
    scene_cfg.ft_all = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/[^/]*")],
    )
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    robot = scene.articulations["robot"]
    cube = scene["cube"]

    # -- ft_base: reorder the feet indices to match the target frames with the _USER suffix removed
    base_feet_indices, base_feet_names = robot.find_bodies(["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"])
    base_frame_names = [name.split("_USER")[0] for name in scene.sensors["ft_base"].data.target_frame_names]
    base_feet_indices = [base_feet_indices[base_feet_names.index(name)] for name in base_frame_names]
    # -- ft_thigh: names are parsed in the same order as the bodies
    thigh_index = robot.find_bodies("LF_THIGH")[0][0]
    thigh_feet_indices, thigh_feet_names = robot.find_bodies(["LF_FOOT", "RF_FOOT"])
    assert scene.sensors["ft_thigh"].data.target_frame_names == [f"{name}_USER" for name in thigh_feet_names]
    # -- ft_all: wildcard frames are named after the bodies
    all_frame_names = scene.sensors["ft_all"].data.target_frame_names
    articulation_body_names = robot.data.body_names
    all_reordering_indices = [all_frame_names.index(name) for name in articulation_body_names]

    # default joint targets
    default_actions = robot.data.default_joint_pos.torch.clone()
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Simulate physics
    for count in range(50):
        # reset every 25 steps so the sensors are checked across a scene reset
        if count % 25 == 0:
            # -- robot
            root_state = torch.cat(
                (robot.data.default_root_pose.torch, robot.data.default_root_vel.torch), dim=-1
            ).clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim_index(root_pose=root_state[:, :7])
            robot.write_root_velocity_to_sim_index(root_velocity=root_state[:, 7:])
            robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch)
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch)
            # -- cube
            cube_state = torch.cat(
                (cube.data.default_root_pose.torch, cube.data.default_root_vel.torch), dim=-1
            ).clone()
            cube_state[:, :3] += scene.env_origins
            cube.write_root_pose_to_sim_index(root_pose=cube_state[:, :7])
            cube.write_root_velocity_to_sim_index(root_velocity=cube_state[:, 7:])
            # reset buffers
            scene.reset()

        # set joint targets
        robot_actions = default_actions + 0.5 * torch.randn_like(default_actions)
        robot.set_joint_position_target_index(target=robot_actions)
        # write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # read data from sim
        scene.update(sim_dt)

        # -- ground-truth
        root_pose_w = robot.data.root_pose_w.torch
        body_pos_w = robot.data.body_pos_w.torch
        body_quat_w = robot.data.body_quat_w.torch
        cube_pos_w_gt = cube.data.root_pos_w.torch
        cube_quat_w_gt = cube.data.root_quat_w.torch

        # -- ft_base: feet w.r.t. the robot base
        data = scene.sensors["ft_base"].data
        torch.testing.assert_close(root_pose_w[:, :3], data.source_pos_w.torch)
        torch.testing.assert_close(root_pose_w[:, 3:], data.source_quat_w.torch)
        torch.testing.assert_close(body_pos_w[:, base_feet_indices], data.target_pos_w.torch)
        torch.testing.assert_close(body_quat_w[:, base_feet_indices], data.target_quat_w.torch)
        _assert_relative_poses(
            root_pose_w[:, :3],
            root_pose_w[:, 3:],
            data.target_pos_w.torch,
            data.target_quat_w.torch,
            data.target_pos_source.torch,
            data.target_quat_source.torch,
        )

        # -- ft_thigh: feet w.r.t. a thigh
        data = scene.sensors["ft_thigh"].data
        source_pose_w_gt = robot.data.body_state_w.torch[:, thigh_index, :7]
        torch.testing.assert_close(source_pose_w_gt[:, :3], data.source_pos_w.torch)
        torch.testing.assert_close(source_pose_w_gt[:, 3:], data.source_quat_w.torch)
        torch.testing.assert_close(body_pos_w[:, thigh_feet_indices], data.target_pos_w.torch)
        torch.testing.assert_close(body_quat_w[:, thigh_feet_indices], data.target_quat_w.torch)
        _assert_relative_poses(
            source_pose_w_gt[:, :3],
            source_pose_w_gt[:, 3:],
            data.target_pos_w.torch,
            data.target_quat_w.torch,
            data.target_pos_source.torch,
            data.target_quat_source.torch,
        )

        # -- ft_cube: the cube w.r.t. the robot base
        data = scene.sensors["ft_cube"].data
        torch.testing.assert_close(root_pose_w[:, :3], data.source_pos_w.torch)
        torch.testing.assert_close(root_pose_w[:, 3:], data.source_quat_w.torch)
        torch.testing.assert_close(cube_pos_w_gt, data.target_pos_w.torch.squeeze())
        torch.testing.assert_close(cube_quat_w_gt, data.target_quat_w.torch.squeeze())
        _assert_relative_poses(
            root_pose_w[:, :3],
            root_pose_w[:, 3:],
            data.target_pos_w.torch,
            data.target_quat_w.torch,
            data.target_pos_source.torch,
            data.target_quat_source.torch,
        )

        # -- ft_offsets: offset frames w.r.t. the cube
        data = scene.sensors["ft_offsets"].data
        target_pos_w_tf = data.target_pos_w.torch
        target_quat_w_tf = data.target_quat_w.torch
        cube_center_idx = data.target_frame_names.index("CUBE_CENTER")
        cube_bottom_idx = data.target_frame_names.index("CUBE_BOTTOM")
        cube_top_idx = data.target_frame_names.index("CUBE_TOP")
        torch.testing.assert_close(cube_pos_w_gt, data.source_pos_w.torch)
        torch.testing.assert_close(cube_quat_w_gt, data.source_quat_w.torch)
        torch.testing.assert_close(cube_pos_w_gt, target_pos_w_tf[:, cube_center_idx])
        torch.testing.assert_close(cube_quat_w_gt, target_quat_w_tf[:, cube_center_idx])
        offset = torch.tensor([0.0, 0.0, 0.1], device=cube_pos_w_gt.device)
        torch.testing.assert_close(target_pos_w_tf[:, cube_top_idx], cube_pos_w_gt + offset)
        torch.testing.assert_close(target_quat_w_tf[:, cube_top_idx], cube_quat_w_gt)
        torch.testing.assert_close(target_pos_w_tf[:, cube_bottom_idx], cube_pos_w_gt - offset)
        torch.testing.assert_close(target_quat_w_tf[:, cube_bottom_idx], cube_quat_w_gt)

        # -- ft_all: every body w.r.t. the robot base
        data = scene.sensors["ft_all"].data
        torch.testing.assert_close(root_pose_w[:, :3], data.source_pos_w.torch)
        torch.testing.assert_close(root_pose_w[:, 3:], data.source_quat_w.torch)
        torch.testing.assert_close(body_pos_w, data.target_pos_w.torch[:, all_reordering_indices])
        torch.testing.assert_close(body_quat_w, data.target_quat_w.torch[:, all_reordering_indices])
        _assert_relative_poses(
            root_pose_w[:, :3],
            root_pose_w[:, 3:],
            data.target_pos_w.torch,
            data.target_quat_w.torch,
            data.target_pos_source.torch,
            data.target_quat_source.torch,
        )


# Each source robot and each path prefix is covered once; the axes select independent branches.
@pytest.mark.parametrize(("source_robot", "path_prefix"), [("Robot", "{ENV_REGEX_NS}"), ("Robot_1", "/World")])
def test_frame_transformer_duplicate_body_names(sim, source_robot, path_prefix):
    """Test tracking bodies with same leaf name at different hierarchy levels.

    This test verifies that bodies with the same leaf name but different paths
    (e.g., Robot/LF_SHANK vs Robot_1/LF_SHANK, or arm/link vs leg/link) are tracked
    separately using their full relative paths internally.

    The test uses 4 target frames to cover both scenarios:

    Explicit frame names (recommended when bodies share the same leaf name):
        User provides unique names like "Robot_LF_SHANK" and "Robot_1_LF_SHANK" to
        distinguish between bodies at different hierarchy levels. This makes it
        easy to identify which transform belongs to which body.

    Implicit frame names (backward compatibility):
        When no name is provided, it defaults to the leaf body name (e.g., "RF_SHANK").
        This preserves backward compatibility for users who may have existing code like
        `idx = target_frame_names.index("RF_SHANK")`. However, when multiple bodies share
        the same leaf name, this results in duplicate frame names. The transforms are
        still distinct because internal body tracking uses full relative paths.

    Args:
        source_robot: The robot to use as the source frame ("Robot" or "Robot_1").
                      This tests that both source frames work correctly when there are
                      duplicate body names.
        path_prefix: The path prefix to use ("{ENV_REGEX_NS}" for env patterns or "/World" for direct paths).
    """

    # Create a custom scene config with two robots
    @configclass
    class MultiRobotSceneCfg(InteractiveSceneCfg):
        """Scene with two robots having bodies with same names."""

        terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

        # Frame transformer will be set after config creation (needs source_robot parameter)
        frame_transformer: FrameTransformerCfg = None  # type: ignore

    # Use multiple envs for env patterns, single env for direct paths
    num_envs = 2 if path_prefix == "{ENV_REGEX_NS}" else 1
    env_spacing = 10.0 if path_prefix == "{ENV_REGEX_NS}" else 0.0

    # Create scene config with appropriate prim paths
    scene_cfg = MultiRobotSceneCfg(num_envs=num_envs, env_spacing=env_spacing, lazy_sensor_update=False)
    scene_cfg.robot = ANYMAL_C_CFG.replace(prim_path=f"{path_prefix}/Robot")
    scene_cfg.robot_1 = ANYMAL_C_CFG.replace(
        prim_path=f"{path_prefix}/Robot_1",
        init_state=ANYMAL_C_CFG.init_state.replace(pos=(2.0, 0.0, 0.6)),
    )

    # Frame transformer tracking same-named bodies from both robots
    # Source frame is parametrized to test both Robot/base and Robot_1/base
    scene_cfg.frame_transformer = FrameTransformerCfg(
        prim_path=f"{path_prefix}/{source_robot}/base",
        target_frames=[
            # Explicit frame names (recommended when bodies share the same leaf name)
            FrameTransformerCfg.FrameCfg(
                name="Robot_LF_SHANK",
                prim_path=f"{path_prefix}/Robot/LF_SHANK",
            ),
            FrameTransformerCfg.FrameCfg(
                name="Robot_1_LF_SHANK",
                prim_path=f"{path_prefix}/Robot_1/LF_SHANK",
            ),
            # Implicit frame names (backward compatibility)
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{path_prefix}/Robot/RF_SHANK",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{path_prefix}/Robot_1/RF_SHANK",
            ),
        ],
    )
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    # Get target frame names
    target_frame_names = scene.sensors["frame_transformer"].data.target_frame_names

    # Verify explicit frame names are present
    assert "Robot_LF_SHANK" in target_frame_names, f"Expected 'Robot_LF_SHANK', got {target_frame_names}"
    assert "Robot_1_LF_SHANK" in target_frame_names, f"Expected 'Robot_1_LF_SHANK', got {target_frame_names}"

    # Without explicit names, both RF_SHANK frames default to same name "RF_SHANK"
    # This results in duplicate frame names (expected behavior for backwards compatibility)
    rf_shank_count = target_frame_names.count("RF_SHANK")
    assert rf_shank_count == 2, f"Expected 2 'RF_SHANK' entries (name collision), got {rf_shank_count}"

    # Get indices for explicit named frames
    robot_lf_idx = target_frame_names.index("Robot_LF_SHANK")
    robot_1_lf_idx = target_frame_names.index("Robot_1_LF_SHANK")

    # Get indices for implicit named frames (both named "RF_SHANK")
    rf_shank_indices = [i for i, name in enumerate(target_frame_names) if name == "RF_SHANK"]
    assert len(rf_shank_indices) == 2, f"Expected 2 RF_SHANK indices, got {rf_shank_indices}"

    # Acquire ground truth body indices
    robot_base_body_idx = scene.articulations["robot"].find_bodies("base")[0][0]
    robot_1_base_body_idx = scene.articulations["robot_1"].find_bodies("base")[0][0]
    robot_lf_shank_body_idx = scene.articulations["robot"].find_bodies("LF_SHANK")[0][0]
    robot_1_lf_shank_body_idx = scene.articulations["robot_1"].find_bodies("LF_SHANK")[0][0]
    robot_rf_shank_body_idx = scene.articulations["robot"].find_bodies("RF_SHANK")[0][0]
    robot_1_rf_shank_body_idx = scene.articulations["robot_1"].find_bodies("RF_SHANK")[0][0]

    # Determine expected source frame based on parameter
    expected_source_robot = "robot" if source_robot == "Robot" else "robot_1"
    expected_source_base_body_idx = robot_base_body_idx if source_robot == "Robot" else robot_1_base_body_idx

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # Simulate physics
    for count in range(20):
        # Reset periodically
        if count % 10 == 0:
            # Reset robot
            root_state = torch.cat(
                (
                    scene.articulations["robot"].data.default_root_pose.torch,
                    scene.articulations["robot"].data.default_root_vel.torch,
                ),
                dim=-1,
            ).clone()
            root_state[:, :3] += scene.env_origins
            scene.articulations["robot"].write_root_pose_to_sim_index(root_pose=root_state[:, :7])
            scene.articulations["robot"].write_root_velocity_to_sim_index(root_velocity=root_state[:, 7:])
            scene.articulations["robot"].write_joint_position_to_sim_index(
                position=scene.articulations["robot"].data.default_joint_pos.torch
            )
            scene.articulations["robot"].write_joint_velocity_to_sim_index(
                velocity=scene.articulations["robot"].data.default_joint_vel.torch
            )
            # Reset robot_1
            root_state_1 = torch.cat(
                (
                    scene.articulations["robot_1"].data.default_root_pose.torch,
                    scene.articulations["robot_1"].data.default_root_vel.torch,
                ),
                dim=-1,
            ).clone()
            root_state_1[:, :3] += scene.env_origins
            scene.articulations["robot_1"].write_root_pose_to_sim_index(root_pose=root_state_1[:, :7])
            scene.articulations["robot_1"].write_root_velocity_to_sim_index(root_velocity=root_state_1[:, 7:])
            scene.articulations["robot_1"].write_joint_position_to_sim_index(
                position=scene.articulations["robot_1"].data.default_joint_pos.torch
            )
            scene.articulations["robot_1"].write_joint_velocity_to_sim_index(
                velocity=scene.articulations["robot_1"].data.default_joint_vel.torch
            )
            scene.reset()

        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Read data from sim
        scene.update(sim_dt)

        # Get frame transformer data
        frame_transformer_data = scene.sensors["frame_transformer"].data
        source_pos_w = frame_transformer_data.source_pos_w.torch
        source_quat_w = frame_transformer_data.source_quat_w.torch
        target_pos_w = frame_transformer_data.target_pos_w.torch

        # Get ground truth positions and orientations (after scene.update() so they're current)
        robot_lf_pos_w = scene.articulations["robot"].data.body_pos_w.torch[:, robot_lf_shank_body_idx]
        robot_1_lf_pos_w = scene.articulations["robot_1"].data.body_pos_w.torch[:, robot_1_lf_shank_body_idx]
        robot_rf_pos_w = scene.articulations["robot"].data.body_pos_w.torch[:, robot_rf_shank_body_idx]
        robot_1_rf_pos_w = scene.articulations["robot_1"].data.body_pos_w.torch[:, robot_1_rf_shank_body_idx]

        # Get expected source frame positions and orientations (after scene.update() so they're current)
        expected_source_base_pos_w = scene.articulations[expected_source_robot].data.body_pos_w.torch[
            :, expected_source_base_body_idx
        ]
        expected_source_base_quat_w = scene.articulations[expected_source_robot].data.body_quat_w.torch[
            :, expected_source_base_body_idx
        ]

        # TEST 1: Verify source frame is correctly resolved
        # The source_pos_w should match the expected source robot's base world position
        torch.testing.assert_close(source_pos_w, expected_source_base_pos_w, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(source_quat_w, expected_source_base_quat_w, rtol=1e-5, atol=1e-5)

        # TEST 2: Explicit named frames (LF_SHANK) should have DIFFERENT world positions
        lf_pos_difference = torch.linalg.norm(target_pos_w[:, robot_lf_idx] - target_pos_w[:, robot_1_lf_idx], dim=-1)
        assert torch.all(lf_pos_difference > 1.0), (
            f"Robot_LF_SHANK and Robot_1_LF_SHANK should have different positions (got diff={lf_pos_difference}). "
            "This indicates body name collision bug."
        )

        # Verify explicit named frames match correct robot bodies
        torch.testing.assert_close(target_pos_w[:, robot_lf_idx], robot_lf_pos_w)
        torch.testing.assert_close(target_pos_w[:, robot_1_lf_idx], robot_1_lf_pos_w)

        # TEST 3: Implicit named frames (RF_SHANK) should also have DIFFERENT world positions
        # Even though they have the same frame name, internal body tracking uses full paths
        rf_pos_difference = torch.linalg.norm(
            target_pos_w[:, rf_shank_indices[0]] - target_pos_w[:, rf_shank_indices[1]], dim=-1
        )
        assert torch.all(rf_pos_difference > 1.0), (
            f"The two RF_SHANK frames should have different positions (got diff={rf_pos_difference}). "
            "This indicates body name collision bug in internal body tracking."
        )

        # Verify implicit named frames match correct robot bodies
        # Note: Order depends on internal processing, so we check both match one of the robots
        rf_positions = [target_pos_w[:, rf_shank_indices[0]], target_pos_w[:, rf_shank_indices[1]]]

        # Each tracked position should match one of the ground truth positions
        for rf_pos in rf_positions:
            matches_robot = torch.allclose(rf_pos, robot_rf_pos_w, atol=1e-5)
            matches_robot_1 = torch.allclose(rf_pos, robot_1_rf_pos_w, atol=1e-5)
            assert matches_robot or matches_robot_1, (
                f"RF_SHANK position {rf_pos} doesn't match either robot's RF_SHANK position"
            )
