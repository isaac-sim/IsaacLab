# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import random
from typing import Any

import pytest

SEED: int = 42
random.seed(SEED)

from isaaclab.app import AppLauncher

headless = True
app_launcher = AppLauncher(headless=headless)
simulation_app: Any = app_launcher.app

from collections.abc import Generator

import gymnasium as gym
import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers

from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env_cfg import FrankaCubeStackIKRelMimicEnvCfg
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

GRIPPER_OPEN_CMD: float = 1.0
GRIPPER_CLOSE_CMD: float = -1.0


def _eef_name(env: ManagerBasedEnv) -> str:
    return list(env.cfg.subtask_configs.keys())[0]


def _action_from_pose(
    env: ManagerBasedEnv, target_pose: torch.Tensor, gripper_binary_action: float, env_id: int = 0
) -> torch.Tensor:
    eef = _eef_name(env)
    play_action = env.target_eef_pose_to_action(
        target_eef_pose_dict={eef: target_pose},
        gripper_action_dict={eef: torch.tensor([gripper_binary_action], device=env.device, dtype=torch.float32)},
        env_id=env_id,
    )
    if play_action.dim() == 1:
        play_action = play_action.unsqueeze(0)
    return play_action


def _env_step_with_action(env: ManagerBasedEnv, action: torch.Tensor) -> None:
    env.step(action)


def _execute_plan(env: ManagerBasedEnv, planner: CuroboPlanner, gripper_binary_action: float, env_id: int = 0) -> None:
    """Execute planner's EEF planned poses using env.step with IK-relative controller actions."""
    planned_poses = planner.get_planned_poses()
    if not planned_poses:
        return
    for pose in planned_poses:
        action = _action_from_pose(env, pose, gripper_binary_action, env_id=env_id)
        _env_step_with_action(env, action)


def _execute_gripper_action(
    env: ManagerBasedEnv, robot: Articulation, gripper_binary_action: float, steps: int = 12, env_id: int = 0
) -> None:
    """Hold current EEF pose and toggle gripper for a few steps."""
    eef = _eef_name(env)
    curr_pose = env.get_robot_eef_pose(eef_name=eef, env_ids=[env_id])[0]
    for _ in range(steps):
        action = _action_from_pose(env, curr_pose, gripper_binary_action, env_id=env_id)
        _env_step_with_action(env, action)


DOWN_FACING_QUAT = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)


@pytest.fixture(scope="class")
def cube_stack_test_env() -> Generator[dict[str, Any], None, None]:
    """Create the environment and motion planner once for the test suite and yield them."""
    random.seed(SEED)
    torch.manual_seed(SEED)

    env_cfg = FrankaCubeStackIKRelMimicEnvCfg()
    env_cfg.scene.num_envs = 1
    for frame in env_cfg.scene.ee_frame.target_frames:
        if frame.name == "end_effector":
            print(f"Setting end effector offset from {frame.offset.pos} to (0.0, 0.0, 0.0) for SkillGen parity")
            frame.offset.pos = (0.0, 0.0, 0.0)

    env: ManagerBasedEnv = gym.make(
        "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0",
        cfg=env_cfg,
        headless=headless,
    ).unwrapped
    env.reset()

    robot: Articulation = env.scene["robot"]
    planner_cfg = CuroboPlannerCfg.franka_stack_cube_config()
    planner_cfg.visualize_plan = False
    planner_cfg.visualize_spheres = False
    planner_cfg.debug_planner = True
    planner_cfg.retreat_distance = 0.05
    planner_cfg.approach_distance = 0.05
    planner_cfg.time_dilation_factor = 1.0

    planner = CuroboPlanner(
        env=env,
        robot=robot,
        config=planner_cfg,
        env_id=0,
    )

    goal_pose_visualizer = None
    if not headless:
        marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/goal_pose")
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        goal_pose_visualizer = VisualizationMarkers(marker_cfg)

    yield {
        "env": env,
        "robot": robot,
        "planner": planner,
        "goal_pose_visualizer": goal_pose_visualizer,
    }

    env.close()


class TestCubeStackPlanner:
    @pytest.fixture(autouse=True)
    def setup(self, cube_stack_test_env) -> None:
        self.env: ManagerBasedEnv = cube_stack_test_env["env"]
        self.robot: Articulation = cube_stack_test_env["robot"]
        self.planner: CuroboPlanner = cube_stack_test_env["planner"]
        self.goal_pose_visualizer: VisualizationMarkers | None = cube_stack_test_env["goal_pose_visualizer"]

    def _visualize_goal_pose(self, pos: torch.Tensor, quat: torch.Tensor) -> None:
        """Visualize the goal frame markers at pos, quat (xyzw)."""
        if headless or self.goal_pose_visualizer is None:
            return
        self.goal_pose_visualizer.visualize(translations=pos.unsqueeze(0), orientations=quat.unsqueeze(0))

    def _pose_from_xy_quat(self, xy: torch.Tensor, z: float, quat: torch.Tensor) -> torch.Tensor:
        """Build a 4×4 pose given xy (Tensor[2]), z, and quaternion."""
        device = xy.device
        dtype = xy.dtype
        pos = torch.cat([xy, torch.tensor([z], dtype=dtype, device=device)])
        rot = math_utils.matrix_from_quat(quat.to(device).unsqueeze(0))[0]
        return math_utils.make_pose(pos, rot)

    def _get_cube_pos(self, cube_name: str) -> torch.Tensor:
        """Return the current world position of a cube's root (x, y, z)."""
        obj: RigidObject = self.env.scene[cube_name]
        return wp.to_torch(obj.data.root_pos_w)[0, :3].clone().detach()

    def _place_pose_over_cube(self, cube_name: str, height_offset: float) -> torch.Tensor:
        """Compute a goal pose directly above the named cube using the latest pose."""
        base_pos = self._get_cube_pos(cube_name)
        return self._pose_from_xy_quat(base_pos[:2], base_pos[2].item() + height_offset, DOWN_FACING_QUAT)

    def test_pick_and_stack(self) -> None:
        """Plan and execute pick-and-place to stack cube_1 on cube_2, then cube_3 on the stack."""
        cube_1_pos = self._get_cube_pos("cube_1")
        cube_2_pos = self._get_cube_pos("cube_2")
        cube_3_pos = self._get_cube_pos("cube_3")
        print(f"Cube 1 position: {cube_1_pos}")
        print(f"Cube 2 position: {cube_2_pos}")
        print(f"Cube 3 position: {cube_3_pos}")

        # Approach above cube_1
        pre_grasp_height = 0.1
        pre_grasp_pose = self._pose_from_xy_quat(cube_1_pos[:2], pre_grasp_height, DOWN_FACING_QUAT)
        print(f"Pre-grasp pose: {pre_grasp_pose}")
        if not headless:
            pos_pg = pre_grasp_pose[:3, 3].detach().cpu()
            quat_pg = math_utils.quat_from_matrix(pre_grasp_pose[:3, :3].unsqueeze(0))[0].detach().cpu()
            self._visualize_goal_pose(pos_pg, quat_pg)

        # Plan to pre-grasp
        assert self.planner.update_world_and_plan_motion(pre_grasp_pose), "Failed to plan to pre-grasp pose"
        _execute_plan(self.env, self.planner, gripper_binary_action=GRIPPER_OPEN_CMD)

        # Close gripper to grasp cube_1 (hold pose while closing)
        _execute_gripper_action(self.env, self.robot, GRIPPER_CLOSE_CMD, steps=16)

        # Plan placement with cube_1 attached (above latest cube_2)
        place_pose = self._place_pose_over_cube("cube_2", 0.15)

        if not headless:
            pos_place = place_pose[:3, 3].detach().cpu()
            quat_place = math_utils.quat_from_matrix(place_pose[:3, :3].unsqueeze(0))[0].detach().cpu()
            self._visualize_goal_pose(pos_place, quat_place)

        # Plan with attached object
        assert self.planner.update_world_and_plan_motion(place_pose, expected_attached_object="cube_1"), (
            "Failed to plan placement trajectory with attached cube"
        )
        _execute_plan(self.env, self.planner, gripper_binary_action=GRIPPER_CLOSE_CMD)

        # Release cube 1
        _execute_gripper_action(self.env, self.robot, GRIPPER_OPEN_CMD, steps=16)

        # Go to cube 3
        cube_3_pos_now = self._get_cube_pos("cube_3")
        pre_grasp_pose = self._pose_from_xy_quat(cube_3_pos_now[:2], pre_grasp_height, DOWN_FACING_QUAT)
        print(f"Pre-grasp pose: {pre_grasp_pose}")
        if not headless:
            pos_pg = pre_grasp_pose[:3, 3].detach().cpu()
            quat_pg = math_utils.quat_from_matrix(pre_grasp_pose[:3, :3].unsqueeze(0))[0].detach().cpu()
            self._visualize_goal_pose(pos_pg, quat_pg)

        assert self.planner.update_world_and_plan_motion(pre_grasp_pose, expected_attached_object=None), (
            "Failed to plan retract motion"
        )
        _execute_plan(self.env, self.planner, gripper_binary_action=GRIPPER_OPEN_CMD)

        # Grasp cube 3
        _execute_gripper_action(self.env, self.robot, GRIPPER_CLOSE_CMD)

        # Plan placement with cube_3 attached, to cube 2 (use latest cube_2 pose)
        place_pose = self._place_pose_over_cube("cube_2", 0.18)

        if not headless:
            pos_place = place_pose[:3, 3].detach().cpu()
            quat_place = math_utils.quat_from_matrix(place_pose[:3, :3].unsqueeze(0))[0].detach().cpu()
            self._visualize_goal_pose(pos_place, quat_place)

        assert self.planner.update_world_and_plan_motion(place_pose, expected_attached_object="cube_3"), (
            "Failed to plan placement trajectory with attached cube"
        )
        _execute_plan(self.env, self.planner, gripper_binary_action=GRIPPER_CLOSE_CMD)

        # Release cube 3
        _execute_gripper_action(self.env, self.robot, GRIPPER_OPEN_CMD)

        print("Pick-and-place stacking test completed successfully!")
