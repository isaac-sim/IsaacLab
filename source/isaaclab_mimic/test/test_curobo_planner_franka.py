# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import random
from collections.abc import Generator
from typing import Any

import pytest

SEED: int = 42
random.seed(SEED)

from isaaclab.app import AppLauncher

headless = True
app_launcher = AppLauncher(headless=headless)
simulation_app: Any = app_launcher.app

import gymnasium as gym
import torch
import warp as wp

import isaaclab.utils.assets as _al_assets
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObjectCfg
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

ISAAC_NUCLEUS_DIR: str = getattr(_al_assets, "ISAAC_NUCLEUS_DIR", "/Isaac")

from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg import FrankaCubeStackEnvCfg

# Predefined EE goals for the test
# Each entry is a tuple of: (goal specification, goal ID)
predefined_ee_goals_and_ids = [
    ({"pos": [0.70, -0.25, 0.25], "quat": [0.707, 0.0, 0.707, 0.0]}, "Behind wall, left"),
    ({"pos": [0.70, 0.25, 0.25], "quat": [0.707, 0.0, 0.707, 0.0]}, "Behind wall, right"),
    ({"pos": [0.65, 0.0, 0.45], "quat": [1.0, 0.0, 0.0, 0.0]}, "Behind wall, center, high"),
    ({"pos": [0.80, -0.15, 0.35], "quat": [0.5, 0.0, 0.866, 0.0]}, "Behind wall, far left"),
    ({"pos": [0.80, 0.15, 0.35], "quat": [0.5, 0.0, 0.866, 0.0]}, "Behind wall, far right"),
]


@pytest.fixture(scope="class")
def curobo_test_env() -> Generator[dict[str, Any], None, None]:
    """Set up the environment for the Curobo test and yield test-critical data."""
    random.seed(SEED)
    torch.manual_seed(SEED)

    env_cfg = FrankaCubeStackEnvCfg()
    env_cfg.scene.num_envs = 1

    # Add a static wall for the robot to avoid
    wall_props = RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
    wall_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_0/moving_wall",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
            scale=(0.5, 4.5, 7.0),
            rigid_props=wall_props,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.80)),
    )
    setattr(env_cfg.scene, "moving_wall", wall_cfg)

    env: ManagerBasedEnv = gym.make("Isaac-Stack-Cube-Franka-v0", cfg=env_cfg, headless=headless).unwrapped
    env.reset()

    robot = env.scene["robot"]
    planner = CuroboPlanner(env=env, robot=robot, config=CuroboPlannerCfg.franka_config())

    goal_pose_visualizer = None
    if not headless:
        goal_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/goal_poses")
        goal_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        goal_pose_visualizer = VisualizationMarkers(goal_marker_cfg)

    # Allow the simulation to settle
    for _ in range(3):
        env.sim.step(render=False)

    planner.update_world()

    # Default joint positions for the Franka arm (7-DOF)
    home_j = torch.tensor([0.0, -0.4, 0.0, -2.1, 0.0, 2.1, 0.7])

    # Yield the necessary objects for the test
    yield {
        "env": env,
        "robot": robot,
        "planner": planner,
        "goal_pose_visualizer": goal_pose_visualizer,
        "home_j": home_j,
    }

    # Teardown: close the environment and simulation app
    env.close()


class TestCuroboPlanner:
    """Test suite for the Curobo motion planner, focusing on obstacle avoidance."""

    @pytest.fixture(autouse=True)
    def setup(self, curobo_test_env) -> None:
        """Inject the test environment into the test class instance."""
        self.env: ManagerBasedEnv = curobo_test_env["env"]
        self.robot: Articulation = curobo_test_env["robot"]
        self.planner: CuroboPlanner = curobo_test_env["planner"]
        self.goal_pose_visualizer: VisualizationMarkers | None = curobo_test_env["goal_pose_visualizer"]
        self.home_j: torch.Tensor = curobo_test_env["home_j"]

    def _visualize_goal_pose(self, pos: torch.Tensor, quat: torch.Tensor) -> None:
        """Visualize the goal pose using frame markers if not in headless mode."""
        if headless or self.goal_pose_visualizer is None:
            return
        pos_vis = pos.unsqueeze(0)
        quat_vis = quat.unsqueeze(0)
        self.goal_pose_visualizer.visualize(translations=pos_vis, orientations=quat_vis)

    def _execute_current_plan(self) -> None:
        """Replay the waypoints of the current plan in the simulator for visualization."""
        if headless or self.planner.current_plan is None:
            return
        for q in self.planner.current_plan.position:
            q_tensor = q if isinstance(q, torch.Tensor) else torch.as_tensor(q, dtype=torch.float32)
            self._set_arm_positions(q_tensor)
            self.env.sim.step(render=True)

    def _set_arm_positions(self, q: torch.Tensor) -> None:
        """Set the joint positions of the robot's arm, appending default gripper values if necessary."""
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if q.shape[-1] == 7:  # Arm only
            fingers = torch.tensor([0.04, 0.04], device=q.device, dtype=q.dtype).repeat(q.shape[0], 1)
            q_full = torch.cat([q, fingers], dim=-1)
        else:
            q_full = q
        self.robot.write_joint_position_to_sim(wp.from_torch(q_full.to(self.env.device)))

    @pytest.mark.parametrize("goal_spec, goal_id", predefined_ee_goals_and_ids)
    def test_plan_to_predefined_goal(self, goal_spec, goal_id) -> None:
        """Test planning to a predefined goal, ensuring the planner can find a path around an obstacle."""
        print(f"Planning for goal: {goal_id}")

        # Reset robot to a known home position before each test
        self._set_arm_positions(self.home_j)
        self.env.sim.step()

        pos = torch.tensor(goal_spec["pos"], dtype=torch.float32)
        quat = torch.tensor(goal_spec["quat"], dtype=torch.float32)

        if not headless:
            self._visualize_goal_pose(pos, quat)

        # Ensure the goal is actually behind the wall
        assert pos[0] > 0.55, f"Goal '{goal_id}' is not behind the wall (x={pos[0]:.3f})"

        rot_matrix = math_utils.matrix_from_quat(quat.unsqueeze(0))[0]
        ee_goal = math_utils.make_pose(pos, rot_matrix)

        result = self.planner.plan_motion(ee_goal)
        print(f"Planning result for '{goal_id}': {'Success' if result else 'Failure'}")

        assert result, f"Failed to find a motion plan for the goal: '{goal_id}'"

        if result and not headless:
            self._execute_current_plan()
