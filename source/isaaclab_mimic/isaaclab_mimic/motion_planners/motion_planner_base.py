# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv


class MotionPlannerBase(ABC):
    """Abstract base class for motion planners.

    This class defines the public interface that all motion planners must implement.
    It focuses on the essential functionality that users interact with, while leaving
    implementation details to specific planner backends.

    The core workflow is:
    1. Initialize planner with environment and robot
    2. Call update_world_and_plan_motion() to plan to a target
    3. Execute plan using has_next_waypoint() and get_next_waypoint_ee_pose()

    Example:
        >>> from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        >>> from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
        >>> config = CuroboPlannerCfg.franka_config()
        >>> planner = CuroboPlanner(env, robot, config)
        >>> success = planner.update_world_and_plan_motion(target_pose)
        >>> if success:
        >>>     while planner.has_next_waypoint():
        >>>         action = planner.get_next_waypoint_ee_pose()
        >>>         obs, info = env.step(action)
    """

    def __init__(
        self, env: ManagerBasedEnv, robot: Articulation, env_id: int = 0, debug: bool = False, **kwargs
    ) -> None:
        """Initialize the motion planner.

        Args:
            env: The environment instance
            robot: Robot articulation to plan motions for
            env_id: Environment ID (0 to num_envs-1)
            debug: Whether to print detailed debugging information
            **kwargs: Additional planner-specific arguments
        """
        self.env = env
        self.robot = robot
        self.env_id = env_id
        self.debug = debug

    @abstractmethod
    def update_world_and_plan_motion(self, target_pose: torch.Tensor, **kwargs: Any) -> bool:
        """Update collision world and plan motion to target pose.

        This is the main entry point for motion planning. It should:
        1. Update the planner's internal world representation
        2. Plan a collision-free path to the target pose
        3. Store the plan internally for execution

        Args:
            target_pose: Target pose to plan motion to (4x4 transformation matrix)
            **kwargs: Planner-specific arguments (e.g., retiming, contact planning)

        Returns:
            bool: True if planning succeeded, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def has_next_waypoint(self) -> bool:
        """Check if there are more waypoints in current plan.

        Returns:
            bool: True if there are more waypoints, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def get_next_waypoint_ee_pose(self) -> Any:
        """Get next waypoint's end-effector pose from current plan.

        This method should only be called after checking has_next_waypoint().

        Returns:
            Any: End-effector pose for the next waypoint in the plan.
        """
        raise NotImplementedError

    def get_planned_poses(self) -> list[Any]:
        """Get all planned poses from current plan.

        Returns:
            list[Any]: List of planned poses.

        Note:
            Default implementation iterates through waypoints.
            Child classes can override for a more efficient implementation.
        """
        planned_poses = []
        # Create a copy of the planner state to not affect the original plan execution
        # This is a placeholder and may need to be implemented by child classes
        # if they manage complex internal state.
        # For now, we assume the planner can be reset and we can iterate through the plan.
        # A more robust solution might involve a dedicated method to get the full plan.
        self.reset_plan()
        while self.has_next_waypoint():
            pose = self.get_next_waypoint_ee_pose()
            planned_poses.append(pose)
        return planned_poses

    @abstractmethod
    def reset_plan(self) -> None:
        """Reset the current plan and execution state.

        This should clear any stored plan and reset the execution index or iterator.
        """
        raise NotImplementedError

    def get_planner_info(self) -> dict[str, Any]:
        """Get information about the planner.

        Returns:
            dict: Information about the planner (name, version, capabilities, etc.)
        """
        return {
            "name": self.__class__.__name__,
            "env_id": self.env_id,
            "debug": self.debug,
        }
