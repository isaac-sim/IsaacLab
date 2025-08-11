# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLEnv


class ManagerBasedRLMimicEnv(ManagerBasedRLEnv):
    """The superclass for the Isaac Lab Mimic environments.

    This class inherits from :class:`ManagerBasedRLEnv` and provides a template for the functions that
    need to be defined to run the Isaac Lab Mimic data generation workflow. The Isaac Lab data generation
    pipeline, inspired by the MimicGen system, enables the generation of new datasets based on a few human
    collected demonstrations. MimicGen is a novel approach designed to automatically synthesize large-scale,
    rich datasets from a sparse set of human demonstrations by adapting them to new contexts. It manages to
    replicate the benefits of large datasets while reducing the immense time and effort usually required to
    gather extensive human demonstrations.

    The MimicGen system works by parsing demonstrations into object-centric segments. It then adapts
    these segments to new scenes by transforming each segment according to the new scene’s context, stitching
    them into a coherent trajectory for a robotic end-effector to execute. This approach allows learners to train
    proficient agents through imitation learning on diverse configurations of scenes, object instances, etc.

    Key Features:
        - Efficient Dataset Generation: Utilizes a small set of human demos to produce large scale demonstrations.
        - Broad Applicability: Capable of supporting tasks that require a range of manipulation skills, such as
          pick-and-place and interacting with articulated objects.
        - Dataset Versatility: The synthetic data retains a quality that compares favorably with additional human demos.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        raise NotImplementedError

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            action_noise_dict: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to compute the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        raise NotImplementedError

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action: Environment action. Shape is (num_envs, action_dim).

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to.
        """
        raise NotImplementedError

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        raise NotImplementedError

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """
        Gets the pose of each object relevant to Isaac Lab Mimic data generation in the current scene.

        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A dictionary that maps object names to object pose matrix (4x4 torch.Tensor)
        """
        if env_ids is None:
            env_ids = slice(None)

        rigid_object_states = self.scene.get_state(is_relative=True)["rigid_object"]
        object_pose_matrix = dict()
        for obj_name, obj_state in rigid_object_states.items():
            object_pose_matrix[obj_name] = PoseUtils.make_pose(
                obj_state["root_pose"][env_ids, :3], PoseUtils.matrix_from_quat(obj_state["root_pose"][env_ids, 3:7])
            )
        return object_pose_matrix

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. The implementation of this method is
        required if intending to enable automatic subtask term signal annotation when running the
        dataset annotation tool. This method can be kept unimplemented if intending to use manual
        subtask term signal annotation.

        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.

        Returns:
            A dictionary termination signal flags (False or True) for each subtask.
        """
        raise NotImplementedError

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.spec.id, type=2, env_kwargs=dict())
