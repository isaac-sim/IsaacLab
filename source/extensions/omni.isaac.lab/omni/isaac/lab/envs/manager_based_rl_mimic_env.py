# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.envs import ManagerBasedRLEnv


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

    def get_robot_eef_pose(self, env_ind=0):
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Returns:
            pose (torch.Tensor): 4x4 eef pose matrix
        """
        raise NotImplementedError

    def target_eef_pose_to_action(self, target_eef_pose, relative=True, env_ind=0):
        """
        Takes a target pose for the end effector controller and returns an action
        to try and achieve that target pose.

        Args:
            target_eef_pose (torch.Tensor): 4x4 target eef pose
            relative (bool): if True, use relative pose actions, else absolute pose actions

        Returns:
            action (torch.Tensor): action compatible with env.step (minus gripper actuation)
        """
        raise NotImplementedError

    def action_to_target_eef_pos(self, action, relative=True, env_ind=0):
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action (torch.Tensor): environment action
            relative (bool): if True, use relative pose actions, else absolute pose actions

        Returns:
            target_eef_pose (torch.Tensor): 4x4 target eef pose that @action corresponds to
        """
        raise NotImplementedError

    def action_to_gripper_action(self, action):
        """
        Extracts the gripper actuation part of an action (compatible with env.step).

        Args:
            action (torch.Tensor): environment action

        Returns:
            gripper_action (torch.Tensor): subset of environment action for gripper actuation
        """
        raise NotImplementedError

    def get_object_poses(self, env_ind=0):
        """
        Gets the pose of each object relevant to Isaac Lab Mimic data generation in the current scene.

        Returns:
            object_poses (dict): dictionary that maps object name (str) to object pose matrix (4x4 torch.Tensor)
        """
        raise NotImplementedError

    def get_subtask_term_signals(self, env_ind=0):
        """
        Gets a dictionary of binary flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise. Isaac Lab Mimic only uses this
        when parsing source demonstrations at the start of data generation, and it only
        uses the first 0 -> 1 transition in this signal to detect the end of a subtask.
        Returns:
            subtask_term_signals (dict): dictionary that maps subtask name to termination flag (0 or 1)
        """
        raise NotImplementedError

    def is_success(self):
        """
        Determines whether the task has succeeded based on internally defined success criteria.

        This method implements the logic to evaluate the task's success by checking relevant
        conditions and constraints derived from the observations in the scene.

        Returns:
            success (bool): True if the task is considered successful based on the defined criteria. False otherwise.
        """
        raise NotImplementedError

    def serialize(self):
        """
        Save all information needed to re-instantiate this environment in a dictionary.
        This is the same as @env_meta - environment metadata stored in hdf5 datasets,
        and used in utils/env_utils.py.
        """
        return dict(env_name=self.spec.id, type=2, env_kwargs=dict())
