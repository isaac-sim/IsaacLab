# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Defines structure of information that is needed from an environment for data generation.
"""
from copy import deepcopy


class DatagenInfo:
    """
    Defines the structure of information required from an environment for data generation processes.
    The `DatagenInfo` class centralizes all essential data elements needed for data generation in one place,
    reducing the overhead and complexity of repeatedly querying the environment whenever this information is needed.

    To allow for flexibility,not all information must be present.

    Core Elements:
    - **eef_pose**: Captures the current 6 dimensional poses of the robot's end-effector.
    - **object_poses**: Captures the 6 dimensional poses of relevant objects in the scene.
    - **subtask_term_signals**: Captures subtask completions signals.
    - **target_eef_pose**: Captures the target 6 dimensional poses for robot's end effector at each time step.
    - **gripper_action**:  Captures the gripper's state.
    """

    def __init__(
        self,
        eef_pose=None,
        object_poses=None,
        subtask_term_signals=None,
        target_eef_pose=None,
        gripper_action=None,
    ):
        """
        Args:
            eef_pose (torch.Tensor or None): robot end effector poses of shape [..., 4, 4]
            object_poses (dict or None): dictionary mapping object name to object poses
                of shape [..., 4, 4]
            subtask_term_signals (dict or None): dictionary mapping subtask name to a binary
                indicator (0 or 1) on whether subtask has been completed. Each value in the
                dictionary could be an int, float, or torch.Tensor of shape [..., 1].
            target_eef_pose (torch.Tensor or None): target end effector poses of shape [..., 4, 4]
            gripper_action (torch.Tensor or None): gripper actions of shape [..., D] where D
                is the dimension of the gripper actuation action for the robot arm
        """
        self.eef_pose = None
        if eef_pose is not None:
            self.eef_pose = eef_pose

        self.object_poses = None
        if object_poses is not None:
            self.object_poses = {k: object_poses[k] for k in object_poses}

        self.subtask_term_signals = None
        if subtask_term_signals is not None:
            self.subtask_term_signals = dict()
            for k in subtask_term_signals:
                if isinstance(subtask_term_signals[k], (float, int)):
                    self.subtask_term_signals[k] = subtask_term_signals[k]
                else:
                    # only create torch tensor if value is not a single value
                    self.subtask_term_signals[k] = subtask_term_signals[k]

        self.target_eef_pose = None
        if target_eef_pose is not None:
            self.target_eef_pose = target_eef_pose

        self.gripper_action = None
        if gripper_action is not None:
            self.gripper_action = gripper_action

    def to_dict(self):
        """
        Convert this instance to a dictionary containing the same information.
        """
        ret = dict()
        if self.eef_pose is not None:
            ret["eef_pose"] = self.eef_pose
        if self.object_poses is not None:
            ret["object_poses"] = deepcopy(self.object_poses)
        if self.subtask_term_signals is not None:
            ret["subtask_term_signals"] = deepcopy(self.subtask_term_signals)
        if self.target_eef_pose is not None:
            ret["target_eef_pose"] = self.target_eef_pose
        if self.gripper_action is not None:
            ret["gripper_action"] = self.gripper_action
        return ret
