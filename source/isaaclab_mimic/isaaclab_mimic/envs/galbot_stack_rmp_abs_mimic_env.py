# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils

from .franka_stack_ik_abs_mimic_env import FrankaCubeStackIKAbsMimicEnv


class RmpFlowGalbotCubeStackAbsMimicEnv(FrankaCubeStackIKAbsMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for Galbot Cube Stack RmpFlow Absolute env.
    """

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """
        Rewrite this function to get the pose of each object in robot base frame,
        relevant to Isaac Lab Mimic data generation in the current scene.

        Args:
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A dictionary that maps object names to object pose matrix in base frame of robot (4x4 torch.Tensor)
        """

        if env_ids is None:
            env_ids = slice(None)

        rigid_object_states = self.scene.get_state(is_relative=True)["rigid_object"]
        robot_states = self.scene.get_state(is_relative=True)["articulation"]["robot"]
        root_pose = robot_states["root_pose"]
        root_pos = root_pose[env_ids, :3]
        root_quat = root_pose[env_ids, 3:7]

        object_pose_matrix = dict()
        for obj_name, obj_state in rigid_object_states.items():
            pos_obj_base, quat_obj_base = PoseUtils.subtract_frame_transforms(
                root_pos, root_quat, obj_state["root_pose"][env_ids, :3], obj_state["root_pose"][env_ids, 3:7]
            )
            rot_obj_base = PoseUtils.matrix_from_quat(quat_obj_base)
            object_pose_matrix[obj_name] = PoseUtils.make_pose(pos_obj_base, rot_obj_base)

        return object_pose_matrix
