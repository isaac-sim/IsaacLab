# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ik_7d task-space action term.

Drives one or both of a G2's 7-DoF arms from end-effector poses plus an explicit
elbow-swivel command, closing the loop on the articulation's measured joint state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from isaaclab_contrib.controllers import Ik7dController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import ik_7d_actions_cfg


class Ik7dAction(ActionTerm):
    r"""ik_7d inverse-kinematics action term.

    The action tensor is ``8`` elements per arm, concatenated in the order of
    :attr:`Ik7dControllerCfg.arms`:

    .. code-block:: text

        [ x, y, z, qx, qy, qz, qw, swivel ]  per arm

    The quaternion is **xyzw**, matching both Isaac Lab 3.0 and AgiBot's
    retargeters, so a teleop pose reaches this term unreordered. Do not "fix" it
    to wxyz.

    ``swivel`` commands the elbow's rotation about the shoulder-wrist axis -- the
    redundancy a 6-DoF pose does not determine. See
    :attr:`Ik7dControllerCfg.apa_mode`.
    """

    cfg: ik_7d_actions_cfg.Ik7dActionCfg
    """Configuration for the ik_7d action term."""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: ik_7d_actions_cfg.Ik7dActionCfg, env: ManagerBasedEnv):
        """Initialize the action term and build one solver per environment.

        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        super().__init__(cfg, env)

        self._env = env

        # One solver per environment: ik_7d holds mutable model state internally,
        # so a shared instance would solve each env against the previous one's torso.
        lab_joint_names = list(self._asset.data.joint_names)
        self._controllers = [Ik7dController(cfg.controller.copy(), lab_joint_names) for _ in range(self.num_envs)]

        reference = self._controllers[0]
        self._arms = list(cfg.controller.arms)
        self._controlled_joint_ids = reference.controlled_lab_indices
        self._controlled_joint_names = [lab_joint_names[i] for i in self._controlled_joint_ids]

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, reference.num_controlled_joints, device=self.device)

        # Targets are resolved into the base-link frame in `process_actions` and
        # consumed by numpy in `apply_actions`; hold both representations.
        self._target_poses = np.zeros((self.num_envs, len(self._arms), 4, 4), dtype=np.float64)
        self._swivel_commands = np.zeros((self.num_envs, len(self._arms)), dtype=np.float64)

        self._base_link_idx = self._env.scene[cfg.controller.articulation_name].data.body_names.index(
            cfg.controller.base_link_name
        )

        if cfg.debug_vis_stats:
            print(f"[ik_7d] {reference.jmap.describe()}")
            for arm in self._arms:
                print(
                    f"[ik_7d] {arm} arm: ee frame {reference.ee_frame[arm]!r} in"
                    f" {reference.base_frame[arm]!r}, home swivel"
                    f" {reference.home_apa[arm]:+.4f} rad, free direction"
                    f" {reference.free_sign[arm]:+d}"
                )

    # ==================== Properties ====================

    @property
    def pose_dim(self) -> int:
        """Position (3) plus orientation (4)."""
        return 7

    @property
    def arm_action_dim(self) -> int:
        """Per-arm action width: a pose plus one swivel command."""
        return self.pose_dim + 1

    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        return len(self.cfg.controller.arms) * self.arm_action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """The action tensor exactly as received."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The joint position targets the IK produced, in controlled-joint order."""
        return self._processed_actions

    @property
    def controlled_joint_names(self) -> list[str]:
        """Names of the joints this term writes, in action order."""
        return self._controlled_joint_names

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term.

        Adds the controlled joint names, the arm ordering and the controller
        configuration to the base descriptor.

        Returns:
            The IO descriptor of the action term.
        """
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "Ik7dAction"
        self._IO_descriptor.controlled_joint_names = self._controlled_joint_names
        self._IO_descriptor.extras["arms"] = self._arms
        self._IO_descriptor.extras["controller_cfg"] = self.cfg.controller.__dict__
        return self._IO_descriptor

    # ==================== Operations ====================

    def process_actions(self, actions: torch.Tensor) -> None:
        """Split the action tensor into per-arm poses and swivel commands.

        Args:
            actions: The input actions tensor, shape ``(num_envs, action_dim)``.
        """
        self._raw_actions[:] = actions

        poses = torch.zeros(self.num_envs, len(self._arms), 4, 4, device=self.device)
        for arm_index in range(len(self._arms)):
            start = arm_index * self.arm_action_dim
            position = actions[:, start : start + 3]
            quaternion = actions[:, start + 3 : start + 7]
            self._swivel_commands[:, arm_index] = actions[:, start + 7].detach().cpu().numpy()
            poses[:, arm_index] = math_utils.make_pose(position, math_utils.matrix_from_quat(quaternion))

        if self.cfg.pose_frame == "world":
            poses = self._to_base_link_frame(poses)

        self._target_poses[:] = poses.detach().cpu().numpy().astype(np.float64)

    def _to_base_link_frame(self, poses: torch.Tensor) -> torch.Tensor:
        """Re-express world-frame poses in the robot's base-link frame.

        Env origins are subtracted first, so this is correct for a tiled scene.

        Args:
            poses: Poses in the world frame, shape ``(num_envs, num_arms, 4, 4)``.

        Returns:
            The same poses in the base-link frame.
        """
        articulation_data = self._env.scene[self.cfg.controller.articulation_name].data
        base_pose_w = articulation_data.body_link_pose_w.torch[:, self._base_link_idx]
        base_pose = math_utils.make_pose(
            base_pose_w[:, :3],
            math_utils.matrix_from_quat(base_pose_w[:, 3:7]),
        )
        # `pose_in_A_to_pose_in_B` broadcasts over the leading env dimension, so
        # transform one arm at a time rather than reshaping.
        base_inv = math_utils.pose_inv(base_pose)
        return torch.stack(
            [math_utils.pose_in_A_to_pose_in_B(poses[:, i], base_inv) for i in range(poses.shape[1])],
            dim=1,
        )

    def apply_actions(self) -> None:
        """Solve the IK for every environment and write the joint position targets."""
        joint_pos_lab = self._asset.data.joint_pos.torch.detach().cpu().numpy()

        solutions = np.empty((self.num_envs, self._controllers[0].num_controlled_joints), dtype=np.float64)
        for env_index, controller in enumerate(self._controllers):
            solutions[env_index] = controller.compute(
                joint_pos_lab[env_index], self._target_poses[env_index], self._swivel_commands[env_index]
            )

        self._processed_actions = torch.as_tensor(solutions, dtype=torch.float32, device=self.device)
        self._asset.set_joint_position_target_index(
            target=self._processed_actions, joint_ids=self._controlled_joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term and re-seed the solvers from the measured state.

        Args:
            env_ids: Environment IDs to reset. If ``None``, all are reset.
        """
        indices = range(self.num_envs) if env_ids is None or isinstance(env_ids, slice) else env_ids
        joint_pos_lab = self._asset.data.joint_pos.torch.detach().cpu().numpy()
        for env_index in indices:
            self._raw_actions[env_index] = 0.0
            self._controllers[env_index].reset(joint_pos_lab[env_index])
