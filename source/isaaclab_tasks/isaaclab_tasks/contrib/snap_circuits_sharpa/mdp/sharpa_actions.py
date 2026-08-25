# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term for AVP-controlled dual Sharpa Wave hands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.envs import ManagerBasedEnv

    from .sharpa_actions_cfg import SharpaWaveBimanualActionCfg


_SHARPA_FINGER_JOINTS = [
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
]


class SharpaWaveBimanualAction(ActionTerm):
    """Drive both Sharpa Wave floating wrists and their 22 finger joints.

    The 58-D input layout is ``left pose (xyz + xyzw)``, ``right pose``,
    ``left fingers (22)``, then ``right fingers (22)``. The official dual
    Sharpa asset represents each floating wrist with XYZ prismatic joints and
    roll/pitch/yaw joints, so wrist quaternions are converted before applying
    the 56 joint-position targets.
    """

    cfg: SharpaWaveBimanualActionCfg
    _asset: Articulation

    def __init__(self, cfg: SharpaWaveBimanualActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._controlled_joint_names = []
        for side in ("left", "right"):
            self._controlled_joint_names.extend(
                [
                    f"{side}_x_joint",
                    f"{side}_y_joint",
                    f"{side}_z_joint",
                    f"{side}_roll_joint",
                    f"{side}_pitch_joint",
                    f"{side}_yaw_joint",
                ]
            )
            self._controlled_joint_names.extend(f"{side}_{name}" for name in _SHARPA_FINGER_JOINTS)

        joint_ids, resolved_names = self._asset.find_joints(
            self._controlled_joint_names, preserve_order=True, as_proxy=True
        )
        if list(resolved_names) != self._controlled_joint_names:
            raise ValueError(
                "The Sharpa Wave asset joint order does not match the teleop contract. "
                f"Expected {self._controlled_joint_names}, got {resolved_names}."
            )
        self._joint_ids = joint_ids.torch

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._controlled_joint_names), device=self.device)
        self._position_offset = torch.tensor(cfg.position_offset, device=self.device)

    @property
    def action_dim(self) -> int:
        return 58

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

        left_pose = actions[:, 0:7]
        right_pose = actions[:, 7:14]
        left_fingers = actions[:, 14:36]
        right_fingers = actions[:, 36:58]

        left_rpy = torch.stack(euler_xyz_from_quat(left_pose[:, 3:7]), dim=-1)
        right_rpy = torch.stack(euler_xyz_from_quat(right_pose[:, 3:7]), dim=-1)
        left_position = left_pose[:, 0:3] * self.cfg.position_scale + self._position_offset
        right_position = right_pose[:, 0:3] * self.cfg.position_scale + self._position_offset

        self._processed_actions[:, 0:6] = torch.cat((left_position, left_rpy), dim=-1)
        self._processed_actions[:, 6:28] = left_fingers
        self._processed_actions[:, 28:34] = torch.cat((right_position, right_rpy), dim=-1)
        self._processed_actions[:, 34:56] = right_fingers

        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        self._processed_actions.clamp_(limits[..., 0], limits[..., 1])

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target_index(target=self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
