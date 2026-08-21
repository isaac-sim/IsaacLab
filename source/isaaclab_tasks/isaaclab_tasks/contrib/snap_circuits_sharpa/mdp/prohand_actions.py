# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action term for one AVP-controlled ProHand."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets.articulation import Articulation
    from isaaclab.envs import ManagerBasedEnv

    from .prohand_actions_cfg import ProHandActionCfg


_FINGER_JOINT_STEMS = [
    "t0_TM_abd",
    "t1_TM",
    "t2_MCP",
    "t3_DIP",
    "i0_CMC_abd",
    "i1_MCP",
    "i2_PIP",
    "i3_DIP",
    "m0_CMC_abd",
    "m1_MCP",
    "m2_PIP",
    "m3_DIP",
    "r0_CMC_abd",
    "r1_MCP",
    "r2_PIP",
    "r3_DIP",
    "p0_CMC_abd",
    "p1_MCP",
    "p2_PIP",
    "p3_DIP",
]


class ProHandAction(ActionTerm):
    """Drive one ProHand free root and its 20 commanded finger joints.

    The 27-D input is ``tracked palm pose (xyz + xyzw)`` followed by the
    ProHand SDK's thumb-first 20-joint finger vector. The published model root
    sits about 25 cm behind the palm, so the tracked palm transform is composed
    with the inverse neutral root-to-palm transform before writing the free
    articulation root. The model's two wrist joints remain at zero.
    """

    cfg: ProHandActionCfg
    _asset: Articulation

    def __init__(self, cfg: ProHandActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        prefix = "L" if cfg.side == "left" else "R"
        self._controlled_joint_names = [f"{prefix}_wrist_abd", f"{prefix}_wrist_flex"]
        self._controlled_joint_names.extend(f"{prefix}_{stem}" for stem in _FINGER_JOINT_STEMS)
        joint_ids, resolved_names = self._asset.find_joints(
            self._controlled_joint_names, preserve_order=True, as_proxy=True
        )
        if list(resolved_names) != self._controlled_joint_names:
            raise ValueError(
                "The ProHand asset joint order does not match the teleop contract. "
                f"Expected {self._controlled_joint_names}, got {resolved_names}."
            )
        self._joint_ids = joint_ids.torch

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._raw_actions[:, 6] = 1.0
        self._processed_actions = torch.zeros(self.num_envs, 29, device=self.device)
        self._root_velocity = torch.zeros(self.num_envs, 6, device=self.device)
        self._position_offset = torch.tensor(cfg.position_offset, device=self.device)

        root_to_palm_position = torch.tensor(cfg.root_to_palm_position, device=self.device).repeat(self.num_envs, 1)
        root_to_palm_quaternion = torch.tensor(cfg.root_to_palm_quaternion, device=self.device).repeat(self.num_envs, 1)
        self._palm_to_root_position, self._palm_to_root_quaternion = subtract_frame_transforms(
            root_to_palm_position, root_to_palm_quaternion
        )

    @property
    def action_dim(self) -> int:
        return 27

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        palm_position = actions[:, 0:3] + self._position_offset
        palm_quaternion = actions[:, 3:7]
        quaternion_norm = torch.linalg.vector_norm(palm_quaternion, dim=-1, keepdim=True)
        identity = torch.zeros_like(palm_quaternion)
        identity[:, 3] = 1.0
        palm_quaternion = torch.where(
            quaternion_norm > 1.0e-6,
            palm_quaternion / quaternion_norm.clamp_min(1.0e-6),
            identity,
        )
        root_position, root_quaternion = combine_frame_transforms(
            palm_position,
            palm_quaternion,
            self._palm_to_root_position,
            self._palm_to_root_quaternion,
        )

        self._processed_actions[:, 0:3] = root_position
        self._processed_actions[:, 3:7] = root_quaternion
        self._processed_actions[:, 7:9] = 0.0
        self._processed_actions[:, 9:29] = actions[:, 7:27]

        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        self._processed_actions[:, 7:29].clamp_(limits[..., 0], limits[..., 1])

    def apply_actions(self) -> None:
        self._asset.write_root_pose_to_sim_index(root_pose=self._processed_actions[:, 0:7])
        self._asset.write_root_velocity_to_sim_index(root_velocity=self._root_velocity)
        self._asset.set_joint_position_target_index(target=self._processed_actions[:, 7:29], joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        indices = slice(None) if env_ids is None else env_ids
        self._raw_actions[indices] = 0.0
        self._raw_actions[indices, 6] = 1.0
