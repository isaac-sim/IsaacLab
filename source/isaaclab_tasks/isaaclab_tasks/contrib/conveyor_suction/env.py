# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka suction sorting on the existing Newton conveyor lifecycle."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab_contrib.surface_gripper.newton import SurfaceGripper

from ..conveyor_franka.conveyor_franka_env import ConveyorFrankaEnv
from .actions_cfg import TOOL_OFFSET
from .scene import BIN_POSITIONS, GOAL_DIRECTIONS, PARCEL_CLASS_IDS, PARCEL_SIZES


class SuctionSortingEnv(ConveyorFrankaEnv):
    """Share conveyor/contact hooks and keep suction attachment state explicit."""

    def _init_sim(self) -> None:
        self.gripper = SurfaceGripper(
            self.cfg.scene.num_envs,
            manager=self.sim.backend_manager,
            tool_body_pattern=r".*/panda_hand",
            target_body_pattern=r".*/Cube[0-9]+",
            target_count_per_env=len(PARCEL_SIZES),
            tool_offset=TOOL_OFFSET,
        )
        try:
            super()._init_sim()
        except Exception:
            self.gripper.close()
            raise

    def load_managers(self) -> None:
        self.parcels = tuple(self.scene[f"parcel_{index}"] for index in range(len(PARCEL_SIZES)))
        self.parcel_sizes = torch.tensor(PARCEL_SIZES, device=self.device)
        self.parcel_classes = torch.tensor(PARCEL_CLASS_IDS, dtype=torch.long, device=self.device)
        self.bin_positions = torch.tensor(BIN_POSITIONS, device=self.device)
        self.bin_directions = torch.tensor(GOAL_DIRECTIONS, device=self.device)
        self.active_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        super().load_managers()
        self._conveyor_tool_offset = torch.tensor(TOOL_OFFSET, device=self.device).expand(self.num_envs, -1)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        super()._reset_idx(env_ids)
        reset = self.event_manager.get_term_cfg("reset_parcels").func
        ids = self.scene._ALL_INDICES[env_ids] if isinstance(env_ids, slice) else env_ids
        self.gripper.set_command(reset.held_rows[ids], self.active_ids[ids], env_ids=ids)

    def _setup_goal_selector(self) -> None:
        """This policy selects a size outlet rather than a user-selected transfer goal."""

    def close(self) -> None:
        self.gripper.close()
        super().close()
