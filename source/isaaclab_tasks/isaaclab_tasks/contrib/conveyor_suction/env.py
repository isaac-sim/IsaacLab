# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka suction sorting on the existing Newton conveyor lifecycle."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..conveyor_franka.conveyor_franka_env import ConveyorFrankaEnv
from .actions_cfg import TOOL_OFFSET
from .scene import (
    PARCEL_CLASS_IDS,
    PARCEL_SIZES,
    SUCTION_PAD_RADIUS,
    SUCTION_PAD_SPACING,
)


class SuctionSortingEnv(ConveyorFrankaEnv):
    """Share conveyor/contact hooks and keep suction attachment state explicit."""

    def _init_sim(self) -> None:
        from isaaclab_contrib.surface_gripper.newton import SurfaceGripper

        for index in range(self.cfg.parcel_count, len(PARCEL_SIZES)):
            setattr(self.cfg.scene, f"parcel_{index}", None)
            setattr(self.cfg.events, f"parcel_{index}_mass", None)
            setattr(self.cfg.events, f"parcel_{index}_material", None)
        self.cfg.conveyor_force.transported_body_count_per_env = self.cfg.parcel_count
        self.parcel_sizes = torch.tensor(PARCEL_SIZES[: self.cfg.parcel_count], device=self.device)
        self.parcel_classes = torch.tensor(
            PARCEL_CLASS_IDS[: self.cfg.parcel_count], dtype=torch.long, device=self.device
        )
        self.gripper = SurfaceGripper(
            self.cfg.scene.num_envs,
            manager=self.sim.backend_manager,
            tool_body_pattern=r".*/panda_hand",
            target_body_pattern=r".*/Cube[0-9]+",
            target_count_per_env=self.cfg.parcel_count,
            tool_offset=TOOL_OFFSET,
            pad_radius=SUCTION_PAD_RADIUS,
            pad_spacing=SUCTION_PAD_SPACING,
        )
        try:
            super()._init_sim()
        except Exception:
            self.gripper.close()
            raise

    def load_managers(self) -> None:
        self.parcels = tuple(self.scene[f"parcel_{index}"] for index in range(self.cfg.parcel_count))
        self.active_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        super().load_managers()
        self._conveyor_tool_offset = torch.tensor(TOOL_OFFSET, device=self.device).expand(self.num_envs, -1)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        super()._reset_idx(env_ids)
        reset = self.event_manager.get_term_cfg("reset_parcels").func
        ids = self.scene._ALL_INDICES[env_ids] if isinstance(env_ids, slice) else env_ids
        self.gripper.set_command(reset.held_rows[ids], self.active_ids[ids], env_ids=ids)
        self.gripper.refresh(env_ids=ids)

    def close(self) -> None:
        self.gripper.close()
        super().close()
