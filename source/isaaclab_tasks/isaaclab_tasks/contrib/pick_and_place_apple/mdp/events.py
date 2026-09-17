# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom events for the H2 pick-and-place apple task."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .rewards import PnpAppleState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger(__name__)


def init_task_phase_state(env: ManagerBasedRLEnv, _env_ids: torch.Tensor | None = None) -> None:
    """Allocate the pick-and-place phase state on ``env``.

    Registered as a ``startup`` event, so it runs once for the whole scene
    before the first reset or step. The env-id argument is part of the event
    term contract but unused here.
    """
    env.pnp_apple_state = PnpAppleState.create(env.num_envs, env.device)


def reset_task_phase(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    apple_cfg: SceneEntityCfg = SceneEntityCfg("apple"),
    print_log: bool = False,
) -> None:
    """Reset phase trackers and capture the post-reset apple height reference."""
    if len(env_ids) == 0:
        return

    state = env.pnp_apple_state
    previous_phase = state.task_phase[env_ids].clone()

    if print_log:
        exact_counts = [int((previous_phase == phase_id).sum().item()) for phase_id in range(5)]
        logger.info(
            "[PNP_PHASE_SUMMARY] total=%d exact_phase_0/1/2/3/4=%s",
            len(env_ids),
            exact_counts,
        )
        for phase_id in (1, 2, 3, 4):
            reached_mask = previous_phase >= phase_id
            reached_env_ids = env_ids[reached_mask].detach().cpu().tolist()
            logger.info(
                "[PNP_PHASE_SUMMARY] reached_phase_%d=%d/%d env_ids=%s",
                phase_id,
                len(reached_env_ids),
                len(env_ids),
                reached_env_ids,
            )

    state.reset(env_ids)

    apple = env.scene[apple_cfg.name]
    apple_z = apple.data.root_pos_w.torch[env_ids, 2]
    state.initial_apple_z[env_ids] = apple_z

    if print_log:
        logger.info("Reset task phase for %d environment(s)", len(env_ids))
