# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from isaaclab_tasks.core.reorient.utils import resolve_actuated_tendons

from ...reorient_warp_env import ReorientDirectWarpEnv, scale

if TYPE_CHECKING:
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_env_cfg import ShadowHandEnvCfg


@wp.kernel
def apply_actions_to_tendon_targets(
    # input
    actions: wp.array2d(dtype=wp.float32),
    first_tendon_action: wp.int32,
    lower_limits: wp.array2d(dtype=wp.float32),
    upper_limits: wp.array2d(dtype=wp.float32),
    tendon_ids: wp.array(dtype=wp.int32),
    # output
    tendon_targets: wp.array2d(dtype=wp.float32),
):
    """Scale tendon actions to each tendon's command range and clamp them to it, without smoothing."""
    env_id, i = wp.tid()
    lower = lower_limits[env_id, i]
    upper = upper_limits[env_id, i]
    target = scale(actions[env_id, first_tendon_action + i], lower, upper)
    tendon_targets[env_id, tendon_ids[i]] = wp.clamp(target, lower, upper)


class ShadowHandDirectWarpEnv(ReorientDirectWarpEnv):
    """Warp twin of the stable Shadow Hand reorientation environment.

    Implements :class:`~isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_env.ShadowHandDirectEnv`:
    four of this hand's twenty motors pull a tendon spanning a finger's middle and distal joints.
    Actions are ordered joints first, then tendons.
    """

    cfg: ShadowHandEnvCfg

    def __init__(self, cfg: ShadowHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        tendon_ids, lower_limits, upper_limits = resolve_actuated_tendons(
            self.hand, cfg.actuated_tendon_names, self.num_envs, self.device, cfg.actuated_tendon_position_limits
        )
        self.num_actuated_tendons = len(tendon_ids)
        self.actuated_tendon_ids = wp.array(tendon_ids, dtype=wp.int32, device=self.device)
        self.actuated_tendon_mask = wp.array(
            [tendon_id in tendon_ids for tendon_id in range(self.hand.num_fixed_tendons)],
            dtype=wp.bool,
            device=self.device,
        )
        self.tendon_lower_limits = wp.from_torch(lower_limits)
        self.tendon_upper_limits = wp.from_torch(upper_limits)
        self.tendon_targets = wp.zeros(
            (self.num_envs, self.hand.num_fixed_tendons), dtype=wp.float32, device=self.device
        )

    def _pre_physics_step(self, actions: wp.array) -> None:
        super()._pre_physics_step(actions)
        # Commanded here, outside the captured action stage: the articulation flags a buffered
        # tendon target for submission on the host, which a graph replay would not repeat.
        wp.launch(
            apply_actions_to_tendon_targets,
            dim=(self.num_envs, self.num_actuated_tendons),
            inputs=[
                self.actions,
                self.num_actuated_dofs,
                self.tendon_lower_limits,
                self.tendon_upper_limits,
                self.actuated_tendon_ids,
                self.tendon_targets,
            ],
            device=self.device,
        )
        self.hand.set_fixed_tendon_position_target_mask(
            target=self.tendon_targets, fixed_tendon_mask=self.actuated_tendon_mask
        )
