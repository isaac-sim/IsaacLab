# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.math import unscale_transform

from isaaclab_tasks.core.reorient.reorient_direct_env import ReorientDirectEnv
from isaaclab_tasks.core.reorient.utils import resolve_actuated_tendons

if TYPE_CHECKING:
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_env_cfg import ShadowHandEnvCfg


class ShadowHandDirectEnv(ReorientDirectEnv):
    """The reorientation task for a hand whose motors also pull tendons.

    Four of this hand's twenty motors pull a tendon spanning a finger's middle and distal joints
    rather than driving a joint. A tendon is not a joint: it has its own index space, resolved with
    :meth:`find_fixed_tendons`, and its own command path. Hands whose motors all drive joints -- the
    Allegro hand, for one -- have nothing to bind here, which is why this lives in a subclass rather
    than in the shared environment.
    """

    cfg: ShadowHandEnvCfg

    def __init__(self, cfg: ShadowHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.actuated_tendon_indices, self.tendon_lower_limits, self.tendon_upper_limits = resolve_actuated_tendons(
            self.hand,
            cfg.actuated_tendon_names,
            self.num_envs,
            self.device,
            cfg.actuated_tendon_position_limits,
        )

    def _apply_action(self) -> None:
        # Actions are ordered joints first, then tendons, matching the manager task's action terms.
        super()._apply_action()
        num_joint_actions = len(self.actuated_dof_indices)
        # No moving average on the tendon target: the manager task's action term applies none, and
        # the two task variants have to stay comparable.
        self.hand.set_fixed_tendon_position_target_index(
            target=unscale_transform(
                self.actions[:, num_joint_actions:], self.tendon_lower_limits, self.tendon_upper_limits
            ),
            fixed_tendon_ids=self.actuated_tendon_indices,
        )
