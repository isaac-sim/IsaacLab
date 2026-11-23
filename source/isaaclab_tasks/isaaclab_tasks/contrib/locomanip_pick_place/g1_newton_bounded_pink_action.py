# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded Pink upper-body action for G1 on Newton."""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction

from .g1_newton_actions import bound_pink_joint_targets


class G1NewtonBoundedPinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """Bound Pink outputs without changing G1's public 28-D action layout."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._pink_joint_limits = None

    def _compute_ik_solutions(self) -> torch.Tensor:
        proposed = super()._compute_ik_solutions()
        current = self._asset.data.joint_pos.torch[:, self._isaaclab_controlled_joint_ids]
        articulation_limits = self._asset.data.joint_pos_limits.torch[:, self._isaaclab_controlled_joint_ids, :]

        # Pink's controlled model/order are immutable for this action's lifetime.
        # Keep articulation limits live: users may update them during a run.
        if self._pink_joint_limits is None:
            self._pink_joint_limits = tuple(
                torch.as_tensor(
                    np.stack(
                        [
                            np.asarray(getattr(controller.pink_configuration.controlled_model, limit_name))[
                                controller.pink_to_isaac_lab_controlled_ordering
                            ]
                            for controller in self._ik_controllers
                        ]
                    ),
                    device=self.device,
                    dtype=proposed.dtype,
                )
                for limit_name in ("lowerPositionLimit", "upperPositionLimit")
            )
        pink_lower, pink_upper = self._pink_joint_limits
        lower = torch.maximum(pink_lower, articulation_limits[..., 0])
        upper = torch.minimum(pink_upper, articulation_limits[..., 1])
        return bound_pink_joint_targets(current, proposed, lower, upper)
