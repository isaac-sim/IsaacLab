# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct environment for the OpenAI Shadow Hand variant.

Reproduces the training regime of `Learning Dexterous In-Hand Manipulation`_. The
observation architecture it pairs with is the core task's ``presets=openai``; the only
behavior that differs from that task is the episode budget, which is spent per goal
rather than per episode.

.. _Learning Dexterous In-Hand Manipulation: https://arxiv.org/pdf/1808.00177.pdf
"""

from __future__ import annotations

import torch

from isaaclab_tasks.core.reorient.reorient_direct_env import ReorientDirectEnv


class ShadowHandOpenAIEnv(ReorientDirectEnv):
    """Reorientation with the paper's per-goal episode budget."""

    def _compute_time_out(self) -> torch.Tensor:
        """Spend the episode budget per goal, and stop at the consecutive-success cap.

        Zeroing :attr:`episode_length_buf` on a reached goal restarts the timer, so
        ``episode_length_s`` bounds the time allowed for *one* goal rather than for the
        episode. Carried over from IsaacGymEnvs' ``shadow_hand`` task.

        A non-positive :attr:`~ShadowHandOpenAIEnvCfg.max_consecutive_success` disables the
        streak cap, and with it the per-goal budget: the comparison below is true from the
        first step when the cap is zero, which would truncate every episode immediately.
        """
        if self.cfg.max_consecutive_success <= 0:
            return super()._compute_time_out()

        self.episode_length_buf = torch.where(
            self._success_flags,
            torch.zeros_like(self.episode_length_buf),
            self.episode_length_buf,
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out | (self.successes >= self.cfg.max_consecutive_success)
