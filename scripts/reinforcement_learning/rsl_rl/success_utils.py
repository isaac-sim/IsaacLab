# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helper for printing task-success statistics during RSL-RL inference/play.

Works with any environment that exposes a ``_compute_success()`` method returning
``(is_success, pos_error, keypoint_dist)`` per environment (see
``deploy.cable_insertion.insertion_env.DisplayportInsertionEnv``). For other
environments the tracker disables itself gracefully.
"""

from __future__ import annotations

import torch


class SuccessTracker:
    """Accumulate and format success statistics across play steps/episodes.

    Two rates are tracked:

    - **instantaneous success rate**: fraction of envs currently in the success
      state (plug mate point within threshold of the socket mate point).
    - **episode success rate**: fraction of completed episodes in which the env
      reached the success state at least once. This is robust to the fact that
      manager-based envs auto-reset inside ``step()`` (the terminal state is not
      visible post-step), because success is accumulated from prior in-episode
      steps.
    """

    def __init__(self, env):
        self.base = env.unwrapped
        self.enabled = hasattr(self.base, "_compute_success")
        if not self.enabled:
            return
        n = self.base.num_envs
        self.device = self.base.device
        self._ever_success = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.num_episodes = 0
        self.num_success_episodes = 0

    def update(self, dones) -> dict | None:
        """Update stats after an ``env.step``. ``dones`` is the per-env done tensor.

        Returns a dict of scalars, or ``None`` if the env does not support success.
        """
        if not self.enabled:
            return None

        is_success, pos_error, keypoint_dist = self.base._compute_success()
        dones_b = torch.as_tensor(dones, device=self.device).bool().view(-1)

        # An episode ends this step for done envs: record whether it reached success
        # at any point (accumulated in prior steps), then reseed with the fresh state.
        if dones_b.any():
            reached = self._ever_success[dones_b]
            self.num_episodes += int(dones_b.sum().item())
            self.num_success_episodes += int(reached.sum().item())
            self._ever_success[dones_b] = is_success[dones_b]
        # Accumulate success for ongoing episodes.
        ongoing = ~dones_b
        self._ever_success[ongoing] |= is_success[ongoing]

        episode_rate = (self.num_success_episodes / self.num_episodes) if self.num_episodes > 0 else float("nan")
        return {
            "instant_success_rate": is_success.float().mean().item(),
            "pos_error_m": pos_error.mean().item(),
            "keypoint_dist_m": keypoint_dist.mean().item(),
            "episode_success_rate": episode_rate,
            "num_episodes": self.num_episodes,
        }

    @staticmethod
    def format(info: dict | None) -> str:
        """Format the stats dict as a compact one-line string (empty if unsupported)."""
        if info is None:
            return ""
        ep = info["episode_success_rate"]
        ep_str = f"{ep:.3f}" if ep == ep else "n/a"  # NaN check
        return (
            f"[success] instant={info['instant_success_rate']:.3f} | "
            f"episode={ep_str} (n={info['num_episodes']}) | "
            f"pos_err={info['pos_error_m'] * 1000:.2f}mm | "
            f"kp_dist={info['keypoint_dist_m'] * 1000:.2f}mm"
        )
