# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from isaaclab_experimental.managers import ManagerTermBase, SceneEntityCfg
from isaaclab_experimental.utils.warp.utils import wrap_to_pi

if TYPE_CHECKING:
    import torch

    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


@wp.kernel
def _joint_pos_target_l2_kernel(
    joint_pos: wp.array(dtype=wp.float32, ndim=2),
    joint_mask: wp.array(dtype=wp.bool),
    out: wp.array(dtype=wp.float32),
    target: float,
):
    """Accumulate squared wrapped-to-pi joint-position deviation from target over masked joints."""
    i = wp.tid()
    s = float(0.0)
    for j in range(joint_pos.shape[1]):
        if joint_mask[j]:
            a = wrap_to_pi(joint_pos[i, j])
            d = a - target
            s += d * d
    out[i] = s


def joint_pos_target_l2(env: ManagerBasedRLEnv, out, target: float, asset_cfg: SceneEntityCfg) -> None:
    """Penalize joint position deviation from a target value. Writes into ``out``."""
    asset: Articulation = env.scene[asset_cfg.name]
    assert asset.data.joint_pos.warp.shape[1] == asset_cfg.joint_mask.shape[0]
    wp.launch(
        kernel=_joint_pos_target_l2_kernel,
        dim=env.num_envs,
        inputs=[asset.data.joint_pos.warp, asset_cfg.joint_mask, out, target],
        device=env.device,
    )


@wp.kernel
def _survival_counts_kernel(
    env_mask: wp.array(dtype=wp.bool),
    time_outs: wp.array(dtype=wp.bool),
    counts: wp.array(dtype=wp.int32),
):
    """Atomically count masked (just-reset) envs and how many of them timed out."""
    env_index = wp.tid()
    if env_mask[env_index]:
        wp.atomic_add(counts, 0, 1)
        if time_outs[env_index]:
            wp.atomic_add(counts, 1, 1)


@wp.kernel
def _survival_rate_kernel(
    counts: wp.array(dtype=wp.int32),
    success_rate: wp.array(dtype=wp.float32),
):
    """Compute the survival success rate as the fraction of just-reset envs that timed out."""
    if counts[0] > 0:
        success_rate[0] = wp.float32(counts[1]) / wp.float32(counts[0])
    else:
        success_rate[0] = 0.0


class survival_success_rate(ManagerTermBase):
    """Tracks episode survival as the success metric (Warp-first).

    Twin of :class:`isaaclab_tasks.core.cartpole.mdp.rewards.survival_success_rate`.
    Returns zero reward (pure metric tracking). On reset, computes the fraction of
    just-reset environments that timed out (survived the full episode) entirely
    on-device and exposes it as ``Metrics/success_rate`` through the reward
    manager's reset extras. Unlike the stable term, there is no host readback, so
    the computation stays CUDA-graph capturable.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # [0] number of just-reset envs, [1] number of those that timed out
        self._counts_wp = wp.zeros((2,), dtype=wp.int32, device=env.device)
        self._success_rate_wp = wp.zeros((1,), dtype=wp.float32, device=env.device)
        # persistent 0-dim tensor view: kernels refresh the value on every reset/replay
        self._reset_extras = {"Metrics/success_rate": wp.to_torch(self._success_rate_wp)[0]}

    def reset(self, env_mask: wp.array | None = None) -> dict[str, torch.Tensor]:
        if env_mask is None:
            env_mask = self._env.resolve_env_mask()
        self._counts_wp.zero_()
        wp.launch(
            kernel=_survival_counts_kernel,
            dim=self.num_envs,
            inputs=[env_mask, self._env.termination_manager.time_outs_wp, self._counts_wp],
            device=self.device,
        )
        wp.launch(
            kernel=_survival_rate_kernel,
            dim=1,
            inputs=[self._counts_wp, self._success_rate_wp],
            device=self.device,
        )
        return self._reset_extras

    def __call__(self, env: ManagerBasedRLEnv, out: wp.array) -> None:
        out.zero_()
