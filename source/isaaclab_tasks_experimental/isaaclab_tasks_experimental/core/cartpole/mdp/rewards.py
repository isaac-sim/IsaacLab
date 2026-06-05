# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp
from isaaclab_experimental.managers import SceneEntityCfg
from isaaclab_experimental.managers.manager_base import ManagerTermBase
from isaaclab_experimental.utils.warp.utils import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab_experimental.managers.manager_term_cfg import RewardTermCfg

    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


@wp.kernel
def _joint_pos_target_l2_kernel(
    joint_pos: wp.array(dtype=wp.float32, ndim=2),
    joint_mask: wp.array(dtype=wp.bool),
    out: wp.array(dtype=wp.float32),
    target: float,
):
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


class survival_success_rate(ManagerTermBase):
    """Logs the mean time-out (survival) rate of the resetting envs; contributes zero reward.

    Mirrors the stable term: the reward value is always zero (so it is registered with
    ``weight=0.0``) and the only effect is logging ``Metrics/success_rate`` on reset, where
    success is defined as the episode ending by time-out rather than an early termination.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def reset(self, env_mask: wp.array | None = None) -> None:
        # ``time_outs`` is exposed as a torch tensor by the warp termination manager.
        time_outs = self._env.termination_manager.time_outs
        if env_mask is None:
            survived = time_outs.float().mean()
        else:
            selected = time_outs[wp.to_torch(env_mask).bool()]
            survived = selected.float().mean() if selected.numel() > 0 else time_outs.new_zeros(())
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = float(survived.item())

    def __call__(self, env: ManagerBasedRLEnv, out) -> None:
        # Pure logging term: the reward contribution is zero. The reward manager pre-zeroes
        # ``out`` each step, but zero it explicitly so the term is correct independent of that.
        out.zero_()
