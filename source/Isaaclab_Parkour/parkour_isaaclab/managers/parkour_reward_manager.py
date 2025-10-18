
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import omni.kit.app
from isaaclab.managers import RewardManager

if TYPE_CHECKING:
    from parkour_isaaclab.envs import ParkourManagerBasedRLEnv

class ParkourRewardManager(RewardManager):
    _env: ParkourManagerBasedRLEnv

    def __init__(self, cfg: object, env: ParkourManagerBasedRLEnv):
        super().__init__(cfg, env)

    def compute(self, dt: float) -> torch.Tensor:
        """
        Same to Legged Gym 
        """
        # reset computation
        self._reward_buf[:] = 0.0
        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total reward
            self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value
            # Update current reward for this step.
            self._step_reward[:, term_idx] = value / dt
        self._reward_buf[:] = torch.clip(self._reward_buf[:], min=0.)
        return self._reward_buf
