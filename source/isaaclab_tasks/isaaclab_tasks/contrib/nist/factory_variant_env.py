# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep the variant task's policy boundary finite when a reset produces an invalid world."""

from typing import Any

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn


class FactoryVariantEnv(ManagerBasedRLEnv):
    """Reset only worlds whose action, reward, or post-reset observation is nonfinite."""

    def _invalid_observation_rows(self, observations: VecEnvObs) -> torch.Tensor:
        invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for group in observations.values():
            invalid |= ~torch.isfinite(group).reshape(self.num_envs, -1).all(dim=1)
        return invalid

    def _recover(self, observations: VecEnvObs, invalid: torch.Tensor) -> None:
        for _ in range(3):
            env_ids = invalid.nonzero(as_tuple=False).squeeze(-1).int()
            self._reset_idx(env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            refreshed = self.observation_manager.compute(update_history=True)
            for name, group in observations.items():
                group[env_ids] = refreshed[name][env_ids]
            invalid = self._invalid_observation_rows(observations)
            if not invalid.any():
                return
        raise RuntimeError(f"Variant reset still produced nonfinite observations in {invalid.sum().item()} worlds.")

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        invalid_action = ~torch.isfinite(action).all(dim=1)
        if invalid_action.any():
            action = action.clone()
            action[invalid_action] = 0.0
        observations, reward, terminated, timeouts, extras = super().step(action)
        invalid = invalid_action | ~torch.isfinite(reward) | self._invalid_observation_rows(observations)
        if invalid.any():
            self._recover(observations, invalid)
            reward[invalid] = 0.0
            terminated[invalid] = True
            timeouts[invalid] = False
            self.reset_buf[invalid] = True
            extras.setdefault("log", {})["Info/NonfiniteRecovery"] = invalid.sum().item()
        return observations, reward, terminated, timeouts, extras

    def reset(
        self,
        env_ids: torch.Tensor | slice | None = slice(None),
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VecEnvObs, dict]:
        observations, extras = super().reset(env_ids=env_ids, seed=seed, options=options)
        invalid = self._invalid_observation_rows(observations)
        if invalid.any():
            self._recover(observations, invalid)
        return observations, extras
