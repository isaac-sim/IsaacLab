# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout timing for the three-camera Cosmos trocar task."""

from weakref import proxy

from isaaclab.envs import ManagerBasedRLEnv

from .assemble_trocar_grid import _GridGroupCfg


class TrocarCosmosEnv(ManagerBasedRLEnv):
    """Schedule grid transfers on the policy's environment steps across warmup and episode resets."""

    def reset(self, env_ids=None, *, seed=None, options=None):
        ids = range(*env_ids.indices(self.num_envs)) if isinstance(env_ids, slice) else env_ids
        if self.cfg.cosmos.generation_mode == "single_frame" and (ids is None or len(ids) == self.num_envs):
            model = self.sim.get_or_create_backend(self.cfg.cosmos)
            group = self.sim.get_or_create_backend(_GridGroupCfg(model=self.cfg.cosmos), model=model)
            # RLinf consumes the observation after its reset hold. Ordinary Gym callers consume reset() directly.
            first_step = self.common_step_counter + (options or {}).get("steps_before_first_action", 0)
            env = proxy(self)
            group.policy_step = lambda: env.common_step_counter - first_step
            group.last_decision_step = None
        return super().reset(env_ids=env_ids, seed=seed, options=options)
