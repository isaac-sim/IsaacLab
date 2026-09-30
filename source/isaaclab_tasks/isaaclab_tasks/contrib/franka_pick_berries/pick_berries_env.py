# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single workcell environment with resettable tissue, damage, and contact history."""

from isaaclab.envs import ManagerBasedRLEnv

from .physics.collection import advance_berries, berry_configs
from .physics.pair import BerryInstance, pair_offsets
from .physics.runtime import BerryRuntime


class BerryPickEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        if cfg.scene.num_envs != 1:
            raise ValueError("Berry teleoperation currently supports one environment")
        if cfg.berry_count not in (1, 2, 3):
            raise ValueError("Berry count must be 1, 2 or 3")
        cfg.pair = cfg.pair or cfg.berry_count > 1
        if cfg.pair and cfg.berry_count == 1:
            cfg.berry_count = 2
        super().__init__(cfg, **kwargs)
        if cfg.pair:
            placements = pair_offsets(cfg)
            owner = BerryRuntime(cfg, self.scene["robot"], placements=placements)
            self.berries = {
                f"raspberry_{i + 1}": BerryInstance(owner, i, shift) for i, shift in enumerate(owner.placements)
            }
            self.berry_systems = [owner]
            self.berry = self.berries["raspberry_1"]
        else:
            self.berries = {item.berry: BerryRuntime(item, self.scene["robot"]) for item in berry_configs(cfg)}
            self.berry_systems = list(self.berries.values())
            self.berry = self.berries[cfg.target_berry if cfg.berry == "all" else cfg.berry]

    def advance_berries(self) -> None:
        """Advance all berries over one robot physics step and sum their contact forces."""
        advance_berries(self.berry_systems, self.scene["robot"])

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        for berry in getattr(self, "berry_systems", []):
            berry.reset(self.scene["robot"])

    def step(self, action):
        result = super().step(action)
        for berry in self.berry_systems:
            try:
                berry.checker.check()
            except RuntimeError as error:
                # Read full state only on failure; normal teleop keeps the small
                # device-side check. Bounds are local to this solver's offset.
                points = berry.sim.x.numpy()
                lower = [float(x) + 0.5 * berry.sim.h for x in berry.sim.origin]
                upper = [float(x) + (int(n) - 1.5) * berry.sim.h for x, n in zip(berry.sim.origin, berry.sim.res)]
                raise RuntimeError(
                    f"{error}; particle bounds [m]: {points.min(0).tolist()} .. {points.max(0).tolist()};"
                    f" safe grid bounds [m]: {lower} .. {upper} (upper exclusive);"
                    f" world offset [m]: {berry.offset.tolist()}"
                ) from error
        return result
