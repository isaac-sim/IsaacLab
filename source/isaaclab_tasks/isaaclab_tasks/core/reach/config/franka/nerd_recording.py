# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-position exploration of the Franka arm."""

import torch
from isaaclab_neural.record.recording import NeRDExplorationAction
from newton_neural.data import CommandHeld, CommandNearState, CommandSmooth, CommandSwitch, SamplerUniform


class FrankaNeRDExploration(NeRDExplorationAction):
    """Sample the arm state and mix held, near-state and smooth targets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        low, high = self.binding.state_bounds()["joint_q"]
        arm = torch.arange(len(low), device=self.device) < cfg.arm_joint_count
        default = self.default_joint_pos
        self.states = SamplerUniform(
            {
                "joint_q": (torch.where(arm, low, default), torch.where(arm, high, default)),
                "joint_qd": (-cfg.state_speed * arm, cfg.state_speed * arm),
            }
        )
        closed = torch.zeros_like(low)
        self.commands = CommandSwitch(
            {
                CommandHeld(cfg.hold_frames): 1.0,
                CommandNearState(cfg.explore_reach * arm): 1.0,
                CommandNearState(cfg.refine_reach * arm): 1.0,
                CommandSmooth(cfg.drift_timescale): 1.0,
            },
            low=torch.where(arm, low - cfg.target_margin, closed),
            high=torch.where(arm, high + cfg.target_margin, closed),
        )
        self.initialize_sampling()
