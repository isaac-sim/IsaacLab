# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_tasks.core.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1RoughPPORunnerCfg
from isaaclab_tasks.utils import resolve_presets


def test_g1_rough_newton_preset_uses_extended_training_schedule():
    """Verify MJWarp selects the tuned G1 Newton training duration."""
    agent_cfg = resolve_presets(G1RoughPPORunnerCfg(), selected=("newton_mjwarp",))

    assert agent_cfg.max_iterations == 5000
