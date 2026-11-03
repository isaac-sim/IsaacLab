# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VelStand PPO and two-teacher imitation configuration."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from isaaclab_tasks.contrib.microduck.backlash import backlash_agent_preset
from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg

from .bc_config import default_bc_cfg


@configclass
class MicroDuckVelStandAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO plus source-compatible post-PPO BC and checkpoint lifecycle settings."""

    class_name = "isaaclab_tasks.contrib.microduck.velstand.agents.ppo:MicroDuckVelStandPPO"
    bc_cfg: dict | None = default_bc_cfg()
    checkpoint_mode: str = "auto"
    artifact_cache_dir: str | None = None


@configclass
class MicroDuckVelStandPPORunnerCfg(MicroDuckPPORunnerCfg):
    """6000 PPO updates with a walking warm start and two frozen teachers."""

    experiment_name = "microduck_velstand_flat"
    max_iterations = 6000
    actor = MicroDuckPPORunnerCfg().actor.replace(
        class_name="isaaclab_tasks.contrib.microduck.velstand.agents.model:MicroDuckVelStandMLPModel"
    )
    critic = MicroDuckPPORunnerCfg().critic.replace(
        class_name="isaaclab_tasks.contrib.microduck.velstand.agents.model:MicroDuckVelStandMLPModel"
    )
    algorithm = MicroDuckVelStandAlgorithmCfg(
        **{k: v for k, v in MicroDuckPPORunnerCfg().algorithm.to_dict().items() if k != "class_name"}
    )

    def __post_init__(self):
        backlash_agent_preset(self)


@configclass
class MicroDuckVelStandRoughPPORunnerCfg(MicroDuckVelStandPPORunnerCfg):
    """Separate rough and backlash checkpoint trees using the standard preset interface."""

    experiment_name = "microduck_velstand_rough"
