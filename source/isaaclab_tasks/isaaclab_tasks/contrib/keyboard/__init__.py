# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-101 keyboard-typing environments.

The MDP implementation is based primarily on OmniReset:

@inproceedings{yin2026emergent,
  title={Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning},
  author={Patrick Yin and Tyler Westenbroek and Zhengyu Zhang and Joshua Tran and Ignacio Dagnino and Eeshani Shilamkar and Numfor Mbiziwo-Tiapo and Simran Bagaria and Xinlei Liu and Galen Mullins and Andrey Kolobov and Abhishek Gupta},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2603.15789},
}
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-Keyboard-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.so101_env_cfg:SO101KeyboardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101PPORunnerCfg",
    },
)
