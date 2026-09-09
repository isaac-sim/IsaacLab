# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SO-101 keyboard-typing environments.

The MDP implementation is heavily inspired by OmniReset:

@inproceedings{yin2026emergent,
  title={Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning},
  author={Patrick Yin and Tyler Westenbroek and Zhengyu Zhang and Joshua Tran and Ignacio Dagnino and Eeshani Shilamkar and Numfor Mbiziwo-Tiapo and Simran Bagaria and Xinlei Liu and Galen Mullins and Andrey Kolobov and Abhishek Gupta},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2603.15789},
}
"""

__all__ = [
    "LetterTypingCommand",
    "LetterTypingCommandCfg",
    "letter_typing_progress",
    "typing_success",
    "typing_mistake",
    "typing_complete",
    "reach_key",
    "key_positions_b",
    "target_keys_onehot",
    "typed_keys_onehot",
    "mechanical_power",
]

from .commands import LetterTypingCommand, LetterTypingCommandCfg
from .observations import key_positions_b, target_keys_onehot, typed_keys_onehot
from .rewards import letter_typing_progress, mechanical_power, reach_key, typing_success
from .terminations import typing_complete, typing_mistake
from isaaclab.envs.mdp import *
