# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental MDP terms.

This package forwards all stable MDP terms from :mod:`isaaclab.envs.mdp`, but overrides reward
functions with Warp-first implementations from :mod:`isaaclab_experimental.envs.mdp.rewards`.
"""

# Forward stable MDP terms (actions/observations/terminations/etc.) but *exclude* rewards.
# Rewards are provided by this experimental package to keep a Warp-compatible signature
# (`func(env, out, **params) -> None`) for the experimental RewardManager.
from isaaclab.envs.mdp.actions import *  # noqa: F401, F403
from isaaclab.envs.mdp.commands import *  # noqa: F401, F403
from isaaclab.envs.mdp.curriculums import *  # noqa: F401, F403
from isaaclab.envs.mdp.events import *  # noqa: F401, F403
from isaaclab.envs.mdp.observations import *  # noqa: F401, F403
from isaaclab.envs.mdp.recorders import *  # noqa: F401, F403
from isaaclab.envs.mdp.terminations import *  # noqa: F401, F403

# Override reward terms with experimental implementations.
from .rewards import *  # noqa: F401, F403
