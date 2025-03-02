# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package for environment wrappers to different learning frameworks.

Wrappers allow you to modify the behavior of an environment without modifying the environment itself.
This is useful for modifying the observation space, action space, or reward function. Additionally,
they can be used to cast a given environment into the respective environment class definition used by
different learning frameworks. This operation may include handling of asymmetric actor-critic observations,
casting the data between different backends such `numpy` and `pytorch`, or organizing the returned data
into the expected data structure by the learning framework.

All wrappers work similar to the :class:`gymnasium.Wrapper` class. Using a wrapper is as simple as passing
the initialized environment instance to the wrapper constructor. However, since learning frameworks
expect different input and output data structures, their wrapper classes are not compatible with each other.
Thus, they should always be used in conjunction with the respective learning framework.

For instance, to wrap an environment in the `Stable-Baselines3`_ wrapper, you can do the following:

.. code-block:: python

   from isaaclab_rl.sb3 import Sb3VecEnvWrapper

   env = Sb3VecEnvWrapper(env)


.. _Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3

"""

from . import sb3
from . import skrl
from . import rsl_rl
from . import rl_games

__all__ = ["sb3", "skrl", "rsl_rl", "rl_games"]
