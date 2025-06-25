# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
"""
