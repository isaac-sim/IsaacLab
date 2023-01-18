# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Classic environments for control.

These environments are based on the MuJoCo environments provided by OpenAI.

Reference:
    https://github.com/openai/gym/tree/master/gym/envs/mujoco
"""

from .ant import AntEnv
from .cartpole import CartpoleEnv
from .humanoid import HumanoidEnv

__all__ = ["CartpoleEnv", "AntEnv", "HumanoidEnv"]
