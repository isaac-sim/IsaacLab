# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-based locomotion environments for legged robots."""

from .locomotion_cfg import LocomotionEnvCfg, LocomotionEnvRoughCfg, LocomotionEnvRoughCfg_PLAY
from .locomotion_env import LocomotionEnv

__all__ = ["LocomotionEnv", "LocomotionEnvRoughCfg", "LocomotionEnvRoughCfg_PLAY"]
