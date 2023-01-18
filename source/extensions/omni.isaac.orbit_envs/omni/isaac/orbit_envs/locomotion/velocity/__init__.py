# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-based locomotion environments for legged robots."""

from .velocity_cfg import VelocityEnvCfg
from .velocity_env import VelocityEnv

__all__ = ["VelocityEnv", "VelocityEnvCfg"]
