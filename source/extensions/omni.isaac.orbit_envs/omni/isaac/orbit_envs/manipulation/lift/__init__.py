# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment for lifting objects with fixed-arm robots."""

from .lift_cfg import LiftEnvCfg
from .lift_env import LiftEnv

__all__ = ["LiftEnv", "LiftEnvCfg"]
