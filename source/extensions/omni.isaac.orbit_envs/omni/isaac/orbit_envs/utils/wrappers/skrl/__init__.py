# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module includes wrappers built in PyTorch and JAX for skrl.

Reference:
    https://github.com/Toni-SM/skrl

"""

from .skrl_jax import SkrlJaxVecEnvWrapper, SkrlJaxVecTrainer
from .skrl_torch import SkrlTorchVecEnvWrapper, SkrlTorchVecTrainer

__all__ = [
    "SkrlJaxVecEnvWrapper",
    "SkrlTorchVecEnvWrapper",
    "SkrlJaxVecTrainer",
    "SkrlTorchVecTrainer",
]
