# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`IsaacEnv` instance to skrl environment.

The following example shows how to wrap an environment for skrl:

.. tabs::

    .. group-tab:: PyTorch

        .. code-block:: python

            from omni.isaac.orbit_envs.utils.wrappers.skrl import SkrlTorchVecEnvWrapper

            env = SkrlTorchVecEnvWrapper(env)

    .. group-tab:: JAX

        .. code-block:: python

            from omni.isaac.orbit_envs.utils.wrappers.skrl import SkrlJaxVecEnvWrapper

            env = SkrlJaxVecEnvWrapper(env)

Or, equivalently, by directly calling the skrl library API as follows:

.. tabs::

    .. group-tab:: PyTorch

        .. code-block:: python

            from skrl.envs.wrappers.torch import wrap_env

            env = wrap_env(env, wrapper="isaac-orbit")

    .. group-tab:: JAX

        .. code-block:: python

            from skrl.envs.wrappers.jax import wrap_env

            env = wrap_env(env, wrapper="isaac-orbit")

"""

from .skrl_jax import SkrlJaxVecEnvWrapper, SkrlJaxVecTrainer
from .skrl_torch import SkrlTorchVecEnvWrapper, SkrlTorchVecTrainer

__all__ = [
    "SkrlJaxVecEnvWrapper",
    "SkrlTorchVecEnvWrapper",
    "SkrlJaxVecTrainer",
    "SkrlTorchVecTrainer",
]
