# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native noise functions and models (experimental).

Each noise function takes a ``wp.array`` as its first argument and operates **in-place**
via ``wp.launch``.  The calling convention mirrors the stable noise interface::

    noise_cfg.func(data_wp, noise_cfg) -> None

Random noise kernels (gaussian, uniform) consume the shared per-env Warp RNG state
(``rng_state_wp``) that is set on the config at manager prep time from
``env.rng_state_wp``.  See :func:`initialize_rng_state` in
``isaaclab_experimental.envs.manager_based_env_warp`` for the initialization pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    from . import noise_cfg

##
# Operation mode mapping.
##

_OPERATION_MAP: dict[str, int] = {"add": 0, "scale": 1, "abs": 2}

##
# Noise as functions.
##


# -- constant -----------------------------------------------------------------


@wp.kernel
def _apply_constant_noise(
    out: wp.array(dtype=wp.float32, ndim=2),
    bias: wp.float32,
    operation: wp.int32,
):
    env_id = wp.tid()
    for j in range(out.shape[1]):
        if operation == 0:
            out[env_id, j] = out[env_id, j] + bias
        elif operation == 1:
            out[env_id, j] = out[env_id, j] * bias
        else:
            out[env_id, j] = bias


def constant_noise(data: wp.array, cfg: noise_cfg.ConstantNoiseCfg) -> None:
    """Applies a constant noise bias to a given data set in-place.

    Warp-native drop-in replacement for :func:`isaaclab.utils.noise.constant_noise`.

    Args:
        data: The data buffer to modify. Shape ``(num_envs, D)``.
        cfg: The configuration parameters for constant noise.
    """
    wp.launch(
        _apply_constant_noise,
        dim=data.shape[0],
        inputs=[data, float(cfg.bias), _OPERATION_MAP[cfg.operation]],
        device=data.device,
    )


# -- uniform ------------------------------------------------------------------


@wp.kernel
def _apply_uniform_noise(
    out: wp.array(dtype=wp.float32, ndim=2),
    rng_state: wp.array(dtype=wp.uint32),
    n_min: wp.float32,
    n_max: wp.float32,
    operation: wp.int32,
):
    env_id = wp.tid()
    state = rng_state[env_id]
    for j in range(out.shape[1]):
        n = wp.randf(state, n_min, n_max)
        if operation == 0:
            out[env_id, j] = out[env_id, j] + n
        elif operation == 1:
            out[env_id, j] = out[env_id, j] * n
        else:
            out[env_id, j] = n
    rng_state[env_id] = state


def uniform_noise(data: wp.array, cfg: noise_cfg.UniformNoiseCfg) -> None:
    """Applies uniform noise to a given data set in-place.

    Warp-native drop-in replacement for :func:`isaaclab.utils.noise.uniform_noise`.

    Args:
        data: The data buffer to modify. Shape ``(num_envs, D)``.
        cfg: The configuration parameters for uniform noise.
    """
    wp.launch(
        _apply_uniform_noise,
        dim=data.shape[0],
        inputs=[data, cfg.rng_state_wp, float(cfg.n_min), float(cfg.n_max), _OPERATION_MAP[cfg.operation]],
        device=data.device,
    )


# -- gaussian -----------------------------------------------------------------


@wp.kernel
def _apply_gaussian_noise(
    out: wp.array(dtype=wp.float32, ndim=2),
    rng_state: wp.array(dtype=wp.uint32),
    mean: wp.float32,
    std: wp.float32,
    operation: wp.int32,
):
    env_id = wp.tid()
    state = rng_state[env_id]
    for j in range(out.shape[1]):
        n = mean + std * wp.randn(state)
        if operation == 0:
            out[env_id, j] = out[env_id, j] + n
        elif operation == 1:
            out[env_id, j] = out[env_id, j] * n
        else:
            out[env_id, j] = n
    rng_state[env_id] = state


def gaussian_noise(data: wp.array, cfg: noise_cfg.GaussianNoiseCfg) -> None:
    """Applies gaussian noise to a given data set in-place.

    Warp-native drop-in replacement for :func:`isaaclab.utils.noise.gaussian_noise`.

    Args:
        data: The data buffer to modify. Shape ``(num_envs, D)``.
        cfg: The configuration parameters for gaussian noise.
    """
    wp.launch(
        _apply_gaussian_noise,
        dim=data.shape[0],
        inputs=[data, cfg.rng_state_wp, float(cfg.mean), float(cfg.std), _OPERATION_MAP[cfg.operation]],
        device=data.device,
    )


##
# Noise models as classes.
##


class NoiseModel:
    """Warp-native base class for noise models.

    Experimental fork of :class:`isaaclab.utils.noise.NoiseModel` adapted for the
    Warp-first calling convention where noise is applied **in-place** on ``wp.array``.
    """

    def __init__(self, noise_model_cfg: noise_cfg.NoiseModelCfg, num_envs: int, device: str):
        """Initialize the noise model.

        Args:
            noise_model_cfg: The noise configuration to use.
            num_envs: The number of environments.
            device: The device to use for the noise model.
        """
        self._noise_model_cfg = noise_model_cfg
        self._num_envs = num_envs
        self._device = device

    def reset(self, env_mask: wp.array | None = None):
        """Reset the noise model.

        This method can be implemented by derived classes to reset the noise model.
        This is useful when implementing temporal noise models such as random walk.

        Args:
            env_mask: Boolean env mask of shape ``(num_envs,)`` selecting environments
                to reset. Defaults to None, in which case all environments are
                considered.
        """
        pass

    def __call__(self, data: wp.array) -> None:
        """Apply the noise to the data in-place.

        Args:
            data: The data to apply the noise to. Shape is ``(num_envs, ...)``.
        """
        self._noise_model_cfg.noise_cfg.func(data, self._noise_model_cfg.noise_cfg)
