# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sampling shared by core and backend event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from ...managers import EventTermCfg, ManagerTermBase
from ...utils import math as math_utils

if TYPE_CHECKING:
    from ..manager_based_env import ManagerBasedEnv


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | slice | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    if dim_0_ids is None:
        dim_0_ids = slice(None)
    n_dim_0 = len(range(data.shape[0])[dim_0_ids]) if isinstance(dim_0_ids, slice) else len(dim_0_ids)
    n_dim_1 = len(range(data.shape[1])[dim_1_ids]) if isinstance(dim_1_ids, slice) else len(dim_1_ids)
    if not isinstance(dim_0_ids, slice) and not isinstance(dim_1_ids, slice):
        dim_0_ids = dim_0_ids[:, None]

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def _validate_scale_range(
    params: tuple[float, float] | None,
    name: str,
    *,
    allow_negative: bool = False,
    allow_zero: bool = True,
) -> None:
    """
    Validates a (low, high) tuple used in scale-based randomization.

    This function ensures the tuple follows expected rules when applying a 'scale'
    operation. It performs type and value checks, optionally allowing negative or
    zero lower bounds.

    Args:
        params (tuple[float, float] | None): The (low, high) range to validate. If None,
            validation is skipped.
        name (str): The name of the parameter being validated, used for error messages.
        allow_negative (bool, optional): If True, allows the lower bound to be negative.
            Defaults to False.
        allow_zero (bool, optional): If True, allows the lower bound to be zero.
            Defaults to True.

    Raises:
        TypeError: If `params` is not a tuple of two numbers.
        ValueError: If the lower bound is negative or zero when not allowed.
        ValueError: If the upper bound is less than the lower bound.

    Example:
        _validate_scale_range((0.5, 1.5), "mass_scale")
    """
    if params is None:  # caller didn’t request randomisation for this field
        return
    low, high = params
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise TypeError(f"{name}: expected (low, high) to be a tuple of numbers, got {params}.")
    if not allow_negative and not allow_zero and low <= 0:
        raise ValueError(f"{name}: lower bound must be > 0 when using the 'scale' operation (got {low}).")
    if not allow_negative and allow_zero and low < 0:
        raise ValueError(f"{name}: lower bound must be ≥ 0 when using the 'scale' operation (got {low}).")
    if high < low:
        raise ValueError(f"{name}: upper bound ({high}) must be ≥ lower bound ({low}).")


class _GravityRandomization(ManagerTermBase):
    """Shared distribution state; native terms own gravity scope and writes."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv, device: str):
        super().__init__(cfg, env)
        self._sampling_device = device
        self._distribution = cfg.params.get("distribution", "uniform")
        if self._distribution not in ("uniform", "log_uniform", "gaussian"):
            raise NotImplementedError(f"Unknown gravity distribution: {self._distribution!r}.")
        if cfg.params["operation"] not in ("add", "scale", "abs"):
            raise NotImplementedError(f"Unknown gravity operation: {cfg.params['operation']!r}.")
        self._last_gravity_params = None

    def _sample_gravity(
        self,
        gravity: torch.Tensor,
        params: tuple[list[float], list[float]],
        operation: Literal["add", "scale", "abs"],
    ) -> torch.Tensor:
        # Curricula can change bounds without rebuilding the term.
        key = (tuple(params[0]), tuple(params[1]))
        if key != self._last_gravity_params:
            self._last_gravity_params = key
            self._bounds = torch.tensor(params, device=self._sampling_device, dtype=torch.float32)
        return _randomize_prop_by_op(
            gravity.clone(),
            (self._bounds[0], self._bounds[1]),
            None,
            slice(None),
            operation=operation,
            distribution=self._distribution,
        )
