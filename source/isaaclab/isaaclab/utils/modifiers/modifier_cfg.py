# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

from isaaclab.utils import configclass

from . import modifier


@configclass
class ModifierCfg:
    """Configuration parameters modifiers"""

    func: Callable[..., torch.Tensor] = MISSING
    """Function or callable class used by modifier.

    The function must take a torch tensor as the first argument. The remaining arguments are specified
    in the :attr:`params` attribute.

    It also supports `callable classes <https://docs.python.org/3/reference/datamodel.html#object.__call__>`_,
    i.e. classes that implement the ``__call__()`` method. In this case, the class should inherit from the
    :class:`ModifierBase` class and implement the required methods.
    """

    params: dict[str, Any] = dict()
    """The parameters to be passed to the function or callable class as keyword arguments. Defaults to
    an empty dictionary."""


@configclass
class DigitalFilterCfg(ModifierCfg):
    """Configuration parameters for a digital filter modifier.

    For more information, please check the :class:`DigitalFilter` class.
    """

    func: type[modifier.DigitalFilter] = modifier.DigitalFilter
    """The digital filter function to be called for applying the filter."""

    A: list[float] = MISSING
    """The coefficients corresponding the the filter's response to past outputs.

    These correspond to the weights of the past outputs of the filter. The first element is the coefficient
    for the output at the previous time step, the second element is the coefficient for the output at two
    time steps ago, and so on.

    It is the denominator coefficients of the transfer function of the filter.
    """

    B: list[float] = MISSING
    """The coefficients corresponding the the filter's response to current and past inputs.

    These correspond to the weights of the current and past inputs of the filter. The first element is the
    coefficient for the current input, the second element is the coefficient for the input at the previous
    time step, and so on.

    It is the numerator coefficients of the transfer function of the filter.
    """


@configclass
class IntegratorCfg(ModifierCfg):
    """Configuration parameters for an integrator modifier.

    For more information, please check the :class:`Integrator` class.
    """

    func: type[modifier.Integrator] = modifier.Integrator
    """The integrator function to be called for applying the integrator."""

    dt: float = MISSING
    """The time step of the integrator."""


@configclass
class DelayedObservationCfg(ModifierCfg):
    """Configuration parameters for a delayed observation modifier.

    For more information, please check the :class:`DelayedObservation` class.
    """

    func: type[modifier.DelayedObservation] = modifier.DelayedObservation
    """The delayed observation function to be called for applying the delay."""

    # Lag parameters
    min_lag: int = 0
    """The minimum lag (in number of policy steps) to be applied to the observations. Defaults to 0."""

    max_lag: int = 3
    """The maximum lag (in number of policy steps) to be applied to the observations.

    This value must be greater than or equal to :attr:`min_lag`.
    """

    per_env: bool = True
    """Whether to use a separate lag for each environment."""

    hold_prob: float = 0.0
    """The probability of holding the previous lag when updating the lag."""

    # multi-rate emulation parameters (optional)
    update_period: int = 1
    """The period (in number of policy steps) at which the lag is updated.

    If set to 0, the lag is sampled once at the beginning and remains constant throughout the simulation.
    If set to a positive integer, the lag is updated every `update_period` policy steps. Defaults to 1.

    This value must be less than or equal to :attr:`max_lag` if it is greater than 0.
    """

    per_env_phase: bool = True
    """Whether to use a separate phase for each environment when updating the lag.

    If set to True, each environment will have its own phase when updating the lag. If set to False, all
    environments will share the same phase. Defaults to True.
    """
