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
