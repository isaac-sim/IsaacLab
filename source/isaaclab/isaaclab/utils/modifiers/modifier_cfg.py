# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .modifier import DigitalFilter, Integrator
    from .modifier_base import ModifierBase


@configclass
class ModifierCfg:
    """Configuration parameters for stateless function modifiers."""

    func: Callable[..., torch.Tensor] = MISSING
    """Function used by the modifier.

    The function must take a torch tensor as the first argument. The remaining arguments are specified
    in the :attr:`params` attribute.
    """

    params: dict[str, Any] = dict()
    """The parameters passed to the function as keyword arguments. Defaults to an empty dictionary."""


@configclass
class ModifierBaseCfg(ModifierCfg):
    """Configuration parameters for stateful class modifiers implemented by :class:`ModifierBase`."""

    func: type[ModifierBase] | str = MISSING
    """Class implementing the modifier.

    The observation manager constructs this class with the configuration, observation dimensions, and device.
    """

    params: dict[str, Any] = dict()
    """Parameters available to the constructor through this configuration.

    These values are not forwarded when calling the constructed modifier instance.
    """


@configclass
class DigitalFilterCfg(ModifierBaseCfg):
    """Configuration parameters for a digital filter modifier.

    For more information, please check the :class:`DigitalFilter` class.
    """

    func: type[DigitalFilter] | str = "{DIR}.modifier:DigitalFilter"
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
class IntegratorCfg(ModifierBaseCfg):
    """Configuration parameters for an integrator modifier.

    For more information, please check the :class:`Integrator` class.
    """

    func: type[Integrator] | str = "{DIR}.modifier:Integrator"
    """The integrator function to be called for applying the integrator."""

    dt: float = MISSING
    """The time step of the integrator."""
