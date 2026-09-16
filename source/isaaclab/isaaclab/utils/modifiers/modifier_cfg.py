# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.utils import config_field

if TYPE_CHECKING:
    from .modifier import DigitalFilter, Integrator
    from .modifier_base import ModifierBase


@dataclass
class ModifierCfg:
    """Configuration parameters for function and class modifiers."""

    func: Callable[..., torch.Tensor] | type[ModifierBase] | str = config_field(MISSING)
    """Function or :class:`ModifierBase` class used by the modifier.

    Functions must take a tensor as their first argument. Classes must inherit from :class:`ModifierBase`; the
    observation manager constructs them with the configuration, observation dimensions, and device.
    """

    params: dict[str, Any] = config_field(dict())
    """Parameters used by the modifier. Defaults to an empty dictionary.

    Function modifiers receive them as keyword arguments on each call. Class modifiers access them through this
    configuration during construction; they are not forwarded to the constructed instance.
    """


@dataclass
class DigitalFilterCfg(ModifierCfg):
    """Configuration parameters for a digital filter modifier.

    For more information, please check the :class:`DigitalFilter` class.
    """

    func: type[DigitalFilter] | str = config_field("{DIR}.modifier:DigitalFilter")
    """The digital filter function to be called for applying the filter."""

    A: list[float] = config_field(MISSING)
    """The coefficients corresponding the the filter's response to past outputs.

    These correspond to the weights of the past outputs of the filter. The first element is the coefficient
    for the output at the previous time step, the second element is the coefficient for the output at two
    time steps ago, and so on.

    It is the denominator coefficients of the transfer function of the filter.
    """

    B: list[float] = config_field(MISSING)
    """The coefficients corresponding the the filter's response to current and past inputs.

    These correspond to the weights of the current and past inputs of the filter. The first element is the
    coefficient for the current input, the second element is the coefficient for the input at the previous
    time step, and so on.

    It is the numerator coefficients of the transfer function of the filter.
    """


@dataclass
class IntegratorCfg(ModifierCfg):
    """Configuration parameters for an integrator modifier.

    For more information, please check the :class:`Integrator` class.
    """

    func: type[Integrator] | str = config_field("{DIR}.modifier:Integrator")
    """The integrator function to be called for applying the integrator."""

    dt: float = config_field(MISSING)
    """The time step of the integrator."""
