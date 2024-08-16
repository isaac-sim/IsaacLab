# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

from omni.isaac.lab.utils import configclass

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
    """Configuration parameters for a digital filter modifier"""

    func: type[modifier.DigitalFilter] = modifier.DigitalFilter
    """The digital filter function to be called for applying the filter."""

    A: list[float] = MISSING
    """The denominator coefficients of the digital filter."""

    B: list[float] = MISSING
    """The numerator coefficients of the digital filter."""


@configclass
class IntegratorCfg(ModifierCfg):
    """Configuration parameters for an integrator modifier"""

    func: type[modifier.Integrator] = modifier.Integrator
    """The integrator function to be called for applying the integrator."""

    dt: float = MISSING
    """The time step of the integrator."""
