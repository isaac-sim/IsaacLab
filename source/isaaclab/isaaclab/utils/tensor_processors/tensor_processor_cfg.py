# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from .. import configclass

if TYPE_CHECKING:
    from .processors import DigitalFilter, Integrator
    from .tensor_processor import TensorProcessor, TensorProcessorData


@configclass
class TensorProcessorCfg:
    """Configuration parameters for function and class tensor processors."""

    func: Callable[..., TensorProcessorData] | type[TensorProcessor] | str = MISSING
    """Function or :class:`TensorProcessor` class used by the processor.

    Functions take their input data as the first argument. Classes inherit from :class:`TensorProcessor`.
    Each implementation defines the tensor or nested collection structures it accepts and returns.
    Construct classes with ``cfg.func(cfg, ...)``, supplying the arguments their constructor requires.
    Import strings resolve lazily through ``configclass``. The observation manager constructs
    single-tensor processors with the configuration, observation dimensions, and device.
    """

    params: dict[str, Any] = {}
    """Parameters used by the processor. Defaults to an empty dictionary.

    Functions receive them as keyword arguments on each call. Classes access them through this
    configuration during construction; they are not forwarded to the constructed instance.
    """

    inputs: list[str] = []
    """Named input keys for callers that route dictionaries. Empty means unspecified.

    This is metadata only; the configuration does not select or validate input data.
    """

    outputs: list[str] = []
    """Named output keys for callers that route dictionaries. Empty means unspecified.

    This is metadata only; the processor implementation defines the returned structure.
    """


@configclass
class DigitalFilterCfg(TensorProcessorCfg):
    """Configuration parameters for a digital filter processor.

    For more information, please check the :class:`DigitalFilter` class.
    """

    func: type[DigitalFilter] | str = "{DIR}.processors:DigitalFilter"
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
class IntegratorCfg(TensorProcessorCfg):
    """Configuration parameters for an integrator processor.

    For more information, please check the :class:`Integrator` class.
    """

    func: type[Integrator] | str = "{DIR}.processors:Integrator"
    """The integrator function to be called for applying the integrator."""

    dt: float = MISSING
    """The time step of the integrator."""
