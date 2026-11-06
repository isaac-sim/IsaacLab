# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native processor configuration (experimental)."""

from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

from isaaclab.utils import configclass


@configclass
class TensorProcessorCfg:
    """Configuration parameters for Warp-native tensor processors.

    Experimental fork of :class:`isaaclab.utils.tensor_processors.TensorProcessorCfg` adapted for the
    Warp-first calling convention where processor functions operate **in-place** on a
    ``wp.array`` buffer and return ``None``.
    """

    func: Callable[..., None] = MISSING
    """Function or callable class used by processor.

    The function must take a ``wp.array`` as the first argument and operate on it
    **in-place** (no return value).  The remaining arguments are specified in the
    :attr:`params` attribute.

    It also supports callable classes that implement ``__call__()``.  In this case the
    class should inherit from :class:`TensorProcessor` and implement the required methods.
    """

    params: dict[str, Any] = dict()
    """The parameters to be passed to the function or callable class as keyword arguments.

    Defaults to an empty dictionary.
    """
