# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

from omni.isaac.lab.utils import configclass


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
    """The parameters to be passed to the function or Callable class as keyword arguments. Defaults to
    an empty dictionary."""
