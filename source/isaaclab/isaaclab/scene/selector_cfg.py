# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for scene env/view selectors."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .selector import Selector


@configclass
class SelectorTermCfg:
    """Configuration for one named scene selector term."""

    func: callable = MISSING
    """Function that resolves scene asset names from the raw scene asset cfgs."""

    params: dict[str, Any] = {}
    """Additional keyword parameters passed to :attr:`func`."""


@configclass
class SelectorCfg:
    """Configuration for runtime env/view selectors.

    Selector terms are declared as config attributes whose values are
    :class:`SelectorTermCfg`. Each term function receives the raw scene asset
    config dictionary and returns one or more asset names that define the
    selected environment rows.
    """

    class_type: type[Selector] | str = "{DIR}.selector:Selector"
    """Selector implementation class."""
