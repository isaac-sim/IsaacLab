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
    """Function or Callable class used by modifier."""

    params: dict[str, Any] = dict()
    """The parameters to be passed to the function or Callable class as keyword arguments. Defaults to an empty dict."""
