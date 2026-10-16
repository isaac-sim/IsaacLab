# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from ..configclass import configclass

if TYPE_CHECKING:
    from .dict_tensor_processor import DictTensorProcessor


@configclass
class DictTensorProcessorCfg:
    """Configure one dictionary processor and declare its producer dependencies.

    Names are actual dictionary keys, with no separate operation roles. Cameras use these
    declarations before renderer initialization. The class receives ``(cfg, data)`` and
    owns its output storage, state, and lifecycle, just like a single-tensor processor.
    """

    class_type: type[DictTensorProcessor] | str = MISSING
    """Processor class, or a resolvable import string."""

    inputs: list[str] = MISSING
    """Dictionary keys required from the producer or preceding processors."""

    outputs: list[str] = MISSING
    """Dictionary keys produced or replaced by this processor."""

    params: dict[str, Any] = {}
    """Processor-specific settings, accessible through ``cfg.params``.

    Configurations are copied. Explicitly shared device resources must preserve identity
    through ``__deepcopy__``; do not place raw Warp allocations in this dictionary.
    """
