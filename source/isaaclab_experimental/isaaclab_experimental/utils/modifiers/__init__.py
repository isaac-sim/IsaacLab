# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated import path for Warp-native tensor processors."""

import sys
import warnings

from .. import tensor_processors as _processors

__all__ = ["ModifierBase", "ModifierCfg", "bias", "clip", "scale"]

warnings.warn(
    "isaaclab_experimental.utils.modifiers is deprecated; use isaaclab_experimental.utils.tensor_processors."
    " ModifierBase and ModifierCfg are now TensorProcessor and TensorProcessorCfg.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    aliases = {"ModifierBase": "TensorProcessor", "ModifierCfg": "TensorProcessorCfg"}
    return getattr(_processors, aliases.get(name, name))


def __dir__():
    return __all__


for _name in ("modifier", "modifier_base", "modifier_cfg"):
    sys.modules[f"{__name__}.{_name}"] = sys.modules[__name__]
    setattr(sys.modules[__name__], _name, sys.modules[__name__])
