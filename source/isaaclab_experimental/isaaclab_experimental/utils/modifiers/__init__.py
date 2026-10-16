# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated import path for Warp-native tensor processors."""

from isaaclab.utils.modifiers import _install_aliases

from .. import tensor_processors

__all__ = ["ModifierBase", "ModifierCfg", "bias", "clip", "scale"]

_install_aliases(__name__, tensor_processors)
