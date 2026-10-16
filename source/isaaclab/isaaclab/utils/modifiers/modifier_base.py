# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated name for the single-tensor processor interface."""

from ..tensor_processors.tensor_processor import TensorProcessor as ModifierBase

__all__ = ["ModifierBase"]
