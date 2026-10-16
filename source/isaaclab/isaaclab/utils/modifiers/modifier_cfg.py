# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated import path for tensor processor configurations."""

from ..tensor_processors.tensor_processor_cfg import DigitalFilterCfg, IntegratorCfg
from ..tensor_processors.tensor_processor_cfg import TensorProcessorCfg as ModifierCfg

__all__ = ["ModifierCfg", "DigitalFilterCfg", "IntegratorCfg"]
