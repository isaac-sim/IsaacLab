# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Submodule containing configuration instances for commonly used robots.
"""

from .anydrive import Anydrive3LSTMCfg, Anydrive3SimpleCfg

__all__ = [
    # ANYmal actuators
    "Anydrive3LSTMCfg",
    "Anydrive3SimpleCfg",
]
