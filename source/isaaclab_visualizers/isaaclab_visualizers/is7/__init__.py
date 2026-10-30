# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim 7 viewer visualizer backend.

This package keeps imports lazy so configuration-only imports avoid pulling the
Isaac Sim 7 libraries before they are needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .is7_visualizer_cfg import IS7VisualizerCfg

if TYPE_CHECKING:
    from .is7_visualizer import IS7Visualizer

__all__ = ["IS7Visualizer", "IS7VisualizerCfg"]


def __getattr__(name: str):
    if name == "IS7Visualizer":
        from .is7_visualizer import IS7Visualizer

        return IS7Visualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
