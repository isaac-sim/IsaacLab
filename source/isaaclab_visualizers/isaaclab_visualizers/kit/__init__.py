# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit visualizer backend (Isaac Sim viewport).

This package keeps imports lazy so configuration-only imports avoid pulling the
full runtime backend before it is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .kit_visualizer_cfg import KitVisualizerCfg

if TYPE_CHECKING:
    from .kit_visualizer import KitVisualizer

__all__ = ["KitVisualizer", "KitVisualizerCfg"]


def __getattr__(name: str):
    if name == "KitVisualizer":
        from .kit_visualizer import KitVisualizer

        return KitVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
