# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton OpenGL visualizer backend.

This package keeps imports lazy so configuration-only imports do not pull in
the heavy viewer/runtime stack before Isaac Sim has finished bootstrapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .newton_visualizer_cfg import NewtonVisualizerCfg

if TYPE_CHECKING:
    from .newton_visualizer import NewtonVisualizer

__all__ = ["NewtonVisualizer", "NewtonVisualizerCfg"]


def __getattr__(name: str):
    if name == "NewtonVisualizer":
        from .newton_visualizer import NewtonVisualizer

        return NewtonVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
