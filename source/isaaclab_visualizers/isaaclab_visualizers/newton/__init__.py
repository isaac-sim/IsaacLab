# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton visualizer backends (GL and RTX).

This package keeps imports lazy so configuration-only imports do not pull in
the heavy viewer/runtime stack before Isaac Sim has finished bootstrapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .newton_visualizer_cfg import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg, NewtonVisualizerCfg

if TYPE_CHECKING:
    from .newton_visualizer import NewtonGLVisualizer, NewtonRTXVisualizer, NewtonVisualizer

__all__ = [
    # Base config (shared fields, not directly instantiable as a visualizer)
    "NewtonVisualizerCfg",
    # GL backend
    "NewtonGLVisualizer",
    "NewtonGLVisualizerCfg",
    # RTX backend
    "NewtonRTXVisualizer",
    "NewtonRTXVisualizerCfg",
]


def __getattr__(name: str):
    if name in ("NewtonVisualizer", "NewtonGLVisualizer", "NewtonRTXVisualizer"):
        from .newton_visualizer import NewtonGLVisualizer, NewtonRTXVisualizer, NewtonVisualizer

        if name == "NewtonGLVisualizer":
            return NewtonGLVisualizer
        if name == "NewtonRTXVisualizer":
            return NewtonRTXVisualizer
        import warnings

        warnings.warn(
            "NewtonVisualizer is deprecated and will be removed in a future release. Use NewtonGLVisualizer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return NewtonGLVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
