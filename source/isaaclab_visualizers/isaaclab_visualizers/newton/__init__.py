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
    from .newton_visualizer import NewtonGLVisualizer, NewtonRTXVisualizer

__all__ = [
    # Deprecated GL configuration
    "NewtonVisualizerCfg",
    # GL backend
    "NewtonGLVisualizer",
    "NewtonGLVisualizerCfg",
    # RTX backend
    "NewtonRTXVisualizer",
    "NewtonRTXVisualizerCfg",
]


def __getattr__(name: str):
    if name == "NewtonRTXVisualizer":
        from .newton_visualizer import NewtonRTXVisualizer

        return NewtonRTXVisualizer
    if name in ("NewtonVisualizer", "NewtonGLVisualizer"):
        from .newton_visualizer import NewtonGLVisualizer

        if name == "NewtonGLVisualizer":
            return NewtonGLVisualizer
        import warnings

        warnings.warn(
            "NewtonVisualizer is deprecated and will be removed in a future release. Use NewtonGLVisualizer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return NewtonGLVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
