# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton visualizer backend (OpenGL and RTX rendering).

This package keeps imports lazy so configuration-only imports do not pull in
the heavy viewer/runtime stack before Isaac Sim has finished bootstrapping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .newton_visualizer_cfg import (
    NewtonGLBackendCfg,
    NewtonGLVisualizerCfg,
    NewtonRecordingCfg,
    NewtonVisualizerCfg,
)

if TYPE_CHECKING:
    from .newton_visualizer import NewtonGLVisualizer
    from .particle_surface import ParticleSurfaceRenderer

__all__ = [
    # Deprecated combined configurations
    "NewtonGLVisualizerCfg",
    "NewtonVisualizerCfg",
    # GL backend
    "NewtonGLVisualizer",
    "NewtonGLBackendCfg",
    "NewtonGLVisualizerCfg",
    "NewtonRecordingCfg",
    # MPM particle-surface rendering
    "ParticleSurfaceRenderer",
]


def __getattr__(name: str):
    if name == "ParticleSurfaceRenderer":
        from .particle_surface import ParticleSurfaceRenderer

        return ParticleSurfaceRenderer
    if name in ("NewtonRTXVisualizer", "NewtonRTXBackendCfg", "NewtonRTXVisualizerCfg"):
        raise AttributeError(
            f"{name} was removed. Use NewtonGLBackendCfg, which renders with OpenGL or RTX;"
            " set initial_renderer='rtx' to start in RTX."
        )
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
