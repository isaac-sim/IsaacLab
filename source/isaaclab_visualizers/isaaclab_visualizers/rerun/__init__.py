# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rerun visualizer backend.

.. deprecated:: 3.2.0
    The Rerun visualizer will be removed in Isaac Lab 3.3. Use the Viser visualizer
    (:mod:`isaaclab_visualizers.viser`) instead.

This package keeps imports lazy so configuration-only imports avoid pulling the
full runtime backend before it is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .rerun_visualizer_cfg import RerunBackendCfg, RerunVisualizerCfg

if TYPE_CHECKING:
    from .rerun_visualizer import RerunVisualizer

__all__ = ["RerunVisualizer", "RerunBackendCfg", "RerunVisualizerCfg"]


def __getattr__(name: str):
    if name == "RerunVisualizer":
        from .rerun_visualizer import RerunVisualizer

        return RerunVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
