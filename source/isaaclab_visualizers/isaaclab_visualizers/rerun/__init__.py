# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rerun visualizer backend.

This package keeps imports lazy so configuration-only imports avoid pulling the
full runtime backend before it is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .rerun_visualizer_cfg import RerunVisualizerCfg

if TYPE_CHECKING:
    from .rerun_visualizer import RerunVisualizer

__all__ = ["RerunVisualizer", "RerunVisualizerCfg"]


def __getattr__(name: str):
    if name == "RerunVisualizer":
        from .rerun_visualizer import RerunVisualizer

        return RerunVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
