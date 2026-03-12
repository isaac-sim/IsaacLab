# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Viser visualizer backend.

This package keeps imports lazy so configuration-only imports avoid pulling the
full runtime backend before it is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .viser_visualizer_cfg import ViserVisualizerCfg

if TYPE_CHECKING:
    from .viser_visualizer import ViserVisualizer

__all__ = ["ViserVisualizer", "ViserVisualizerCfg"]


def __getattr__(name: str):
    if name == "ViserVisualizer":
        from .viser_visualizer import ViserVisualizer

        return ViserVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
