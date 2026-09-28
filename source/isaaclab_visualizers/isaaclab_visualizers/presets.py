# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer choices resolved before simulation startup."""

from typing import ClassVar

from isaaclab.utils import configclass
from isaaclab.utils.presets import PresetCfg
from isaaclab.visualizers import VisualizerCfg

from .kit import KitVisualizerCfg
from .newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg
from .rerun import RerunVisualizerCfg
from .viser import ViserVisualizerCfg


@configclass
class MultiBackendVisualizerCfg(PresetCfg):
    """Built-in visualizers selected with ``visualizer=NAME``; no viewer opens by default."""

    kit: KitVisualizerCfg = KitVisualizerCfg()
    newton_gl: NewtonGLVisualizerCfg = NewtonGLVisualizerCfg()
    newton_rtx: NewtonRTXVisualizerCfg = NewtonRTXVisualizerCfg()
    rerun: RerunVisualizerCfg = RerunVisualizerCfg()
    viser: ViserVisualizerCfg = ViserVisualizerCfg()
    none: list[VisualizerCfg] = []
    default: None = None

    _aliases: ClassVar[dict[str, str]] = {"newton": "newton_gl"}
