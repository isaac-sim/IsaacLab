# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task presentation settings without importing renderer runtimes."""

from isaaclab_visualizers.newton import NewtonRTXVisualizerCfg

from isaaclab.utils import configclass


@configclass
class CoffeeRTXVisualizerCfg(NewtonRTXVisualizerCfg):
    """Borrow the scene while optionally replacing its particle display with grains."""

    class_type: str = "{DIR}.presentation:CoffeeRTXVisualizer"
    show_particles: bool = True
    """Show native scene particles; disable when passive grains and a surface replace them."""
