# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coffee Kit settings without importing the renderer before app startup."""

from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg
from isaaclab_visualizers.kit import KitVisualizerCfg

from isaaclab.utils import configclass


@configclass
class CoffeeKitVisualizerCfg(KitVisualizerCfg):
    """Coffee presentation with an optional render-only public kitchen."""

    class_type: str = "isaaclab_tasks.contrib.rizon_sharpa_coffee.kit_presentation:CoffeeKitVisualizer"
    backdrop: str = "kitchen"
    """Presentation backdrop: ``kitchen`` or the original ``studio``."""
    render_settings: IsaacRtxRendererGlobalSettingsCfg = IsaacRtxRendererGlobalSettingsCfg()
    """Global RTX settings for the Kit viewport and its recorded frames."""
