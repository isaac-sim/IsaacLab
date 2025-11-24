# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from .utils.lazy_importer import LazyImporter

# Lazily import heavy dependencies so importing ``isaaclab`` does not
# immediately import them. Currently only ``isaacsim`` is required.
isaacsim = LazyImporter("isaacsim")
isaacsim_core_utils_viewports = LazyImporter("isaacsim.core.utils.viewports")
omni = LazyImporter("omni")
omni_metrics_assembler_core = LazyImporter("omni.metrics.assembler.core")
omni_kit_viewport_utility = LazyImporter("omni.kit.viewport.utility")

__all__ = [
    "isaacsim",
    "isaacsim_core_utils_viewports",
    "omni",
    "omni_metrics_assembler_core",
    "omni_kit_viewport_utility",
]
