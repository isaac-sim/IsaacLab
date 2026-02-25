# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##
# Configuration for different assets.
##

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "gelsight": ["GELSIGHT_R15_CFG", "GELSIGHT_MINI_CFG"],
        "velodyne": ["VELODYNE_VLP_16_RAYCASTER_CFG"],
    },
)
