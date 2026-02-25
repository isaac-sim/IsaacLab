# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for rigid articulated assets."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "base_articulation": ["BaseArticulation"],
        "base_articulation_data": ["BaseArticulationData"],
        "articulation": ["Articulation"],
        "articulation_cfg": ["ArticulationCfg"],
        "articulation_data": ["ArticulationData"],
    },
)
