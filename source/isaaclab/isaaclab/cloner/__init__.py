# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sam/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "cloner_cfg": ["TemplateCloneCfg"],
        "cloner_strategies": ["random", "sequential"],
        "cloner_utils": [
            "clone_from_template",
            "make_clone_plan",
            "usd_replicate",
            "filter_collisions",
            "grid_transforms",
        ],
    },
)
