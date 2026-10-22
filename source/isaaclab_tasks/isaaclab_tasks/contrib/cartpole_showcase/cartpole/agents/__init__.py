# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from isaaclab.utils.io import load_yaml

from isaaclab_tasks.utils import preset


def skrl_cfg():
    """Load the SKRL configuration family keyed by the task's space presets."""
    configs = {
        path.stem.removeprefix("skrl_").removesuffix("_ppo_cfg"): load_yaml(path)
        for path in Path(__file__).parent.glob("skrl_*_ppo_cfg.yaml")
    }
    return preset(default=configs.pop("box_box"), **configs)
