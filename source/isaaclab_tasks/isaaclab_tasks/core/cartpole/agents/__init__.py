# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from isaaclab.utils.io import load_yaml

from isaaclab_tasks.utils import preset


def rl_games_camera_cfg():
    """Load the RL-Games configuration family keyed by camera pipeline presets."""
    agent_dir = Path(__file__).parent
    feature_cfg = load_yaml(agent_dir / "rl_games_manager_feature_ppo_cfg.yaml")
    return preset(
        default=load_yaml(agent_dir / "rl_games_camera_ppo_cfg.yaml"),
        resnet18=feature_cfg,
        theia_tiny=feature_cfg,
    )
