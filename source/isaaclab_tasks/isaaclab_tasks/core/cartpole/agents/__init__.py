# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from isaaclab.utils.io import load_yaml

from isaaclab_tasks.utils import preset


def _rl_games_lidar_cfg(name: str) -> dict:
    """Load the camera CNN and name the run for the lidar range image.

    The camera config leaves input normalization off. The range image is already in ``[0, 1]``, and
    per-coordinate normalization would blow up the mostly-zero misses.

    Args:
        name: Experiment name stored in the RL-Games config.

    Returns:
        RL-Games configuration for the stacked lidar range image.
    """
    lidar_cfg = load_yaml(Path(__file__).parent / "rl_games_camera_ppo_cfg.yaml")
    lidar_cfg["params"]["config"]["name"] = name
    return lidar_cfg


def rl_games_camera_cfg():
    """Load the RL-Games configuration family keyed by camera pipeline presets.

    Image presets use the convolutional policy. ``lidar`` uses that same CNN on the stacked range image.
    """
    agent_dir = Path(__file__).parent
    feature_cfg = load_yaml(agent_dir / "rl_games_manager_feature_ppo_cfg.yaml")
    return preset(
        default=load_yaml(agent_dir / "rl_games_camera_ppo_cfg.yaml"),
        resnet18=feature_cfg,
        theia_tiny=feature_cfg,
        lidar=_rl_games_lidar_cfg("cartpole_lidar"),
    )


def rl_games_direct_camera_cfg():
    """Load the RL-Games configuration family for the direct cartpole camera task.

    Image presets use the convolutional policy. ``lidar`` uses that same CNN on the stacked range image.
    """
    agent_dir = Path(__file__).parent
    return preset(
        default=load_yaml(agent_dir / "rl_games_camera_ppo_cfg.yaml"),
        lidar=_rl_games_lidar_cfg("cartpole_lidar_direct"),
    )


def skrl_direct_camera_cfg():
    """Load the SKRL configuration family for the direct cartpole camera task.

    Image presets use the convolutional policy. ``lidar`` uses that same CNN on the stacked range image.
    """
    agent_dir = Path(__file__).parent
    return preset(
        default=load_yaml(agent_dir / "skrl_direct_camera_ppo_cfg.yaml"),
        lidar=load_yaml(agent_dir / "skrl_direct_lidar_ppo_cfg.yaml"),
    )
