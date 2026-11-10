# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from isaaclab.utils.io import load_yaml

from isaaclab_tasks.utils import preset


def _rl_games_lidar_cfg(name: str) -> dict:
    """Load the feature-pipeline MLP and scale its minibatch with the environment count.

    Args:
        name: Experiment name stored in the RL-Games config.

    Returns:
        RL-Games configuration for a vector observation.
    """
    lidar_cfg = load_yaml(Path(__file__).parent / "rl_games_manager_feature_ppo_cfg.yaml")
    lidar_config = lidar_cfg["params"]["config"]
    lidar_config["name"] = name
    # The feature config's fixed minibatch is larger than a small test rollout. Scale it with the
    # environment count so horizon_length * num_envs always divides the minibatch.
    lidar_config.pop("minibatch_size")
    lidar_config["minibatch_size_per_env"] = 4
    return lidar_cfg


def rl_games_camera_cfg():
    """Load the RL-Games configuration family keyed by camera pipeline presets.

    Image presets use the convolutional policy. ``lidar`` reuses the feature-pipeline MLP because its
    observation is the joint state plus a point cloud, not an image.
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

    Image presets use the convolutional policy. ``lidar`` reuses the feature-pipeline MLP because its
    observation is the joint state plus a point cloud, not an image.
    """
    agent_dir = Path(__file__).parent
    return preset(
        default=load_yaml(agent_dir / "rl_games_camera_ppo_cfg.yaml"),
        lidar=_rl_games_lidar_cfg("cartpole_lidar_direct"),
    )


def skrl_direct_camera_cfg():
    """Load the SKRL configuration family for the direct cartpole camera task.

    Image presets use the convolutional policy. ``lidar`` uses an MLP because its observation is the
    joint state plus a point cloud.
    """
    agent_dir = Path(__file__).parent
    return preset(
        default=load_yaml(agent_dir / "skrl_direct_camera_ppo_cfg.yaml"),
        lidar=load_yaml(agent_dir / "skrl_direct_lidar_ppo_cfg.yaml"),
    )
