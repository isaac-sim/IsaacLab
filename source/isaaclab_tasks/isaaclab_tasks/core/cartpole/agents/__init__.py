# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from isaaclab.utils.io import load_yaml

from isaaclab_tasks.utils import preset


def rl_games_camera_cfg():
    """Load the RL-Games configuration family keyed by camera pipeline presets.

    Image presets use the convolutional policy. ``lidar`` reuses the feature-pipeline MLP because its
    observation is the joint state plus a point cloud, not an image.
    """
    agent_dir = Path(__file__).parent
    feature_cfg = load_yaml(agent_dir / "rl_games_manager_feature_ppo_cfg.yaml")
    lidar_cfg = load_yaml(agent_dir / "rl_games_manager_feature_ppo_cfg.yaml")
    lidar_config = lidar_cfg["params"]["config"]
    lidar_config["name"] = "cartpole_lidar"
    # The feature config's fixed minibatch is larger than a small test rollout. Scale it with the
    # environment count so horizon_length * num_envs always divides the minibatch.
    lidar_config.pop("minibatch_size")
    lidar_config["minibatch_size_per_env"] = 4
    return preset(
        default=load_yaml(agent_dir / "rl_games_camera_ppo_cfg.yaml"),
        resnet18=feature_cfg,
        theia_tiny=feature_cfg,
        lidar=lidar_cfg,
    )
