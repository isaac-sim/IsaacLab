# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pretrained-checkpoint declarations for camera-based Cartpole tasks."""

from typing import cast

from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg
from isaaclab.sensors import CameraCfg

from isaaclab_rl.utils.pretrained_checkpoint import (
    PRETRAINED_CHECKPOINT_DEFAULT_VARIANT,
    PretrainedCheckpointCfg,
    PretrainedCheckpointSetCfg,
)

import isaaclab_tasks.core.cartpole.mdp as mdp

_POLICY_IMAGE_SIZE = (96, 96)
_POLICY_FRAME_STACK = 2
_POLICY_CHANNELS = {"depth": 1, "rgb": 3}

_CHECKPOINTS = (
    PretrainedCheckpointCfg(workflow="rsl_rl"),
    PretrainedCheckpointCfg(workflow="rl_games", smoke_num_envs=32),
    PretrainedCheckpointCfg(workflow="rl_games", variant="depth", training_presets=("depth",), smoke_num_envs=32),
)


def _raw_camera_policy_variant(
    camera_cfg: CameraCfg,
    data_type: str,
    frame_stack: int,
    channels: int | None = None,
) -> str | None:
    """Return the variant for a compatible raw-camera policy contract."""
    if len(camera_cfg.data_types) != 1:
        return None
    if camera_cfg.data_types[0] != data_type:
        return None
    if (camera_cfg.height, camera_cfg.width) != _POLICY_IMAGE_SIZE or max(1, frame_stack) != _POLICY_FRAME_STACK:
        return None
    if channels is not None and channels != _POLICY_CHANNELS.get(data_type):
        return None
    if data_type == "rgb":
        return PRETRAINED_CHECKPOINT_DEFAULT_VARIANT
    return data_type if data_type == "depth" else None


def _direct_camera_policy_variant(env_cfg: DirectRLEnvCfg) -> str | None:
    """Resolve the direct environment's policy variant from its final policy inputs."""
    observation_space = cast(list[int], env_cfg.observation_space)
    if len(observation_space) != 3:
        return None
    camera_cfg = cast(CameraCfg, getattr(env_cfg, "tiled_camera"))
    if len(camera_cfg.data_types) != 1:
        return None
    return _raw_camera_policy_variant(
        camera_cfg,
        camera_cfg.data_types[0],
        cast(int, getattr(env_cfg, "frame_stack")),
        observation_space[0],
    )


def _manager_camera_policy_variant(env_cfg: ManagerBasedRLEnvCfg) -> str | None:
    """Resolve the manager environment's policy variant from its final policy observation term."""
    policy_cfg = getattr(env_cfg.observations, "policy")
    image_cfg = cast(ObservationTermCfg, getattr(policy_cfg, "image"))
    if image_cfg.func is not mdp.CameraImageStack:
        return None
    data_type = image_cfg.params.get("data_type")
    if not isinstance(data_type, str):
        return None
    return _raw_camera_policy_variant(
        cast(CameraCfg, getattr(env_cfg.scene, "tiled_camera")),
        data_type,
        cast(int, getattr(env_cfg, "frame_stack")),
    )


def cartpole_camera_direct_pretrained_checkpoint_cfg() -> PretrainedCheckpointSetCfg:
    """Return published policy contracts for direct camera Cartpole."""
    return PretrainedCheckpointSetCfg(variant_resolver=_direct_camera_policy_variant, checkpoints=_CHECKPOINTS)


def cartpole_camera_manager_pretrained_checkpoint_cfg() -> PretrainedCheckpointSetCfg:
    """Return published policy contracts for manager-based camera Cartpole."""
    return PretrainedCheckpointSetCfg(variant_resolver=_manager_camera_policy_variant, checkpoints=_CHECKPOINTS)
