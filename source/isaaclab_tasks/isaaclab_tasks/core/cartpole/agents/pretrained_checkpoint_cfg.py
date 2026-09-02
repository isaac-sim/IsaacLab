# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pretrained-checkpoint declarations for camera-based Cartpole tasks."""

from isaaclab_rl.utils.pretrained_checkpoint import PretrainedCheckpointCfg, PretrainedCheckpointSetCfg

_RAW_CAMERA_POLICY_PRESETS = (
    "albedo",
    "depth",
    "rgb",
    "semantic_segmentation",
    "simple_shading_constant_diffuse",
    "simple_shading_diffuse_mdl",
    "simple_shading_full_mdl",
)


def _cartpole_camera_pretrained_checkpoint_cfg(policy_presets: tuple[str, ...]) -> PretrainedCheckpointSetCfg:
    """Build the shared camera-policy declarations."""
    return PretrainedCheckpointSetCfg(
        policy_presets=policy_presets,
        checkpoints=(
            PretrainedCheckpointCfg(workflow="rsl_rl", preset_aliases=(("rgb",),)),
            PretrainedCheckpointCfg(workflow="rl_games", preset_aliases=(("rgb",),), smoke_num_envs=32),
            PretrainedCheckpointCfg(workflow="rl_games", presets=("depth",), variant="depth", smoke_num_envs=32),
        ),
    )


def cartpole_camera_direct_pretrained_checkpoint_cfg() -> PretrainedCheckpointSetCfg:
    """Return published policy contracts for direct camera Cartpole."""
    return _cartpole_camera_pretrained_checkpoint_cfg(_RAW_CAMERA_POLICY_PRESETS)


def cartpole_camera_manager_pretrained_checkpoint_cfg() -> PretrainedCheckpointSetCfg:
    """Return published policy contracts for manager-based camera Cartpole."""
    return _cartpole_camera_pretrained_checkpoint_cfg((*_RAW_CAMERA_POLICY_PRESETS, "resnet18", "theia_tiny"))
