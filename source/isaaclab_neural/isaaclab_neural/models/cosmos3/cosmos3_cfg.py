# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the Cosmos3 world model."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass

from ...world_model import ModelCfg


@configclass
class JointMappingCfg:
    """Robot-specific conversion of joint targets to Cosmos3 controls.

    ``class_type`` is built as ``class_type(cfg)`` and provides ``encode(joint_targets, start_joint_state)``, which
    maps joint targets [rad], shape ``(T, J)``, and the joint positions at the start of the chunk, shape ``(J,)``, to
    normalized ``unified_v1`` controls, shape ``(T, 59)``. The mapping belongs to the task, which owns the robot
    description and the normalization data.
    """

    class_type: type | str = MISSING


@configclass
class Cosmos3Cfg(ModelCfg):
    """Cosmos3 predicting camera frames from joint targets with one local ``unified_v1`` checkpoint.

    ``window`` mode conditions each prediction on the last generated image (Stage 0 checkpoints) and serves any
    number of environments. ``session`` mode keeps the native causal history of distilled checkpoints and serves one
    environment. ``chunk_size / fps`` is the duration of one model step [s]. With ``None`` sampler overrides, window
    mode uses the forward-dynamics defaults of 30 denoising steps and shift 10.
    """

    class_type: type | str = "{DIR}.cosmos3_model:Cosmos3Model"

    checkpoint_path: str = MISSING
    """Directory of a local HF export, or of a DCP checkpoint together with :attr:`config_file`."""

    output_dir: str = MISSING
    """Directory the Cosmos Framework writes inference artifacts to."""

    config_file: str | None = None
    """Saved YAML configuration of a DCP checkpoint."""

    mode: Literal["window", "session"] = "window"

    image_size: tuple[int, int] = (480, 640)
    """Height and width of the frames [pixels]."""

    chunk_size: int = 16
    """Frames predicted, and joint targets consumed, per model step. Divisible by 4."""

    fps: int = 30
    """Frame rate [Hz]."""

    domain_name: str = "hand_pose"

    resolution: str = "480"
    """Resolution tier of the Cosmos Framework."""

    num_inference_steps: int | None = None

    guidance: float = 1.0

    shift: float | None = None

    use_ema_weights: bool = True

    joint_mapping: JointMappingCfg | None = None
    """Joint mapping of the robot. Needed to drive the model with joint targets; replaying recorded controls does
    not need it."""
