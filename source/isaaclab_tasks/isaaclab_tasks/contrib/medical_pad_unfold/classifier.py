# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-owned ordered-class ResNet18 reward model, independent of the RL trainer."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn


class MedicalPadClassifier(nn.Module):
    """An FP32 ResNet18 image classifier whose ordered classes each carry a reward.

    The checkpoint holds the ResNet18 parameters with an optional ``backbone.`` prefix; the YAML describes the
    preprocessing, the two-layer head and the reward of each class.
    """

    def __init__(self, checkpoint: str, config_file: str, device: torch.device | str):
        super().__init__()
        from torchvision.models import resnet18

        cfg = yaml.safe_load(Path(config_file).read_text())
        self.image_size = tuple(cfg["image_size"][1:])
        self.pre_resize = cfg["pre_resize"]
        self.channels_last = cfg["channels_last"]
        self.model = resnet18(weights=None)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, cfg["hidden_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["hidden_dim"], cfg["num_classes"]),
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict({key.removeprefix("backbone."): value for key, value in state.items()}, strict=True)
        self.register_buffer("reward_values", torch.tensor(cfg["reward_values"], dtype=torch.float32))
        mean = [0.485, 0.456, 0.406] if cfg["normalize"] else [0.0] * 3
        std = [0.229, 0.224, 0.225] if cfg["normalize"] else [1.0] * 3
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval().to(device)

    @torch.inference_mode()
    def score_frames(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score uint8 RGB frames ``(N, T, H, W, 3)`` as expected rewards ``(N, T)`` and probabilities."""
        inputs = frames.flatten(0, 1).permute(0, 3, 1, 2).to(self.mean.device).float() / 255
        if self.channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        if self.pre_resize and inputs.shape[-2:] != tuple(self.pre_resize):
            inputs = F.interpolate(inputs, size=self.pre_resize, mode="bilinear", align_corners=False)
        if inputs.shape[-2:] != self.image_size:
            inputs = F.interpolate(inputs, size=self.image_size, mode="bilinear", align_corners=False)
        inputs = (inputs - self.mean) / self.std
        with torch.autocast(self.mean.device.type, enabled=False):
            probabilities = torch.cat([self.model(batch).softmax(-1) for batch in inputs.split(64)])
        probabilities = probabilities.reshape(*frames.shape[:2], len(self.reward_values))
        return (probabilities * self.reward_values).sum(-1), probabilities
