# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :func:`~isaaclab.envs.mdp.resnet_utils.prepare_resnet_model`.

No simulation context required — the module under test has no dependency on
Isaac Lab runtime, env managers, or any simulation infrastructure.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from isaaclab.envs.mdp.observations import image_features
from isaaclab.envs.mdp.resnet_utils import prepare_resnet_model


def _make_image_features_term(freeze: bool) -> image_features:
    """Create an image_features term without initializing a full manager env."""
    term = image_features.__new__(image_features)
    term.freeze = freeze
    term._model = nn.Linear(3, 3)

    def _inference(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
        return model(images.reshape(-1, 3))

    term._inference_fn = _inference
    return term


class TestResNetModelPreparation:
    """Verify the behavioral contract of :func:`prepare_resnet_model`."""

    def test_resnet18_fc_removed(self):
        """ResNet18 FC layer should be replaced with Identity."""
        d = prepare_resnet_model("resnet18", "cpu")
        model = d["model"]()
        assert isinstance(model.fc, nn.Identity), f"Expected Identity, got {type(model.fc)}"

    def test_resnet18_feature_dim(self):
        """ResNet18 should produce 512-dim features."""
        d = prepare_resnet_model("resnet18", "cpu")
        model = d["model"]()
        dummy = torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8)
        features = d["inference"](model, dummy)
        assert features.shape == (2, 512), f"Expected (2, 512), got {features.shape}"

    def test_resnet18_resizes_camera_input_to_imagenet_resolution(self):
        """Camera frames should be resized to the ImageNet pretraining resolution."""
        d = prepare_resnet_model("resnet18", "cpu")
        model = d["model"]()
        seen_shape = None

        def _capture_input(images: torch.Tensor) -> torch.Tensor:
            nonlocal seen_shape
            seen_shape = images.shape[-2:]
            return torch.zeros(images.shape[0], 512)

        model.forward = _capture_input
        dummy = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)

        features = d["inference"](model, dummy)

        assert seen_shape == (224, 224)
        assert features.shape == (2, 512)

    def test_resnet50_feature_dim(self):
        """ResNet50 should produce 2048-dim features."""
        d = prepare_resnet_model("resnet50", "cpu")
        model = d["model"]()
        dummy = torch.randint(0, 256, (1, 224, 224, 3), dtype=torch.uint8)
        features = d["inference"](model, dummy)
        assert features.shape == (1, 2048), f"Expected (1, 2048), got {features.shape}"

    def test_inference_no_grad(self):
        """Frozen inference should not accumulate gradients."""
        d = prepare_resnet_model("resnet18", "cpu", freeze=True)
        model = d["model"]()
        dummy = torch.randint(0, 256, (1, 224, 224, 3), dtype=torch.uint8)
        features = d["inference"](model, dummy)
        assert not features.requires_grad, "Features should not require grad when freeze=True"

    def test_model_eval_mode(self):
        """Frozen model should be in eval mode."""
        d = prepare_resnet_model("resnet18", "cpu", freeze=True)
        model = d["model"]()
        assert not model.training, "Model should be in eval mode when freeze=True"

    def test_freeze_false_training_mode(self):
        """Unfrozen model should remain in training mode."""
        d = prepare_resnet_model("resnet18", "cpu", freeze=False)
        model = d["model"]()
        assert model.training, "Model should be in training mode when freeze=False"

    def test_freeze_false_requires_grad(self):
        """Unfrozen inference should produce gradients."""
        d = prepare_resnet_model("resnet18", "cpu", freeze=False)
        model = d["model"]()
        # Use float input so gradients can flow
        dummy = torch.rand(1, 224, 224, 3) * 255.0
        features = d["inference"](model, dummy)
        assert features.requires_grad, "Features should require grad when freeze=False"

    def test_image_features_detaches_when_frozen(self, monkeypatch):
        """Frozen observation term should detach feature tensors."""
        from isaaclab.envs.mdp import observations

        monkeypatch.setattr(observations, "image", lambda **_: torch.rand(1, 1, 1, 3))
        term = _make_image_features_term(freeze=True)

        features = term(env=None)

        assert not features.requires_grad, "Features should be detached when freeze=True"

    def test_image_features_preserves_grad_when_unfrozen(self, monkeypatch):
        """Unfrozen observation term should preserve feature tensor gradients."""
        from isaaclab.envs.mdp import observations

        monkeypatch.setattr(observations, "image", lambda **_: torch.rand(1, 1, 1, 3))
        term = _make_image_features_term(freeze=False)

        features = term(env=None)

        assert features.requires_grad, "Features should preserve gradients when freeze=False"

    def test_unsupported_model_raises(self):
        """Unsupported model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported ResNet model"):
            prepare_resnet_model("resnet152", "cpu")
