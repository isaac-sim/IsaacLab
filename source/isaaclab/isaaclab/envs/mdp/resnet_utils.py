# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure torchvision helpers for ResNet feature extraction.

No simulation context, env manager, or Isaac Lab runtime required.
These utilities can be imported and tested in a plain Python environment.
"""

from __future__ import annotations

import torch


def prepare_resnet_model(model_name: str, model_device: str, freeze: bool = True) -> dict:
    """Prepare a pretrained ResNet model for feature extraction.

    Loads a pretrained ResNet model and removes the final fully-connected
    classification layer, replacing it with an Identity layer so that the
    model outputs feature vectors instead of classification logits.

    Feature dimensions by variant:
        - ResNet18: 512-dim
        - ResNet34: 512-dim
        - ResNet50: 2048-dim
        - ResNet101: 2048-dim

    Args:
        model_name: One of ``"resnet18"``, ``"resnet34"``, ``"resnet50"``, ``"resnet101"``.
        model_device: The device to place the model on (e.g. ``"cpu"`` or ``"cuda:0"``).
        freeze: If ``True`` (default), sets the model to eval mode and wraps inference in
            :func:`torch.no_grad` to avoid building an autograd graph through the backbone.
            If ``False``, the model remains in training mode and gradients flow through it.

    Returns:
        A dict with keys ``"model"`` (a zero-arg callable returning the loaded model)
        and ``"inference"`` (a callable ``(model, images) -> features``).

    Raises:
        ValueError: If ``model_name`` is not one of the supported ResNet variants.
    """
    from torchvision import models

    resnet_weights_map = {
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
        "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
        "resnet101": models.ResNet101_Weights.IMAGENET1K_V1,
    }

    if model_name not in resnet_weights_map:
        raise ValueError(
            f"Unsupported ResNet model: {model_name}. Supported models: {list(resnet_weights_map.keys())}"
        )

    def _load_model() -> torch.nn.Module:
        model = getattr(models, model_name)(weights=resnet_weights_map[model_name])
        # Replace the final FC classification layer with Identity to extract features
        model.fc = torch.nn.Identity()
        if freeze:
            model.eval()
        return model.to(model_device)

    normalization_tensors: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_model_device(model: torch.nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            try:
                return next(model.buffers()).device
            except StopIteration:
                return torch.device(model_device)

    def _get_normalization_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if device not in normalization_tensors:
            mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).view(1, 3, 1, 1)
            normalization_tensors[device] = (mean, std)
        return normalization_tensors[device]

    def _inference(model, images: torch.Tensor) -> torch.Tensor:
        """Run inference on the ResNet model.

        Args:
            model: ResNet model with FC replaced by Identity.
            images: Input tensor of shape ``(N, H, W, C)`` in ``[0, 255]`` range.

        Returns:
            Feature tensor of shape ``(N, feature_dim)``.
        """
        device = _get_model_device(model)
        image_proc = images.to(device)
        image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
        mean, std = _get_normalization_tensors(device)
        image_proc = (image_proc - mean) / std

        if freeze:
            with torch.no_grad():
                features = model(image_proc)
        else:
            features = model(image_proc)
        return features

    return {"model": _load_model, "inference": _inference}
