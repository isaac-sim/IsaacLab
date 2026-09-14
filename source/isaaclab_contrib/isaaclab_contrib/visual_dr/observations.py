# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera observation boundary, before normalization and observation history."""

from typing import TYPE_CHECKING

import torch

from .runtime import DRFrame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def preserve_mask(segmentation: torch.Tensor, labels: dict, background_classes: tuple[str, ...]) -> torch.Tensor:
    """Allow only explicitly named background classes; unknown IDs stay protected."""
    if segmentation.dtype != torch.int32 or segmentation.shape[-1] != 1 or not labels:
        raise ValueError("DR needs uncolored semantic IDs and a nonempty idToLabels mapping")
    preserve = torch.ones_like(segmentation, dtype=torch.bool)
    for semantic_id, entry in labels.items():
        if isinstance(entry, dict) and entry.get("class") in background_classes:
            preserve &= segmentation != int(semantic_id)
    return preserve


def image_runtime_dr(
    env: "ManagerBasedRLEnv", camera: str, background_classes: tuple[str, ...] = ("ground", "BACKGROUND")
) -> torch.Tensor:
    """Read raw uint8 RGB, or apply an attached ``env.visual_dr_runtime``.

    Attach only after environment construction, so dimension probes stay raw.
    This prototype reads one camera at a time and does not support partial resets.
    """
    sensor = env.scene.sensors[camera]
    data = sensor.data
    # CameraData in this revision exposes Warp-backed ProxyArray; .torch is zero-copy.
    rgb = data.output["rgb"].torch
    runtime = getattr(env, "visual_dr_runtime", None)
    if runtime is None:
        return rgb.clone()
    info = data.info.get("semantic_segmentation") or {}
    mask = preserve_mask(data.output["semantic_segmentation"].torch, info.get("idToLabels", {}), background_classes)
    return runtime.process(camera, DRFrame(rgb, data.output["distance_to_image_plane"].torch, mask))
