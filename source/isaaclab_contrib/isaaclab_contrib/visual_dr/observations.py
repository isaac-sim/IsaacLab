# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The observation boundary: where a camera's pixels become a policy's input.

DR is applied here, before normalization and observation history, so the policy
sees a randomized frame everywhere it would have seen the raw one -- including in
whatever the rollout buffer stores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .backends import DRFrame

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .cfg import CameraDRCfg


def semantic_id(key: str) -> int:
    """Convert an ``idToLabels`` key to the value the segmentation buffer stores.

    With ``colorize_semantic_segmentation`` disabled the buffer holds uncolored
    ``int32`` IDs, but the renderer still reports keys as ``"(r, g, b, a)"``
    strings. Those channels are the little-endian bytes of the stored ID, read as
    signed, so ``"(33, 243, 3, 255)"`` is ``-16518367``. Plain integer keys are
    passed through for renderers that report them directly.
    """
    text = key.strip()
    if not text.startswith("("):
        return int(text)
    channels = [int(part) for part in text.strip("()").split(",")]
    if len(channels) != 4:
        raise ValueError(f"Expected an RGBA semantic key, got {key!r}")
    packed = channels[0] | channels[1] << 8 | channels[2] << 16 | channels[3] << 24
    return packed - (1 << 32) if packed >= 1 << 31 else packed


def preserve_mask(segmentation: torch.Tensor, labels: dict, cfg: CameraDRCfg) -> torch.Tensor:
    """Mark pixels that must survive generation untouched.

    Classes name the foreground, and anything unrecognized follows
    ``unknown_policy`` -- which defaults to keeping it. An asset nobody tagged
    then shows up unchanged instead of dissolving into the background, which is
    the failure mode that is invisible until a policy has already learned around it.
    """
    if segmentation.ndim != 4 or segmentation.shape[-1] != 1:
        raise ValueError(f"Visual DR needs NHW1 semantic IDs, got {tuple(segmentation.shape)}")
    if segmentation.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"Visual DR needs uncolored integer semantic IDs, got {segmentation.dtype}")
    if not labels:
        raise ValueError(
            "Visual DR needs a non-empty idToLabels mapping; set "
            "colorize_semantic_segmentation=False on the camera and tag the scene's assets"
        )

    keep_unknown = cfg.unknown_policy == "preserve"
    preserve = torch.full_like(segmentation, keep_unknown, dtype=torch.bool)
    matched = False
    for key, entry in labels.items():
        name = entry.get("class") if isinstance(entry, dict) else entry
        if name is None:
            continue
        is_class = segmentation == semantic_id(key)
        if name in cfg.preserve_classes:
            preserve |= is_class
            matched = True
        elif keep_unknown:
            preserve &= ~is_class
    if not matched:
        raise ValueError(
            f"None of preserve_classes={cfg.preserve_classes} appear in this camera's labels "
            f"({sorted({e.get('class') for e in labels.values() if isinstance(e, dict)})}); "
            "the whole image would be regenerated"
        )

    if cfg.boundary_px:
        width = 2 * cfg.boundary_px + 1
        dilated = F.max_pool2d(preserve.permute(0, 3, 1, 2).float(), width, stride=1, padding=cfg.boundary_px)
        preserve = dilated.permute(0, 2, 3, 1).bool()
    return preserve


def image_runtime_dr(env: ManagerBasedRLEnv, camera: str) -> torch.Tensor:
    """Camera RGB, randomized by ``env.visual_dr_runtime`` when one is attached.

    Without a runtime this is the raw image, so a task config carrying DR terms
    still behaves like the original task when DR is off -- and observation
    dimension probes never load a model.
    """
    sensor = env.scene.sensors[camera]
    data = sensor.data
    # CameraData exposes Warp-backed ProxyArray in this revision; .torch is zero-copy.
    rgb = data.output["rgb"].torch
    runtime = getattr(env, "visual_dr_runtime", None)
    if runtime is None:
        return rgb

    runtime.sync(env)
    camera_cfg = runtime.cfg.cameras.get(camera)
    if camera_cfg is None:
        return rgb

    def make_frame() -> DRFrame:
        info = data.info.get("semantic_segmentation") or {}
        segmentation = data.output["semantic_segmentation"].torch
        mask = preserve_mask(segmentation, info.get("idToLabels", {}), camera_cfg)
        return DRFrame(rgb, data.output["distance_to_image_plane"].torch, mask, segmentation)

    return runtime.read(camera, rgb, make_frame)
