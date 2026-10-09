# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos camera control recipes as camera modifier chains.

Each factory returns ``(input_name, modifiers)`` for :attr:`CameraCfg.modifiers`: the camera
output the chain reads and the control modifier followed by the Cosmos transfer modifier.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.modifiers import ModifierBase, ModifierCfg

from isaaclab_experimental.image_transfer import depth_to_control

from .cosmos_modifier_cfg import CosmosTransferModifierCfg

if TYPE_CHECKING:
    from isaaclab.sensors import Camera

    from .cosmos_model_cfg import CosmosModelCfg


def edge_control(data: torch.Tensor, lower: int = 100, upper: int = 200) -> torch.Tensor:
    """Extract Canny edges from uint8 sRGB images using optional OpenCV on CPU."""
    import cv2

    edges = np.stack([cv2.Canny(np.ascontiguousarray(frame), lower, upper) for frame in data.cpu().numpy()])
    return torch.from_numpy(edges).to(data.device).unsqueeze(-1).expand(-1, -1, -1, 3)


def region_control(data: torch.Tensor, palette: dict[int, tuple[int, int, int]]) -> torch.Tensor:
    """Colorize raw segmentation IDs with an explicit episode-stable region palette."""
    ids = data[..., 0]
    output = torch.zeros((*ids.shape, 3), dtype=torch.uint8, device=ids.device)
    known = (ids == 0) | (ids == 1)
    for label, color in palette.items():
        label = int(label)
        if label < 2 or len(color) != 3 or any(not 0 <= c <= 255 for c in color):
            raise ValueError("Palette entries require nonreserved IDs and three byte-valued channels.")
        selected = ids == label
        output[selected] = torch.tensor(tuple(color), dtype=torch.uint8, device=ids.device)
        known |= selected
    if not bool(known.all()):
        raise ValueError("Segmentation contains IDs missing from the fixed palette.")
    return output


def regional_edges(
    rgb: torch.Tensor, labels: torch.Tensor, foreground_ids: Sequence[int], background_sigma: float = 4.0
) -> torch.Tensor:
    """Keep foreground edges and blur background detail before extracting edges.

    This constructs one binary edge image, not regional model guidance weights. Dilating the
    foreground selection retains object outlines outside the label. Generation remains
    full-frame and does not guarantee pixel preservation.
    """
    import cv2

    result = []
    for frame, ids in zip(rgb.cpu().numpy(), labels[..., 0].cpu().numpy(), strict=True):
        frame = np.ascontiguousarray(frame)
        selected = np.isin(ids, tuple(foreground_ids)).astype(np.uint8)
        selected = cv2.dilate(selected, np.ones((7, 7), np.uint8)).astype(bool)
        fine = cv2.Canny(frame, 100, 200)
        coarse = cv2.Canny(cv2.GaussianBlur(frame, (0, 0), background_sigma), 100, 200)
        result.append(np.where(selected, fine, coarse))
    return torch.from_numpy(np.stack(result)).to(rgb.device).unsqueeze(-1).expand(-1, -1, -1, 3)


class RegionalEdgeControl(ModifierBase):
    """Edge control from the camera's RGB plus its raw ``semantic_segmentation`` output.

    A modifier chain reads one tensor, so this modifier takes RGB as its input and reads the
    segmentation of the same capture from the camera it is bound to. List ``semantic_segmentation``
    in the camera's ``data_types`` with ``colorize_semantic_segmentation=False``.
    """

    def __init__(self, cfg: RegionalEdgeControlCfg, data_dim: tuple[int, ...], device: str):
        super().__init__(cfg, data_dim, device)
        self._camera = None

    def bind_sensor(self, sensor: Camera) -> None:
        """Read segmentation from the camera that owns this modifier."""
        self._camera = sensor

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the stateless control modifier."""
        pass

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self._camera is None:
            raise RuntimeError("Regional edges need the camera's segmentation; apply them as a camera modifier.")
        # Read the camera's buffers directly: the camera is publishing this capture, and its public data
        # property would try to update it again.
        labels = self._camera._data.output.get("semantic_segmentation")
        if labels is None:
            raise RuntimeError("Add 'semantic_segmentation' to the camera's data_types for regional edges.")
        return regional_edges(data, labels.torch, self._cfg.foreground_ids, self._cfg.background_sigma)


@configclass
class RegionalEdgeControlCfg(ModifierCfg):
    """Configuration of :class:`RegionalEdgeControl`."""

    func: type[RegionalEdgeControl] = RegionalEdgeControl
    foreground_ids: tuple[int, ...] = ()
    """Nonreserved segmentation IDs kept in full detail; verify them against the camera label map."""
    background_sigma: float = 4.0
    """Gaussian blur sigma [px] applied to the background before edge extraction."""


def _check_modality(backend: CosmosModelCfg, modality: str) -> None:
    """Reject a model configured for another control type than the chain prepares."""
    configured = backend.modality
    if configured != modality:
        raise ValueError(f"This chain prepares {modality} controls, but the model is set to modality={configured!r}.")


def edge_processor(
    backend: CosmosModelCfg, *, lower: int = 100, upper: int = 200, **cosmos: Any
) -> tuple[str, list[ModifierCfg]]:
    """Edge-guided Cosmos chain on the camera's ``rgb`` output."""
    _check_modality(backend, "edge")
    if not 0 <= lower < upper:
        raise ValueError("Edge thresholds must satisfy 0 <= lower < upper.")
    control = ModifierCfg(func=edge_control, params={"lower": lower, "upper": upper})
    return "rgb", [control, CosmosTransferModifierCfg(backend=backend, **cosmos)]


def blur_processor(backend: CosmosModelCfg, **cosmos: Any) -> tuple[str, list[ModifierCfg]]:
    """Blur-guided Cosmos chain on the camera's ``rgb`` output.

    The camera sends its RGB unchanged; the service blurs it with the Framework's own filter, so the control matches
    the model's training data.
    """
    _check_modality(backend, "blur")
    return "rgb", [CosmosTransferModifierCfg(backend=backend, **cosmos)]


def segmentation_processor(
    backend: CosmosModelCfg, palette: dict[int, tuple[int, int, int]], **cosmos: Any
) -> tuple[str, list[ModifierCfg]]:
    """Region-color Cosmos chain on the camera's uncolorized ``semantic_segmentation`` output.

    Keep the palette fixed for the whole episode so each region keeps its color.
    """
    _check_modality(backend, "seg")
    palette = {int(key): tuple(value) for key, value in palette.items()}
    control = ModifierCfg(func=region_control, params={"palette": palette})
    return "semantic_segmentation", [control, CosmosTransferModifierCfg(backend=backend, **cosmos)]


def regional_edge_processor(
    backend: CosmosModelCfg, *, foreground_ids: Sequence[int], background_sigma: float = 4.0, **cosmos: Any
) -> tuple[str, list[ModifierCfg]]:
    """Edge-guided Cosmos chain on ``rgb`` that keeps detail only on the selected segmentation IDs."""
    _check_modality(backend, "edge")
    if not foreground_ids or any(type(i) is not int or i < 2 for i in foreground_ids):
        raise ValueError("Supply nonreserved foreground segmentation IDs from the camera label map.")
    if not np.isfinite(background_sigma) or background_sigma <= 0:
        raise ValueError("Background blur sigma must be finite and positive.")
    control = RegionalEdgeControlCfg(foreground_ids=tuple(foreground_ids), background_sigma=background_sigma)
    return "rgb", [control, CosmosTransferModifierCfg(backend=backend, **cosmos)]


def depth_processor(
    backend: CosmosModelCfg, *, near: float = 1.2, far: float = 3.5, **cosmos: Any
) -> tuple[str, list[ModifierCfg]]:
    """Depth-guided Cosmos chain on the camera's ``distance_to_image_plane`` output.

    Near hits become white and far hits black. Set the metric range to match the camera's scene.
    """
    _check_modality(backend, "depth")
    control = ModifierCfg(func=depth_to_control, params={"near": near, "far": far})
    return "distance_to_image_plane", [control, CosmosTransferModifierCfg(backend=backend, **cosmos)]
