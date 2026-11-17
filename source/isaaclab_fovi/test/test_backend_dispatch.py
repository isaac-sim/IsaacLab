# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unsupported renderers fail before importing simulator backends."""

from dataclasses import dataclass

import pytest

from isaaclab_fovi.sensors import FoveatedTiledCamera


@dataclass
class RendererSelection:
    renderer_type: str


@dataclass
class CameraSelection:
    renderer_cfg: RendererSelection
    tiled: bool


def test_unknown_backend_fails_explicitly() -> None:
    cfg = CameraSelection(RendererSelection("unknown"), True)
    with pytest.raises(ValueError, match="Unsupported foveated renderer"):
        FoveatedTiledCamera(cfg)


@pytest.mark.parametrize("renderer_type", ["isaac_rtx", "ovrtx"])
def test_non_tiled_rtx_fails_explicitly(renderer_type: str) -> None:
    cfg = CameraSelection(RendererSelection(renderer_type), False)
    with pytest.raises(NotImplementedError, match="Non-tiled.*not implemented"):
        FoveatedTiledCamera(cfg)
