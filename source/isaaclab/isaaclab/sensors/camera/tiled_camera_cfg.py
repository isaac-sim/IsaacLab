# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .camera_cfg import CameraCfg
from .tiled_camera import TiledCamera


@configclass
class TiledCameraCfg(CameraCfg):
    """Configuration for a tiled rendering-based camera sensor."""

    class_type: type = TiledCamera

    renderer_type: str | None = None
    """Renderer backend. Default is ``None`` (RTX). If ``"newton_warp"``, uses Newton Warp ray
    tracing (PhysX sim + Newton state sync). If ``None`` or anything else, uses Omniverse RTX
    tiled rendering (Replicator annotators). The training script's ``--renderer_backend`` sets
    ``env.scene`` so the task's scene variant supplies this value (e.g. rtx vs newton_warp)."""
