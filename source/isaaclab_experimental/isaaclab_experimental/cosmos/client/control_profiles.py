# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos camera control recipe as a camera modifier chain.

The factory returns ``(input_name, modifiers)`` for :attr:`CameraCfg.modifiers`: the camera
output the chain reads and the control modifier followed by the Cosmos transfer modifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab.utils.modifiers import ModifierCfg

from isaaclab_experimental.image_transfer import depth_to_control

from .cosmos_modifier_cfg import CosmosTransferModifierCfg

if TYPE_CHECKING:
    from .cosmos_model_cfg import CosmosModelCfg


def depth_processor(
    backend: CosmosModelCfg, *, near: float = 1.2, far: float = 3.5, **cosmos: Any
) -> tuple[str, list[ModifierCfg]]:
    """Depth-guided Cosmos chain on the camera's ``distance_to_image_plane`` output.

    Near hits become white and far hits black. Set the metric range to match the camera's scene.
    """
    control = ModifierCfg(func=depth_to_control, params={"near": near, "far": far})
    return "distance_to_image_plane", [control, CosmosTransferModifierCfg(backend=backend, **cosmos)]
