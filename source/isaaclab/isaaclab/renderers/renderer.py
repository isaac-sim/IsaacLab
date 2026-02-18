# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from isaaclab.sensors import SensorBase


class Renderer:
    def __init__(self):
        pass

    def create_render_data(self, sensor: SensorBase) -> Any:
        return None

    def set_outputs(self, render_data: Any, output_data: dict[str, torch.Tensor]):
        pass

    def update_transforms(self):
        pass

    def update_camera(
        self, render_data: Any, positions: torch.Tensor, orientations: torch.Tensor, intrinsics: torch.Tensor
    ):
        pass

    def render(self, render_data: Any):
        pass

    def write_output(self, render_data: Any, output_name: str, output_data: torch.Tensor):
        pass
