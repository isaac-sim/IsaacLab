# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab.utils import config_field

if TYPE_CHECKING:
    from .base_renderer import BaseRenderer


@dataclass
class RendererCfg:
    """Configuration for a renderer."""

    class_type: type[BaseRenderer] | str | None = config_field(None)
    """Renderer implementation class. Concrete configs must set this field."""

    renderer_type: str = config_field("default")
