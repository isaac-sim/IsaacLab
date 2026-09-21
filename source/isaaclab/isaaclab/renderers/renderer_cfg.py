# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for renderers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .output_contract import RenderBufferKind, RenderBufferSpec

if TYPE_CHECKING:
    from .base_renderer import BaseRenderer


@configclass
class RendererCfg:
    """Configuration for a renderer."""

    class_type: type[BaseRenderer] | str | None = None
    """Renderer implementation class. Concrete configs must set this field."""

    renderer_type: str = "default"

    def supported_output_types(self) -> dict[RenderBufferKind, RenderBufferSpec] | None:
        """Return the camera output layouts supported by this renderer configuration.

        Concrete renderer configurations override this method when their output contract can be
        determined without importing or instantiating the renderer implementation. Returning
        ``None`` defers compatibility validation until the renderer is created.

        Returns:
            Mapping from supported output types to their buffer layouts, or ``None`` when the
            renderer is selected dynamically.
        """
        return None
