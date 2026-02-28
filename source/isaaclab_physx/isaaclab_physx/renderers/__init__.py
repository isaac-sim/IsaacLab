# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for PhysX renderer backends (Isaac RTX / Omniverse Replicator)."""

from .isaac_rtx_renderer import IsaacRtxRenderer

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)

Renderer = IsaacRtxRenderer
