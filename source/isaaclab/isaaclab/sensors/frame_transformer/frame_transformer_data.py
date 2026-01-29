# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-exports the base frame transformer data class for backwards compatibility."""

from .base_frame_transformer_data import BaseFrameTransformerData

# Re-export for backwards compatibility
FrameTransformerData = BaseFrameTransformerData

__all__ = ["BaseFrameTransformerData", "FrameTransformerData"]
