# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for frame transformer sensor."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "base_frame_transformer": ["BaseFrameTransformer"],
        "base_frame_transformer_data": ["BaseFrameTransformerData"],
        "frame_transformer": ["FrameTransformer"],
        "frame_transformer_cfg": ["FrameTransformerCfg", "OffsetCfg"],
        "frame_transformer_data": ["FrameTransformerData"],
    },
)
