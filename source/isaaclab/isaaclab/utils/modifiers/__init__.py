# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Migration error for the removed modifiers API."""

raise ValueError(
    "isaaclab.utils.modifiers was removed. Use isaaclab.utils.tensor_processors instead:"
    " ModifierBase is now TensorProcessor, and ModifierCfg is now TensorProcessorCfg."
)
