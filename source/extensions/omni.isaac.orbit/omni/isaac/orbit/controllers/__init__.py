# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# enable motion generation extension
from omni.isaac.core.utils.extensions import enable_extension

# TODO: Maybe keep this extension enabled by default? -- Fix the app experience.
enable_extension("omni.isaac.motion_generation")


# get module libraries
from .differential_inverse_kinematics import DifferentialInverseKinematics
from .rmp_flow import RmpFlowController

__all__ = ["DifferentialInverseKinematics", "RmpFlowController"]
