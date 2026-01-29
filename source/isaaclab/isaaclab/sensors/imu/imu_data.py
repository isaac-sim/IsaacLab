# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-exports the base IMU data class for backwards compatibility."""

from .base_imu_data import BaseImuData

# Re-export for backwards compatibility
ImuData = BaseImuData

__all__ = ["BaseImuData", "ImuData"]
