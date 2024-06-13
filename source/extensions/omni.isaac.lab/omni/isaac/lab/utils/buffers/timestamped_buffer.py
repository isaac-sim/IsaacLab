# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass


@dataclass
class TimestampedBuffer:
    """Buffer to hold timestamped data.

    Such a buffer is useful to check whether data is outdated and needs to be refreshed to create lazy buffers.
    """

    data: torch.Tensor = None
    """Data stored in the buffer."""

    update_timestamp: float = -1.0
    """Timestamp of the last update of the buffer."""
