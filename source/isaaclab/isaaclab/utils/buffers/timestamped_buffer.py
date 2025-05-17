# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import dataclass


@dataclass
class TimestampedBuffer:
    """A buffer class containing data and its timestamp.

    This class is a simple data container that stores a tensor and its timestamp. The timestamp is used to
    track the last update of the buffer. The timestamp is set to -1.0 by default, indicating that the buffer
    has not been updated yet. The timestamp should be updated whenever the data in the buffer is updated. This
    way the buffer can be used to check whether the data is outdated and needs to be refreshed.

    The buffer is useful for creating lazy buffers that only update the data when it is outdated. This can be
    useful when the data is expensive to compute or retrieve. For example usage, refer to the data classes in
    the :mod:`isaaclab.assets` module.
    """

    data: torch.Tensor = None  # type: ignore
    """The data stored in the buffer. Default is None, indicating that the buffer is empty."""

    timestamp: float = -1.0
    """Timestamp at the last update of the buffer. Default is -1.0, indicating that the buffer has not been updated."""
