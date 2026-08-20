# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp


class TimestampedBufferWarp:
    """A buffer class containing data and its timestamp.

    This class is a simple data container that stores a tensor and its timestamp. The timestamp is used to
    track the last update of the buffer. The timestamp is set to -1.0 by default, indicating that the buffer
    has not been updated yet. The timestamp should be updated whenever the data in the buffer is updated. This
    way the buffer can be used to check whether the data is outdated and needs to be refreshed.

    The buffer is useful for creating lazy buffers that only update the data when it is outdated. This can be
    useful when the data is expensive to compute or retrieve. For example usage, refer to the data classes in
    the :mod:`isaaclab.assets` module.
    """

    def __init__(self, shape: tuple, device: str, dtype: type) -> None:
        """Initializes the timestamped buffer.

        .. note:: Unlike the :class:`TimestampedBuffer` class in the :mod:`isaaclab.utils.buffers` module,
            this class allocates the memory on init. Ideally, users should avoid to overwrite the data after
            initialization and should use data.assign(...) whenever possible.

        Args:
            shape: The shape of the data.
            device: The device used for the data.
            dtype: The data type of the data.
        """
        self.data = wp.zeros(shape, dtype=dtype, device=device)
        self.timestamp = -1.0
