# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warp as wp

from .timestamped_buffer import TimestampedBuffer


class TimestampedBufferWarp(TimestampedBuffer[wp.array]):
    """A :class:`TimestampedBuffer` with preallocated Warp storage.

    Allocation does not make the data fresh. The owner commits the timestamp after computing it.
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
        super().__init__(data=wp.zeros(shape, dtype=dtype, device=device))
