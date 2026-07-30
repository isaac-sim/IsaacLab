# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod

from isaaclab.utils.warp import ProxyArray


class BaseCableObjectData(ABC):
    """Abstract data container for cable segment state."""

    def __init__(self, device: str) -> None:
        """Initialize the cable object data.

        Args:
            device: The device used for processing.
        """
        self.device = device

    default_segment_pose_w: ProxyArray | None = None
    """Default segment actor-frame poses in simulation world frame [m].

    The Warp view has shape (num_instances, num_segments), dtype ``wp.transformf``. The Torch view has shape
    (num_instances, num_segments, 7), with position [m] and quaternion in ``(x, y, z, w)`` format.
    """

    default_segment_velocity_w: ProxyArray | None = None
    """Default segment center-of-mass velocities in simulation world frame [m/s, rad/s].

    The Warp view has shape (num_instances, num_segments), dtype ``wp.spatial_vectorf``. The Torch view has shape
    (num_instances, num_segments, 6), with linear [m/s] followed by angular [rad/s] velocity.
    """

    @property
    @abstractmethod
    def segment_pose_w(self) -> ProxyArray:
        """Segment actor-frame poses in the simulation world frame.

        Shape is (num_instances, num_segments), dtype ``wp.transformf``. The Torch view has shape
        (num_instances, num_segments, 7), with position [m] and quaternion in ``(x, y, z, w)`` format.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def segment_velocity_w(self) -> ProxyArray:
        """Segment center-of-mass velocities in the simulation world frame.

        Shape is (num_instances, num_segments), dtype ``wp.spatial_vectorf``. The Torch view has shape
        (num_instances, num_segments, 6), with linear [m/s] followed by angular [rad/s] velocity.
        """
        raise NotImplementedError()

    @abstractmethod
    def update(self, dt: float) -> None:
        """Update the cable data.

        Args:
            dt: The time step for the update [s].
        """
        raise NotImplementedError()
