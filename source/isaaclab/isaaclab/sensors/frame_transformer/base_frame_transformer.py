# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

import isaaclab.utils.string as string_utils

from ..sensor_base import SensorBase
from .base_frame_transformer_data import BaseFrameTransformerData

if TYPE_CHECKING:
    from .frame_transformer_cfg import FrameTransformerCfg


class BaseFrameTransformer(SensorBase):
    """A sensor for reporting frame transforms.

    This class provides an interface for reporting the transform of one or more frames (target frames)
    with respect to another frame (source frame). The source frame is specified by the user as a prim path
    (:attr:`FrameTransformerCfg.prim_path`) and the target frames are specified by the user as a list of
    prim paths (:attr:`FrameTransformerCfg.target_frames`).

    The source frame and target frames are assumed to be rigid bodies. The transform of the target frames
    with respect to the source frame is computed by first extracting the transform of the source frame
    and target frames from the physics engine and then computing the relative transform between the two.

    Additionally, the user can specify an offset for the source frame and each target frame. This is useful
    for specifying the transform of the desired frame with respect to the body's center of mass, for instance.

    A common example of using this sensor is to track the position and orientation of the end effector of a
    robotic manipulator. In this case, the source frame would be the body corresponding to the base frame of the
    manipulator, and the target frame would be the body corresponding to the end effector. Since the end-effector is
    typically a fictitious body, the user may need to specify an offset from the end-effector to the body of the
    manipulator.

    """

    cfg: FrameTransformerCfg
    """The configuration parameters."""

    __backend_name__: str = "base"
    """The name of the backend for the frame transformer sensor."""

    def __init__(self, cfg: FrameTransformerCfg):
        """Initializes the frame transformer object.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)

    """
    Properties
    """

    @property
    @abstractmethod
    def data(self) -> BaseFrameTransformerData:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_bodies(self) -> int:
        """Returns the number of target bodies being tracked.

        .. deprecated::
            Use ``len(data.target_frame_names)`` instead. This property will be removed in a future release.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def body_names(self) -> list[str]:
        """Returns the names of the target bodies being tracked.

        .. deprecated::
            Use ``data.target_frame_names`` instead. This property will be removed in a future release.
        """
        raise NotImplementedError

    """
    Operations
    """

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self._target_frame_names, preserve_order)

    """
    Implementation - Abstract methods to be implemented by backend-specific subclasses.
    """

    @abstractmethod
    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        Subclasses should call ``super()._initialize_impl()`` first to initialize
        the common sensor infrastructure from :class:`SensorBase`.
        """
        super()._initialize_impl()

    @abstractmethod
    def _update_buffers_impl(self, env_mask: wp.array):
        raise NotImplementedError

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)
