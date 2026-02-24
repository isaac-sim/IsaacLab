# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

from ..sensor_base import SensorBase

if TYPE_CHECKING:
    from .base_frame_transformer_data import BaseFrameTransformerData
    from .frame_transformer_cfg import FrameTransformerCfg


class BaseFrameTransformer(SensorBase):
    """Sensor reporting transforms of target frames relative to a source frame.

    Source frame is set via :attr:`FrameTransformerCfg.prim_path`; target frames via
    :attr:`FrameTransformerCfg.target_frames`. Both must be rigid bodies. Relative
    transforms are derived from world-frame poses obtained from the physics engine.

    Optional offsets on source and target frames allow measuring from points other
    than the body origin (e.g. a tool-tip offset from the body center of mass).
    """

    cfg: FrameTransformerCfg
    """The configuration parameters."""

    def __init__(self, cfg: FrameTransformerCfg):
        """Initializes the frame transformer object.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"FrameTransformer @ '{self.cfg.prim_path}': \n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of bodies     : {self.num_bodies}\n"
            f"\tbody names           : {self.body_names}\n"
        )

    """
    Properties
    """

    @property
    @abstractmethod
    def data(self) -> BaseFrameTransformerData:
        """Data from the sensor."""
        raise NotImplementedError

    @property
    def num_bodies(self) -> int:
        """Number of target bodies tracked."""
        return len(self._target_frame_body_names)

    @property
    def body_names(self) -> list[str]:
        """Ordered names of target bodies tracked."""
        return self._target_frame_body_names

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        # reset the timers and counters
        super().reset(env_ids, env_mask)

    @abstractmethod
    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find target bodies by name.

        Args:
            name_keys: Regex or list of regexes to match target body names.
            preserve_order: Whether to preserve the order of the name keys. Defaults to False.

        Returns:
            Tuple of (indices, names) for matched bodies.
        """
        raise NotImplementedError

    """
    Implementation.
    """

    @abstractmethod
    def _initialize_impl(self):
        super()._initialize_impl()

    @abstractmethod
    def _update_buffers_impl(self, env_mask: wp.array | None):
        """Fills the buffers of the sensor data.

        Args:
            env_mask: Mask of the environments to update. None: update all environments.
        """
        raise NotImplementedError

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)
