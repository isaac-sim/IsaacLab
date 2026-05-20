# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.frame_transformer import BaseFrameTransformer
from isaaclab.utils.math import is_identity_pose

import isaaclab_ovphysx.tensor_types as TT
from isaaclab_ovphysx.physics import OvPhysxManager

from .frame_transformer_data import FrameTransformerData
from .kernels import frame_transformer_update_kernel, gather_body_pose_kernel

if TYPE_CHECKING:
    from isaaclab.sensors.frame_transformer import FrameTransformerCfg

logger = logging.getLogger(__name__)


class FrameTransformer(BaseFrameTransformer):
    """An OVPhysX sensor for reporting frame transforms.

    Reports the world-frame transform of one or more target frames relative to a source frame.
    Both the source frame (:attr:`FrameTransformerCfg.prim_path`) and target frames
    (:attr:`FrameTransformerCfg.target_frames`) must attach to rigid bodies — either
    articulation links or standalone rigid bodies. The two cases are handled uniformly
    via ``TT.RIGID_BODY_POSE`` tensor bindings.

    Per-frame offsets (position + quaternion) are applied to the source and to each target.
    The relative transforms are computed on GPU by the same warp kernel the PhysX backend uses.
    """

    cfg: FrameTransformerCfg
    """The configuration parameters."""

    __backend_name__: str = "ovphysx"
    """The name of the backend for the frame transformer sensor."""

    def __init__(self, cfg: FrameTransformerCfg):
        """Initializes the frame transformer object.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)
        self._data: FrameTransformerData = FrameTransformerData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"FrameTransformer @ '{self.cfg.prim_path}': \n"
            f"\ttracked body frames: {[self._source_frame_body_name] + self._target_frame_body_names} \n"
            f"\tnumber of envs: {self._num_envs}\n"
            f"\tsource body frame: {self._source_frame_body_name}\n"
            f"\ttarget frames (count: {len(self._target_frame_names)}): {self._target_frame_names}\n"
        )

    """
    Properties
    """

    @property
    def data(self) -> FrameTransformerData:
        self._update_outdated_buffers()
        return self._data

    @property
    def num_bodies(self) -> int:
        """Returns the number of target body frames being tracked."""
        warnings.warn(
            "The `num_bodies` property will be deprecated in a future release."
            " Please use `len(data.target_frame_names)` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return len(self._target_frame_body_names)

    @property
    def body_names(self) -> list[str]:
        """Returns the names of the target body frames being tracked."""
        warnings.warn(
            "The `body_names` property will be deprecated in a future release."
            " Please use `data.target_frame_names` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._target_frame_body_names

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(None, env_mask)

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        raise NotImplementedError("FrameTransformer._initialize_impl lands in the next commit.")

    def _update_buffers_impl(self, env_mask: wp.array | None = None):
        raise NotImplementedError("FrameTransformer._update_buffers_impl lands in the next commit.")

    @staticmethod
    def _get_relative_body_path(prim_path: str) -> str:
        """Strip the ``/envs/env_<id>/`` prefix from a prim path so paths can be compared across environments.

        Args:
            prim_path: Absolute USD prim path that may contain an ``/envs/env_<digits>/`` segment.

        Returns:
            The prim path with that segment collapsed to ``/envs/``, so prim paths from any env compare equal.
        """
        pattern = re.compile(r"/envs/env_[^/]+/")
        return pattern.sub("/envs/", prim_path)
