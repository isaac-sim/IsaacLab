# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.sensors.frame_transformer.base_frame_transformer import BaseFrameTransformer
from isaaclab.utils.string import resolve_matching_names

from isaaclab_newton.physics import NewtonManager

from .frame_transformer_data import FrameTransformerData
from .frame_transformer_kernels import compose_target_world_kernel, copy_from_newton_kernel

if TYPE_CHECKING:
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg

logger = logging.getLogger(__name__)


def _offset_to_transform(offset: OffsetCfg) -> wp.transform:
    """Convert an OffsetCfg (pos, rot_wxyz) to wp.transform (pos, quat_xyzw)."""
    return wp.transform(
        offset.pos[0],
        offset.pos[1],
        offset.pos[2],
        offset.rot[1],
        offset.rot[2],
        offset.rot[3],
        offset.rot[0],
    )


class FrameTransformer(BaseFrameTransformer):
    """Newton frame transformer wrapping :class:`newton.sensors.SensorFrameTransform`.

    Creates per-env sites for the source and all target frames, backed by a single
    :class:`SensorFrameTransform` with 1:1 shape/reference pairs:

    * Entry 0 per env — source site measured w.r.t. a world-origin site.
    * Entries 1..M per env — target sites measured w.r.t. source site.

    Flat sensor output is indexed with stride ``1 + num_targets``:
    ``[i * stride]`` is the source world transform, ``[i * stride + 1 + j]``
    is target *j* relative to source in env *i*.
    """

    cfg: FrameTransformerCfg
    """The configuration parameters."""

    __backend_name__: str = "newton"
    """The name of the backend for the frame transformer sensor."""

    def __init__(self, cfg: FrameTransformerCfg):
        """Initializes the frame transformer.

        Registers site requests via :meth:`NewtonManager.cl_register_site` for
        the source frame, each target frame, and a shared world-origin reference.
        Sites are injected into prototype builders by ``newton_replicate`` before
        replication, so they end up correctly in each world.

        Args:
            cfg: Configuration parameters.
        """
        # initialize base class (registers PHYSICS_READY callback for _initialize_impl)
        super().__init__(cfg)

        self._data: FrameTransformerData = FrameTransformerData()

        self._sensor_index: int | None = None
        self._source_frame_body_name: str = cfg.prim_path.rsplit("/", 1)[-1]

        # Register world-origin reference site
        self._world_origin_label = NewtonManager.cl_register_site(None, wp.transform())

        # Register source site
        source_offset = _offset_to_transform(cfg.source_frame_offset)
        self._source_label = NewtonManager.cl_register_site(cfg.prim_path, source_offset)

        # Register target sites
        self._target_labels: list[str] = []
        self._target_frame_body_names: list[str] = []
        self._num_targets: int = 0

        for tgt_idx, target_frame in enumerate(cfg.target_frames):
            target_offset = _offset_to_transform(target_frame.offset)
            label = NewtonManager.cl_register_site(target_frame.prim_path, target_offset)

            self._target_labels.append(label)
            body_name = target_frame.prim_path.rsplit("/", 1)[-1]
            self._target_frame_body_names.append(target_frame.name or body_name)
            self._num_targets += 1

        # Set target frame names for base class find_bodies() and data container
        self._target_frame_names = [t.name or t.prim_path.rsplit("/", 1)[-1] for t in cfg.target_frames]
        self._data._target_frame_names = self._target_frame_names

        logger.info(
            f"FrameTransformer '{cfg.prim_path}': source='{self._source_frame_body_name}', "
            f"{self._num_targets} target(s) registered"
        )

    """
    Properties
    """

    @property
    def data(self) -> FrameTransformerData:
        # update sensors if needed
        self._update_outdated_buffers()
        return self._data

    @property
    def num_bodies(self) -> int:
        """Returns the number of target bodies being tracked."""
        return len(self._target_frame_body_names)

    @property
    def body_names(self) -> list[str]:
        """Returns the names of the target bodies being tracked."""
        return self._target_frame_body_names

    """
    Implementation
    """

    def _initialize_impl(self):
        """PHYSICS_READY callback: resolves site indices and creates the SensorFrameTransform."""
        super()._initialize_impl()

        model = NewtonManager.get_model()
        shape_labels = list(model.shape_label)
        num_envs = self._num_envs

        # Resolve world-origin site
        world_origin_indices, _ = resolve_matching_names(self._world_origin_label, shape_labels)
        world_origin_idx = world_origin_indices[0]

        # Resolve source sites (one per env)
        source_pattern = f"{self.cfg.prim_path}/{self._source_label}"
        source_indices, _ = resolve_matching_names(source_pattern, shape_labels)

        # Resolve target sites (one per env per target)
        target_indices_per_target: list[list[int]] = []
        for tgt_idx, target_frame in enumerate(self.cfg.target_frames):
            target_pattern = f"{target_frame.prim_path}/{self._target_labels[tgt_idx]}"
            indices, _ = resolve_matching_names(target_pattern, shape_labels)
            target_indices_per_target.append(indices)

        # Build ordered 1:1 shape/reference index lists
        # Layout per env: [source, target_0, target_1, ..., target_M-1]
        shapes_list: list[int] = []
        references_list: list[int] = []

        for env_idx in range(num_envs):
            # Source site measured w.r.t. world origin
            source_idx = source_indices[env_idx]
            shapes_list.append(source_idx)
            references_list.append(world_origin_idx)

            # Each target measured w.r.t. source
            for tgt_idx in range(self._num_targets):
                target_idx = target_indices_per_target[tgt_idx][env_idx]
                shapes_list.append(target_idx)
                references_list.append(source_idx)

        # Create SensorFrameTransform via NewtonManager
        self._sensor_index = NewtonManager.add_frame_transform_sensor(shapes_list, references_list)

        # Store reference to Newton sensor's flat transforms array
        sensor = NewtonManager._newton_frame_transform_sensors[self._sensor_index]
        self._newton_transforms = sensor.transforms
        self._data._stride = 1 + self._num_targets

        # Allocate owned buffers
        self._data._create_buffers(num_envs, self._num_targets, self._device)

        logger.info(
            f"FrameTransformer initialized: {num_envs} envs, "
            f"{self._num_targets} targets, sensor_index={self._sensor_index}"
        )

    def _update_buffers_impl(self, env_mask: wp.array):
        """Copies transforms from Newton sensor into owned buffers."""
        wp.launch(
            copy_from_newton_kernel,
            dim=(self._num_envs, 1 + self._num_targets),
            inputs=[env_mask, self._newton_transforms, self._data._stride, self._num_targets],
            outputs=[self._data._source_transforms, self._data._target_transforms],
            device=self._device,
        )

        # Compose target world transforms: source_world * target_relative
        if self._num_targets > 0:
            wp.launch(
                compose_target_world_kernel,
                dim=(self._num_envs, self._num_targets),
                inputs=[self._data._source_transforms, self._data._target_transforms],
                outputs=[self._data._target_transforms_w],
                device=self._device,
            )

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)
        self._newton_transforms = None
        self._sensor_index = None
