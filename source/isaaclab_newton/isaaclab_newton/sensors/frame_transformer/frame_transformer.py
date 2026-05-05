# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.sensors.frame_transformer.base_frame_transformer import BaseFrameTransformer

from isaaclab_newton.physics import NewtonManager

from .frame_transformer_data import FrameTransformerData
from .frame_transformer_kernels import compose_target_world_kernel, copy_from_newton_kernel

if TYPE_CHECKING:
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg

logger = logging.getLogger(__name__)


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
        self._newton_transforms = None
        self._stride: int = 0

        self._sensor_index: int | None = None
        self._source_frame_body_name: str = cfg.prim_path.rsplit("/", 1)[-1]

        # Register world-origin reference site
        self._world_origin_label = NewtonManager.cl_register_site(None, wp.transform())

        # Register source site
        source_offset = wp.transform(cfg.source_frame_offset.pos, cfg.source_frame_offset.rot)
        self._source_label = NewtonManager.cl_register_site(cfg.prim_path, source_offset)

        # Register target sites
        self._target_labels: list[str] = []
        self._target_frame_body_names: list[str] = []
        self._num_targets: int = 0

        for target_frame in cfg.target_frames:
            target_offset = wp.transform(target_frame.offset.pos, target_frame.offset.rot)
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

        num_envs = self._num_envs
        site_map = NewtonManager._cl_site_index_map

        # Resolve and validate per-env site indices
        assert self._world_origin_label in site_map
        world_origin_idx, _ = site_map[self._world_origin_label]
        source_indices, target_per_world = self._validate_site_map(
            self._source_label,
            self.cfg.prim_path,
            self._target_labels,
            [t.prim_path for t in self.cfg.target_frames],
            site_map,
            num_envs,
        )

        # Expand targets and build sensor index lists
        expanded_names, target_indices_per_target, shapes_list, references_list = self._build_sensor_index_lists(
            source_indices,
            target_per_world,
            self._target_frame_body_names,
            NewtonManager._builder.shape_label,
            world_origin_idx,
            num_envs,
        )

        # Update instance state with expanded values
        self._num_targets = len(target_indices_per_target)
        self._target_frame_names = expanded_names
        self._target_frame_body_names = expanded_names
        self._data._target_frame_names = expanded_names

        # Create SensorFrameTransform via NewtonManager
        self._sensor_index = NewtonManager.add_frame_transform_sensor(shapes_list, references_list)

        # Store reference to Newton sensor's flat transforms array
        sensor = NewtonManager._newton_frame_transform_sensors[self._sensor_index]
        self._newton_transforms = sensor.transforms
        self._stride = 1 + self._num_targets

        # Allocate owned buffers
        self._data._create_buffers(num_envs, self._num_targets, self._device)

        logger.info(
            f"FrameTransformer initialized: {num_envs} envs, "
            f"{self._num_targets} targets, sensor_index={self._sensor_index}"
        )

    @staticmethod
    def _validate_site_map(
        source_label: str,
        source_prim_path: str,
        target_labels: list[str],
        target_prim_paths: list[str],
        site_map: dict,
        num_envs: int,
    ) -> tuple[list[int], list[list[list[int]]]]:
        """Validate per-env site counts and return resolved index arrays.

        Args:
            source_label: Site label for the source frame.
            source_prim_path: Config prim path used in error messages.
            target_labels: Site labels for each target frame (in order).
            target_prim_paths: Config prim paths used in error messages.
            site_map: ``NewtonManager._cl_site_index_map``.
            num_envs: Expected number of environments.

        Returns:
            ``(source_indices, target_per_world)`` where ``source_indices[e]`` is the
            single source site index for env ``e``, and ``target_per_world[t][e]`` is
            the list of site indices for target ``t`` in env ``e``.

        Raises:
            ValueError: If the source has the wrong world count, or any env has a
                count other than 1. If any target has zero matches, non-uniform
                counts across envs, or a world-count mismatch.
        """
        assert source_label in site_map, (
            f"FrameTransformer source '{source_prim_path}' (site label '{source_label}') "
            "not found in NewtonManager._cl_site_index_map."
        )
        _, source_per_world = site_map[source_label]
        if len(source_per_world) != num_envs:
            raise ValueError(
                f"FrameTransformer source '{source_prim_path}' has {len(source_per_world)} "
                f"world entries in the site map, expected {num_envs}."
            )
        for env_idx, world_sites in enumerate(source_per_world):
            if len(world_sites) != 1:
                raise ValueError(
                    f"FrameTransformer source pattern '{source_prim_path}' matched "
                    f"{len(world_sites)} bodies in env {env_idx}, expected exactly 1. "
                    f"Source patterns must resolve to a single rigid body per environment."
                )
        source_indices: list[int] = [w[0] for w in source_per_world]

        target_per_world: list[list[list[int]]] = []
        for tgt_idx, label in enumerate(target_labels):
            assert label in site_map, (
                f"FrameTransformer target '{target_prim_paths[tgt_idx]}' (site label '{label}') "
                "not found in NewtonManager._cl_site_index_map."
            )
            _, per_world = site_map[label]
            if len(per_world) != num_envs:
                raise ValueError(
                    f"FrameTransformer target '{target_prim_paths[tgt_idx]}' has "
                    f"{len(per_world)} world entries, expected {num_envs}."
                )
            lengths = [len(w) for w in per_world]
            if len(set(lengths)) != 1:
                raise ValueError(
                    f"FrameTransformer target pattern '{target_prim_paths[tgt_idx]}' matched "
                    f"different numbers of bodies across envs: {lengths}. "
                    f"All environments must have the same number of matches."
                )
            if lengths[0] == 0:
                raise ValueError(
                    f"FrameTransformer target pattern '{target_prim_paths[tgt_idx]}' "
                    f"matched no bodies in any environment."
                )
            target_per_world.append(per_world)

        return source_indices, target_per_world

    @staticmethod
    def _build_sensor_index_lists(
        source_indices: list[int],
        target_per_world: list[list[list[int]]],
        target_frame_body_names: list[str],
        shape_labels: list[str],
        world_origin_idx: int,
        num_envs: int,
    ) -> tuple[list[str], list[list[int]], list[int], list[int]]:
        """Expand per-world target sublists and build sensor index lists.

        Args:
            source_indices: Per-env source site indices, length ``num_envs``.
            target_per_world: Per-target-config, per-world, per-body site indices.
                Shape: ``[num_target_cfgs][num_envs][n_bodies_per_env]``.
            target_frame_body_names: Config-level name for each target config entry.
            shape_labels: ``builder.shape_label`` — maps shape index to its label string.
                Site labels have the form ``"{body_name}/{site_label}"``; the body name
                is extracted for wildcard expansion.
            world_origin_idx: Global world-origin site index.
            num_envs: Number of environments.

        Returns:
            ``(expanded_names, target_indices_per_target, shapes_list, references_list)``
            where ``expanded_names[k]`` is the resolved name for expanded target ``k``,
            ``target_indices_per_target[k][e]`` is the site index for expanded target ``k``
            in env ``e``, ``shapes_list`` and ``references_list`` are 1:1 sensor inputs.
        """
        target_indices_per_target: list[list[int]] = []
        expanded_names: list[str] = []

        for tgt_idx, per_world in enumerate(target_per_world):
            n_bodies = len(per_world[0])  # uniform across envs (validated)
            for k in range(n_bodies):
                per_env = [per_world[env_idx][k] for env_idx in range(num_envs)]
                target_indices_per_target.append(per_env)
                # For wildcards (n_bodies > 1), derive the bare body name from the
                # site label ("{body_path}/{site_label}") using env 0.
                if n_bodies > 1:
                    site_idx = per_world[0][k]
                    expanded_names.append(shape_labels[site_idx].rsplit("/", 2)[-2])
                else:
                    expanded_names.append(target_frame_body_names[tgt_idx])

        num_targets = len(target_indices_per_target)
        shapes_list: list[int] = []
        references_list: list[int] = []

        for env_idx in range(num_envs):
            source_idx = source_indices[env_idx]
            shapes_list.append(source_idx)
            references_list.append(world_origin_idx)
            for tgt_idx in range(num_targets):
                target_idx = target_indices_per_target[tgt_idx][env_idx]
                shapes_list.append(target_idx)
                references_list.append(source_idx)

        return expanded_names, target_indices_per_target, shapes_list, references_list

    def _update_buffers_impl(self, env_mask: wp.array):
        """Copies transforms from Newton sensor into owned buffers."""
        if self._newton_transforms is None:
            raise RuntimeError(f"FrameTransformer '{self.cfg.prim_path}': sensor is not initialized")
        wp.launch(
            copy_from_newton_kernel,
            dim=(self._num_envs, 1 + self._num_targets),
            inputs=[env_mask, self._newton_transforms, self._stride],
            outputs=[self._data._source_transforms, self._data._target_transforms],
            device=self._device,
        )

        # Compose target world transforms: source_world * target_relative
        if self._num_targets > 0:
            wp.launch(
                compose_target_world_kernel,
                dim=(self._num_envs, self._num_targets),
                inputs=[env_mask, self._data._source_transforms, self._data._target_transforms],
                outputs=[self._data._target_transforms_w],
                device=self._device,
            )

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Clears references to the native sensor and re-registers sites.

        ``NewtonManager.close()`` calls ``clear()`` before dispatching ``STOP``,
        so ``_cl_pending_sites`` is already empty when this callback fires.
        Re-registering here ensures sites survive a close/reinit cycle.
        """
        super()._invalidate_initialize_callback(event)
        self._newton_transforms = None
        self._sensor_index = None

        # Re-register sites so a subsequent start_simulation picks them up.
        self._world_origin_label = NewtonManager.cl_register_site(None, wp.transform())

        source_offset = wp.transform(self.cfg.source_frame_offset.pos, self.cfg.source_frame_offset.rot)
        self._source_label = NewtonManager.cl_register_site(self.cfg.prim_path, source_offset)

        self._target_labels = []
        for target_frame in self.cfg.target_frames:
            target_offset = wp.transform(target_frame.offset.pos, target_frame.offset.rot)
            label = NewtonManager.cl_register_site(target_frame.prim_path, target_offset)
            self._target_labels.append(label)
