# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import warp as wp

from isaaclab.sensors.pva import BasePva

from .kernels import make_pva_update
from .pva_data import PvaData

if TYPE_CHECKING:
    from isaaclab.sensors.pva import PvaCfg

logger = logging.getLogger(__name__)


class Pva(BasePva):
    """Newton Pose Velocity Acceleration (PVA) sensor.

    Reads body transforms, velocities, and accelerations directly from
    Newton's simulation state (``body_q``, ``body_qd``, ``body_qdd``) to
    provide world-frame pose and body-frame velocities/accelerations.
    """

    cfg: PvaCfg
    """The configuration parameters."""

    __backend_name__: str = "newton"
    """The name of the backend for the PVA sensor."""

    supported_data_types: ClassVar[frozenset[str]] = frozenset(
        {"pose", "velocity", "acceleration", "projected_gravity"}
    )

    def __init__(self, cfg: PvaCfg):
        """Initializes the Newton PVA sensor.

        Registers a site request and, when acceleration is enabled, ``body_qdd`` with
        :class:`NewtonManager`. The site is injected into prototype builders
        before replication so it ends up in each world.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)

        self._data = PvaData(self.cfg.data_types)
        self._update_kernel = make_pva_update(self._data.data_types)
        self._site_indices: wp.array | None = None
        self._newton_model = None

        offset_xform = wp.transform(cfg.offset.pos, cfg.offset.rot)
        self._site_label = self._backend_manager.register_site(cfg.prim_path, offset_xform)
        if "acceleration" in self._data.data_types:
            self._backend_manager.request_extended_state_attribute("body_qdd")

        logger.info(f"Pva '{cfg.prim_path}': site registered (label='{self._site_label}')")

    def __str__(self) -> str:
        """String representation of the sensor instance."""
        return (
            f"Pva sensor @ '{self.cfg.prim_path}': \n"
            f"\tbackend           : newton\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self._num_envs}\n"
        )

    """
    Implementation
    """

    def _initialize_impl(self):
        """PHYSICS_READY callback: resolves site indices and stores model reference."""
        super()._initialize_impl()

        site_map = self._backend_manager.get_site_index_map()
        num_envs = self._num_envs

        if self._site_label not in site_map:
            raise ValueError(
                f"Pva '{self.cfg.prim_path}': site label '{self._site_label}' not found in the Newton site map."
            )

        global_idx, per_world = site_map[self._site_label]

        if per_world is None:
            site_indices = [global_idx] * num_envs
        else:
            if len(per_world) != num_envs:
                raise ValueError(
                    f"Pva '{self.cfg.prim_path}': site has {len(per_world)} world entries, expected {num_envs}."
                )

            site_indices: list[int] = []
            for env_idx, world_sites in enumerate(per_world):
                if len(world_sites) != 1:
                    raise ValueError(
                        f"Pva '{self.cfg.prim_path}': pattern matched {len(world_sites)} "
                        f"bodies in env {env_idx}, expected exactly 1."
                    )
                site_indices.append(world_sites[0])

        self._site_indices = wp.array(site_indices, dtype=int, device=self._device)
        self._newton_model = self._backend_manager.backend.model

        self._data.create_buffers(num_envs=num_envs, device=self._device)

        logger.info(f"Pva initialized: {num_envs} envs")

    def _update_buffers_impl(self, env_mask: wp.array):
        """Read Newton body state and compute the enabled PVA quantities."""
        if self._newton_model is None:
            raise RuntimeError(
                f"Pva '{self.cfg.prim_path}': sensor not initialized. "
                "Access sensor data only after sim.reset() has been called."
            )
        state = self._backend_manager.get_scene_data_backend().state

        wp.launch(
            self._update_kernel,
            dim=self._num_envs,
            inputs=[
                env_mask,
                self._site_indices,
                self._newton_model.shape_body,
                self._newton_model.shape_transform,
                self._newton_model.body_com,
                self._newton_model.gravity,
                self._newton_model.body_world,
                state.body_q,
                state.body_qd,
                state.body_qdd if "acceleration" in self._data.data_types else None,
            ],
            outputs=[
                self._data.pose_w.warp,
                self._data._velocity_b,
                self._data._acceleration_b,
                self._data._gravity_b,
            ],
            device=self._device,
        )

    def _invalidate_initialize_callback(self, event):
        """Release native handles; the site and ``body_qdd`` request persist on the builder."""
        super()._invalidate_initialize_callback(event)
        self._newton_model = None
        self._site_indices = None
