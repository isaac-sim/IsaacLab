# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from pxr import UsdGeom

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.pva import BasePva

from isaaclab_newton.physics import NewtonManager

from .kernels import pva_reset_kernel, pva_update_kernel
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

    def __init__(self, cfg: PvaCfg):
        """Initializes the Newton PVA sensor.

        Registers a site request and the ``body_qdd`` state attribute with
        :class:`NewtonManager`. The site is injected into prototype builders
        before replication so it ends up in each world.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)

        self._data = PvaData()
        self._site_indices: wp.array | None = None
        self._newton_model = None

        offset_xform = wp.transform(cfg.offset.pos, cfg.offset.rot)
        self._site_label = NewtonManager.cl_register_site(cfg.prim_path, offset_xform)
        NewtonManager.request_extended_state_attribute("body_qdd")

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
    Properties
    """

    @property
    def data(self) -> PvaData:
        """The PVA sensor data."""
        self._update_outdated_buffers()
        return self._data

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        """Reset the sensor for the given environments.

        Zeroes out all PVA buffers for the specified environments.

        Args:
            env_ids: Environment indices to reset. Defaults to all environments.
            env_mask: Boolean mask of environments to reset. Mutually exclusive with *env_ids*.
        """
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(None, env_mask)
        wp.launch(
            pva_reset_kernel,
            dim=self._num_envs,
            inputs=[
                env_mask,
                self._data._pose_w,
                self._data._pos_w,
                self._data._quat_w,
                self._data._projected_gravity_b,
                self._data._lin_vel_b,
                self._data._ang_vel_b,
                self._data._lin_acc_b,
                self._data._ang_acc_b,
            ],
            device=self._device,
        )

    """
    Implementation
    """

    def _initialize_impl(self):
        """PHYSICS_READY callback: resolves site indices and stores model reference."""
        super()._initialize_impl()

        site_map = NewtonManager._cl_site_index_map
        num_envs = self._num_envs

        if self._site_label not in site_map:
            raise ValueError(
                f"Pva '{self.cfg.prim_path}': site label '{self._site_label}' "
                "not found in NewtonManager._cl_site_index_map."
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
        self._newton_model = NewtonManager._model

        self._data.create_buffers(num_envs=num_envs, device=self._device)

        logger.info(f"Pva initialized: {num_envs} envs")

    def _update_buffers_impl(self, env_mask: wp.array):
        """Reads Newton body state and computes all PVA quantities."""
        if self._newton_model is None:
            raise RuntimeError(
                f"Pva '{self.cfg.prim_path}': sensor not initialized. "
                "Access sensor data only after sim.reset() has been called."
            )
        state = NewtonManager._state_0

        wp.launch(
            pva_update_kernel,
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
                state.body_qdd,
            ],
            outputs=[
                self._data._pose_w,
                self._data._pos_w,
                self._data._quat_w,
                self._data._projected_gravity_b,
                self._data._lin_vel_b,
                self._data._ang_vel_b,
                self._data._lin_acc_b,
                self._data._ang_acc_b,
            ],
            device=self._device,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "acceleration_visualizer"):
                self.acceleration_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.acceleration_visualizer.set_visibility(True)
        else:
            if hasattr(self, "acceleration_visualizer"):
                self.acceleration_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self._newton_model is None:
            return
        # base position (offset upward for visibility)
        base_pos_w = self._data.pos_w.torch.clone()
        base_pos_w[:, 2] += 0.5
        # arrow scale
        default_scale = self.acceleration_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(self._data.lin_acc_b.torch.shape[0], 1)
        # arrow direction from acceleration
        up_axis = UsdGeom.GetStageUpAxis(self.stage)
        pos_w_torch = self._data.pos_w.torch
        quat_w_torch = self._data.quat_w.torch
        lin_acc_b_torch = self._data.lin_acc_b.torch
        quat_opengl = math_utils.quat_from_matrix(
            math_utils.create_rotation_matrix_from_view(
                pos_w_torch,
                pos_w_torch + math_utils.quat_apply(quat_w_torch, lin_acc_b_torch),
                up_axis=up_axis,
                device=self._device,
            )
        )
        quat_w = math_utils.convert_camera_frame_orientation_convention(quat_opengl, "opengl", "world")
        self.acceleration_visualizer.visualize(base_pos_w, quat_w, arrow_scale)

    def _invalidate_initialize_callback(self, event):
        """Clears references for re-initialization and re-registers with NewtonManager."""
        super()._invalidate_initialize_callback(event)
        self._newton_model = None
        self._site_indices = None

        # Zero out data buffers so stale data is not served between STOP and reinit.
        for buf in [
            self._data._pose_w,
            self._data._pos_w,
            self._data._quat_w,
            self._data._projected_gravity_b,
            self._data._lin_vel_b,
            self._data._ang_vel_b,
            self._data._lin_acc_b,
            self._data._ang_acc_b,
        ]:
            if buf is not None:
                buf.zero_()

        # Re-register so a subsequent start_simulation picks them up.
        offset_xform = wp.transform(self.cfg.offset.pos, self.cfg.offset.rot)
        self._site_label = NewtonManager.cl_register_site(self.cfg.prim_path, offset_xform)
        NewtonManager.request_extended_state_attribute("body_qdd")
