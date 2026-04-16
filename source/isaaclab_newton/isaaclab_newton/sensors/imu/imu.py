# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.sensors.imu import BaseImu

from isaaclab_newton.physics import NewtonManager

from .imu_data import ImuData
from .kernels import imu_copy_kernel, imu_reset_kernel

if TYPE_CHECKING:
    from newton.sensors import SensorIMU as NewtonSensorIMU

    from isaaclab.sensors.imu import ImuCfg

logger = logging.getLogger(__name__)


class Imu(BaseImu):
    """Newton Inertial Measurement Unit (IMU) sensor.

    Wrapper around ``newton.sensors.SensorIMU`` providing angular velocity
    (gyroscope) and linear acceleration (accelerometer) in the sensor's
    body frame.
    """

    cfg: ImuCfg
    """The configuration parameters."""

    __backend_name__: str = "newton"
    """The name of the backend for the IMU sensor."""

    def __init__(self, cfg: ImuCfg):
        """Initializes the Newton IMU sensor.

        Registers a site request and the ``body_qdd`` state attribute with
        :class:`NewtonManager`. The site is injected into prototype builders
        before replication so it ends up in each world.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)

        self._data = ImuData()
        self._sensor_index: int | None = None
        self._newton_sensor: NewtonSensorIMU | None = None

        offset_xform = wp.transform(cfg.offset.pos, cfg.offset.rot)
        self._site_label: str = NewtonManager.cl_register_site(cfg.prim_path, offset_xform)
        NewtonManager.request_extended_state_attribute("body_qdd")

        logger.info(f"IMU '{cfg.prim_path}': site registered (label='{self._site_label}')")

    def __str__(self) -> str:
        """String representation of the sensor instance."""
        return (
            f"IMU sensor @ '{self.cfg.prim_path}': \n"
            f"\tbackend           : newton\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self._num_envs}\n"
        )

    """
    Properties
    """

    @property
    def data(self) -> ImuData:
        """The IMU sensor data."""
        self._update_outdated_buffers()
        return self._data

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        """Reset the sensor for the given environments.

        Zeroes out angular velocity and linear acceleration buffers for the
        specified environments.

        Args:
            env_ids: Environment indices to reset. Defaults to all environments.
            env_mask: Boolean mask of environments to reset. Mutually exclusive with *env_ids*.
        """
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(None, env_mask)
        wp.launch(
            imu_reset_kernel,
            dim=self._num_envs,
            inputs=[env_mask, self._data._lin_acc_b, self._data._ang_vel_b],
            device=self._device,
        )

    """
    Implementation
    """

    def _initialize_impl(self):
        """PHYSICS_READY callback: resolves site indices and creates the native SensorIMU."""
        super()._initialize_impl()

        site_map = NewtonManager._cl_site_index_map
        num_envs = self._num_envs

        if self._site_label not in site_map:
            raise ValueError(
                f"IMU '{self.cfg.prim_path}': site label '{self._site_label}' "
                "not found in NewtonManager._cl_site_index_map."
            )

        global_idx, per_world = site_map[self._site_label]

        if per_world is None:
            # Global site (body=-1, i.e. world frame): replicate across envs.
            site_indices = [global_idx] * num_envs
        else:
            if len(per_world) != num_envs:
                raise ValueError(
                    f"IMU '{self.cfg.prim_path}': site has {len(per_world)} world entries, expected {num_envs}."
                )

            site_indices: list[int] = []
            for env_idx, world_sites in enumerate(per_world):
                if len(world_sites) != 1:
                    raise ValueError(
                        f"IMU '{self.cfg.prim_path}': pattern matched {len(world_sites)} "
                        f"bodies in env {env_idx}, expected exactly 1."
                    )
                site_indices.append(world_sites[0])

        self._sensor_index = NewtonManager.add_imu_sensor(site_indices)
        self._newton_sensor = NewtonManager._newton_imu_sensors[self._sensor_index]

        self._data.create_buffers(num_envs=num_envs, device=self._device)

        logger.info(f"IMU initialized: {num_envs} envs, sensor_index={self._sensor_index}")

    def _update_buffers_impl(self, env_mask: wp.array):
        """Copies accelerometer/gyroscope data from native Newton sensor into owned buffers."""
        if self._newton_sensor is None:
            raise RuntimeError(
                f"IMU '{self.cfg.prim_path}': sensor not initialized. "
                "Access sensor data only after sim.reset() has been called."
            )
        wp.launch(
            imu_copy_kernel,
            dim=self._num_envs,
            inputs=[env_mask, self._newton_sensor.accelerometer, self._newton_sensor.gyroscope],
            outputs=[self._data._lin_acc_b, self._data._ang_vel_b],
            device=self._device,
        )

    def _invalidate_initialize_callback(self, event):
        """Clears references to the native Newton sensor and re-registers site/attributes.

        ``NewtonManager.close()`` calls ``clear()`` before dispatching ``STOP``,
        so ``_cl_pending_sites`` and ``_pending_extended_state_attributes`` are
        already empty when this callback fires.  Re-registering here ensures the
        site and ``body_qdd`` attribute survive a close/reinit cycle.
        """
        super()._invalidate_initialize_callback(event)
        self._newton_sensor = None
        self._sensor_index = None

        # Zero out data buffers so stale data is not served between STOP and reinit.
        if self._data._ang_vel_b is not None:
            self._data._ang_vel_b.zero_()
        if self._data._lin_acc_b is not None:
            self._data._lin_acc_b.zero_()

        # Re-register so a subsequent start_simulation picks them up.
        offset_xform = wp.transform(self.cfg.offset.pos, self.cfg.offset.rot)
        self._site_label = NewtonManager.cl_register_site(self.cfg.prim_path, offset_xform)
        NewtonManager.request_extended_state_attribute("body_qdd")
