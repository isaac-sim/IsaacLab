# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=false

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.sensors.joint_wrench import BaseJointWrenchSensor

from isaaclab_physx.physics import PhysxManager as SimulationManager

from .joint_wrench_sensor_data import JointWrenchSensorData
from .kernels import joint_wrench_reset_kernel, joint_wrench_split_kernel

if TYPE_CHECKING:
    import omni.physics.tensors.api as physx

    from isaaclab.sensors.joint_wrench import JointWrenchSensorCfg

logger = logging.getLogger(__name__)


class JointWrenchSensor(BaseJointWrenchSensor):
    """PhysX joint reaction wrench sensor.

    The sensor reads PhysX's incoming joint wrench for every articulation link
    and exposes the linear force [N] and angular torque [N·m] components. PhysX
    reports each wrench in the frame of ``body1`` from the USD joint, which is
    the child body's frame for the standard ``body0`` = parent, ``body1`` =
    child convention. The root body's entry is included.

    :attr:`~isaaclab.sensors.SensorBaseCfg.prim_path` must point at the
    articulation root prim in every environment.
    """

    cfg: JointWrenchSensorCfg
    """The configuration parameters."""

    __backend_name__: str = "physx"
    """The name of the backend for the joint wrench sensor."""

    def __init__(self, cfg: JointWrenchSensorCfg):
        """Initialize the PhysX joint-wrench sensor.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)

        self._data = JointWrenchSensorData()
        self._physics_sim_view = None
        self._root_view: physx.ArticulationView | None = None
        self._num_bodies: int = 0

    def __str__(self) -> str:
        """String representation of the sensor instance."""
        return (
            f"Joint wrench sensor @ '{self.cfg.prim_path}': \n"
            f"\tbackend           : physx\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of bodies  : {self._num_bodies}\n"
            f"\tbody names        : {self.body_names}\n"
        )

    """
    Properties
    """

    @property
    def body_names(self) -> list[str]:
        """Ordered names of the bodies whose incoming joint wrench is reported."""
        return self._data._body_names

    @property
    def data(self) -> JointWrenchSensorData:
        """The joint-wrench sensor data."""
        self._update_outdated_buffers()
        return self._data

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        """Reset the sensor buffers for the given environments.

        Args:
            env_ids: The environment ids to reset.
            env_mask: The mask used to reset the environments. Shape is ``(num_envs,)``.
        """
        if self._data._force is None or self._data._torque is None:
            return
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(None, env_mask)
        wp.launch(
            joint_wrench_reset_kernel,
            dim=(self._num_envs, self._num_bodies),
            inputs=[env_mask, self._data._force, self._data._torque],
            device=self._device,
        )

    """
    Implementation
    """

    def _initialize_impl(self) -> None:
        """PHYSICS_READY callback: builds the articulation view and allocates buffers."""
        super()._initialize_impl()

        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        self._root_view = self._physics_sim_view.create_articulation_view(self.cfg.prim_path.replace(".*", "*"))
        if self._root_view._backend is None:
            raise RuntimeError(f"Failed to create articulation view at: {self.cfg.prim_path}. Check PhysX logs.")

        self._num_bodies = self._root_view.shared_metatype.link_count
        if self._num_bodies == 0:
            raise RuntimeError(f"Joint wrench sensor matched zero bodies at '{self.cfg.prim_path}'.")

        self._data._body_names = list(self._root_view.shared_metatype.link_names)
        self._data.create_buffers(num_envs=self._num_envs, num_bodies=self._num_bodies, device=self._device)

        logger.info(f"Joint wrench sensor initialized: {self._num_envs} envs, {self._num_bodies} bodies")

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        """Read PhysX incoming joint wrenches and split them into force / torque buffers.

        Args:
            env_mask: A mask containing which environments need to be updated. Shape is ``(num_envs,)``.
        """
        if self._root_view is None:
            raise RuntimeError(
                f"Joint wrench sensor '{self.cfg.prim_path}': not initialized."
                " Access sensor data only after sim.reset() has been called."
            )

        incoming_joint_wrench = self._root_view.get_link_incoming_joint_force().view(wp.spatial_vectorf)
        wp.launch(
            joint_wrench_split_kernel,
            dim=(self._num_envs, self._num_bodies),
            inputs=[env_mask, incoming_joint_wrench, self._data._force, self._data._torque],
            device=self._device,
        )

    def _invalidate_initialize_callback(self, event) -> None:
        """Drop view, cached sizes, and buffers when physics stops.

        Args:
            event: An invalidate event.
        """
        super()._invalidate_initialize_callback(event)
        self._physics_sim_view = None
        self._root_view = None
        self._num_bodies = 0
        self._data._force = None
        self._data._torque = None
        self._data._body_names = []
        self._data._force_ta = None
        self._data._torque_ta = None
