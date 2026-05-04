# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp
from newton import JointType
from newton.selection import ArticulationView

from isaaclab.sensors.joint_wrench import BaseJointWrenchSensor

from isaaclab_newton.physics import NewtonManager

from .joint_wrench_sensor_data import JointWrenchSensorData
from .kernels import joint_wrench_reset_kernel, joint_wrench_to_incoming_joint_frame_kernel

if TYPE_CHECKING:
    from isaaclab.sensors.joint_wrench import JointWrenchSensorCfg

logger = logging.getLogger(__name__)


class JointWrenchSensor(BaseJointWrenchSensor):
    """Newton joint reaction wrench sensor.

    Reads Newton's ``body_parent_f`` (world-frame wrench at child COM) and
    converts each entry to the ``INCOMING_JOINT_FRAME`` convention
    (child-side joint frame, child-side joint anchor as reference point)
    before storing it in per-joint force / torque buffers.

    :attr:`~isaaclab.sensors.SensorBaseCfg.prim_path` must point at the
    articulation root prim (the one carrying ``ArticulationRootAPI``) in
    every environment; the sensor uses it as the
    :class:`~newton.selection.ArticulationView` pattern directly. ``FREE``
    and ``FIXED`` joints are excluded — neither has a meaningful joint
    anchor.
    """

    cfg: JointWrenchSensorCfg
    """The configuration parameters."""

    __backend_name__: str = "newton"
    """The name of the backend for the joint wrench sensor."""

    def __init__(self, cfg: JointWrenchSensorCfg):
        """Initialize the Newton joint-wrench sensor.

        Requests the ``body_parent_f`` extended state attribute from :class:`NewtonManager` so the
        model builder allocates it during simulation startup.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)

        self._data = JointWrenchSensorData()
        self._root_view: ArticulationView | None = None
        self._sim_bind_body_parent_f: wp.array | None = None
        self._sim_bind_body_q: wp.array | None = None
        self._sim_bind_body_com: wp.array | None = None
        self._sim_bind_joint_X_c: wp.array | None = None
        self._joint_child: wp.array | None = None
        self._num_joints: int = 0

        NewtonManager.request_extended_state_attribute("body_parent_f")

    def __str__(self) -> str:
        """String representation of the sensor instance."""
        return (
            f"Joint wrench sensor @ '{self.cfg.prim_path}': \n"
            f"\tbackend           : newton\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of joints  : {self._num_joints}\n"
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
            env_ids: the environment ids to reset.
            env_mask: the mask used to reset the environments. Shape is (num_envs)."""
        if self._data._force is None or self._data._torque is None:
            return
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        super().reset(None, env_mask)
        wp.launch(
            joint_wrench_reset_kernel,
            dim=(self._num_envs, self._num_joints),
            inputs=[env_mask, self._data._force, self._data._torque],
            device=self._device,
        )

    """
    Implementation
    """

    def _initialize_impl(self) -> None:
        """PHYSICS_READY callback: builds the articulation view and binds model / state arrays."""
        super()._initialize_impl()

        model = NewtonManager.get_model()
        state_0 = NewtonManager.get_state_0()

        self._root_view = ArticulationView(
            model,
            self.cfg.prim_path.replace(".*", "*"),
            verbose=False,
            exclude_joint_types=[JointType.FREE, JointType.FIXED],
        )
        self._num_joints = self._root_view.joint_count
        if self._num_joints == 0:
            raise RuntimeError(
                "Joint wrench sensor matched zero reportable joints (all joints are FREE or FIXED)."
                f" Check the articulation at '{self.cfg.prim_path}'."
            )

        try:
            body_parent_f = self._root_view.get_attribute("body_parent_f", state_0)
        except AttributeError as err:
            raise RuntimeError(
                f"Joint wrench sensor '{self.cfg.prim_path}': Newton state does not expose"
                " 'body_parent_f'. Construct the sensor before sim startup so the extended-state"
                " request is forwarded to the model builder."
            ) from err

        self._sim_bind_body_parent_f = body_parent_f[:, 0]
        self._sim_bind_body_q = self._root_view.get_link_transforms(state_0)[:, 0]
        self._sim_bind_body_com = self._root_view.get_attribute("body_com", model)[:, 0]
        self._sim_bind_joint_X_c = self._root_view.get_attribute("joint_X_c", model)[:, 0]

        # joint_child is per-articulation; topology is identical across envs,
        # so we take the first-env mapping as the 1-D kernel input.
        joint_child_full = self._root_view.get_attribute("joint_child", model)[:, 0]
        joint_child_np = joint_child_full.numpy()[0]
        if not all(0 <= b < self._sim_bind_body_parent_f.shape[1] for b in joint_child_np):
            raise RuntimeError(f"joint_child contains out-of-range body indices for '{self.cfg.prim_path}'")
        self._joint_child = wp.array(joint_child_np, dtype=wp.int32, device=self._device)

        link_names = list(self._root_view.link_names)
        self._data._body_names = [link_names[int(b)] for b in joint_child_np]

        self._data.create_buffers(num_envs=self._num_envs, num_joints=self._num_joints, device=self._device)

        logger.info(f"Joint wrench sensor initialized: {self._num_envs} envs, {self._num_joints} joints")

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        """Convert Newton's body_parent_f into INCOMING_JOINT_FRAME force and torque buffers.

        Args:
            env_mask: A mask containing which environments need to be updated. Shape is (num_envs)
        """
        if self._sim_bind_body_parent_f is None:
            raise RuntimeError(
                f"Joint wrench sensor '{self.cfg.prim_path}': not initialized."
                " Access sensor data only after sim.reset() has been called."
            )
        wp.launch(
            joint_wrench_to_incoming_joint_frame_kernel,
            dim=(self._num_envs, self._num_joints),
            inputs=[
                env_mask,
                self._sim_bind_body_parent_f,
                self._sim_bind_body_q,
                self._sim_bind_body_com,
                self._sim_bind_joint_X_c,
                self._joint_child,
            ],
            outputs=[self._data._force, self._data._torque],
            device=self._device,
        )

    def _invalidate_initialize_callback(self, event) -> None:
        """Drop view, cached sizes, and buffers; re-register the extended-state request.

        Args:
            event: An invalidate event.
        """
        super()._invalidate_initialize_callback(event)
        self._root_view = None
        self._sim_bind_body_parent_f = None
        self._sim_bind_body_q = None
        self._sim_bind_body_com = None
        self._sim_bind_joint_X_c = None
        self._joint_child = None
        self._num_joints = 0
        self._data._force = None
        self._data._torque = None
        self._data._body_names = []
        self._data._force_ta = None
        self._data._torque_ta = None
        NewtonManager.request_extended_state_attribute("body_parent_f")
