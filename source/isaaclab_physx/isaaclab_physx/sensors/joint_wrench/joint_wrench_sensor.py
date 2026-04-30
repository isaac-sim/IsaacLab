# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=false

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from pxr import Usd, UsdPhysics

from isaaclab.sensors.joint_wrench import BaseJointWrenchSensor
from isaaclab.sim.utils.queries import find_first_matching_prim, get_all_matching_child_prims

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
    and exposes the linear force [N] and angular torque [N·m] components in
    the child-side joint frame, with torque referenced at the child-side joint
    anchor. The root body's entry is included.

    :attr:`~isaaclab.sensors.SensorBaseCfg.prim_path` must point at either
    the articulation root prim or a parent prim containing a single
    articulation root in every environment.
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
        self._joint_pos_b: wp.array | None = None
        self._joint_quat_b: wp.array | None = None
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
        root_prim_path_expr = self._resolve_articulation_root_prim_path()
        self._root_view = self._physics_sim_view.create_articulation_view(root_prim_path_expr.replace(".*", "*"))
        if self._root_view._backend is None:
            raise RuntimeError(f"Failed to create articulation view at: {root_prim_path_expr}. Check PhysX logs.")

        self._num_bodies = self._root_view.shared_metatype.link_count
        if self._num_bodies == 0:
            raise RuntimeError(f"Joint wrench sensor matched zero bodies at '{self.cfg.prim_path}'.")

        self._data._body_names = list(self._root_view.shared_metatype.link_names)
        self._create_joint_frame_buffers()
        self._data.create_buffers(num_envs=self._num_envs, num_bodies=self._num_bodies, device=self._device)

        logger.info(f"Joint wrench sensor initialized: {self._num_envs} envs, {self._num_bodies} bodies")

    def _resolve_articulation_root_prim_path(self) -> str:
        """Resolve the articulation root prim path expression from the configured asset prim path."""
        first_env_matching_prim = find_first_matching_prim(self.cfg.prim_path)
        if first_env_matching_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        first_env_matching_prim_path = first_env_matching_prim.GetPath().pathString

        first_env_root_prims = get_all_matching_child_prims(
            first_env_matching_prim_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            and prim.GetAttribute("physxArticulation:articulationEnabled").Get() is not False,
            traverse_instance_prims=False,
        )
        if len(first_env_root_prims) == 0:
            raise RuntimeError(
                f"Failed to find an articulation when resolving '{first_env_matching_prim_path}'."
                " Please ensure that the prim has 'USD ArticulationRootAPI' applied."
            )
        if len(first_env_root_prims) > 1:
            raise RuntimeError(
                f"Failed to find a single articulation when resolving '{first_env_matching_prim_path}'."
                f" Found multiple '{first_env_root_prims}' under '{first_env_matching_prim_path}'."
                " Please ensure that there is only one articulation in the prim path tree."
            )

        first_env_root_prim_path = first_env_root_prims[0].GetPath().pathString
        root_prim_path_relative_to_prim_path = first_env_root_prim_path[len(first_env_matching_prim_path) :]
        return self.cfg.prim_path + root_prim_path_relative_to_prim_path

    def _create_joint_frame_buffers(self) -> None:
        """Create child-side joint frame transforms indexed by PhysX link order."""
        joint_pos_b = np.zeros((self._num_bodies, 3), dtype=np.float32)
        joint_quat_b = np.zeros((self._num_bodies, 4), dtype=np.float32)
        joint_quat_b[:, 3] = 1.0

        first_env_matching_prim = find_first_matching_prim(self.cfg.prim_path)
        if first_env_matching_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        link_name_to_index = {name: index for index, name in enumerate(self._data._body_names)}

        for prim in Usd.PrimRange(first_env_matching_prim):
            joint = UsdPhysics.Joint(prim)
            if not joint or joint.GetJointEnabledAttr().Get() is False:
                continue
            body1_targets = joint.GetBody1Rel().GetTargets()
            if len(body1_targets) == 0:
                continue
            body_index = link_name_to_index.get(body1_targets[0].name)
            if body_index is None:
                continue

            local_pos1 = joint.GetLocalPos1Attr().Get()
            if local_pos1 is not None:
                joint_pos_b[body_index] = (float(local_pos1[0]), float(local_pos1[1]), float(local_pos1[2]))

            local_rot1 = joint.GetLocalRot1Attr().Get()
            if local_rot1 is not None:
                local_rot1_imag = local_rot1.GetImaginary()
                joint_quat_b[body_index] = (
                    float(local_rot1_imag[0]),
                    float(local_rot1_imag[1]),
                    float(local_rot1_imag[2]),
                    float(local_rot1.GetReal()),
                )

        self._joint_pos_b = wp.array(joint_pos_b, dtype=wp.vec3f, device=self._device)
        self._joint_quat_b = wp.array(joint_quat_b, dtype=wp.quatf, device=self._device)

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
        if self._joint_pos_b is None or self._joint_quat_b is None:
            raise RuntimeError(f"Joint wrench sensor '{self.cfg.prim_path}': joint frame buffers are not initialized.")

        incoming_joint_wrench = self._root_view.get_link_incoming_joint_force().view(wp.spatial_vectorf)
        wp.launch(
            joint_wrench_split_kernel,
            dim=(self._num_envs, self._num_bodies),
            inputs=[
                env_mask,
                incoming_joint_wrench,
                self._joint_pos_b,
                self._joint_quat_b,
                self._data._force,
                self._data._torque,
            ],
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
        self._joint_pos_b = None
        self._joint_quat_b = None
        self._num_bodies = 0
        self._data._force = None
        self._data._torque = None
        self._data._body_names = []
        self._data._force_ta = None
        self._data._torque_ta = None
