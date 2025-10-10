# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false


import warp as wp
import weakref

from newton.selection import ArticulationView as NewtonArticulationView
from isaaclab.utils.helpers import warn_overhead_cost
from isaaclab.assets.core.root_properties.root_data import RootData
from isaaclab.assets.core.kernels import (
    project_link_velocity_to_com_frame_masked,
    transform_CoM_pose_to_link_frame_masked,
    update_wrench_array_with_force,
    update_wrench_array_with_torque,
    update_array_with_value_masked,
    update_array_with_array_masked,
)


class Root:
    def __init__(self, root_newton_view, root_data: RootData, device: str):
        self._root_newton_view = weakref.proxy(root_newton_view)
        self._root_data = root_data
        self._device = device

    """
    Properties
    """

    @property
    def data(self) -> RootData:
        return self._root_data

    @property
    def num_instances(self) -> int:
        return self._root_newton_view.count

    @property
    def root_body_names(self) -> list[str]:
        return self._root_newton_view.body_names[0]

    @property
    def root_newton_view(self) -> NewtonArticulationView:
        """Articulation view for the asset (Newton).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_newton_view

    """
    Operations.
    """

    def reset(self, mask: wp.array):
        # use ellipses object to skip initial indices.
        if mask is None:
            mask = self._ALL_ENV_MASK

        # reset external wrench
        wp.launch(
            update_array_with_value_masked,
            dim=(self.num_instances,),
            inputs=[
                wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                self._external_wrench,
                mask,
            ],
        )

    def write_data_to_sim(self):
        """Write external wrenches and joint commands to the simulation.

        If any explicit actuators are present, then the actuator models are used to compute the
        joint commands. Otherwise, the joint commands are directly set into the simulation.

        Note:
            We write external wrench to the simulation here since this function is called before the simulation step.
            This ensures that the external wrench is applied at every simulation step.
        """
        # Wrenches are automatically applied by set_external_force_and_torque.
        # apply actuator models
        pass

    def update(self, dt: float):
        self._root_data.update(dt)

    """
    Operations - State Writers.
    """

    @warn_overhead_cost(
        "write_root_link_pose_to_sim and or write_root_com_velocity_to_sim",
        "This function splits the root state into pose and velocity. Consider using write_root_link_pose_to_sim and"
        " write_root_com_velocity_to_sim instead. In general there is no good reasons to be using states with Newton.",
    )
    def write_root_state_to_sim(self, root_state: wp.array, env_mask: wp.array | None = None):
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (num_instances, 13).
            env_mask: Environment mask. Shape is (num_instances,).
        """

        if env_mask is None:
            env_mask = self._ALL_ENV_MASK

        self.write_root_link_pose_to_sim(root_state[:, :7], env_mask=env_mask)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_mask=env_mask)

    @warn_overhead_cost(
        "write_root_state_to_sim",
        "This function splits the root state into pose and velocity. Consider using write_root_link_pose_to_sim and"
        " write_root_com_velocity_to_sim instead. In general there is no good reasons to be using states with Newton.",
    )
    def write_root_com_state_to_sim(self, root_state: wp.array, env_mask: wp.array | None = None):
        """Set the root center of mass state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (num_instances, 13).
            env_mask: Environment mask. Shape is (num_instances,).
        """

        if env_mask is None:
            env_mask = self._ALL_ENV_MASK

        self.write_root_com_pose_to_sim(root_state[:, :7], env_mask=env_mask)
        self.write_root_com_velocity_to_sim(root_state[:, 7:], env_mask=env_mask)

    @warn_overhead_cost(
        "write_root_state_to_sim",
        "This function splits the root state into pose and velocity. Consider using write_root_link_pose_to_sim and"
        " write_root_com_velocity_to_sim instead. In general there is no good reasons to be using states with Newton.",
    )
    def write_root_link_state_to_sim(self, root_state: wp.array, env_mask: wp.array | None = None):
        """Set the root link state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (x, y, z, w), and angular
        and linear velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (num_instances, 13).
            env_mask: Environment mask. Shape is (num_instances,).
        """

        if env_mask is None:
            env_mask = self._ALL_ENV_MASK

        self.write_root_link_pose_to_sim(root_state[:, :7], env_mask=env_mask)
        self.write_root_link_velocity_to_sim(root_state[:, 7:], env_mask=env_mask)

    def write_root_pose_to_sim(self, root_pose: wp.array, env_mask: wp.array | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7).
            env_mask: Environment mask. Shape is (num_instances,).
        """
        self.write_root_link_pose_to_sim(root_pose, env_mask=env_mask)

    def write_root_link_pose_to_sim(self, pose: wp.array, env_mask: wp.array | None = None):
        """Set the root link pose over selected environment indices into the simulation.


        The root pose ``wp.transformf`` comprises of the cartesian position and quaternion orientation in (x, y, z, w).

        Args:
            root_pose: Root poses in simulation frame. Shape is (num_instances, 7).
            env_mask: Environment mask. Shape is (num_instances,).
        """
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK

        # set into internal buffers
        wp.launch(
            update_array_with_array_masked,
            dim=self.num_instances,
            inputs=[
                pose,
                self._root_data.root_link_pose_w,
                env_mask,
            ],
        )
        # Need to invalidate the buffer to trigger the update with the new state.
        self._root_data._root_com_pose_w.timestamp = -1.0
        self._body_data._body_com_pose_w.timestamp = -1.0

    def write_root_com_pose_to_sim(self, root_pose: wp.array, env_mask: wp.array | None = None) -> None:
        """Set the root center of mass pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        The orientation is the orientation of the principle axes of inertia.

        Args:
            root_pose: Root center of mass poses in simulation frame. Shape is (num_instances, 7).
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # resolve all indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK

        # set into internal buffers
        wp.launch(
            update_array_with_array_masked,
            dim=self.num_instances,
            inputs=[
                root_pose,
                self._root_data.root_com_pose_w,
                env_mask,
            ],
        )
        # set link frame poses
        wp.launch(
            transform_CoM_pose_to_link_frame_masked,
            dim=self.num_instances,
            inputs=[
                self._root_data.root_com_pose_w,
                self._root_data.root_com_pos_b,
                self._root_data.root_link_pose_w,
                env_mask,
            ],
        )
        # Need to invalidate the buffer to trigger the update with the new state.
        self._body_data._body_com_pose_w.timestamp = -1.0

    def write_root_velocity_to_sim(self, root_velocity: wp.array, env_mask: wp.array | None = None) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises angular velocity (x, y, z) and linear velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6).
            env_mask: Environment mask. Shape is (num_instances,).
        """
        self.write_root_com_velocity_to_sim(root_velocity=root_velocity, env_mask=env_mask)

    def write_root_com_velocity_to_sim(self, root_velocity: wp.array, env_mask: wp.array | None = None) -> None:
        """Set the root center of mass velocity over selected environment indices into the simulation.

        The velocity comprises angular velocity (x, y, z) and linear velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's center of mass rather than the roots frame.

        Args:
            root_velocity: Root center of mass velocities in simulation world frame. Shape is (num_instances, 6).
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # resolve all indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK

        # set into internal buffers
        wp.launch(
            update_array_with_array_masked,
            dim=self.num_instances,
            inputs=[
                root_velocity,
                self._root_data.root_com_vel_w,
                env_mask,
            ],
        )
        # Need to invalidate the buffer to trigger the update with the new state.
        self._root_data._root_link_vel_w.timestamp = -1.0
        self._root_data._root_link_vel_b.timestamp = -1.0
        self._root_data._root_com_vel_b.timestamp = -1.0

    def write_root_link_velocity_to_sim(self, root_velocity: wp.array, env_mask: wp.array | None = None) -> None:
        """Set the root link velocity over selected environment indices into the simulation.

        The velocity comprises angular velocity (x, y, z) and linear velocity (x, y, z) in that order.
        NOTE: This sets the velocity of the root's frame rather than the roots center of mass.

        Args:
            root_velocity: Root frame velocities in simulation world frame. Shape is (num_instances, 6).
            env_mask: Environment mask. Shape is (num_instances,).
        """
        # resolve all indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        # update the root link velocity
        wp.launch(
            update_array_with_array_masked,
            dim=self.num_instances,
            inputs=[
                root_velocity,
                self._root_data.root_link_vel_w,
                env_mask,
            ],
        )
        # set into internal buffers
        wp.launch(
            project_link_velocity_to_com_frame_masked,
            dim=self.num_instances,
            inputs=[
                root_velocity,
                self._root_data.root_link_pose_w,
                self._root_data.root_com_pos_b,
                self._root_data.root_com_vel_w,
                env_mask,
            ],
        )
        # Need to invalidate the buffer to trigger the update with the new state.
        self._root_data._root_link_vel_b.timestamp = -1.0
        self._root_data._root_com_vel_b.timestamp = -1.0

    """
    Operations - Setters.
    """

    def set_external_force_and_torque(
        self,
        forces: wp.array,
        torques: wp.array,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set external force and torque to apply on the asset's bodies in their local frame.

        For many applications, we want to keep the applied external force on rigid bodies constant over a period of
        time (for instance, during the policy control). This function allows us to store the external force and torque
        into buffers which are then applied to the simulation at every step.

        .. caution::
            If the function is called with empty forces and torques, then this function disables the application
            of external wrench to the simulation.

            .. code-block:: python

                # example of disabling external wrench
                asset.set_external_force_and_torque(forces=wp.zeros(0, 3), torques=wp.zeros(0, 3))

        .. note::
            This function does not apply the external wrench to the simulation. It only fills the buffers with
            the desired values. To apply the external wrench, call the :meth:`write_data_to_sim` function
            right before the simulation step.

        Args:
            forces: External forces in bodies' local frame. Shape is (num_instances, num_bodies, 3).
            torques: External torques in bodies' local frame. Shape is (num_instances, num_bodies, 3).
            body_mask: The body mask. Shape is (num_bodies).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # resolve indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        # Check if there are any external forces or torques
        if (forces is not None) or (torques is not None):
            self.has_external_wrench = True
            if forces is not None:
                wp.launch(
                    update_wrench_array_with_force,
                    dim=self.num_instances,
                    inputs=[
                        forces,
                        self._external_wrench,
                        env_mask,
                    ],
                )
            if torques is not None:
                wp.launch(
                    update_wrench_array_with_torque,
                    dim=self.num_instances,
                    inputs=[
                        torques,
                        self._external_wrench,
                        env_mask,
                    ],
                )

    def _create_buffers(self):
        # constants
        self._ALL_ENV_MASK = wp.ones((self.num_instances,), dtype=wp.bool, device=self._device)
        # masks
        self._ENV_MASK = wp.zeros((self.num_instances,), dtype=wp.bool, device=self._device)
        # external wrench
        self._external_wrench = wp.zeros((self.num_instances), dtype=wp.spatial_vectorf, device=self._device)