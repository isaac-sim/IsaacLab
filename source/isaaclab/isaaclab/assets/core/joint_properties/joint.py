# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false


from collections.abc import Sequence

import omni.log
import weakref
import warp as wp
from newton.selection import ArticulationView as NewtonArticulationView
from newton.solvers import SolverMuJoCo, SolverNotifyFlags

import isaaclab.utils.string as string_utils
from isaaclab.sim._impl.newton_manager import NewtonManager

from isaaclab.assets.core.joint_properties.joint_data import JointData
from isaaclab.assets.core.kernels import (
    generate_mask_from_ids,
    update_joint_array,
    update_joint_array_int,
    update_joint_array_with_value,
    update_joint_array_with_value_int,
    update_joint_limits,
    update_joint_limits_value_vec2f,
    update_joint_pos_with_limits,
    update_joint_pos_with_limits_value_vec2f,
)


class Joint:
    def __init__(self, root_newton_view, joint_data: JointData, device: str):
        self._root_newton_view = weakref.proxy(root_newton_view)
        self._data = joint_data
        self._device = device

    @property
    def data(self) -> JointData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._root_newton_view.count

    @property
    def num_joints(self) -> int:
        """Number of joints in articulation."""
        return self._root_newton_view.joint_dof_count

    @property
    def joint_names(self) -> list[str]:
        """Ordered names of joints in articulation."""
        return self._root_newton_view.joint_dof_names

    @property
    def root_newton_view(self) -> NewtonArticulationView:
        """Articulation view for the asset (Newton).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_newton_view

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_joints(
        self, name_keys: str | Sequence[str], joint_subset: list[str] | None = None, preserve_order: bool = False
    ) -> tuple[wp.array, list[str], list[int]]:
        """Find joints in the articulation based on the name keys.

        Please see the :func:`isaaclab.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint names.
            joint_subset: A subset of joints to search for. Defaults to None, which means all joints
                in the articulation are searched.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the joint mask, names, and indices.
        """
        if joint_subset is None:
            joint_subset = self.joint_names
        # find joints
        indices, names = string_utils.resolve_matching_names(name_keys, joint_subset, preserve_order)
        self._JOINT_MASK.fill_(False)
        mask = wp.clone(self._JOINT_MASK)
        wp.launch(
            generate_mask_from_ids,
            dim=(len(indices),),
            inputs=[
                mask,
                wp.array(indices, dtype=wp.int32, device=self.device),
            ],
        )
        return mask, names, indices

    """
    Operations - State Writers.
    """

    def write_joint_state_to_sim(
        self,
        position: wp.array,
        velocity: wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Write joint positions and velocities to the simulation.

        Args:
            position: Joint positions. Shape is (num_instances, num_joints).
            velocity: Joint velocities. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # set into simulation
        self.write_joint_position_to_sim(position, joint_mask=joint_mask, env_mask=env_mask)
        self.write_joint_velocity_to_sim(velocity, joint_mask=joint_mask, env_mask=env_mask)

    def write_joint_position_to_sim(
        self,
        position: wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Write joint positions to the simulation.

        Args:
            position: Joint positions. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # resolve indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if joint_mask is None:
            joint_mask = self._ALL_JOINT_MASK
        # set into internal buffers
        wp.launch(
            update_joint_array,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                position,
                self._data.sim_bind_joint_pos,
                env_mask,
                joint_mask,
            ],
        )
        # invalidate buffers to trigger the update with the new root pose.
        self._data._body_com_pose_w.timestamp = -1.0

    def write_joint_velocity_to_sim(
        self,
        velocity: wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Write joint velocities to the simulation.

        Args:
            velocity: Joint velocities. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if joint_mask is None:
            joint_mask = self._ALL_JOINT_MASK
        # update joint velocity
        wp.launch(
            update_joint_array,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                velocity,
                self._data.sim_bind_joint_vel,
                env_mask,
                joint_mask,
            ],
        )
        # update previous joint velocity
        wp.launch(
            update_joint_array,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                velocity,
                self._data._previous_joint_vel,
                env_mask,
                joint_mask,
            ],
        )
        # Set joint acceleration to 0.0
        wp.launch(
            update_joint_array_with_value,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                0.0,
                self._data._joint_acc.data,
                env_mask,
                joint_mask,
            ],
        )
        # Need to invalidate the buffer to trigger the update with the new root pose.
        self._data._body_link_vel_w.timestamp = -1.0

    """
    Operations - Simulation Parameters Writers.
    """

    def write_joint_control_mode_to_sim(
        self,
        control_mode: wp.array | int,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ):
        """Write joint control mode into the simulation.

        Args:
            control_mode: Joint control mode. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).

        Raises:
            ValueError: If the control mode is invalid.
        """
        # resolve indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if joint_mask is None:
            joint_mask = self._ALL_JOINT_MASK
        # set to simulation
        if isinstance(control_mode, int):
            wp.launch(
                update_joint_array_with_value_int,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    control_mode,
                    self._data.sim_bind_joint_control_mode_sim,
                    env_mask,
                    joint_mask,
                ],
            )
        else:
            wp.launch(
                update_joint_array_int,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    control_mode,
                    self._data.sim_bind_joint_control_mode_sim,
                    env_mask,
                    joint_mask,
                ],
            )
        # tell the physics engine to use the new control mode
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_stiffness_to_sim(
        self,
        stiffness: wp.array | float,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint stiffness into the simulation.

        Args:
            stiffness: Joint stiffness. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if joint_mask is None:
            joint_mask = self._ALL_JOINT_MASK
        # set into internal buffers
        if isinstance(stiffness, float):
            wp.launch(
                update_joint_array_with_value,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    stiffness,
                    self._data.sim_bind_joint_stiffness_sim,
                    env_mask,
                    joint_mask,
                ],
            )
        else:
            wp.launch(
                update_joint_array,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    stiffness,
                    self._data.sim_bind_joint_stiffness_sim,
                    env_mask,
                    joint_mask,
                ],
            )
        # tell the physics engine to use the new stiffness
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_damping_to_sim(
        self,
        damping: wp.array | float,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint damping into the simulation.

        Args:
            damping: Joint damping. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if joint_mask is None:
            joint_mask = self._ALL_JOINT_MASK
        # set into internal buffers
        if isinstance(damping, float):
            wp.launch(
                update_joint_array_with_value,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    damping,
                    self._data.sim_bind_joint_damping_sim,
                    env_mask,
                    joint_mask,
                ],
            )
        else:
            wp.launch(
                update_joint_array,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    damping,
                    self._data.sim_bind_joint_damping_sim,
                    env_mask,
                    joint_mask,
                ],
            )
        # tell the physics engine to use the new damping
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_position_limit_to_sim(
        self,
        upper_limits: wp.array | float,
        lower_limits: wp.array | float,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint position limits into the simulation.

        Args:
            upper_limits: Joint upper limits. Shape is (num_instances, num_joints).
            lower_limits: Joint lower limits. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if joint_mask is None:
            joint_mask = self._ALL_JOINT_MASK
        if isinstance(upper_limits, float) and isinstance(lower_limits, float):
            # update default joint pos to stay within the new limits
            wp.launch(
                update_joint_pos_with_limits_value_vec2f,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    wp.vec2f(upper_limits, lower_limits),
                    self._data.default_joint_pos,
                    env_mask,
                    joint_mask,
                ],
            )
            # set into simulation
            wp.launch(
                update_joint_limits_value_vec2f,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    wp.vec2f(upper_limits, lower_limits),
                    self.cfg.soft_joint_pos_limit_factor,
                    self._data.sim_bind_joint_pos_limits_lower,
                    self._data.sim_bind_joint_pos_limits_upper,
                    self._data.soft_joint_pos_limits,
                    env_mask,
                    joint_mask,
                ],
            )
        elif isinstance(upper_limits, wp.array) and isinstance(lower_limits, wp.array):
            # update default joint pos to stay within the new limits
            wp.launch(
                update_joint_pos_with_limits,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    lower_limits,
                    upper_limits,
                    self._data.default_joint_pos,
                    env_mask,
                    joint_mask,
                ],
            )
            # set into simulation
            wp.launch(
                update_joint_limits,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    lower_limits,
                    upper_limits,
                    self.cfg.soft_joint_pos_limit_factor,
                    self._data.sim_bind_joint_pos_limits_lower,
                    self._data.sim_bind_joint_pos_limits_upper,
                    self._data.soft_joint_pos_limits,
                    env_mask,
                    joint_mask,
                ],
            )
        else:
            raise NotImplementedError("Only float or wp.array of float is supported for upper and lower limits.")
        # tell the physics engine to use the new limits
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_velocity_limit_to_sim(
        self,
        limits: wp.array | float,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint max velocity to the simulation.

        The velocity limit is used to constrain the joint velocities in the physics engine. The joint will only
        be able to reach this velocity if the joint's effort limit is sufficiently large. If the joint is moving
        faster than this velocity, the physics engine will actually try to brake the joint to reach this velocity.

        .. warn:: This function is ignored when using the Mujoco solver.

        Args:
            limits: Joint max velocity. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # Warn if using Mujoco solver
        if isinstance(NewtonManager._solver, SolverMuJoCo):
            omni.log.warn("write_joint_velocity_limit_to_sim is ignored when using the Mujoco solver.")

        # resolve indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if joint_mask is None:
            joint_mask = self._ALL_JOINT_MASK
        # set into internal buffers
        if isinstance(limits, float):
            wp.launch(
                update_joint_array_with_value,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    limits,
                    self._data.sim_bind_joint_vel_limits_sim,
                    env_mask,
                    joint_mask,
                ],
            )
        else:
            wp.launch(
                update_joint_array,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    limits,
                    self._data.sim_bind_joint_vel_limits_sim,
                    env_mask,
                    joint_mask,
                ],
            )
        # tell the physics engine to use the new limits
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_effort_limit_to_sim(
        self,
        limits: wp.array | float,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint effort limits into the simulation.

        The effort limit is used to constrain the computed joint efforts in the physics engine. If the
        computed effort exceeds this limit, the physics engine will clip the effort to this value.

        Args:
            limits: Joint torque limits. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # note: This function isn't setting the values for actuator models. (#128)
        # resolve indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if joint_mask is None:
            joint_mask = self._ALL_JOINT_MASK
        # set into internal buffers
        if isinstance(limits, float):
            wp.launch(
                update_joint_array_with_value,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    limits,
                    self._data.sim_bind_joint_effort_limits_sim,
                    env_mask,
                    joint_mask,
                ],
            )
        else:
            wp.launch(
                update_joint_array,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    limits,
                    self._data.sim_bind_joint_effort_limits_sim,
                    env_mask,
                    joint_mask,
                ],
            )
        # tell the physics engine to use the new limits
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_armature_to_sim(
        self,
        armature: wp.array | float,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Write joint armature into the simulation.

        The armature is directly added to the corresponding joint-space inertia. It helps improve the
        simulation stability by reducing the joint velocities.

        Args:
            armature: Joint armature. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # resolve indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if joint_mask is None:
            joint_mask = self._ALL_JOINT_MASK
        # set into internal buffers
        if isinstance(armature, float):
            wp.launch(
                update_joint_array_with_value,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    armature,
                    self._data.sim_bind_joint_armature,
                    env_mask,
                    joint_mask,
                ],
            )
        else:
            wp.launch(
                update_joint_array,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    armature,
                    self._data.sim_bind_joint_armature,
                    env_mask,
                    joint_mask,
                ],
            )
        # tell the physics engine to use the new armature
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    def write_joint_friction_coefficient_to_sim(
        self,
        joint_friction_coeff: wp.array | float,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        r"""Write joint friction coefficients into the simulation.

        The joint friction is a unitless quantity. It relates the magnitude of the spatial force transmitted
        from the parent body to the child body to the maximal friction force that may be applied by the solver
        to resist the joint motion.

        Mathematically, this means that: :math:`F_{resist} \leq \mu F_{spatial}`, where :math:`F_{resist}`
        is the resisting force applied by the solver and :math:`F_{spatial}` is the spatial force
        transmitted from the parent body to the child body. The simulated friction effect is therefore
        similar to static and Coulomb friction.

        Args:
            joint_friction_coeff: Joint friction coefficients. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # resolve indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if joint_mask is None:
            joint_mask = self._ALL_JOINT_MASK
        # set into internal buffers
        if isinstance(joint_friction_coeff, float):
            wp.launch(
                update_joint_array_with_value,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    joint_friction_coeff,
                    self._data.sim_bind_joint_friction_coeff,
                    env_mask,
                    joint_mask,
                ],
            )
        else:
            wp.launch(
                update_joint_array,
                dim=(self.num_instances, self.num_joints),
                inputs=[
                    joint_friction_coeff,
                    self._data.sim_bind_joint_friction_coeff,
                    env_mask,
                    joint_mask,
                ],
            )
        # tell the physics engine to use the new friction
        NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

    """
    Operations - Setters.
    """

    def set_joint_position_target(
        self,
        target: wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint position targets into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint position targets. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # resolve indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if joint_mask is None:
            joint_mask = self._ALL_JOINT_MASK
        # set targets
        wp.launch(
            update_joint_array,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                target,
                self._data.joint_target,
                env_mask,
                joint_mask,
            ],
        )

    def set_joint_velocity_target(
        self,
        target: wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint velocity targets into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint velocity targets. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # resolve indice
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if joint_mask is None:
            joint_mask = self._ALL_JOINT_MASK
        # set targets
        wp.launch(
            update_joint_array,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                target,
                self._data.joint_target,
                env_mask,
                joint_mask,
            ],
        )

    def set_joint_effort_target(
        self,
        target: wp.array,
        joint_mask: wp.array | None = None,
        env_mask: wp.array | None = None,
    ) -> None:
        """Set joint efforts into internal buffers.

        This function does not apply the joint targets to the simulation. It only fills the buffers with
        the desired values. To apply the joint targets, call the :meth:`write_data_to_sim` function.

        Args:
            target: Joint effort targets. Shape is (num_instances, num_joints).
            joint_mask: The joint mask. Shape is (num_joints).
            env_mask: The environment mask. Shape is (num_instances,).
        """
        # resolve indices
        if env_mask is None:
            env_mask = self._ALL_ENV_MASK
        if joint_mask is None:
            joint_mask = self._ALL_JOINT_MASK
        # set targets
        wp.launch(
            update_joint_array,
            dim=(self.num_instances, self.num_joints),
            inputs=[
                target,
                self._data.joint_effort_target,
                env_mask,
                joint_mask,
            ],
        )

    def _create_buffers(self):
        # constants
        self._ALL_ENV_MASK = wp.ones((self.num_instances,), dtype=wp.bool, device=self.device)
        self._ALL_JOINT_MASK = wp.ones((self.num_joints,), dtype=wp.bool, device=self.device)
        # masks
        self._ENV_MASK = wp.zeros((self.num_instances,), dtype=wp.bool, device=self.device)
        self._JOINT_MASK = wp.zeros((self.num_joints,), dtype=wp.bool, device=self.device)