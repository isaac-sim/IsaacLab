# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Flag for pyright to ignore type errors in this file.
# pyright: reportPrivateUsage=false


from collections.abc import Sequence
import weakref
import warp as wp

import isaaclab.utils.string as string_utils
from newton.selection import ArticulationView as NewtonArticulationView
from isaaclab.assets.core.body_properties.body_data import BodyData
from isaaclab.assets.core.kernels import (
    generate_mask_from_ids,
    update_wrench_array_with_force,
    update_wrench_array_with_torque,
    update_array_with_value_masked,
)


class Body:
    def __init__(self, root_newton_view, device: str):
        self._root_newton_view = weakref.proxy(root_newton_view)
        self._data = BodyData(root_newton_view, device)
        self._device = device

    """
    Properties
    """

    @property
    def data(self) -> BodyData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._root_newton_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in articulation."""
        return self._root_newton_view.link_count

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in articulation."""
        return self._root_newton_view.body_names

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
        # reset external wrench
        wp.launch(
            update_array_with_value_masked,
            dim=(self.num_instances,),
            inputs=[
                wp.spatial_vectorf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                self._data.external_wrench,
                mask,
                self._ALL_BODY_MASK,
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
        pass

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[wp.array, list[str], list[int]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body mask, names, and indices.
        """
        indices, names = string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)
        self._BODY_MASK.fill_(False)
        mask = wp.clone(self._BODY_MASK)
        wp.launch(
            generate_mask_from_ids,
            dim=(len(indices),),
            inputs=[
                mask,
                wp.array(indices, dtype=wp.int32, device=self._device),
            ],
        )
        return mask, names, indices

    """
    Operations - Setters.
    """

    def set_external_force_and_torque(
        self,
        forces: wp.array,
        torques: wp.array,
        body_mask: wp.array | None = None,
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
        if body_mask is None:
            body_mask = self._ALL_BODY_MASK
        # Check if there are any external forces or torques
        if (forces is not None) or (torques is not None):
            self.has_external_wrench = True
            if forces is not None:
                wp.launch(
                    update_wrench_array_with_force,
                    dim=(self.num_instances, self.num_bodies),
                    inputs=[
                        forces,
                        self._data.external_wrench,
                        env_mask,
                        body_mask,
                    ],
                )
            if torques is not None:
                wp.launch(
                    update_wrench_array_with_torque,
                    dim=(self.num_instances, self.num_bodies),
                    inputs=[
                        torques,
                        self._data.external_wrench,
                        env_mask,
                        body_mask,
                    ],
                )

    def _create_buffers(self):
        # constants
        self._ALL_ENV_MASK = wp.ones((self.num_instances,), dtype=wp.bool, device=self._device)
        self._ALL_BODY_MASK = wp.ones((self.num_bodies,), dtype=wp.bool, device=self._device)
        # masks
        self._ENV_MASK = wp.zeros((self.num_instances,), dtype=wp.bool, device=self._device)
        self._BODY_MASK = wp.zeros((self.num_bodies,), dtype=wp.bool, device=self._device)