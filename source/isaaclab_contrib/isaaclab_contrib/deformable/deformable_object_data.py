# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp
from isaaclab_newton.physics import NewtonManager as SimulationManager

from isaaclab.assets.deformable_object.base_deformable_object_data import BaseDeformableObjectData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.warp import ProxyArray

from .kernels import compute_mean_vec3f_over_vertices, compute_nodal_state_w, gather_particles_vec3f, vec6f


class DeformableObjectData(BaseDeformableObjectData):
    """Data container for a deformable object (Newton backend).

    Newton stores all particles in flat arrays (``model.particle_q``, ``state.particle_qd``).
    This data class builds a per-instance view by gathering from the flat arrays using
    precomputed offsets.

    The data is lazily updated, meaning that the data is only updated when it is accessed.
    """

    def __init__(
        self,
        particle_offsets: wp.array,
        particles_per_body: int,
        num_instances: int,
        device: str,
    ):
        """Initialize the Newton deformable object data.

        Args:
            particle_offsets: Per-instance start offset into the flat particle array.
                Shape is (num_instances,) with dtype int32.
            particles_per_body: Number of particles per deformable body instance.
            num_instances: Number of deformable body instances.
            device: The device used for processing.
        """
        super().__init__(device)

        # Store dimensions and indexing
        self._particle_offsets = particle_offsets
        self._particles_per_body = particles_per_body
        self._num_instances = num_instances

        # Initialize lazy buffers
        self._nodal_pos_w = TimestampedBuffer((num_instances, particles_per_body), device, wp.vec3f)
        self._nodal_vel_w = TimestampedBuffer((num_instances, particles_per_body), device, wp.vec3f)
        self._nodal_state_w = TimestampedBuffer((num_instances, particles_per_body), device, vec6f)
        self._root_pos_w = TimestampedBuffer((num_instances,), device, wp.vec3f)
        self._root_vel_w = TimestampedBuffer((num_instances,), device, wp.vec3f)
        self._nodal_pos_w_ta: ProxyArray | None = None
        self._nodal_vel_w_ta: ProxyArray | None = None
        self._nodal_state_w_ta: ProxyArray | None = None
        self._root_pos_w_ta: ProxyArray | None = None
        self._root_vel_w_ta: ProxyArray | None = None

        self._create_simulation_bindings()

    ##
    # Defaults.
    ##

    default_nodal_state_w: ProxyArray = None
    """Default nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.
    Shape is (num_instances, particles_per_body) with dtype vec6f.
    """

    ##
    # Kinematic commands.
    ##

    nodal_kinematic_target: ProxyArray = None
    """Simulation mesh kinematic targets for the deformable bodies.
    Shape is (num_instances, particles_per_body) with dtype vec4f.
    """

    def _create_simulation_bindings(self) -> None:
        """Validate the current Newton particle state and invalidate gathered buffers.

        Newton may swap :attr:`state_0` and :attr:`state_1` across substeps, so deformable data does not keep
        long-lived particle array bindings. Read properties query :meth:`SimulationManager.get_state_0` at gather time
        and materialize object-local views from the current flat particle arrays.
        """
        self._get_current_particle_state()

        # Invalidate lazy buffers gathered from the previous simulation state.
        self._nodal_pos_w.timestamp = -1.0
        self._nodal_vel_w.timestamp = -1.0
        self._nodal_state_w.timestamp = -1.0
        self._root_pos_w.timestamp = -1.0
        self._root_vel_w.timestamp = -1.0

    def _get_current_particle_state(self):
        """Return the current Newton state containing deformable particle arrays."""
        state = SimulationManager.get_state_0()
        if state is None or state.particle_q is None or state.particle_qd is None:
            raise RuntimeError(
                "Failed to access Newton deformable particle state. Ensure the Newton model has been finalized and "
                "contains particle position and velocity arrays."
            )
        return state

    ##
    # Properties.
    ##

    @property
    def nodal_pos_w(self) -> ProxyArray:
        """Nodal positions in simulation world frame [m]. Shape is (num_instances, particles_per_body) vec3f."""
        if self._nodal_pos_w.timestamp < self._sim_timestamp:
            state = self._get_current_particle_state()
            wp.launch(
                gather_particles_vec3f,
                dim=(self._num_instances, self._particles_per_body),
                inputs=[state.particle_q, self._particle_offsets, self._particles_per_body],
                outputs=[self._nodal_pos_w.data],
                device=self.device,
            )
            self._nodal_pos_w.timestamp = self._sim_timestamp
        if self._nodal_pos_w_ta is None:
            self._nodal_pos_w_ta = ProxyArray(self._nodal_pos_w.data)
        return self._nodal_pos_w_ta

    @property
    def nodal_vel_w(self) -> ProxyArray:
        """Nodal velocities in simulation world frame [m/s]. Shape is (num_instances, particles_per_body) vec3f."""
        if self._nodal_vel_w.timestamp < self._sim_timestamp:
            state = self._get_current_particle_state()
            wp.launch(
                gather_particles_vec3f,
                dim=(self._num_instances, self._particles_per_body),
                inputs=[state.particle_qd, self._particle_offsets, self._particles_per_body],
                outputs=[self._nodal_vel_w.data],
                device=self.device,
            )
            self._nodal_vel_w.timestamp = self._sim_timestamp
        if self._nodal_vel_w_ta is None:
            self._nodal_vel_w_ta = ProxyArray(self._nodal_vel_w.data)
        return self._nodal_vel_w_ta

    @property
    def nodal_state_w(self) -> ProxyArray:
        """Nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame [m, m/s].

        Shape is (num_instances, particles_per_body) vec6f.
        """
        if self._nodal_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_nodal_state_w,
                dim=(self._num_instances, self._particles_per_body),
                inputs=[self.nodal_pos_w.warp, self.nodal_vel_w.warp],
                outputs=[self._nodal_state_w.data],
                device=self.device,
            )
            self._nodal_state_w.timestamp = self._sim_timestamp
        if self._nodal_state_w_ta is None:
            self._nodal_state_w_ta = ProxyArray(self._nodal_state_w.data)
        return self._nodal_state_w_ta

    ##
    # Derived properties.
    ##

    @property
    def root_pos_w(self) -> ProxyArray:
        """Root position from nodal positions [m]. Shape is (num_instances,) vec3f.

        This quantity is computed as the mean of the nodal positions.
        """
        if self._root_pos_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_mean_vec3f_over_vertices,
                dim=(self._num_instances,),
                inputs=[self.nodal_pos_w.warp, self._particles_per_body],
                outputs=[self._root_pos_w.data],
                device=self.device,
            )
            self._root_pos_w.timestamp = self._sim_timestamp
        if self._root_pos_w_ta is None:
            self._root_pos_w_ta = ProxyArray(self._root_pos_w.data)
        return self._root_pos_w_ta

    @property
    def root_vel_w(self) -> ProxyArray:
        """Root velocity from nodal velocities [m/s]. Shape is (num_instances,) vec3f.

        This quantity is computed as the mean of the nodal velocities.
        """
        if self._root_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_mean_vec3f_over_vertices,
                dim=(self._num_instances,),
                inputs=[self.nodal_vel_w.warp, self._particles_per_body],
                outputs=[self._root_vel_w.data],
                device=self.device,
            )
            self._root_vel_w.timestamp = self._sim_timestamp
        if self._root_vel_w_ta is None:
            self._root_vel_w_ta = ProxyArray(self._root_vel_w.data)
        return self._root_vel_w_ta
