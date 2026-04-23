# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.assets.deformable_object.base_deformable_object_data import BaseDeformableObjectData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer

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

        # Store direct references (no weakref) — set later by the asset class when state is available
        self._particle_q: wp.array | None = None
        self._particle_qd: wp.array | None = None

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

    def bind_simulation_state(self, particle_q: wp.array, particle_qd: wp.array) -> None:
        """Bind the simulation state arrays for lazy reads.

        Called by the asset class after the Newton model is built and states are available.

        Args:
            particle_q: Flat particle positions from Newton state. Shape is (total_particles,) vec3f.
            particle_qd: Flat particle velocities from Newton state. Shape is (total_particles,) vec3f.
        """
        self._particle_q = particle_q
        self._particle_qd = particle_qd

    ##
    # Properties.
    ##

    @property
    def nodal_pos_w(self) -> wp.array:
        """Nodal positions in simulation world frame [m]. Shape is (num_instances, particles_per_body) vec3f."""
        if self._nodal_pos_w.timestamp < self._sim_timestamp:
            wp.launch(
                gather_particles_vec3f,
                dim=(self._num_instances, self._particles_per_body),
                inputs=[self._particle_q, self._particle_offsets, self._particles_per_body],
                outputs=[self._nodal_pos_w.data],
                device=self.device,
            )
            self._nodal_pos_w.timestamp = self._sim_timestamp
        return self._nodal_pos_w.data

    @property
    def nodal_vel_w(self) -> wp.array:
        """Nodal velocities in simulation world frame [m/s]. Shape is (num_instances, particles_per_body) vec3f."""
        if self._nodal_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                gather_particles_vec3f,
                dim=(self._num_instances, self._particles_per_body),
                inputs=[self._particle_qd, self._particle_offsets, self._particles_per_body],
                outputs=[self._nodal_vel_w.data],
                device=self.device,
            )
            self._nodal_vel_w.timestamp = self._sim_timestamp
        return self._nodal_vel_w.data

    @property
    def nodal_state_w(self) -> wp.array:
        """Nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame [m, m/s].

        Shape is (num_instances, particles_per_body) vec6f.
        """
        if self._nodal_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_nodal_state_w,
                dim=(self._num_instances, self._particles_per_body),
                inputs=[self.nodal_pos_w, self.nodal_vel_w],
                outputs=[self._nodal_state_w.data],
                device=self.device,
            )
            self._nodal_state_w.timestamp = self._sim_timestamp
        return self._nodal_state_w.data

    ##
    # Derived properties.
    ##

    @property
    def root_pos_w(self) -> wp.array:
        """Root position from nodal positions [m]. Shape is (num_instances,) vec3f.

        This quantity is computed as the mean of the nodal positions.
        """
        if self._root_pos_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_mean_vec3f_over_vertices,
                dim=(self._num_instances,),
                inputs=[self.nodal_pos_w, self._particles_per_body],
                outputs=[self._root_pos_w.data],
                device=self.device,
            )
            self._root_pos_w.timestamp = self._sim_timestamp
        return self._root_pos_w.data

    @property
    def root_vel_w(self) -> wp.array:
        """Root velocity from nodal velocities [m/s]. Shape is (num_instances,) vec3f.

        This quantity is computed as the mean of the nodal velocities.
        """
        if self._root_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_mean_vec3f_over_vertices,
                dim=(self._num_instances,),
                inputs=[self.nodal_vel_w, self._particles_per_body],
                outputs=[self._root_vel_w.data],
                device=self.device,
            )
            self._root_vel_w.timestamp = self._sim_timestamp
        return self._root_vel_w.data
