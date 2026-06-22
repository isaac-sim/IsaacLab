# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warp as wp

from isaaclab.assets.deformable_object.base_deformable_object_data import BaseDeformableObjectData
from isaaclab.utils.buffers import TimestampedBufferWarp as TimestampedBuffer
from isaaclab.utils.warp import ProxyArray

from isaaclab_newton.physics import NewtonManager as SimulationManager

from .kernels import compute_mean_vec3f_over_particles, compute_particle_state_w, gather_particles_vec3f, vec6f


class MPMObjectData(BaseDeformableObjectData):
    """Data container for a Newton MPM particle object."""

    __backend_name__: str = "newton"

    def __init__(self, particle_offsets: wp.array, particles_per_object: int, num_instances: int, device: str):
        super().__init__(device)
        self._particle_offsets = particle_offsets
        self._particles_per_object = particles_per_object
        self._num_instances = num_instances

        self._particle_pos_w = TimestampedBuffer((num_instances, particles_per_object), device, wp.vec3f)
        self._particle_vel_w = TimestampedBuffer((num_instances, particles_per_object), device, wp.vec3f)
        self._particle_state_w = TimestampedBuffer((num_instances, particles_per_object), device, vec6f)
        self._root_pos_w = TimestampedBuffer((num_instances,), device, wp.vec3f)
        self._root_vel_w = TimestampedBuffer((num_instances,), device, wp.vec3f)
        self._particle_pos_w_ta = ProxyArray(self._particle_pos_w.data)
        self._particle_vel_w_ta = ProxyArray(self._particle_vel_w.data)
        self._particle_state_w_ta = ProxyArray(self._particle_state_w.data)
        self._root_pos_w_ta = ProxyArray(self._root_pos_w.data)
        self._root_vel_w_ta = ProxyArray(self._root_vel_w.data)

        self.default_nodal_state_w: ProxyArray | None = None
        self.default_particle_state_w: ProxyArray | None = None
        self.nodal_kinematic_target: ProxyArray | None = None

        self._create_simulation_bindings()

    def _create_simulation_bindings(self) -> None:
        """Validate current Newton particle arrays and invalidate gathered buffers."""
        self._get_current_particle_state()
        self._particle_pos_w.timestamp = -1.0
        self._particle_vel_w.timestamp = -1.0
        self._particle_state_w.timestamp = -1.0
        self._root_pos_w.timestamp = -1.0
        self._root_vel_w.timestamp = -1.0

    def _get_current_particle_state(self):
        state = SimulationManager.get_state_0()
        if state is None or state.particle_q is None or state.particle_qd is None:
            raise RuntimeError(
                "Failed to access Newton MPM particle state. Ensure the Newton model has been finalized and contains "
                "particle position and velocity arrays."
            )
        return state

    @property
    def particle_pos_w(self) -> ProxyArray:
        """Particle positions in simulation world frame [m]."""
        if self._particle_pos_w.timestamp < self._sim_timestamp:
            state = self._get_current_particle_state()
            wp.launch(
                gather_particles_vec3f,
                dim=(self._num_instances, self._particles_per_object),
                inputs=[state.particle_q, self._particle_offsets],
                outputs=[self._particle_pos_w.data],
                device=self.device,
            )
            self._particle_pos_w.timestamp = self._sim_timestamp
        return self._particle_pos_w_ta

    @property
    def particle_vel_w(self) -> ProxyArray:
        """Particle velocities in simulation world frame [m/s]."""
        if self._particle_vel_w.timestamp < self._sim_timestamp:
            state = self._get_current_particle_state()
            wp.launch(
                gather_particles_vec3f,
                dim=(self._num_instances, self._particles_per_object),
                inputs=[state.particle_qd, self._particle_offsets],
                outputs=[self._particle_vel_w.data],
                device=self.device,
            )
            self._particle_vel_w.timestamp = self._sim_timestamp
        return self._particle_vel_w_ta

    @property
    def particle_state_w(self) -> ProxyArray:
        """Particle state ``[pos, vel]`` in simulation world frame [m, m/s]."""
        if self._particle_state_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_particle_state_w,
                dim=(self._num_instances, self._particles_per_object),
                inputs=[self.particle_pos_w.warp, self.particle_vel_w.warp],
                outputs=[self._particle_state_w.data],
                device=self.device,
            )
            self._particle_state_w.timestamp = self._sim_timestamp
        return self._particle_state_w_ta

    @property
    def nodal_pos_w(self) -> ProxyArray:
        return self.particle_pos_w

    @property
    def nodal_vel_w(self) -> ProxyArray:
        return self.particle_vel_w

    @property
    def nodal_state_w(self) -> ProxyArray:
        return self.particle_state_w

    @property
    def root_pos_w(self) -> ProxyArray:
        """Mean particle position per instance in simulation world frame [m]."""
        if self._root_pos_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_mean_vec3f_over_particles,
                dim=(self._num_instances,),
                inputs=[self.particle_pos_w.warp, self._particles_per_object],
                outputs=[self._root_pos_w.data],
                device=self.device,
            )
            self._root_pos_w.timestamp = self._sim_timestamp
        return self._root_pos_w_ta

    @property
    def root_vel_w(self) -> ProxyArray:
        """Mean particle velocity per instance in simulation world frame [m/s]."""
        if self._root_vel_w.timestamp < self._sim_timestamp:
            wp.launch(
                compute_mean_vec3f_over_particles,
                dim=(self._num_instances,),
                inputs=[self.particle_vel_w.warp, self._particles_per_object],
                outputs=[self._root_vel_w.data],
                device=self.device,
            )
            self._root_vel_w.timestamp = self._sim_timestamp
        return self._root_vel_w_ta
