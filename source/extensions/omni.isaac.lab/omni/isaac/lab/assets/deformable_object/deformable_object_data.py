# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class DeformableObjectData:
    """Data container for a robot."""

    ##
    # Default states.
    ##

    default_nodal_state_w: torch.Tensor = None
    """Default nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.
    Shape is ``(num_instances, 2 * max_simulation_mesh_vertices_per_body, 3)``.
    """

    ##
    # Frame states.
    ##

    nodal_state_w: torch.Tensor = None
    """Nodal state ``[nodal_pos, nodal_vel]`` in simulation world frame.
    Shape is ``(num_instances, 2 * max_simulation_mesh_vertices_per_body, 3)``.
    """

    ##
    # Element-wise states.
    ##

    sim_element_rotations: torch.Tensor = None
    """Simulation mesh element-wise rotations as quaternions for the deformable bodies.
    Shape is ``(num_instances, max_simulation_mesh_elements_per_body, 4)``.
    """

    collision_element_rotations: torch.Tensor = None
    """Collision mesh element-wise rotations as quaternions for the deformable bodies.
    Shape is ``(num_instances, max_collision_mesh_elements_per_body, 4)``.
    """

    sim_element_deformation_gradients: torch.Tensor = None
    """Simulation mesh element-wise second-order deformation gradient tensors for the deformable bodies.
    Shape is ``(num_instances, max_simulation_mesh_elements_per_body, 3, 3)``.
    """

    collision_element_deformation_gradients: torch.Tensor = None
    """Collision mesh element-wise second-order deformation gradient tensors for the deformable bodies.
    Shape is ``(num_instances, max_collision_mesh_elements_per_body, 3, 3)``.
    """

    sim_element_stresses: torch.Tensor = None
    """Simulation mesh element-wise second-order stress tensors for the deformable bodies.
    Shape is ``(num_instances, max_simulation_mesh_elements_per_body, 3, 3)``.
    """

    collision_element_stresses: torch.Tensor = None
    """Collision mesh element-wise second-order stress tensors for the deformable bodies.
    Shape is ``(num_instances, max_collision_mesh_elements_per_body, 3, 3)``.
    """

    """
    Properties
    """

    @property
    def nodal_pos_w(self) -> torch.Tensor:
        """Nodal positions of the simulation mesh for the deformable bodies in simulation world frame.
        Shape is ``(num_instances, max_simulation_mesh_vertices_per_body, 3)``.
        """
        return self.nodal_state_w[:, :self.nodal_state_w.size(1) // 2, :]

    @property
    def nodal_vel_w(self) -> torch.Tensor:
        """Vertex velocities for the deformable bodies in simulation world frame.
        Shape is ``(num_instances, max_simulation_mesh_vertices_per_body, 3)``.
        """
        return self.nodal_state_w[:, self.nodal_state_w.size(1) // 2:, :]
    
    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position from nodal positions of the simulation mesh for the deformable bodies in simulation world frame.
        Shape is ``(num_instances, 3)``.
        """
        return self.nodal_pos_w.mean(dim=1)

    @property
    def root_vel_w(self) -> torch.Tensor:
        """Root velocity from vertex velocities for the deformable bodies in simulation world frame.
        Shape is ``(num_instances, 3)``.
        """
        return self.nodal_vel_w.mean(dim=1)
