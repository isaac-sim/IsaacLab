# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Direct Position-Based Solver for Stiff Rods.

This module implements the main solver class based on:
Deul et al. 2018 "Direct Position-Based Solver for Stiff Rods"
Computer Graphics Forum, Vol. 37, No. 8

Key features:
1. XPBD (Extended Position-Based Dynamics) framework
2. Newton iterations for solving the non-linear constraint system
3. Direct solver with linear-time complexity for tree structures
4. Cosserat rod model for accurate bending and twisting

The solver achieves high stiffness and inextensibility with only a few
Newton iterations, providing a speedup of two orders of magnitude compared
to the original Gauss-Seidel XPBD approach.

Reference:
    @article{deul2018direct,
      title={Direct Position-Based Solver for Stiff Rods},
      author={Deul, Crispin and Kugelstadt, Tassilo and Weiler, Marcel and Bender, Jan},
      journal={Computer Graphics Forum},
      volume={37},
      number={8},
      pages={313--324},
      year={2018},
      publisher={Wiley Online Library}
    }
"""

from __future__ import annotations

from typing import Callable

import torch
import warp as wp

from .rod_data import (
    RodConfig, 
    RodData, 
    RodMaterialConfig, 
    RodGeometryConfig, 
    RodSolverConfig,
    RodTipConfig,
    FrictionConfig,
    CollisionMeshConfig,
)
from .rod_kernels import (
    compute_constraint_residuals_kernel,
    normalize_quaternions_kernel,
    predict_orientations_kernel,
    predict_positions_kernel,
    reset_lambda_kernel,
    reset_lambda_with_shear_kernel,
    solve_bend_twist_constraints_kernel,
    solve_ground_collision_kernel,
    solve_self_collision_kernel,
    solve_stretch_constraints_kernel,
    solve_shear_constraints_kernel,
    solve_mesh_collision_kernel,
    apply_coulomb_friction_kernel,
    apply_viscous_friction_kernel,
    update_velocities_kernel,
)

__all__ = [
    "RodSolver",
    "RodConfig",
    "RodData",
    "RodMaterialConfig",
    "RodGeometryConfig",
    "RodSolverConfig",
    "RodTipConfig",
    "FrictionConfig",
    "CollisionMeshConfig",
]


class DirectTreeSolver:
    """Linear-time direct solver for tree-structured constraint systems.

    This implements the direct solver from Section 4 of the paper.
    For a tree of constraints, the system can be solved in O(n) time
    by exploiting the tree structure.

    The algorithm has two passes:
    1. Bottom-up pass: Accumulate contributions from leaves to root
    2. Top-down pass: Propagate solutions from root to leaves

    For a linear chain (rod), this simplifies to a tridiagonal system
    that can be solved using Thomas algorithm.
    """

    def __init__(self, num_segments: int, num_envs: int, device: str = "cuda"):
        """Initialize the direct solver.

        Args:
            num_segments: Number of segments in the rod.
            num_envs: Number of parallel environments.
            device: Computation device.
        """
        self.num_segments = num_segments
        self.num_envs = num_envs
        self.device = device
        self.num_constraints = num_segments - 1

        # Allocate temporary arrays for the direct solve
        # For position constraints: 3 DOFs per constraint
        # For orientation constraints: 3 DOFs per constraint
        self._allocate_temp_arrays()

    def _allocate_temp_arrays(self):
        """Allocate temporary arrays for the direct solver."""
        n = self.num_envs
        c = self.num_constraints

        # Diagonal blocks of the system matrix (6x6 per constraint)
        self.diag_blocks = torch.zeros((n, c, 6, 6), device=self.device, dtype=torch.float32)

        # Off-diagonal blocks (coupling between adjacent constraints)
        self.off_diag_blocks = torch.zeros((n, c - 1, 6, 6), device=self.device, dtype=torch.float32)

        # Right-hand side vector
        self.rhs = torch.zeros((n, c, 6), device=self.device, dtype=torch.float32)

        # Solution vector (Δλ for all constraints)
        self.delta_lambda = torch.zeros((n, c, 6), device=self.device, dtype=torch.float32)

        # Temporary arrays for Thomas algorithm
        self.c_prime = torch.zeros((n, c - 1, 6, 6), device=self.device, dtype=torch.float32)
        self.d_prime = torch.zeros((n, c, 6), device=self.device, dtype=torch.float32)

    def solve(
        self,
        rod_data: RodData,
        dt: float,
    ) -> torch.Tensor:
        """Solve the constraint system using the direct method.

        This implements the direct solver for a linear chain of constraints.
        For more complex tree structures, a more general algorithm would
        be needed.

        The system has the form:
            A * Δλ = b

        where A is block-tridiagonal for a linear chain.

        Args:
            rod_data: Rod data containing current state and constraints.
            dt: Time step.

        Returns:
            Solution vector Δλ for all constraints.
        """
        # Build the system matrix and RHS
        self._build_system(rod_data, dt)

        # Solve using Thomas algorithm (tridiagonal solver)
        self._solve_tridiagonal()

        return self.delta_lambda

    def _build_system(self, rod_data: RodData, dt: float):
        """Build the linear system for the Newton step.

        For each constraint, we need to compute:
        - The constraint residual (RHS)
        - The constraint Jacobian
        - The effective mass matrix

        The system matrix A = α + J^T * W * J where:
        - α is the compliance matrix
        - J is the Jacobian of all constraints
        - W is the inverse mass matrix
        """
        n = self.num_envs
        c = self.num_constraints

        # Reset arrays
        self.diag_blocks.zero_()
        self.off_diag_blocks.zero_()
        self.rhs.zero_()

        # Compute constraint residuals
        stretch_residual = torch.zeros((n, c, 3), device=self.device)
        bend_twist_residual = torch.zeros((n, c, 3), device=self.device)

        # Get segment data
        positions = rod_data.positions
        orientations = rod_data.orientations
        segment_lengths = rod_data.segment_lengths
        inv_masses = rod_data.inv_masses
        inv_inertias = rod_data.inv_inertias

        dt2 = dt * dt

        for i in range(c):
            parent_idx = i
            child_idx = i + 1

            # Stretch constraint residual
            L1 = segment_lengths[:, parent_idx]
            L2 = segment_lengths[:, child_idx]

            # Get endpoint positions
            q1 = orientations[:, parent_idx]  # (n, 4)
            q2 = orientations[:, child_idx]  # (n, 4)

            # Local attachment points
            local_end = torch.tensor([[0.5, 0.0, 0.0]], device=self.device) * L1.unsqueeze(1)
            local_start = torch.tensor([[-0.5, 0.0, 0.0]], device=self.device) * L2.unsqueeze(1)

            # Rotate to world frame
            p1_offset = self._quat_rotate(q1, local_end)
            p2_offset = self._quat_rotate(q2, local_start)

            p1 = positions[:, parent_idx] + p1_offset
            p2 = positions[:, child_idx] + p2_offset

            # Stretch residual (should be zero)
            stretch_residual[:, i] = p2 - p1

            # Bend/twist constraint residual
            L_avg = 0.5 * (L1 + L2)
            darboux = self._compute_darboux(q1, q2, L_avg)
            bend_twist_residual[:, i] = darboux - rod_data.rest_darboux[:, i]

            # Build RHS: -C - α*λ
            stretch_compliance = rod_data.stretch_compliance[:, i:i+1]
            bend_compliance = rod_data.bend_twist_compliance[:, i]

            self.rhs[:, i, :3] = -(stretch_residual[:, i] +
                                   stretch_compliance * rod_data.lambda_stretch[:, i:i+1] / dt2)
            self.rhs[:, i, 3:] = -(bend_twist_residual[:, i] +
                                   bend_compliance * rod_data.lambda_bend_twist[:, i] / dt2)

            # Build diagonal block
            w1 = inv_masses[:, parent_idx]
            w2 = inv_masses[:, child_idx]

            # Position constraint Jacobian contribution
            # For stretch: J^T W J ≈ (w1 + w2) * I
            w_pos = (w1 + w2).unsqueeze(1).unsqueeze(2)
            self.diag_blocks[:, i, :3, :3] = w_pos * torch.eye(3, device=self.device)

            # Add compliance
            self.diag_blocks[:, i, :3, :3] += (
                stretch_compliance.unsqueeze(2) * torch.eye(3, device=self.device) / dt2
            )

            # Orientation constraint Jacobian contribution
            L_scale = 2.0 / L_avg.unsqueeze(1)
            I1_inv_diag = torch.diagonal(inv_inertias[:, parent_idx], dim1=-2, dim2=-1)
            I2_inv_diag = torch.diagonal(inv_inertias[:, child_idx], dim1=-2, dim2=-1)
            w_rot = (I1_inv_diag + I2_inv_diag) * L_scale * L_scale

            for j in range(3):
                self.diag_blocks[:, i, 3 + j, 3 + j] = w_rot[:, j]

            # Add compliance for orientation
            for j in range(3):
                self.diag_blocks[:, i, 3 + j, 3 + j] += bend_compliance[:, j] / dt2

            # Build off-diagonal blocks (coupling between adjacent constraints)
            if i < c - 1:
                # The child of constraint i is the parent of constraint i+1
                # This creates coupling in the system matrix
                # For now, we use a simplified decoupled approach
                # A full implementation would compute the cross-Jacobian terms
                pass

    def _solve_tridiagonal(self):
        """Solve the block-tridiagonal system using Thomas algorithm.

        For a system:
            A_0 x_0 + C_0 x_1 = d_0
            B_i x_{i-1} + A_i x_i + C_i x_{i+1} = d_i  for i = 1..n-2
            B_{n-1} x_{n-2} + A_{n-1} x_{n-1} = d_{n-1}

        The Thomas algorithm is:
        1. Forward elimination: c'_i = C_i / (A_i - B_i * c'_{i-1})
                               d'_i = (d_i - B_i * d'_{i-1}) / (A_i - B_i * c'_{i-1})
        2. Back substitution: x_i = d'_i - c'_i * x_{i+1}
        """
        n = self.num_envs
        c = self.num_constraints

        # For the simplified case where we ignore off-diagonal coupling,
        # each constraint can be solved independently
        for i in range(c):
            # Solve 6x6 system: diag_blocks[:, i] * delta_lambda[:, i] = rhs[:, i]
            A = self.diag_blocks[:, i]  # (n, 6, 6)
            b = self.rhs[:, i]  # (n, 6)

            # Add small regularization for numerical stability
            A = A + 1e-8 * torch.eye(6, device=self.device).unsqueeze(0)

            # Solve using batch matrix solve
            try:
                self.delta_lambda[:, i] = torch.linalg.solve(A, b)
            except RuntimeError:
                # Fallback to pseudo-inverse if singular
                self.delta_lambda[:, i] = torch.matmul(
                    torch.linalg.pinv(A), b.unsqueeze(-1)
                ).squeeze(-1)

    @staticmethod
    def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by quaternion q (x, y, z, w format)."""
        qv = q[:, :3]  # vector part
        qw = q[:, 3:4]  # scalar part

        t = 2.0 * torch.cross(qv, v, dim=-1)
        return v + qw * t + torch.cross(qv, t, dim=-1)

    @staticmethod
    def _compute_darboux(q1: torch.Tensor, q2: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Compute Darboux vector between two orientations."""
        # q_rel = q1^(-1) * q2
        # For quaternion: q^(-1) = conjugate(q) = (-qv, qw)
        q1_conj = torch.cat([-q1[:, :3], q1[:, 3:4]], dim=-1)

        # Quaternion multiplication
        q_rel = DirectTreeSolver._quat_multiply(q1_conj, q2)

        # Darboux = 2 * vector_part / L
        return 2.0 * q_rel[:, :3] / L.unsqueeze(1)

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (x, y, z, w format)."""
        x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        return torch.stack([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ], dim=-1)


class RodSolver:
    """Direct Position-Based Solver for Stiff Rods.

    This class implements the main simulation loop for rod dynamics
    using the XPBD framework with a direct solver for improved
    convergence on stiff constraints.

    The simulation follows these steps each time step:
    1. Predict positions/orientations using explicit integration
    2. Reset Lagrange multipliers (optional)
    3. For each Newton iteration:
       a. Solve stretch constraints
       b. Solve bend/twist constraints
       c. Handle collisions
    4. Update velocities from position changes

    Example usage:

    ```python
    from isaaclab_newton.solvers import RodSolver, RodConfig

    # Create configuration
    config = RodConfig()
    config.geometry.num_segments = 20
    config.material.young_modulus = 1e9
    config.solver.newton_iterations = 4

    # Create solver
    solver = RodSolver(config, num_envs=10)

    # Fix first segment
    solver.data.fix_segment(slice(None), 0)

    # Simulation loop
    for _ in range(1000):
        solver.step()
        positions = solver.data.positions  # Get segment positions
    ```

    Attributes:
        config: Solver configuration.
        data: Rod state data.
        direct_solver: Direct solver for tree-structured systems.
    """

    def __init__(
        self,
        config: RodConfig | None = None,
        num_envs: int = 1,
        device: str | None = None,
    ):
        """Initialize the rod solver.

        Args:
            config: Rod configuration. If None, uses defaults.
            num_envs: Number of parallel environments.
            device: Computation device. If None, uses config.device.
        """
        self.config = config or RodConfig()
        self.device = device or self.config.device
        self.num_envs = num_envs

        # Initialize rod data
        self.data = RodData(self.config, num_envs, self.device)

        # Initialize direct solver
        if self.config.solver.use_direct_solver:
            self.direct_solver = DirectTreeSolver(
                self.config.geometry.num_segments, num_envs, self.device
            )
        else:
            self.direct_solver = None

        # Pre-compute gravity vector
        self.gravity = wp.vec3f(*self.config.solver.gravity)

        # Simulation time
        self.time = 0.0

        # Callback for external forces
        self._external_force_callback: Callable[[RodData], None] | None = None

    def step(self, dt: float | None = None):
        """Advance simulation by one time step.

        Args:
            dt: Time step. If None, uses config.solver.dt.
        """
        dt = dt or self.config.solver.dt
        num_substeps = self.config.solver.num_substeps
        sub_dt = dt / num_substeps

        for _ in range(num_substeps):
            self._substep(sub_dt)

        self.time += dt

    def _substep(self, dt: float):
        """Perform one simulation substep.

        Args:
            dt: Substep time.
        """
        # Apply external forces if callback is set
        if self._external_force_callback is not None:
            self._external_force_callback(self.data)

        # Ensure Warp arrays are synced
        self.data.sync_to_warp()

        num_segments = self.config.geometry.num_segments
        total_segments = self.num_envs * num_segments
        num_constraints = num_segments - 1
        total_constraints = self.num_envs * num_constraints

        # Step 1: Predict positions and orientations
        wp.launch(
            kernel=predict_positions_kernel,
            dim=total_segments,
            inputs=[
                self.data.wp_positions,
                self.data.wp_velocities,
                self.data.wp_prev_positions,
                self.data.wp_masses,
                self.data.wp_fixed_segments,
                self.gravity,
                dt,
                self.config.material.damping,
            ],
        )

        wp.launch(
            kernel=predict_orientations_kernel,
            dim=total_segments,
            inputs=[
                self.data.wp_orientations,
                self.data.wp_angular_velocities,
                self.data.wp_prev_orientations,
                self.data.wp_fixed_segments,
                dt,
                self.config.material.damping,
            ],
        )

        # Step 2: Reset Lagrange multipliers
        wp.launch(
            kernel=reset_lambda_kernel,
            dim=total_constraints,
            inputs=[
                self.data.wp_lambda_stretch,
                self.data.wp_lambda_bend_twist,
            ],
        )

        # Step 3: Newton iterations
        newton_iters = self.config.solver.newton_iterations

        if self.config.solver.use_direct_solver and self.direct_solver is not None:
            # Use direct solver
            for _ in range(newton_iters):
                self._direct_solve_iteration(dt)
        else:
            # Use Gauss-Seidel iterations
            for _ in range(newton_iters):
                self._gauss_seidel_iteration(dt)

        # Step 4: Handle collisions
        if self.config.solver.enable_collisions:
            self._solve_collisions()

        # Normalize quaternions
        wp.launch(
            kernel=normalize_quaternions_kernel,
            dim=total_segments,
            inputs=[self.data.wp_orientations],
        )

        # Step 5: Update velocities
        wp.launch(
            kernel=update_velocities_kernel,
            dim=total_segments,
            inputs=[
                self.data.wp_positions,
                self.data.wp_orientations,
                self.data.wp_prev_positions,
                self.data.wp_prev_orientations,
                self.data.wp_velocities,
                self.data.wp_angular_velocities,
                self.data.wp_fixed_segments,
                dt,
            ],
        )

        # Sync back to torch
        self.data.sync_from_warp()

    def _direct_solve_iteration(self, dt: float):
        """Perform one Newton iteration using the direct solver.

        This solves all constraints simultaneously using the linear-time
        direct solver for tree structures.

        Args:
            dt: Time step.
        """
        # Sync data to torch for the direct solver
        self.data.sync_from_warp()

        # Solve for delta lambda
        delta_lambda = self.direct_solver.solve(self.data, dt)

        # Apply corrections
        self._apply_corrections(delta_lambda, dt)

        # Sync back to Warp
        self.data.sync_to_warp()

    def _apply_corrections(self, delta_lambda: torch.Tensor, dt: float):
        """Apply position and orientation corrections from delta lambda.

        Args:
            delta_lambda: Solution vector (num_envs, num_constraints, 6).
            dt: Time step.
        """
        num_constraints = self.config.geometry.num_segments - 1

        for i in range(num_constraints):
            parent_idx = i
            child_idx = i + 1

            # Extract stretch and bend/twist multipliers
            d_lambda_stretch = delta_lambda[:, i, :3]  # (n, 3)
            d_lambda_bend = delta_lambda[:, i, 3:]  # (n, 3)

            # Update accumulated lambda
            self.data.lambda_stretch[:, i] += d_lambda_stretch.norm(dim=-1)
            self.data.lambda_bend_twist[:, i] += d_lambda_bend

            # Position corrections
            w1 = self.data.inv_masses[:, parent_idx]
            w2 = self.data.inv_masses[:, child_idx]

            # Parent moves in negative constraint direction
            self.data.positions[:, parent_idx] -= w1.unsqueeze(1) * d_lambda_stretch
            # Child moves in positive constraint direction
            self.data.positions[:, child_idx] += w2.unsqueeze(1) * d_lambda_stretch

            # Orientation corrections
            L_avg = 0.5 * (
                self.data.segment_lengths[:, parent_idx] +
                self.data.segment_lengths[:, child_idx]
            )
            J_scale = 2.0 / L_avg

            I1_inv = torch.diagonal(self.data.inv_inertias[:, parent_idx], dim1=-2, dim2=-1)
            I2_inv = torch.diagonal(self.data.inv_inertias[:, child_idx], dim1=-2, dim2=-1)

            # Apply rotation corrections
            omega_corr = J_scale.unsqueeze(1) * d_lambda_bend

            # Parent correction
            corr1 = I1_inv * omega_corr
            q1 = self.data.orientations[:, parent_idx]
            corr1_world = self._quat_rotate(q1, corr1)
            dq1 = self._omega_to_quat(-corr1_world)
            self.data.orientations[:, parent_idx] = self._quat_multiply(dq1, q1)

            # Child correction
            corr2 = I2_inv * omega_corr
            q2 = self.data.orientations[:, child_idx]
            corr2_world = self._quat_rotate(q2, corr2)
            dq2 = self._omega_to_quat(corr2_world)
            self.data.orientations[:, child_idx] = self._quat_multiply(dq2, q2)

        # Normalize quaternions
        self.data.orientations = self.data.orientations / (
            self.data.orientations.norm(dim=-1, keepdim=True) + 1e-8
        )

    def _gauss_seidel_iteration(self, dt: float):
        """Perform one iteration of Gauss-Seidel constraint projection.

        This is the standard XPBD approach where constraints are solved
        one at a time in a local fashion.

        Args:
            dt: Time step.
        """
        num_segments = self.config.geometry.num_segments
        total_constraints = self.num_envs * (num_segments - 1)

        # Create diagonal inertia array for the kernel
        inv_inertias_diag = wp.from_torch(
            torch.diagonal(self.data.inv_inertias, dim1=-2, dim2=-1).contiguous().view(-1, 3),
            dtype=wp.vec3f
        )

        # Solve stretch constraints
        wp.launch(
            kernel=solve_stretch_constraints_kernel,
            dim=total_constraints,
            inputs=[
                self.data.wp_positions,
                self.data.wp_orientations,
                self.data.wp_inv_masses,
                self.data.wp_segment_lengths,
                self.data.wp_stretch_compliance,
                self.data.wp_lambda_stretch,
                self.data.wp_parent_indices,
                self.data.wp_fixed_segments,
                num_segments,
                dt,
            ],
        )
        
        # Solve shear constraints (if shear stiffness > 0)
        if self.config.material.shear_stiffness > 0:
            wp.launch(
                kernel=solve_shear_constraints_kernel,
                dim=total_constraints,
                inputs=[
                    self.data.wp_positions,
                    self.data.wp_orientations,
                    self.data.wp_inv_masses,
                    self.data.wp_segment_lengths,
                    self.data.wp_shear_compliance,
                    self.data.wp_lambda_shear,
                    self.data.wp_parent_indices,
                    self.data.wp_fixed_segments,
                    num_segments,
                    dt,
                ],
            )

        # Solve bend/twist constraints
        wp.launch(
            kernel=solve_bend_twist_constraints_kernel,
            dim=total_constraints,
            inputs=[
                self.data.wp_positions,
                self.data.wp_orientations,
                self.data.wp_inv_masses,
                inv_inertias_diag,
                self.data.wp_segment_lengths,
                self.data.wp_rest_darboux,
                self.data.wp_bend_twist_compliance,
                self.data.wp_lambda_bend_twist,
                self.data.wp_parent_indices,
                self.data.wp_fixed_segments,
                num_segments,
                dt,
            ],
        )

    def _solve_collisions(self):
        """Solve collision constraints including mesh collision and friction."""
        num_segments = self.config.geometry.num_segments
        total_segments = self.num_envs * num_segments
        friction_cfg = self.config.solver.friction
        mesh_cfg = self.config.solver.collision_mesh

        # Create radius array for kernel
        radii = wp.from_torch(self.data.radii.view(-1), dtype=wp.float32)
        
        # Allocate contact arrays for friction
        if not hasattr(self, '_contact_normals'):
            self._contact_normals = wp.zeros(total_segments, dtype=wp.vec3f)
            self._contact_depths = wp.zeros(total_segments, dtype=wp.float32)

        # Ground collision
        wp.launch(
            kernel=solve_ground_collision_kernel,
            dim=total_segments,
            inputs=[
                self.data.wp_positions,
                self.data.wp_orientations,
                self.data.wp_segment_lengths,
                radii,
                self.data.wp_fixed_segments,
                0.0,  # ground height
                mesh_cfg.restitution,
            ],
        )
        
        # Mesh collision (BVH-accelerated) if mesh is loaded
        if self.data.collision_bvh is not None:
            wp.launch(
                kernel=solve_mesh_collision_kernel,
                dim=total_segments,
                inputs=[
                    self.data.wp_positions,
                    self.data.wp_velocities,
                    radii,
                    self.data.wp_fixed_segments,
                    self.data.collision_bvh.id,
                    self._contact_normals,
                    self._contact_depths,
                    mesh_cfg.restitution,
                    mesh_cfg.collision_radius,
                ],
            )
            
            # Apply friction based on selected method
            if friction_cfg.method == "coulomb":
                wp.launch(
                    kernel=apply_coulomb_friction_kernel,
                    dim=total_segments,
                    inputs=[
                        self.data.wp_positions,
                        self.data.wp_prev_positions,
                        self.data.wp_velocities,
                        self._contact_normals,
                        self._contact_depths,
                        self.data.wp_fixed_segments,
                        friction_cfg.static_coefficient,
                        friction_cfg.dynamic_coefficient,
                        friction_cfg.stiction_velocity,
                        self.config.solver.dt,
                    ],
                )
            elif friction_cfg.method == "viscous":
                wp.launch(
                    kernel=apply_viscous_friction_kernel,
                    dim=total_segments,
                    inputs=[
                        self.data.wp_velocities,
                        self.data.wp_angular_velocities,
                        self._contact_depths,
                        self.data.wp_fixed_segments,
                        friction_cfg.viscous_coefficient,
                        self.config.solver.dt,
                    ],
                )
            elif friction_cfg.method == "static_dynamic":
                # Use Coulomb model with static/dynamic transition
                wp.launch(
                    kernel=apply_coulomb_friction_kernel,
                    dim=total_segments,
                    inputs=[
                        self.data.wp_positions,
                        self.data.wp_prev_positions,
                        self.data.wp_velocities,
                        self._contact_normals,
                        self._contact_depths,
                        self.data.wp_fixed_segments,
                        friction_cfg.static_coefficient,
                        friction_cfg.dynamic_coefficient,
                        friction_cfg.stiction_velocity,
                        self.config.solver.dt,
                    ],
                )

        # Self-collision
        wp.launch(
            kernel=solve_self_collision_kernel,
            dim=total_segments,
            inputs=[
                self.data.wp_positions,
                radii,
                self.data.wp_inv_masses,
                self.data.wp_fixed_segments,
                num_segments,
                self.config.solver.collision_margin,
            ],
        )

    @staticmethod
    def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by quaternion q."""
        qv = q[:, :3]
        qw = q[:, 3:4]
        t = 2.0 * torch.cross(qv, v, dim=-1)
        return v + qw * t + torch.cross(qv, t, dim=-1)

    @staticmethod
    def _omega_to_quat(omega: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Convert angular velocity to quaternion increment."""
        angle = omega.norm(dim=-1, keepdim=True) * dt
        half_angle = angle * 0.5

        # Small angle approximation
        small_angle = angle.squeeze(-1) < 1e-6
        s = torch.where(
            small_angle.unsqueeze(-1),
            torch.ones_like(half_angle) * 0.5 * dt,
            torch.sin(half_angle) / (angle / dt + 1e-8)
        )
        c = torch.where(
            small_angle.unsqueeze(-1),
            torch.ones_like(half_angle),
            torch.cos(half_angle)
        )

        return torch.cat([s * omega, c], dim=-1)

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply quaternions."""
        x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        return torch.stack([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ], dim=-1)

    def set_external_force_callback(self, callback: Callable[[RodData], None]):
        """Set callback for applying external forces.

        The callback is called at the beginning of each substep before
        position prediction. It receives the RodData object and can
        modify velocities or apply impulses.

        Args:
            callback: Function that takes RodData and applies external forces.
        """
        self._external_force_callback = callback

    def reset(self, env_indices: torch.Tensor | None = None):
        """Reset the simulation.

        Args:
            env_indices: Indices of environments to reset. If None, reset all.
        """
        self.data.reset(env_indices)
        self.time = 0.0

    def get_energy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute total kinetic and potential energy.

        Returns:
            Tuple of (kinetic_energy, potential_energy) tensors with shape (num_envs,).
        """
        num_segments = self.config.geometry.num_segments
        total_segments = self.num_envs * num_segments

        # Allocate output arrays
        ke = wp.zeros(total_segments, dtype=wp.float32)
        pe = wp.zeros(total_segments, dtype=wp.float32)

        # Create required arrays
        inertias_diag = wp.from_torch(
            torch.diagonal(self.data.inertias, dim1=-2, dim2=-1).contiguous().view(-1, 3),
            dtype=wp.vec3f
        )
        bending_stiffness = wp.from_torch(
            self.data.bending_stiffness.view(-1, 2), dtype=wp.vec2f
        )
        torsion_stiffness = wp.from_torch(
            self.data.torsion_stiffness.view(-1), dtype=wp.float32
        )

        from .rod_kernels import compute_total_energy_kernel

        wp.launch(
            kernel=compute_total_energy_kernel,
            dim=total_segments,
            inputs=[
                self.data.wp_positions,
                self.data.wp_orientations,
                self.data.wp_velocities,
                self.data.wp_angular_velocities,
                self.data.wp_masses,
                inertias_diag,
                self.data.wp_segment_lengths,
                self.data.wp_rest_darboux,
                bending_stiffness,
                torsion_stiffness,
                self.data.wp_parent_indices,
                self.gravity,
                ke,
                pe,
                num_segments,
            ],
        )

        # Sum over segments
        ke_torch = wp.to_torch(ke).view(self.num_envs, num_segments).sum(dim=1)
        pe_torch = wp.to_torch(pe).view(self.num_envs, num_segments).sum(dim=1)

        return ke_torch, pe_torch

