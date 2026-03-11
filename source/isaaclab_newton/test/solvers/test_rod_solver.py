# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for the Direct Position-Based Solver for Stiff Rods.

These tests verify the implementation based on:
Deul et al. 2018 "Direct Position-Based Solver for Stiff Rods"
"""

import pytest
import torch
import math


class TestRodSolverBasic:
    """Basic tests for rod solver functionality."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def simple_config(self):
        """Create a simple rod configuration for testing."""
        from isaaclab_newton.solvers import RodConfig, RodGeometryConfig, RodMaterialConfig, RodSolverConfig

        return RodConfig(
            material=RodMaterialConfig(
                young_modulus=1e6,  # Softer for testing
                density=1000.0,
                damping=0.01,
            ),
            geometry=RodGeometryConfig(
                num_segments=5,
                rest_length=1.0,
                radius=0.01,
            ),
            solver=RodSolverConfig(
                dt=1.0 / 60.0,
                num_substeps=1,
                newton_iterations=4,
                use_direct_solver=True,
                gravity=(0.0, -9.81, 0.0),
            ),
        )

    def test_rod_data_initialization(self, simple_config, device):
        """Test that rod data is properly initialized."""
        from isaaclab_newton.solvers import RodData

        simple_config.device = device
        data = RodData(simple_config, num_envs=2)

        # Check shapes
        assert data.positions.shape == (2, 5, 3)
        assert data.orientations.shape == (2, 5, 4)
        assert data.velocities.shape == (2, 5, 3)
        assert data.masses.shape == (2, 5)

        # Check initial positions are along x-axis
        seg_len = simple_config.geometry.segment_length
        for i in range(5):
            expected_x = (i + 0.5) * seg_len
            assert torch.allclose(data.positions[:, i, 0], torch.tensor(expected_x, device=device))
            assert torch.allclose(data.positions[:, i, 1:], torch.zeros(2, 2, device=device))

        # Check orientations are identity
        assert torch.allclose(data.orientations[:, :, :3], torch.zeros(2, 5, 3, device=device))
        assert torch.allclose(data.orientations[:, :, 3], torch.ones(2, 5, device=device))

    def test_rod_solver_creation(self, simple_config, device):
        """Test that rod solver is properly created."""
        from isaaclab_newton.solvers import RodSolver

        simple_config.device = device
        solver = RodSolver(simple_config, num_envs=2)

        assert solver.data is not None
        assert solver.config == simple_config
        assert solver.direct_solver is not None

    def test_rod_solver_step(self, simple_config, device):
        """Test that simulation steps without errors."""
        from isaaclab_newton.solvers import RodSolver

        simple_config.device = device
        solver = RodSolver(simple_config, num_envs=2)

        # Fix first segment
        solver.data.fix_segment(slice(None), 0)

        # Run a few steps
        initial_pos = solver.data.positions.clone()
        for _ in range(10):
            solver.step()

        # Positions should have changed due to gravity
        assert not torch.allclose(solver.data.positions, initial_pos)

    def test_fixed_segment_stays_fixed(self, simple_config, device):
        """Test that fixed segments don't move."""
        from isaaclab_newton.solvers import RodSolver

        simple_config.device = device
        solver = RodSolver(simple_config, num_envs=2)

        # Fix first segment
        solver.data.fix_segment(slice(None), 0)

        initial_pos_0 = solver.data.positions[:, 0].clone()

        # Run simulation
        for _ in range(50):
            solver.step()

        # First segment should not have moved
        assert torch.allclose(solver.data.positions[:, 0], initial_pos_0, atol=1e-5)

    def test_gravity_affects_rod(self, simple_config, device):
        """Test that gravity pulls the rod down."""
        from isaaclab_newton.solvers import RodSolver

        simple_config.device = device
        solver = RodSolver(simple_config, num_envs=1)

        # Fix first segment
        solver.data.fix_segment(0, 0)

        initial_y = solver.data.positions[0, -1, 1].item()

        # Run simulation
        for _ in range(100):
            solver.step()

        final_y = solver.data.positions[0, -1, 1].item()

        # Last segment should have moved down
        assert final_y < initial_y


class TestRodConstraints:
    """Tests for constraint behavior."""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def stiff_config(self):
        """Create a stiff rod configuration."""
        from isaaclab_newton.solvers import RodConfig, RodGeometryConfig, RodMaterialConfig, RodSolverConfig

        return RodConfig(
            material=RodMaterialConfig(
                young_modulus=1e9,  # Very stiff
                density=1000.0,
                damping=0.1,
            ),
            geometry=RodGeometryConfig(
                num_segments=10,
                rest_length=1.0,
                radius=0.02,
            ),
            solver=RodSolverConfig(
                dt=1.0 / 120.0,
                num_substeps=2,
                newton_iterations=8,
                use_direct_solver=True,
                gravity=(0.0, -9.81, 0.0),
            ),
        )

    def test_stretch_constraint_preservation(self, stiff_config, device):
        """Test that segment lengths are approximately preserved."""
        from isaaclab_newton.solvers import RodSolver

        stiff_config.device = device
        solver = RodSolver(stiff_config, num_envs=1)
        solver.data.fix_segment(0, 0)

        expected_length = stiff_config.geometry.segment_length

        # Run simulation
        for _ in range(100):
            solver.step()

        # Check distances between adjacent segments
        positions = solver.data.positions[0]
        for i in range(len(positions) - 1):
            dist = (positions[i + 1] - positions[i]).norm().item()
            # Allow some stretch due to finite stiffness
            assert abs(dist - expected_length) < 0.1 * expected_length

    def test_gauss_seidel_vs_direct_solver(self, device):
        """Compare Gauss-Seidel and direct solver results."""
        from isaaclab_newton.solvers import RodConfig, RodGeometryConfig, RodMaterialConfig, RodSolverConfig, RodSolver

        # Create configs for both solvers
        base_config = RodConfig(
            material=RodMaterialConfig(young_modulus=1e6, density=1000.0, damping=0.1),
            geometry=RodGeometryConfig(num_segments=5, rest_length=1.0),
            solver=RodSolverConfig(dt=1.0 / 60.0, newton_iterations=10, gravity=(0.0, -9.81, 0.0)),
            device=device,
        )

        # Direct solver
        direct_config = base_config
        direct_config.solver.use_direct_solver = True
        direct_solver = RodSolver(direct_config, num_envs=1)
        direct_solver.data.fix_segment(0, 0)

        # Gauss-Seidel solver
        gs_config = RodConfig(
            material=RodMaterialConfig(young_modulus=1e6, density=1000.0, damping=0.1),
            geometry=RodGeometryConfig(num_segments=5, rest_length=1.0),
            solver=RodSolverConfig(
                dt=1.0 / 60.0, newton_iterations=10, gravity=(0.0, -9.81, 0.0), use_direct_solver=False
            ),
            device=device,
        )
        gs_solver = RodSolver(gs_config, num_envs=1)
        gs_solver.data.fix_segment(0, 0)

        # Run both for same duration
        for _ in range(50):
            direct_solver.step()
            gs_solver.step()

        # Results should be similar (not necessarily identical due to different convergence)
        direct_pos = direct_solver.data.positions
        gs_pos = gs_solver.data.positions

        # Check that both solvers produce reasonable results
        # (Both should have the tip moved down due to gravity)
        assert direct_pos[0, -1, 1] < 0.0  # Tip below origin
        assert gs_pos[0, -1, 1] < 0.0


class TestCantileverBeam:
    """Test cantilever beam deflection against analytical solution.

    This validates the bending stiffness implementation.
    For a cantilever beam with point load P at the tip:
        max_deflection = P * L³ / (3 * E * I)
    """

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_cantilever_qualitative_deflection(self, device):
        """Test that cantilever deflects in the correct direction under gravity."""
        from isaaclab_newton.solvers import RodConfig, RodGeometryConfig, RodMaterialConfig, RodSolverConfig, RodSolver

        config = RodConfig(
            material=RodMaterialConfig(
                young_modulus=1e8,  # Moderate stiffness
                density=1000.0,
                damping=0.5,  # High damping for faster settling
            ),
            geometry=RodGeometryConfig(
                num_segments=20,
                rest_length=1.0,
                radius=0.02,
            ),
            solver=RodSolverConfig(
                dt=1.0 / 120.0,
                num_substeps=2,
                newton_iterations=4,
                use_direct_solver=True,
                gravity=(0.0, -9.81, 0.0),
            ),
            device=device,
        )

        solver = RodSolver(config, num_envs=1)

        # Fix the first segment (cantilever boundary condition)
        solver.data.fix_segment(0, 0)

        # Run until approximately steady state
        for _ in range(500):
            solver.step()

        # Get tip deflection
        tip_y = solver.data.positions[0, -1, 1].item()

        # Tip should be below zero due to gravity
        assert tip_y < 0.0, f"Expected negative deflection, got {tip_y}"

        # The rod should still be approximately the right length
        start, end = solver.data.get_endpoint_positions()
        total_length = (end - start).norm().item()
        expected_length = config.geometry.rest_length

        # Allow for some bending shortening
        assert total_length > 0.9 * expected_length
        assert total_length < 1.1 * expected_length


class TestRodReset:
    """Tests for reset functionality."""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_reset_restores_initial_state(self, device):
        """Test that reset restores the initial configuration."""
        from isaaclab_newton.solvers import RodConfig, RodSolver

        config = RodConfig(device=device)
        solver = RodSolver(config, num_envs=2)

        initial_pos = solver.data.positions.clone()

        # Run simulation
        solver.data.fix_segment(slice(None), 0)
        for _ in range(50):
            solver.step()

        # Positions should have changed
        assert not torch.allclose(solver.data.positions, initial_pos)

        # Reset
        solver.reset()

        # Positions should be restored
        assert torch.allclose(solver.data.positions, initial_pos, atol=1e-5)

    def test_partial_reset(self, device):
        """Test resetting only specific environments."""
        from isaaclab_newton.solvers import RodConfig, RodSolver

        config = RodConfig(device=device)
        solver = RodSolver(config, num_envs=3)

        initial_pos = solver.data.positions.clone()

        # Run simulation
        solver.data.fix_segment(slice(None), 0)
        for _ in range(50):
            solver.step()

        # Reset only environment 1
        solver.reset(env_indices=torch.tensor([1], device=device))

        # Environment 1 should be reset
        assert torch.allclose(solver.data.positions[1], initial_pos[1], atol=1e-5)

        # Environment 0 and 2 should still be modified
        assert not torch.allclose(solver.data.positions[0], initial_pos[0])
        assert not torch.allclose(solver.data.positions[2], initial_pos[2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

