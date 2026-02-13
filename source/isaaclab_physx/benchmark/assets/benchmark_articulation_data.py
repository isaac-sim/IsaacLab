# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for ArticulationData class.

This module provides a benchmarking framework to measure the performance of all functions
in the ArticulationData class. Each function is run multiple times with randomized mock data,
and timing statistics (mean and standard deviation) are reported.

Usage:
    python benchmark_articulation_data.py [--num_iterations N] [--warmup_steps W]
        [--num_instances I] [--num_bodies B] [--num_joints J]

Example:
    python benchmark_articulation_data.py --num_iterations 10000 --warmup_steps 10
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Micro-benchmarking framework for ArticulationData class.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations")
parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
parser.add_argument("--num_instances", type=int, default=4096, help="Number of instances")
parser.add_argument("--num_bodies", type=int, default=12, help="Number of bodies")
parser.add_argument("--num_joints", type=int, default=11, help="Number of joints")
parser.add_argument("--output_dir", type=str, default=".", help="Output directory for results")
parser.add_argument("--backend", type=str, default="json", choices=["json", "osmo", "omniperf"], help="Metrics backend")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True, args=args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import warnings
from unittest.mock import MagicMock

import torch

# Mock SimulationManager.get_physics_sim_view() to return a mock object with gravity
# This is needed because the Data classes call SimulationManager.get_physics_sim_view().get_gravity()
# but there's no actual physics scene when running benchmarks
_mock_physics_sim_view = MagicMock()
_mock_physics_sim_view.get_gravity.return_value = (0.0, 0.0, -9.81)

from isaacsim.core.simulation_manager import SimulationManager

SimulationManager.get_physics_sim_view = MagicMock(return_value=_mock_physics_sim_view)

from isaaclab_physx.assets.articulation.articulation_data import ArticulationData
from isaaclab_physx.test.mock_interfaces.views import MockArticulationView

from isaaclab.test.benchmark import MethodBenchmarkRunner, MethodBenchmarkRunnerConfig

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Skip Lists
# =============================================================================

# List of deprecated properties (for backward compatibility) - skip these
DEPRECATED_PROPERTIES = {
    "default_root_state",
    "root_pose_w",
    "root_pos_w",
    "root_quat_w",
    "root_vel_w",
    "root_lin_vel_w",
    "root_ang_vel_w",
    "root_lin_vel_b",
    "root_ang_vel_b",
    "body_pose_w",
    "body_pos_w",
    "body_quat_w",
    "body_vel_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "body_acc_w",
    "body_lin_acc_w",
    "body_ang_acc_w",
    "com_pos_b",
    "com_quat_b",
    "joint_limits",
    "joint_friction",
    "fixed_tendon_limit",
    "applied_torque",
    "computed_torque",
    "joint_dynamic_friction",
    "joint_effort_target",
    "joint_viscous_friction",
    "joint_velocity_limits",
    # Also skip the combined state properties marked as deprecated
    "root_state_w",
    "root_link_state_w",
    "root_com_state_w",
    "body_state_w",
    "body_link_state_w",
    "body_com_state_w",
}

# List of properties that raise NotImplementedError - skip these
NOT_IMPLEMENTED_PROPERTIES = {
    "fixed_tendon_stiffness",
    "fixed_tendon_damping",
    "fixed_tendon_limit_stiffness",
    "fixed_tendon_rest_length",
    "fixed_tendon_offset",
    "fixed_tendon_pos_limits",
    "spatial_tendon_stiffness",
    "spatial_tendon_damping",
    "spatial_tendon_limit_stiffness",
    "spatial_tendon_offset",
    "body_incoming_joint_wrench_b",
}

# Removed default_* properties that raise RuntimeError
REMOVED_PROPERTIES = {
    "default_fixed_tendon_damping",
    "default_fixed_tendon_limit",
    "default_fixed_tendon_limit_stiffness",
    "default_fixed_tendon_offset",
    "default_fixed_tendon_pos_limits",
    "default_fixed_tendon_rest_length",
    "default_fixed_tendon_stiffness",
    "default_inertia",
    "default_joint_armature",
    "default_joint_damping",
    "default_joint_dynamic_friction_coeff",
    "default_joint_friction",
    "default_joint_friction_coeff",
    "default_joint_limits",
    "default_joint_pos_limits",
    "default_joint_stiffness",
    "default_joint_viscous_friction_coeff",
    "default_mass",
    "default_spatial_tendon_damping",
    "default_spatial_tendon_limit_stiffness",
    "default_spatial_tendon_offset",
    "default_spatial_tendon_stiffness",
}

# Private/internal properties and methods to skip
INTERNAL_PROPERTIES = {
    "_create_simulation_bindings",
    "_create_buffers",
    "update",
    "is_primed",
    "device",
    "body_names",
    "joint_names",
    "fixed_tendon_names",
    "spatial_tendon_names",
    "GRAVITY_VEC_W",
    "GRAVITY_VEC_W_TORCH",
    "FORWARD_VEC_B",
    "FORWARD_VEC_B_TORCH",
    "ALL_ENV_MASK",
    "ALL_BODY_MASK",
    "ALL_JOINT_MASK",
    "ENV_MASK",
    "BODY_MASK",
    "JOINT_MASK",
}

# Dependency mapping for properties
PROPERTY_DEPENDENCIES = {
    "root_link_lin_vel_w": ["root_link_vel_w"],
    "root_link_ang_vel_w": ["root_link_vel_w"],
    "root_link_lin_vel_b": ["root_link_vel_b"],
    "root_link_ang_vel_b": ["root_link_vel_b"],
    "root_com_pos_w": ["root_com_pose_w"],
    "root_com_quat_w": ["root_com_pose_w"],
    "root_com_lin_vel_b": ["root_com_vel_b"],
    "root_com_ang_vel_b": ["root_com_vel_b"],
    "root_com_lin_vel_w": ["root_com_vel_w"],
    "root_com_ang_vel_w": ["root_com_vel_w"],
    "root_link_pos_w": ["root_link_pose_w"],
    "root_link_quat_w": ["root_link_pose_w"],
    "body_link_lin_vel_w": ["body_link_vel_w"],
    "body_link_ang_vel_w": ["body_link_vel_w"],
    "body_link_pos_w": ["body_link_pose_w"],
    "body_link_quat_w": ["body_link_pose_w"],
    "body_com_pos_w": ["body_com_pose_w"],
    "body_com_quat_w": ["body_com_pose_w"],
    "body_com_lin_vel_w": ["body_com_vel_w"],
    "body_com_ang_vel_w": ["body_com_vel_w"],
    "body_com_lin_acc_w": ["body_com_acc_w"],
    "body_com_ang_acc_w": ["body_com_acc_w"],
    "body_com_quat_b": ["body_com_pose_b"],
}


# =============================================================================
# Benchmark Functions
# =============================================================================


def get_benchmarkable_properties(articulation_data: ArticulationData) -> list[str]:
    """Get list of properties that can be benchmarked."""
    all_properties = []

    for name in dir(articulation_data):
        if name.startswith("_"):
            continue
        if name in DEPRECATED_PROPERTIES:
            continue
        if name in NOT_IMPLEMENTED_PROPERTIES:
            continue
        if name in REMOVED_PROPERTIES:
            continue
        if name in INTERNAL_PROPERTIES:
            continue

        try:
            attr = getattr(type(articulation_data), name, None)
            if isinstance(attr, property):
                all_properties.append(name)
        except Exception:
            pass

    return sorted(all_properties)


def setup_mock_environment(config: MethodBenchmarkRunnerConfig) -> MockArticulationView:
    """Set up the mock environment for benchmarking."""
    mock_view = MockArticulationView(
        count=config.num_instances,
        num_links=config.num_bodies,
        num_dofs=config.num_joints,
        device=config.device,
    )
    return mock_view


def main():
    """Main entry point for the benchmarking script."""
    config = MethodBenchmarkRunnerConfig(
        num_iterations=args.num_iterations,
        warmup_steps=args.warmup_steps,
        num_instances=args.num_instances,
        num_bodies=args.num_bodies,
        num_joints=args.num_joints,
        device=args.device,
    )

    # Setup mock environment
    mock_view = setup_mock_environment(config)
    mock_view.set_random_mock_data()

    # Create ArticulationData instance
    articulation_data = ArticulationData(mock_view, config.device)

    # Get list of properties to benchmark
    properties = get_benchmarkable_properties(articulation_data)

    # Generator that updates mock data and invalidates timestamp
    def gen_mock_data(cfg: MethodBenchmarkRunnerConfig) -> dict:
        mock_view.set_mock_coms(torch.randn(cfg.num_instances, cfg.num_bodies, 7, device=cfg.device))
        mock_view.set_mock_inertias(torch.randn(cfg.num_instances, cfg.num_bodies, 3, 3, device=cfg.device))
        mock_view.set_mock_root_transforms(torch.randn(cfg.num_instances, 7, device=cfg.device))
        mock_view.set_mock_root_velocities(torch.randn(cfg.num_instances, 6, device=cfg.device))
        mock_view.set_mock_link_transforms(torch.randn(cfg.num_instances, cfg.num_bodies, 7, device=cfg.device))
        mock_view.set_mock_link_velocities(torch.randn(cfg.num_instances, cfg.num_bodies, 6, device=cfg.device))
        mock_view.set_mock_link_accelerations(torch.randn(cfg.num_instances, cfg.num_bodies, 6, device=cfg.device))
        mock_view.set_mock_dof_positions(torch.randn(cfg.num_instances, cfg.num_joints, device=cfg.device))
        mock_view.set_mock_dof_velocities(torch.randn(cfg.num_instances, cfg.num_joints, device=cfg.device))
        articulation_data._sim_timestamp += 1.0
        return {}

    # Create runner
    runner = MethodBenchmarkRunner(
        benchmark_name="articulation_data_benchmark",
        config=config,
        backend_type=args.backend,
        output_path=args.output_dir,
        use_recorders=True,
    )

    # Run property benchmarks
    runner.run_property_benchmarks(
        target_data=articulation_data,
        properties=properties,
        gen_mock_data=gen_mock_data,
        dependencies=PROPERTY_DEPENDENCIES,
        category="property",
    )

    runner.finalize()

    # Close the simulation app
    simulation_app.close()


if __name__ == "__main__":
    main()
