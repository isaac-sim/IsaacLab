# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for RigidObjectData class (Newton backend).

This module provides a benchmarking framework to measure the performance of all properties
in the Newton RigidObjectData class. Each property is run multiple times with randomized mock data,
and timing statistics (mean and standard deviation) are reported.

Usage:
    python benchmark_rigid_object_data.py [--num_iterations N] [--warmup_steps W]
        [--num_instances I]

Example:
    python benchmark_rigid_object_data.py --num_iterations 10000 --warmup_steps 10
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Micro-benchmarking framework for RigidObjectData class (Newton backend).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations")
parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
parser.add_argument("--num_instances", type=int, default=4096, help="Number of instances")
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

import numpy as np
import warp as wp
from isaaclab_newton.test.mock_interfaces import MockNewtonArticulationView, create_mock_newton_manager

from isaaclab.test.benchmark import MethodBenchmarkRunner, MethodBenchmarkRunnerConfig

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Skip Lists
# =============================================================================

# List of deprecated properties - skip these
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
    # Combined state properties marked as deprecated
    "root_state_w",
    "root_link_state_w",
    "root_com_state_w",
    "body_state_w",
    "body_link_state_w",
    "body_com_state_w",
}

# List of properties that raise NotImplementedError - skip these
NOT_IMPLEMENTED_PROPERTIES = set()

# Removed default_* properties that raise RuntimeError
REMOVED_PROPERTIES = {
    "default_inertia",
    "default_mass",
}

# Private/internal properties and methods to skip
INTERNAL_PROPERTIES = {
    "_create_simulation_bindings",
    "_create_buffers",
    "update",
    "is_primed",
    "device",
    "body_names",
    "GRAVITY_VEC_W",
    "GRAVITY_VEC_W_TORCH",
    "FORWARD_VEC_B",
    "FORWARD_VEC_B_TORCH",
    "ALL_ENV_MASK",
    "ENV_MASK",
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


def get_benchmarkable_properties(rigid_object_data) -> list[str]:
    """Get list of properties that can be benchmarked."""
    all_properties = []

    for name in dir(rigid_object_data):
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
            attr = getattr(type(rigid_object_data), name, None)
            if isinstance(attr, property):
                all_properties.append(name)
        except Exception:
            pass

    return sorted(all_properties)


def setup_mock_environment(config: MethodBenchmarkRunnerConfig) -> MockNewtonArticulationView:
    """Set up the mock environment for benchmarking."""
    mock_view = MockNewtonArticulationView(
        num_instances=config.num_instances,
        num_bodies=config.num_bodies,
        num_joints=0,
        device=config.device,
    )
    return mock_view


def main():
    """Main entry point for the benchmarking script."""
    config = MethodBenchmarkRunnerConfig(
        num_iterations=args.num_iterations,
        warmup_steps=args.warmup_steps,
        num_instances=args.num_instances,
        num_bodies=1,
        num_joints=0,
        device=args.device,
    )

    # Patch the NewtonManager for the rigid_object_data module
    with create_mock_newton_manager(
        "isaaclab_newton.assets.rigid_object.rigid_object_data.SimulationManager",
        gravity=(0.0, 0.0, -9.81),
    ):
        # Setup mock environment
        mock_view = setup_mock_environment(config)
        mock_view.set_random_mock_data()

        # Import RigidObjectData inside the patch context
        from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData

        # Create RigidObjectData instance
        rigid_object_data = RigidObjectData(mock_view, config.device)

        # Get list of properties to benchmark
        properties = get_benchmarkable_properties(rigid_object_data)

        # Generator that updates mock data and invalidates timestamp
        def gen_mock_data(cfg: MethodBenchmarkRunnerConfig) -> dict:
            N, L = cfg.num_instances, cfg.num_bodies
            dev = cfg.device

            # Update root transforms
            root_tf_np = np.random.randn(N, 1, 7).astype(np.float32)
            root_tf_np[..., 3:7] /= np.linalg.norm(root_tf_np[..., 3:7], axis=-1, keepdims=True)
            mock_view.set_mock_root_transforms(wp.array(root_tf_np, dtype=wp.transformf, device=dev))

            # Update root velocities
            root_vel_np = np.random.randn(N, 1, 6).astype(np.float32)
            mock_view.set_mock_root_velocities(wp.array(root_vel_np, dtype=wp.spatial_vectorf, device=dev))

            # Update link transforms
            link_tf_np = np.random.randn(N, 1, L, 7).astype(np.float32)
            link_tf_np[..., 3:7] /= np.linalg.norm(link_tf_np[..., 3:7], axis=-1, keepdims=True)
            mock_view.set_mock_link_transforms(wp.array(link_tf_np, dtype=wp.transformf, device=dev))

            # Update link velocities
            link_vel_np = np.random.randn(N, 1, L, 6).astype(np.float32)
            mock_view.set_mock_link_velocities(wp.array(link_vel_np, dtype=wp.spatial_vectorf, device=dev))

            # Update body properties
            mock_view.set_mock_coms(
                wp.array(np.random.randn(N, 1, L, 3).astype(np.float32), dtype=wp.vec3f, device=dev)
            )
            mock_view.set_mock_inertias(
                wp.array(np.random.randn(N, 1, L, 9).astype(np.float32), dtype=wp.mat33f, device=dev)
            )
            mock_view.set_mock_masses(
                wp.array((np.random.rand(N, 1, L) * 10 + 0.1).astype(np.float32), dtype=wp.float32, device=dev)
            )

            # Invalidate timestamp to trigger recomputation
            rigid_object_data._sim_timestamp += 1.0
            return {}

        # Create runner
        runner = MethodBenchmarkRunner(
            benchmark_name="newton_rigid_object_data_benchmark",
            config=config,
            backend_type=args.backend,
            output_path=args.output_dir,
            use_recorders=True,
        )

        # Run property benchmarks
        runner.run_property_benchmarks(
            target_data=rigid_object_data,
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
