# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for RigidObjectData class (OVPhysX backend).

This module provides a benchmarking framework to measure the performance of all properties
in the OVPhysX RigidObjectData class. Each property is run multiple times after invalidating
its cached data, and timing statistics (mean and standard deviation) are reported.

Usage:
    python benchmark_rigid_object_data.py [--num_iterations N] [--warmup_steps W]
        [--num_instances I]

Example:
    python benchmark_rigid_object_data.py --num_iterations 10000 --warmup_steps 10
"""

from __future__ import annotations

import argparse

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Micro-benchmarking framework for RigidObjectData class (OVPhysX backend).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations")
parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
parser.add_argument("--num_instances", type=int, default=4096, help="Number of instances")
parser.add_argument("--device", type=str, default="cuda:0", help="Device for tensors")
parser.add_argument(
    "--output_path", "--output_dir", dest="output_path", type=str, default=".", help="Output directory for results"
)
parser.add_argument(
    "--benchmark_formatter",
    "--backend",
    dest="benchmark_formatter",
    type=str,
    default="json",
    choices=["json", "osmo", "omniperf", "summary"],
    help="Metrics formatter",
)

args = parser.parse_args()


import warnings

from isaaclab_ovphysx.test.mock_interfaces import MockOvPhysxBindingSet, MockOvPhysxView

from isaaclab.benchmark import MethodBenchmarkRunner, MethodBenchmarkRunnerConfig

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
    "root_link_lin_vel_b": ["root_link_lin_vel_w", "root_link_quat_w"],
    "root_link_ang_vel_b": ["root_link_ang_vel_w", "root_link_quat_w"],
    "root_com_pos_w": ["root_com_pose_w"],
    "root_com_quat_w": ["root_com_pose_w"],
    "root_com_lin_vel_b": ["root_com_lin_vel_w", "root_link_quat_w"],
    "root_com_ang_vel_b": ["root_com_ang_vel_w", "root_link_quat_w"],
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

        attr = getattr(type(rigid_object_data), name, None)
        if isinstance(attr, property):
            try:
                getattr(rigid_object_data, name)
            except NotImplementedError:
                continue
            all_properties.append(name)

    return sorted(all_properties)


def setup_mock_environment(config: MethodBenchmarkRunnerConfig) -> MockOvPhysxView:
    """Set up the mock environment for benchmarking."""
    bindings = MockOvPhysxBindingSet(
        num_instances=config.num_instances,
        num_bodies=1,
        num_joints=0,
        asset_kind="rigid_object",
        benchmark_mode=True,
    )
    bindings.set_random_data()
    return bindings.view


def main():
    """Main entry point for the benchmarking script."""
    from isaaclab_ovphysx.assets.rigid_object.rigid_object_data import RigidObjectData

    config = MethodBenchmarkRunnerConfig(
        num_iterations=args.num_iterations,
        warmup_steps=args.warmup_steps,
        num_instances=args.num_instances,
        num_bodies=1,
        num_joints=0,
        device=args.device,
    )
    mock_view = setup_mock_environment(config)
    rigid_object_data = RigidObjectData(mock_view, config.device)
    properties = get_benchmarkable_properties(rigid_object_data)

    def gen_mock_data(_cfg: MethodBenchmarkRunnerConfig) -> dict:
        rigid_object_data._sim_timestamp += 1.0
        return {}

    runner = MethodBenchmarkRunner(
        benchmark_name="ovphysx_rigid_object_data_benchmark",
        config=config,
        backend_type=args.benchmark_formatter,
        output_path=args.output_path,
        use_recorders=True,
    )
    runner.run_property_benchmarks(
        target_data=rigid_object_data,
        properties=properties,
        gen_mock_data=gen_mock_data,
        dependencies=PROPERTY_DEPENDENCIES,
        category="property",
    )
    runner.finalize()


if __name__ == "__main__":
    main()
