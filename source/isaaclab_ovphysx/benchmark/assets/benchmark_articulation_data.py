# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for ArticulationData class (OVPhysX backend).

This module provides a benchmarking framework to measure the performance of all properties
in the OVPhysX ArticulationData class. Each property is run multiple times after invalidating
its cached data, and timing statistics (mean and standard deviation) are reported.

Usage:
    python benchmark_articulation_data.py [--num_iterations N] [--warmup_steps W]
        [--num_instances I] [--num_bodies B] [--num_joints J]

Example:
    python benchmark_articulation_data.py --num_iterations 10000 --warmup_steps 10
"""

from __future__ import annotations

import argparse

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Micro-benchmarking framework for ArticulationData class (OVPhysX backend).",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations")
parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
parser.add_argument("--num_instances", type=int, default=4096, help="Number of instances")
parser.add_argument("--num_bodies", type=int, default=12, help="Number of bodies")
parser.add_argument("--num_joints", type=int, default=11, help="Number of joints")
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
    "joint_limits",
    "joint_friction",
    "fixed_tendon_limit",
    "applied_torque",
    "computed_torque",
    "joint_dynamic_friction",
    "joint_effort_target",
    "joint_viscous_friction",
    "joint_velocity_limits",
    # Combined state properties marked as deprecated
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
    "gravity_compensation_forces",
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
    "has_body_ordering",
    "has_joint_ordering",
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


def get_benchmarkable_properties(articulation_data) -> list[str]:
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

        attr = getattr(type(articulation_data), name, None)
        if isinstance(attr, property):
            try:
                getattr(articulation_data, name)
            except NotImplementedError:
                continue
            all_properties.append(name)

    return sorted(all_properties)


def setup_mock_environment(config: MethodBenchmarkRunnerConfig) -> MockOvPhysxView:
    """Set up the mock environment for benchmarking."""
    bindings = MockOvPhysxBindingSet(
        num_instances=config.num_instances,
        num_bodies=config.num_bodies,
        num_joints=config.num_joints,
        benchmark_mode=True,
    )
    bindings.set_random_data()
    return bindings.view


def main():
    """Main entry point for the benchmarking script."""
    from isaaclab_ovphysx.assets.articulation.articulation_data import ArticulationData

    config = MethodBenchmarkRunnerConfig(
        num_iterations=args.num_iterations,
        warmup_steps=args.warmup_steps,
        num_instances=args.num_instances,
        num_bodies=args.num_bodies,
        num_joints=args.num_joints,
        device=args.device,
    )
    mock_view = setup_mock_environment(config)
    articulation_data = ArticulationData(mock_view, config.device)
    articulation_data._apply_ordering_maps_after_resolve()
    properties = get_benchmarkable_properties(articulation_data)

    def gen_mock_data(_cfg: MethodBenchmarkRunnerConfig) -> dict:
        articulation_data._sim_timestamp += 1.0
        articulation_data._fk_timestamp = articulation_data._sim_timestamp
        return {}

    runner = MethodBenchmarkRunner(
        benchmark_name="ovphysx_articulation_data_benchmark",
        config=config,
        backend_type=args.benchmark_formatter,
        output_path=args.output_path,
        use_recorders=True,
    )
    runner.run_property_benchmarks(
        target_data=articulation_data,
        properties=properties,
        gen_mock_data=gen_mock_data,
        dependencies=PROPERTY_DEPENDENCIES,
        category="property",
    )
    runner.finalize()


if __name__ == "__main__":
    main()
