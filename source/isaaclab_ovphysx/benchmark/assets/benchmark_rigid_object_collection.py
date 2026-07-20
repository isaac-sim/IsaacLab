# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for RigidObjectCollection class (OVPhysX backend).

This module provides a benchmarking framework to measure the performance of setter and writer
methods in the RigidObjectCollection class. Each method is benchmarked under three scenarios:

1. **Torch List**: Inputs are PyTorch tensors with list indices (via deprecated wrappers).
2. **Torch Tensor**: Inputs are PyTorch tensors with tensor indices (via deprecated wrappers).
3. **Warp Mask**: Inputs are warp arrays with boolean masks (via ``_mask`` methods).

Usage:
    python benchmark_rigid_object_collection.py [--num_iterations N] [--warmup_steps W]
        [--num_instances I] [--num_bodies B]

Example:
    python benchmark_rigid_object_collection.py --num_iterations 1000 --warmup_steps 10
    python benchmark_rigid_object_collection.py --mode torch_list  # Only run list-based benchmarks
    python benchmark_rigid_object_collection.py --mode warp_mask   # Only run warp mask benchmarks
"""

from __future__ import annotations

import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Benchmark RigidObjectCollection methods (OVPhysX backend).")
parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations")
parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
parser.add_argument("--num_instances", type=int, default=4096, help="Number of instances")
parser.add_argument("--num_bodies", type=int, default=3, help="Number of bodies (object types)")
parser.add_argument(
    "--mode",
    type=str,
    default="all",
    choices=["all", "torch_list", "torch_tensor", "warp_mask"],
    help="Benchmark mode",
)
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
parser.add_argument("--no_shape_checks", action="store_true", help="Disable shape/dtype assertions")

args = parser.parse_args()


import logging
import warnings

import torch
import warp as wp
from isaaclab_ovphysx.test.mock_interfaces import MockOvPhysxBindingSet

from isaaclab.assets.rigid_object_collection.rigid_object_collection_cfg import RigidObjectCollectionCfg
from isaaclab.benchmark import MethodBenchmarkDefinition, MethodBenchmarkRunner, MethodBenchmarkRunnerConfig

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Also suppress logging warnings
logging.getLogger("isaaclab_ovphysx").setLevel(logging.ERROR)
logging.getLogger("isaaclab").setLevel(logging.ERROR)


# =============================================================================
# Index Helpers
# =============================================================================


def make_tensor_env_ids(num_instances: int, device: str) -> torch.Tensor:
    """Create a tensor of environment IDs."""
    return torch.arange(num_instances, dtype=torch.int32, device=device)


def make_tensor_body_ids(num_bodies: int, device: str) -> torch.Tensor:
    """Create a tensor of body IDs."""
    return torch.arange(num_bodies, dtype=torch.int32, device=device)


# =============================================================================
# Test RigidObjectCollection Factory
# =============================================================================


def create_test_collection(
    num_instances: int = 2,
    num_bodies: int = 3,
    device: str = "cuda:0",
):
    """Create a test RigidObjectCollection instance with mocked dependencies."""
    from isaaclab_ovphysx.assets.rigid_object_collection.rigid_object_collection import RigidObjectCollection
    from isaaclab_ovphysx.assets.rigid_object_collection.rigid_object_collection_data import RigidObjectCollectionData

    from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg

    object_names = [f"object_{i}" for i in range(num_bodies)]
    binding_set = MockOvPhysxBindingSet(
        num_instances=num_instances,
        num_joints=0,
        num_bodies=num_bodies,
        body_names=object_names,
        benchmark_mode=True,
    )
    binding_set.set_random_data()
    mock_view = binding_set.view

    collection = object.__new__(RigidObjectCollection)
    rigid_objects = {name: RigidObjectCfg(prim_path=f"/World/{name}") for name in object_names}
    collection.cfg = RigidObjectCollectionCfg(rigid_objects=rigid_objects)
    object.__setattr__(collection, "_initialize_handle", None)
    object.__setattr__(collection, "_invalidate_initialize_handle", None)
    object.__setattr__(collection, "_prim_deletion_handle", None)
    object.__setattr__(collection, "_debug_vis_handle", None)
    object.__setattr__(collection, "_root_view", mock_view)
    object.__setattr__(collection, "_device", device)
    object.__setattr__(collection, "_check_shapes", not args.no_shape_checks)
    object.__setattr__(collection, "_num_instances", num_instances)
    object.__setattr__(collection, "_num_bodies", num_bodies)
    object.__setattr__(collection, "_body_names_list", object_names)
    object.__setattr__(collection, "_object_names", object_names)

    data = RigidObjectCollectionData(mock_view, num_bodies, device)
    object.__setattr__(collection, "_data", data)
    collection._create_buffers()

    return collection, mock_view


# Input Generators (Torch-only for OVPhysX backend)
# =============================================================================


# --- Body Link Pose ---
def gen_body_link_pose_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "body_poses": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_body_link_pose_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "body_poses": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- Body COM Pose ---
def gen_body_com_pose_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "body_poses": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_body_com_pose_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "body_poses": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- Body Link Velocity ---
def gen_body_link_velocity_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "body_velocities": torch.rand(
            config.num_instances, config.num_bodies, 6, device=config.device, dtype=torch.float32
        ),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_body_link_velocity_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "body_velocities": torch.rand(
            config.num_instances, config.num_bodies, 6, device=config.device, dtype=torch.float32
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- Body COM Velocity ---
def gen_body_com_velocity_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "body_velocities": torch.rand(
            config.num_instances, config.num_bodies, 6, device=config.device, dtype=torch.float32
        ),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_body_com_velocity_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "body_velocities": torch.rand(
            config.num_instances, config.num_bodies, 6, device=config.device, dtype=torch.float32
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- Set Masses ---
def gen_set_masses_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "masses": torch.rand(config.num_instances, config.num_bodies, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_set_masses_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "masses": torch.rand(config.num_instances, config.num_bodies, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- Set CoMs ---
def gen_set_coms_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_set_coms_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- Set Inertias ---
def gen_set_inertias_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "inertias": torch.rand(config.num_instances, config.num_bodies, 9, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_set_inertias_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "inertias": torch.rand(config.num_instances, config.num_bodies, 9, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- Set External Force and Torque ---
def gen_set_external_force_and_torque_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "forces": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_set_external_force_and_torque_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "forces": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


# =============================================================================
# Warp Mask Input Generators (for _mask methods)
# =============================================================================


def _env_mask(config: MethodBenchmarkRunnerConfig) -> wp.array:
    return wp.ones((config.num_instances,), dtype=wp.bool, device=config.device)


def _body_mask(config: MethodBenchmarkRunnerConfig) -> wp.array:
    return wp.ones((config.num_bodies,), dtype=wp.bool, device=config.device)


# --- Body Link Pose (mask) ---
def gen_body_link_pose_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "body_poses": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "env_mask": _env_mask(config),
    }


# --- Body COM Pose (mask) ---
def gen_body_com_pose_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "body_poses": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "env_mask": _env_mask(config),
    }


# --- Body Link Velocity (mask) ---
def gen_body_link_velocity_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "body_velocities": torch.rand(
            config.num_instances, config.num_bodies, 6, device=config.device, dtype=torch.float32
        ),
        "env_mask": _env_mask(config),
    }


# --- Body COM Velocity (mask) ---
def gen_body_com_velocity_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "body_velocities": torch.rand(
            config.num_instances, config.num_bodies, 6, device=config.device, dtype=torch.float32
        ),
        "env_mask": _env_mask(config),
    }


# --- Set Masses (mask) ---
def gen_set_masses_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "masses": torch.rand(config.num_instances, config.num_bodies, device=config.device, dtype=torch.float32),
        "body_mask": _body_mask(config),
        "env_mask": _env_mask(config),
    }


# --- Set CoMs (mask) ---
def gen_set_coms_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "body_mask": _body_mask(config),
        "env_mask": _env_mask(config),
    }


# --- Set Inertias (mask) ---
def gen_set_inertias_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "inertias": torch.rand(config.num_instances, config.num_bodies, 9, device=config.device, dtype=torch.float32),
        "body_mask": _body_mask(config),
        "env_mask": _env_mask(config),
    }


# =============================================================================
# Benchmarks
# =============================================================================

BENCHMARKS = [
    # --- Body Link Pose ---
    MethodBenchmarkDefinition(
        name="write_body_link_pose_to_sim",
        method_name="write_body_link_pose_to_sim",
        input_generators={
            "torch_list": gen_body_link_pose_torch_list,
            "torch_tensor": gen_body_link_pose_torch_tensor,
        },
        category="body_pose",
    ),
    MethodBenchmarkDefinition(
        name="write_body_link_pose_to_sim_mask",
        method_name="write_body_link_pose_to_sim_mask",
        input_generators={"warp_mask": gen_body_link_pose_warp_mask},
        category="body_pose",
    ),
    # --- Body COM Pose ---
    MethodBenchmarkDefinition(
        name="write_body_com_pose_to_sim",
        method_name="write_body_com_pose_to_sim",
        input_generators={
            "torch_list": gen_body_com_pose_torch_list,
            "torch_tensor": gen_body_com_pose_torch_tensor,
        },
        category="body_pose",
    ),
    MethodBenchmarkDefinition(
        name="write_body_com_pose_to_sim_mask",
        method_name="write_body_com_pose_to_sim_mask",
        input_generators={"warp_mask": gen_body_com_pose_warp_mask},
        category="body_pose",
    ),
    # --- Body Link Velocity ---
    MethodBenchmarkDefinition(
        name="write_body_link_velocity_to_sim",
        method_name="write_body_link_velocity_to_sim",
        input_generators={
            "torch_list": gen_body_link_velocity_torch_list,
            "torch_tensor": gen_body_link_velocity_torch_tensor,
        },
        category="body_velocity",
    ),
    MethodBenchmarkDefinition(
        name="write_body_link_velocity_to_sim_mask",
        method_name="write_body_link_velocity_to_sim_mask",
        input_generators={"warp_mask": gen_body_link_velocity_warp_mask},
        category="body_velocity",
    ),
    # --- Body COM Velocity ---
    MethodBenchmarkDefinition(
        name="write_body_com_velocity_to_sim",
        method_name="write_body_com_velocity_to_sim",
        input_generators={
            "torch_list": gen_body_com_velocity_torch_list,
            "torch_tensor": gen_body_com_velocity_torch_tensor,
        },
        category="body_velocity",
    ),
    MethodBenchmarkDefinition(
        name="write_body_com_velocity_to_sim_mask",
        method_name="write_body_com_velocity_to_sim_mask",
        input_generators={"warp_mask": gen_body_com_velocity_warp_mask},
        category="body_velocity",
    ),
    # --- Body Properties ---
    MethodBenchmarkDefinition(
        name="set_masses",
        method_name="set_masses",
        input_generators={
            "torch_list": gen_set_masses_torch_list,
            "torch_tensor": gen_set_masses_torch_tensor,
        },
        category="body_props",
    ),
    MethodBenchmarkDefinition(
        name="set_masses_mask",
        method_name="set_masses_mask",
        input_generators={"warp_mask": gen_set_masses_warp_mask},
        category="body_props",
    ),
    MethodBenchmarkDefinition(
        name="set_coms",
        method_name="set_coms",
        input_generators={
            "torch_list": gen_set_coms_torch_list,
            "torch_tensor": gen_set_coms_torch_tensor,
        },
        category="body_props",
    ),
    MethodBenchmarkDefinition(
        name="set_coms_mask",
        method_name="set_coms_mask",
        input_generators={"warp_mask": gen_set_coms_warp_mask},
        category="body_props",
    ),
    MethodBenchmarkDefinition(
        name="set_inertias",
        method_name="set_inertias",
        input_generators={
            "torch_list": gen_set_inertias_torch_list,
            "torch_tensor": gen_set_inertias_torch_tensor,
        },
        category="body_props",
    ),
    MethodBenchmarkDefinition(
        name="set_inertias_mask",
        method_name="set_inertias_mask",
        input_generators={"warp_mask": gen_set_inertias_warp_mask},
        category="body_props",
    ),
    # --- External Force and Torque ---
    MethodBenchmarkDefinition(
        name="set_external_force_and_torque",
        method_name="set_external_force_and_torque",
        input_generators={
            "torch_list": gen_set_external_force_and_torque_torch_list,
            "torch_tensor": gen_set_external_force_and_torque_torch_tensor,
        },
        category="external_wrench",
    ),
]


def main():
    """Main entry point for the benchmarking script."""
    config = MethodBenchmarkRunnerConfig(
        num_iterations=args.num_iterations,
        warmup_steps=args.warmup_steps,
        num_instances=args.num_instances,
        num_bodies=args.num_bodies,
        num_joints=0,
        device=args.device,
        mode=args.mode,
    )

    # Create the test collection
    collection, _ = create_test_collection(
        num_instances=config.num_instances,
        num_bodies=config.num_bodies,
        device=config.device,
    )

    print(
        f"Benchmarking RigidObjectCollection (OVPhysX) with {config.num_instances} instances, "
        f"{config.num_bodies} bodies..."
    )

    # Create runner and run benchmarks
    runner = MethodBenchmarkRunner(
        benchmark_name="ovphysx_rigid_object_collection_benchmark",
        config=config,
        backend_type=args.benchmark_formatter,
        output_path=args.output_path,
        use_recorders=True,
    )

    runner.run_benchmarks(BENCHMARKS, collection)
    runner.finalize()


if __name__ == "__main__":
    main()
