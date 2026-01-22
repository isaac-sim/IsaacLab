# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for RigidObject class.

This module provides a benchmarking framework to measure the performance of setter and writer
methods in the RigidObject class. Each method is benchmarked under two scenarios:

1. **Best Case (Warp)**: Inputs are Warp arrays with masks - this is the optimal path that
   avoids any data conversion overhead.

2. **Worst Case (Torch)**: Inputs are PyTorch tensors with indices - this path requires
   conversion from Torch to Warp and from indices to masks.

Usage:
    python benchmark_rigid_object.py [--num_iterations N] [--warmup_steps W] [--num_instances I]

Example:
    python benchmark_rigid_object.py --num_iterations 1000 --warmup_steps 10
    python benchmark_rigid_object.py --mode warp  # Only run Warp benchmarks
    python benchmark_rigid_object.py --mode torch  # Only run Torch benchmarks
"""

from __future__ import annotations

import argparse
import sys
import torch
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import warp as wp
from isaaclab_newton.assets.rigid_object.rigid_object import RigidObject
from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab_newton.kernels import vec13f

from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg

# Add test directory to path for common module imports
_TEST_DIR = Path(__file__).resolve().parents[2]
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

# Import shared utilities from common module
from common.benchmark_core import (
    BenchmarkConfig,
    MethodBenchmark,
    benchmark_method,
    make_tensor_body_ids,
    make_tensor_env_ids,
    make_warp_body_mask,
    make_warp_env_mask,
)
from common.benchmark_io import (
    export_results_json,
    get_default_output_filename,
    get_hardware_info,
    print_hardware_info,
    print_results,
)

# Import mock classes from common test utilities
from common.mock_newton import MockNewtonArticulationView, MockNewtonModel, MockWrenchComposer

# Initialize Warp
wp.init()

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def create_test_rigid_object(
    num_instances: int = 2,
    num_bodies: int = 1,
    device: str = "cuda:0",
) -> tuple[RigidObject, MockNewtonArticulationView, MagicMock]:
    """Create a test RigidObject instance with mocked dependencies."""
    body_names = [f"body_{i}" for i in range(num_bodies)]

    rigid_object = object.__new__(RigidObject)

    rigid_object.cfg = RigidObjectCfg(
        prim_path="/World/Object",
    )

    mock_view = MockNewtonArticulationView(
        num_instances=num_instances,
        num_bodies=num_bodies,
        num_joints=0,
        device=device,
        is_fixed_base=False,
        joint_names=[],
        body_names=body_names,
    )
    mock_view.set_mock_data()

    object.__setattr__(rigid_object, "_root_view", mock_view)
    object.__setattr__(rigid_object, "_device", device)

    mock_newton_manager = MagicMock()
    mock_model = MockNewtonModel()
    mock_state = MagicMock()
    mock_control = MagicMock()
    mock_newton_manager.get_model.return_value = mock_model
    mock_newton_manager.get_state_0.return_value = mock_state
    mock_newton_manager.get_control.return_value = mock_control
    mock_newton_manager.get_dt.return_value = 0.01

    with patch("isaaclab_newton.assets.rigid_object.rigid_object_data.NewtonManager", mock_newton_manager):
        data = RigidObjectData(mock_view, device)
        object.__setattr__(rigid_object, "_data", data)

    # Call _create_buffers() with MockWrenchComposer patched in
    with patch("isaaclab_newton.assets.rigid_object.rigid_object.WrenchComposer", MockWrenchComposer):
        rigid_object._create_buffers()

    return rigid_object, mock_view, mock_newton_manager


# =============================================================================
# Input Generators
# =============================================================================


# --- Root State (Deprecated) ---
def gen_root_state_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_root_state_to_sim."""
    return {
        "root_state": wp.from_torch(
            torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
            dtype=vec13f,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
    }


def gen_root_state_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_state_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


# --- Root COM State (Deprecated) ---
def gen_root_com_state_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_root_com_state_to_sim."""
    return {
        "root_state": wp.from_torch(
            torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
            dtype=vec13f,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
    }


def gen_root_com_state_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_com_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_com_state_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_com_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


# --- Root Link State (Deprecated) ---
def gen_root_link_state_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_root_link_state_to_sim."""
    return {
        "root_state": wp.from_torch(
            torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
            dtype=vec13f,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
    }


def gen_root_link_state_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_link_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_link_state_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_link_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


# --- Root Link Pose ---
def gen_root_link_pose_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_root_link_pose_to_sim."""
    return {
        "pose": wp.from_torch(
            torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
            dtype=wp.transformf,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
    }


def gen_root_link_pose_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_link_pose_to_sim."""
    return {
        "pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_link_pose_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_link_pose_to_sim."""
    return {
        "pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


# --- Root COM Pose ---
def gen_root_com_pose_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_root_com_pose_to_sim."""
    return {
        "root_pose": wp.from_torch(
            torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
            dtype=wp.transformf,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
    }


def gen_root_com_pose_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_com_pose_to_sim."""
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_com_pose_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_com_pose_to_sim."""
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


# --- Root Link Velocity ---
def gen_root_link_velocity_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_root_link_velocity_to_sim."""
    return {
        "root_velocity": wp.from_torch(
            torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
            dtype=wp.spatial_vectorf,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
    }


def gen_root_link_velocity_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_link_velocity_to_sim."""
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_link_velocity_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_link_velocity_to_sim."""
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


# --- Root COM Velocity ---
def gen_root_com_velocity_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_root_com_velocity_to_sim."""
    return {
        "root_velocity": wp.from_torch(
            torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
            dtype=wp.spatial_vectorf,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
    }


def gen_root_com_velocity_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_com_velocity_to_sim."""
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_com_velocity_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_com_velocity_to_sim."""
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


# --- Masses ---
def gen_masses_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for set_masses."""
    return {
        "masses": wp.from_torch(
            torch.rand(config.num_instances, config.num_bodies, device=config.device, dtype=torch.float32),
            dtype=wp.float32,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "body_mask": make_warp_body_mask(config.num_bodies, config.device),
    }


def gen_masses_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_masses."""
    return {
        "masses": torch.rand(config.num_instances, config.num_bodies, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_masses_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_masses."""
    return {
        "masses": torch.rand(config.num_instances, config.num_bodies, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- CoMs ---
def gen_coms_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for set_coms."""
    return {
        "coms": wp.from_torch(
            torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "body_mask": make_warp_body_mask(config.num_bodies, config.device),
    }


def gen_coms_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_coms."""
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_coms_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_coms."""
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- Inertias ---
def gen_inertias_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for set_inertias."""
    return {
        "inertias": wp.from_torch(
            torch.rand(config.num_instances, config.num_bodies, 3, 3, device=config.device, dtype=torch.float32),
            dtype=wp.mat33f,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "body_mask": make_warp_body_mask(config.num_bodies, config.device),
    }


def gen_inertias_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_inertias."""
    return {
        "inertias": torch.rand(
            config.num_instances, config.num_bodies, 3, 3, device=config.device, dtype=torch.float32
        ),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_inertias_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_inertias."""
    return {
        "inertias": torch.rand(
            config.num_instances, config.num_bodies, 3, 3, device=config.device, dtype=torch.float32
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- External Wrench ---
def gen_external_force_and_torque_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for set_external_force_and_torque."""
    return {
        "forces": wp.from_torch(
            torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "torques": wp.from_torch(
            torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "body_mask": make_warp_body_mask(config.num_bodies, config.device),
    }


def gen_external_force_and_torque_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_external_force_and_torque."""
    return {
        "forces": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_external_force_and_torque_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_external_force_and_torque."""
    return {
        "forces": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# =============================================================================
# Benchmarks
# =============================================================================

BENCHMARK_DEPENDENCIES = {}

BENCHMARKS = [
    # --- Root State (Deprecated) ---
    MethodBenchmark(
        name="write_root_state_to_sim",
        method_name="write_root_state_to_sim",
        input_generators={
            "warp": gen_root_state_warp,
            "torch_list": gen_root_state_torch_list,
            "torch_tensor": gen_root_state_torch_tensor,
        },
        category="root_state",
    ),
    MethodBenchmark(
        name="write_root_com_state_to_sim",
        method_name="write_root_com_state_to_sim",
        input_generators={
            "warp": gen_root_com_state_warp,
            "torch_list": gen_root_com_state_torch_list,
            "torch_tensor": gen_root_com_state_torch_tensor,
        },
        category="root_state",
    ),
    MethodBenchmark(
        name="write_root_link_state_to_sim",
        method_name="write_root_link_state_to_sim",
        input_generators={
            "warp": gen_root_link_state_warp,
            "torch_list": gen_root_link_state_torch_list,
            "torch_tensor": gen_root_link_state_torch_tensor,
        },
        category="root_state",
    ),
    # --- Root Pose / Velocity ---
    MethodBenchmark(
        name="write_root_link_pose_to_sim",
        method_name="write_root_link_pose_to_sim",
        input_generators={
            "warp": gen_root_link_pose_warp,
            "torch_list": gen_root_link_pose_torch_list,
            "torch_tensor": gen_root_link_pose_torch_tensor,
        },
        category="root_pose",
    ),
    MethodBenchmark(
        name="write_root_com_pose_to_sim",
        method_name="write_root_com_pose_to_sim",
        input_generators={
            "warp": gen_root_com_pose_warp,
            "torch_list": gen_root_com_pose_torch_list,
            "torch_tensor": gen_root_com_pose_torch_tensor,
        },
        category="root_pose",
    ),
    MethodBenchmark(
        name="write_root_link_velocity_to_sim",
        method_name="write_root_link_velocity_to_sim",
        input_generators={
            "warp": gen_root_link_velocity_warp,
            "torch_list": gen_root_link_velocity_torch_list,
            "torch_tensor": gen_root_link_velocity_torch_tensor,
        },
        category="root_velocity",
    ),
    MethodBenchmark(
        name="write_root_com_velocity_to_sim",
        method_name="write_root_com_velocity_to_sim",
        input_generators={
            "warp": gen_root_com_velocity_warp,
            "torch_list": gen_root_com_velocity_torch_list,
            "torch_tensor": gen_root_com_velocity_torch_tensor,
        },
        category="root_velocity",
    ),
    # --- Body Properties ---
    MethodBenchmark(
        name="set_masses",
        method_name="set_masses",
        input_generators={
            "warp": gen_masses_warp,
            "torch_list": gen_masses_torch_list,
            "torch_tensor": gen_masses_torch_tensor,
        },
        category="body_props",
    ),
    MethodBenchmark(
        name="set_coms",
        method_name="set_coms",
        input_generators={
            "warp": gen_coms_warp,
            "torch_list": gen_coms_torch_list,
            "torch_tensor": gen_coms_torch_tensor,
        },
        category="body_props",
    ),
    MethodBenchmark(
        name="set_inertias",
        method_name="set_inertias",
        input_generators={
            "warp": gen_inertias_warp,
            "torch_list": gen_inertias_torch_list,
            "torch_tensor": gen_inertias_torch_tensor,
        },
        category="body_props",
    ),
    # --- External Wrench ---
    MethodBenchmark(
        name="set_external_force_and_torque",
        method_name="set_external_force_and_torque",
        input_generators={
            "warp": gen_external_force_and_torque_warp,
            "torch_list": gen_external_force_and_torque_torch_list,
            "torch_tensor": gen_external_force_and_torque_torch_tensor,
        },
        category="wrench",
    ),
]


def run_benchmark(config: BenchmarkConfig):
    """Run all benchmarks."""
    results = []

    # Check if we should run all modes or specific ones
    modes_to_run = []
    if isinstance(config.mode, str):
        if config.mode == "all":
            # Will be populated dynamically based on available generators
            modes_to_run = None
        else:
            modes_to_run = [config.mode]
    elif isinstance(config.mode, list):
        modes_to_run = config.mode

    # Setup
    rigid_object, mock_view, _ = create_test_rigid_object(
        num_instances=config.num_instances,
        num_bodies=config.num_bodies,
        device=config.device,
    )

    print(f"Benchmarking RigidObject with {config.num_instances} instances, {config.num_bodies} bodies...")
    print(f"Device: {config.device}")
    print(f"Iterations: {config.num_iterations}, Warmup: {config.warmup_steps}")
    print(f"Modes: {modes_to_run if modes_to_run else 'All available'}")

    print(f"\nBenchmarking {len(BENCHMARKS)} methods...")
    for i, benchmark in enumerate(BENCHMARKS):
        method = getattr(rigid_object, benchmark.method_name, None)

        # Determine which modes to run for this benchmark
        available_modes = list(benchmark.input_generators.keys())
        current_modes = modes_to_run if modes_to_run is not None else available_modes

        # Filter modes that are available for this benchmark
        current_modes = [m for m in current_modes if m in available_modes]

        for mode in current_modes:
            generator = benchmark.input_generators[mode]
            print(f"[{i + 1}/{len(BENCHMARKS)}] [{mode.upper()}] {benchmark.name}...", end=" ", flush=True)

            result = benchmark_method(
                method=method,
                method_name=benchmark.name,
                generator=generator,
                config=config,
                dependencies=BENCHMARK_DEPENDENCIES,
            )
            result.mode = mode
            results.append(result)

            if result.skipped:
                print(f"SKIPPED ({result.skip_reason})")
            else:
                print(f"{result.mean_time_us:.2f} ± {result.std_time_us:.2f} µs")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RigidObject methods.")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--num_instances", type=int, default=4096, help="Number of instances")
    parser.add_argument("--num_bodies", type=int, default=12, help="Number of bodies")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--mode", type=str, default="all", help="Benchmark mode (all, warp, torch_list, torch_tensor)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON filename")
    parser.add_argument("--no_csv", action="store_true", help="Disable CSV output")

    args = parser.parse_args()

    config = BenchmarkConfig(
        num_iterations=args.num_iterations,
        warmup_steps=args.warmup_steps,
        num_instances=args.num_instances,
        num_bodies=args.num_bodies,
        device=args.device,
        mode=args.mode,
    )

    results = run_benchmark(config)

    hardware_info = get_hardware_info()
    print_hardware_info(hardware_info)
    print_results(results)

    if args.output:
        json_filename = args.output
    else:
        json_filename = get_default_output_filename("rigid_object_benchmark")

    export_results_json(results, config, hardware_info, json_filename)

    if not args.no_csv:
        csv_filename = json_filename.replace(".json", ".csv")
        from common.benchmark_io import export_results_csv

        export_results_csv(results, csv_filename)
