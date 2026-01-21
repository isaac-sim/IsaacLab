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
import contextlib
import numpy as np
import sys
import time
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
    BenchmarkResult,
    InputMode,
    MethodBenchmark,
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
# Method Benchmark Definitions
# =============================================================================

METHOD_BENCHMARKS = [
    # Root State Writers
    MethodBenchmark(
        name="write_root_link_pose_to_sim",
        method_name="write_root_link_pose_to_sim",
        input_generator_warp=gen_root_link_pose_warp,
        input_generator_torch_list=gen_root_link_pose_torch_list,
        input_generator_torch_tensor=gen_root_link_pose_torch_tensor,
        category="root_state",
    ),
    MethodBenchmark(
        name="write_root_com_pose_to_sim",
        method_name="write_root_com_pose_to_sim",
        input_generator_warp=gen_root_com_pose_warp,
        input_generator_torch_list=gen_root_com_pose_torch_list,
        input_generator_torch_tensor=gen_root_com_pose_torch_tensor,
        category="root_state",
    ),
    MethodBenchmark(
        name="write_root_link_velocity_to_sim",
        method_name="write_root_link_velocity_to_sim",
        input_generator_warp=gen_root_link_velocity_warp,
        input_generator_torch_list=gen_root_link_velocity_torch_list,
        input_generator_torch_tensor=gen_root_link_velocity_torch_tensor,
        category="root_state",
    ),
    MethodBenchmark(
        name="write_root_com_velocity_to_sim",
        method_name="write_root_com_velocity_to_sim",
        input_generator_warp=gen_root_com_velocity_warp,
        input_generator_torch_list=gen_root_com_velocity_torch_list,
        input_generator_torch_tensor=gen_root_com_velocity_torch_tensor,
        category="root_state",
    ),
    # Root State Writers (Deprecated)
    MethodBenchmark(
        name="write_root_state_to_sim (deprecated)",
        method_name="write_root_state_to_sim",
        input_generator_warp=gen_root_state_warp,
        input_generator_torch_list=gen_root_state_torch_list,
        input_generator_torch_tensor=gen_root_state_torch_tensor,
        category="root_state_deprecated",
    ),
    MethodBenchmark(
        name="write_root_com_state_to_sim (deprecated)",
        method_name="write_root_com_state_to_sim",
        input_generator_warp=gen_root_com_state_warp,
        input_generator_torch_list=gen_root_com_state_torch_list,
        input_generator_torch_tensor=gen_root_com_state_torch_tensor,
        category="root_state_deprecated",
    ),
    MethodBenchmark(
        name="write_root_link_state_to_sim (deprecated)",
        method_name="write_root_link_state_to_sim",
        input_generator_warp=gen_root_link_state_warp,
        input_generator_torch_list=gen_root_link_state_torch_list,
        input_generator_torch_tensor=gen_root_link_state_torch_tensor,
        category="root_state_deprecated",
    ),
    # Body Properties
    MethodBenchmark(
        name="set_masses",
        method_name="set_masses",
        input_generator_warp=gen_masses_warp,
        input_generator_torch_list=gen_masses_torch_list,
        input_generator_torch_tensor=gen_masses_torch_tensor,
        category="body_properties",
    ),
    MethodBenchmark(
        name="set_coms",
        method_name="set_coms",
        input_generator_warp=gen_coms_warp,
        input_generator_torch_list=gen_coms_torch_list,
        input_generator_torch_tensor=gen_coms_torch_tensor,
        category="body_properties",
    ),
    MethodBenchmark(
        name="set_inertias",
        method_name="set_inertias",
        input_generator_warp=gen_inertias_warp,
        input_generator_torch_list=gen_inertias_torch_list,
        input_generator_torch_tensor=gen_inertias_torch_tensor,
        category="body_properties",
    ),
    MethodBenchmark(
        name="set_external_force_and_torque",
        method_name="set_external_force_and_torque",
        input_generator_warp=gen_external_force_and_torque_warp,
        input_generator_torch_list=gen_external_force_and_torque_torch_list,
        input_generator_torch_tensor=gen_external_force_and_torque_torch_tensor,
        category="body_properties",
    ),
]


def benchmark_method(
    rigid_object: RigidObject,
    method_benchmark: MethodBenchmark,
    mode: InputMode,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark a single method of RigidObject.

    Args:
        rigid_object: The RigidObject instance.
        method_benchmark: The method benchmark definition.
        mode: Input mode (WARP or TORCH).
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    method_name = method_benchmark.method_name

    # Check if method exists
    if not hasattr(rigid_object, method_name):
        return BenchmarkResult(
            name=method_benchmark.name,
            mode=mode,
            mean_time_us=0.0,
            std_time_us=0.0,
            num_iterations=0,
            skipped=True,
            skip_reason="Method not found",
        )

    method = getattr(rigid_object, method_name)
    if mode == InputMode.WARP:
        input_generator = method_benchmark.input_generator_warp
    elif mode == InputMode.TORCH_TENSOR:
        # Use tensor generator if available, otherwise fall back to list generator
        input_generator = method_benchmark.input_generator_torch_tensor or method_benchmark.input_generator_torch_list
    else:  # TORCH_LIST
        input_generator = method_benchmark.input_generator_torch_list

    # Try to call the method once to check for errors
    try:
        inputs = input_generator(config)
        method(**inputs)
    except NotImplementedError as e:
        return BenchmarkResult(
            name=method_benchmark.name,
            mode=mode,
            mean_time_us=0.0,
            std_time_us=0.0,
            num_iterations=0,
            skipped=True,
            skip_reason=f"NotImplementedError: {e}",
        )
    except Exception as e:
        return BenchmarkResult(
            name=method_benchmark.name,
            mode=mode,
            mean_time_us=0.0,
            std_time_us=0.0,
            num_iterations=0,
            skipped=True,
            skip_reason=f"Error: {type(e).__name__}: {e}",
        )

    # Warmup phase
    for _ in range(config.warmup_steps):
        inputs = input_generator(config)
        with contextlib.suppress(Exception):
            method(**inputs)
        if config.device.startswith("cuda"):
            wp.synchronize()

    # Timing phase
    times = []
    for _ in range(config.num_iterations):
        inputs = input_generator(config)

        # Sync before timing
        if config.device.startswith("cuda"):
            wp.synchronize()

        start_time = time.perf_counter()
        try:
            method(**inputs)
        except Exception:
            continue

        # Sync after to ensure kernel completion
        if config.device.startswith("cuda"):
            wp.synchronize()

        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1e6)  # Convert to microseconds

    if not times:
        return BenchmarkResult(
            name=method_benchmark.name,
            mode=mode,
            mean_time_us=0.0,
            std_time_us=0.0,
            num_iterations=0,
            skipped=True,
            skip_reason="No successful iterations",
        )

    return BenchmarkResult(
        name=method_benchmark.name,
        mode=mode,
        mean_time_us=float(np.mean(times)),
        std_time_us=float(np.std(times)),
        num_iterations=len(times),
    )


def run_benchmarks(config: BenchmarkConfig) -> tuple[list[BenchmarkResult], dict]:
    """Run all benchmarks for RigidObject.

    Args:
        config: Benchmark configuration.

    Returns:
        Tuple of (List of BenchmarkResults, hardware_info dict).
    """
    results = []

    # Gather and print hardware info
    hardware_info = get_hardware_info()
    print_hardware_info(hardware_info)

    # Create rigid object
    rigid_object, mock_view, _ = create_test_rigid_object(
        num_instances=config.num_instances,
        num_bodies=config.num_bodies,
        device=config.device,
    )

    # Determine modes to run
    modes = []
    if config.mode in ("all", "warp"):
        modes.append(InputMode.WARP)
    if config.mode in ("all", "torch", "torch_list"):
        modes.append(InputMode.TORCH_LIST)
    if config.mode in ("all", "torch", "torch_tensor"):
        modes.append(InputMode.TORCH_TENSOR)

    print(f"\nBenchmarking {len(METHOD_BENCHMARKS)} methods...")
    print(f"Config: {config.num_iterations} iterations, {config.warmup_steps} warmup steps")
    print(f"        {config.num_instances} instances, {config.num_bodies} bodies")
    print(f"Modes:  {', '.join(m.value for m in modes)}")
    print("-" * 100)

    for i, method_benchmark in enumerate(METHOD_BENCHMARKS):
        for mode in modes:
            mode_str = f"[{mode.value.upper():5}]"
            print(f"[{i + 1}/{len(METHOD_BENCHMARKS)}] {mode_str} {method_benchmark.name}...", end=" ", flush=True)

            result = benchmark_method(rigid_object, method_benchmark, mode, config)
            results.append(result)

            if result.skipped:
                print(f"SKIPPED ({result.skip_reason})")
            else:
                print(f"{result.mean_time_us:.2f} ± {result.std_time_us:.2f} µs")

    return results, hardware_info


def main():
    """Main entry point for the benchmarking script."""
    parser = argparse.ArgumentParser(
        description="Micro-benchmarking framework for RigidObject class.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10000,
        help="Number of iterations to run each method.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warmup steps before timing.",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=16384,
        help="Number of rigid object instances.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run benchmarks on.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["warp", "torch", "torch_list", "torch_tensor", "all"],
        default="all",
        help="Benchmark mode: 'warp', 'torch_list', 'torch_tensor', 'torch' (both torch modes), or 'all'.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file for benchmark results.",
    )
    parser.add_argument(
        "--no_json",
        action="store_true",
        help="Disable JSON output.",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        num_iterations=args.num_iterations,
        warmup_steps=args.warmup_steps,
        num_instances=args.num_instances,
        num_bodies=1,
        num_joints=0,
        device=args.device,
        mode=args.mode,
    )

    # Run benchmarks
    results, hardware_info = run_benchmarks(config)

    # Print results
    print_results(results)

    # Export to JSON
    if not args.no_json:
        output_filename = args.output if args.output else get_default_output_filename("rigid_object_benchmark")
        export_results_json(results, config, hardware_info, output_filename)


if __name__ == "__main__":
    main()
