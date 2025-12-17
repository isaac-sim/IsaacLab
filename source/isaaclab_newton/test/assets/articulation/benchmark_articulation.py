# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for Articulation class.

This module provides a benchmarking framework to measure the performance of setter and writer
methods in the Articulation class. Each method is benchmarked under two scenarios:

1. **Best Case (Warp)**: Inputs are Warp arrays with masks - this is the optimal path that
   avoids any data conversion overhead.

2. **Worst Case (Torch)**: Inputs are PyTorch tensors with indices - this path requires
   conversion from Torch to Warp and from indices to masks.

Usage:
    python benchmark_articulation.py [--num_iterations N] [--warmup_steps W] [--num_instances I] [--num_bodies B] [--num_joints J]

Example:
    python benchmark_articulation.py --num_iterations 1000 --warmup_steps 10
    python benchmark_articulation.py --mode warp  # Only run Warp benchmarks
    python benchmark_articulation.py --mode torch  # Only run Torch benchmarks
"""

from __future__ import annotations

import argparse
import contextlib
import numpy as np
import time
import torch
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
from unittest.mock import MagicMock, patch

import warp as wp
from isaaclab_newton.assets.articulation.articulation_data import ArticulationData
from isaaclab_newton.assets.articulation.articulation import Articulation
from isaaclab_newton.kernels import vec13f
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg

# Import mock classes from shared module
from mock_interface import MockNewtonArticulationView, MockNewtonModel

# Initialize Warp
wp.init()

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class InputMode(Enum):
    """Input mode for benchmarks."""

    WARP = "warp"
    TORCH = "torch"


def get_git_info() -> dict:
    """Get git repository information.

    Returns:
        Dictionary containing git commit hash, branch, and other info.
    """
    import os
    import subprocess

    git_info = {
        "commit_hash": "Unknown",
        "commit_hash_short": "Unknown",
        "branch": "Unknown",
        "commit_date": "Unknown",
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["commit_hash"] = result.stdout.strip()
            git_info["commit_hash_short"] = result.stdout.strip()[:8]

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["commit_date"] = result.stdout.strip()

    except Exception:
        pass

    return git_info


def get_hardware_info() -> dict:
    """Gather hardware information for the benchmark.

    Returns:
        Dictionary containing CPU, GPU, and memory information.
    """
    import os
    import platform

    hardware_info = {
        "cpu": {
            "name": platform.processor() or "Unknown",
            "physical_cores": os.cpu_count(),
        },
        "gpu": {},
        "memory": {},
        "system": {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
        },
    }

    # Try to get more detailed CPU info on Linux
    with contextlib.suppress(Exception):
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    hardware_info["cpu"]["name"] = line.split(":")[1].strip()
                    break

    # Memory info
    try:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
            for line in meminfo.split("\n"):
                if "MemTotal" in line:
                    mem_kb = int(line.split()[1])
                    hardware_info["memory"]["total_gb"] = round(mem_kb / (1024 * 1024), 2)
                    break
    except Exception:
        try:
            import psutil

            mem = psutil.virtual_memory()
            hardware_info["memory"]["total_gb"] = round(mem.total / (1024**3), 2)
        except ImportError:
            hardware_info["memory"]["total_gb"] = "Unknown"

    # GPU info using PyTorch
    if torch.cuda.is_available():
        hardware_info["gpu"]["available"] = True
        hardware_info["gpu"]["count"] = torch.cuda.device_count()
        hardware_info["gpu"]["devices"] = []

        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            hardware_info["gpu"]["devices"].append({
                "index": i,
                "name": gpu_props.name,
                "total_memory_gb": round(gpu_props.total_memory / (1024**3), 2),
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                "multi_processor_count": gpu_props.multi_processor_count,
            })

        current_device = torch.cuda.current_device()
        hardware_info["gpu"]["current_device"] = current_device
        hardware_info["gpu"]["current_device_name"] = torch.cuda.get_device_name(current_device)
    else:
        hardware_info["gpu"]["available"] = False

    hardware_info["gpu"]["pytorch_version"] = torch.__version__
    if torch.cuda.is_available():
        try:
            import torch.version as torch_version

            cuda_version = getattr(torch_version, "cuda", None)
            hardware_info["gpu"]["cuda_version"] = cuda_version if cuda_version else "Unknown"
        except Exception:
            hardware_info["gpu"]["cuda_version"] = "Unknown"
    else:
        hardware_info["gpu"]["cuda_version"] = "N/A"

    try:
        warp_version = getattr(wp.config, "version", None)
        hardware_info["warp"] = {"version": warp_version if warp_version else "Unknown"}
    except Exception:
        hardware_info["warp"] = {"version": "Unknown"}

    return hardware_info


def print_hardware_info(hardware_info: dict):
    """Print hardware information to console."""
    print("\n" + "=" * 80)
    print("HARDWARE INFORMATION")
    print("=" * 80)

    sys_info = hardware_info.get("system", {})
    print(f"\nSystem: {sys_info.get('platform', 'Unknown')} {sys_info.get('platform_release', '')}")
    print(f"Python: {sys_info.get('python_version', 'Unknown')}")

    cpu_info = hardware_info.get("cpu", {})
    print(f"\nCPU: {cpu_info.get('name', 'Unknown')}")
    print(f"     Cores: {cpu_info.get('physical_cores', 'Unknown')}")

    mem_info = hardware_info.get("memory", {})
    print(f"\nMemory: {mem_info.get('total_gb', 'Unknown')} GB")

    gpu_info = hardware_info.get("gpu", {})
    if gpu_info.get("available", False):
        print(f"\nGPU: {gpu_info.get('current_device_name', 'Unknown')}")
        for device in gpu_info.get("devices", []):
            print(f"     [{device['index']}] {device['name']}")
            print(f"         Memory: {device['total_memory_gb']} GB")
            print(f"         Compute Capability: {device['compute_capability']}")
            print(f"         SM Count: {device['multi_processor_count']}")
        print(f"\nPyTorch: {gpu_info.get('pytorch_version', 'Unknown')}")
        print(f"CUDA: {gpu_info.get('cuda_version', 'Unknown')}")
    else:
        print("\nGPU: Not available")

    warp_info = hardware_info.get("warp", {})
    print(f"Warp: {warp_info.get('version', 'Unknown')}")

    repo_info = get_git_info()
    print("\nRepository:")
    print(f"     Commit: {repo_info.get('commit_hash_short', 'Unknown')}")
    print(f"     Branch: {repo_info.get('branch', 'Unknown')}")
    print(f"     Date:   {repo_info.get('commit_date', 'Unknown')}")
    print("=" * 80)


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmarking framework."""

    num_iterations: int = 1000
    """Number of iterations to run each function."""

    warmup_steps: int = 10
    """Number of warmup steps before timing."""

    num_instances: int = 4096
    """Number of articulation instances."""

    num_bodies: int = 12
    """Number of bodies per articulation."""

    num_joints: int = 11
    """Number of joints per articulation."""

    device: str = "cuda:0"
    """Device to run benchmarks on."""

    mode: str = "both"
    """Benchmark mode: 'warp', 'torch', or 'both'."""


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    """Name of the benchmarked method."""

    mode: InputMode
    """Input mode used (WARP or TORCH)."""

    mean_time_us: float
    """Mean execution time in microseconds."""

    std_time_us: float
    """Standard deviation of execution time in microseconds."""

    num_iterations: int
    """Number of iterations run."""

    skipped: bool = False
    """Whether the benchmark was skipped."""

    skip_reason: str = ""
    """Reason for skipping the benchmark."""


@dataclass
class MethodBenchmark:
    """Definition of a method to benchmark."""

    name: str
    """Name of the method."""

    method_name: str
    """Actual method name on the Articulation class."""

    input_generator_warp: Callable
    """Function to generate Warp inputs."""

    input_generator_torch: Callable
    """Function to generate Torch inputs."""

    category: str = "general"
    """Category of the method (e.g., 'root_state', 'joint_state', 'joint_params')."""


def create_test_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    device: str = "cuda:0",
) -> tuple[Articulation, MockNewtonArticulationView, MagicMock]:
    """Create a test Articulation instance with mocked dependencies."""
    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]

    articulation = object.__new__(Articulation)

    articulation.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        soft_joint_pos_limit_factor=1.0,
        actuators={},
    )

    mock_view = MockNewtonArticulationView(
        num_instances=num_instances,
        num_bodies=num_bodies,
        num_joints=num_joints,
        device=device,
        is_fixed_base=False,
        joint_names=joint_names,
        body_names=body_names,
    )
    mock_view.set_mock_data()

    object.__setattr__(articulation, "_root_view", mock_view)
    object.__setattr__(articulation, "_device", device)

    mock_newton_manager = MagicMock()
    mock_model = MockNewtonModel()
    mock_state = MagicMock()
    mock_control = MagicMock()
    mock_newton_manager.get_model.return_value = mock_model
    mock_newton_manager.get_state_0.return_value = mock_state
    mock_newton_manager.get_control.return_value = mock_control
    mock_newton_manager.get_dt.return_value = 0.01

    with patch("isaaclab_newton.assets.articulation.articulation_data.NewtonManager", mock_newton_manager):
        data = ArticulationData(mock_view, device)
        object.__setattr__(articulation, "_data", data)

    return articulation, mock_view, mock_newton_manager


# =============================================================================
# Input Generators
# =============================================================================


def make_warp_env_mask(num_instances: int, device: str) -> wp.array:
    """Create an all-true environment mask."""
    return wp.ones((num_instances,), dtype=wp.bool, device=device)


def make_warp_joint_mask(num_joints: int, device: str) -> wp.array:
    """Create an all-true joint mask."""
    return wp.ones((num_joints,), dtype=wp.bool, device=device)


def make_warp_body_mask(num_bodies: int, device: str) -> wp.array:
    """Create an all-true body mask."""
    return wp.ones((num_bodies,), dtype=wp.bool, device=device)


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


def gen_root_link_pose_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_root_link_pose_to_sim."""
    return {
        "pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
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


def gen_root_com_pose_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_root_com_pose_to_sim."""
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
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


def gen_root_link_velocity_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_root_link_velocity_to_sim."""
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
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


def gen_root_com_velocity_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_root_com_velocity_to_sim."""
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
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


def gen_root_state_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_root_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
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


def gen_root_com_state_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_root_com_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
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


def gen_root_link_state_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_root_link_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


# --- Joint State ---
def gen_joint_state_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_joint_state_to_sim."""
    return {
        "position": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
            dtype=wp.float32,
        ),
        "velocity": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
            dtype=wp.float32,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "joint_mask": make_warp_joint_mask(config.num_joints, config.device),
    }


def gen_joint_state_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_joint_state_to_sim."""
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


# --- Joint Position ---
def gen_joint_position_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_joint_position_to_sim."""
    return {
        "position": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
            dtype=wp.float32,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "joint_mask": make_warp_joint_mask(config.num_joints, config.device),
    }


def gen_joint_position_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_joint_position_to_sim."""
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


# --- Joint Velocity ---
def gen_joint_velocity_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_joint_velocity_to_sim."""
    return {
        "velocity": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
            dtype=wp.float32,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "joint_mask": make_warp_joint_mask(config.num_joints, config.device),
    }


def gen_joint_velocity_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_joint_velocity_to_sim."""
    return {
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


# --- Joint Stiffness ---
def gen_joint_stiffness_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_joint_stiffness_to_sim."""
    return {
        "stiffness": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
            dtype=wp.float32,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "joint_mask": make_warp_joint_mask(config.num_joints, config.device),
    }


def gen_joint_stiffness_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_joint_stiffness_to_sim."""
    return {
        "stiffness": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


# --- Joint Damping ---
def gen_joint_damping_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_joint_damping_to_sim."""
    return {
        "damping": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
            dtype=wp.float32,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "joint_mask": make_warp_joint_mask(config.num_joints, config.device),
    }


def gen_joint_damping_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_joint_damping_to_sim."""
    return {
        "damping": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


# --- Joint Position Limit ---
def gen_joint_position_limit_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_joint_position_limit_to_sim."""
    return {
        "lower_limits": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * -3.14,
            dtype=wp.float32,
        ),
        "upper_limits": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 3.14,
            dtype=wp.float32,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "joint_mask": make_warp_joint_mask(config.num_joints, config.device),
    }


def gen_joint_position_limit_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_joint_position_limit_to_sim."""
    return {
        "lower_limits": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32)
        * -3.14,
        "upper_limits": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32)
        * 3.14,
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


# --- Joint Velocity Limit ---
def gen_joint_velocity_limit_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_joint_velocity_limit_to_sim."""
    return {
        "limits": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 10.0,
            dtype=wp.float32,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "joint_mask": make_warp_joint_mask(config.num_joints, config.device),
    }


def gen_joint_velocity_limit_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_joint_velocity_limit_to_sim."""
    return {
        "limits": torch.rand(
            config.num_instances, config.num_joints, device=config.device, dtype=torch.float32
        )
        * 10.0,
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


# --- Joint Effort Limit ---
def gen_joint_effort_limit_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_joint_effort_limit_to_sim."""
    return {
        "limits": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 100.0,
            dtype=wp.float32,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "joint_mask": make_warp_joint_mask(config.num_joints, config.device),
    }


def gen_joint_effort_limit_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_joint_effort_limit_to_sim."""
    return {
        "limits": torch.rand(
            config.num_instances, config.num_joints, device=config.device, dtype=torch.float32
        )
        * 100.0,
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


# --- Joint Armature ---
def gen_joint_armature_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_joint_armature_to_sim."""
    return {
        "armature": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.1,
            dtype=wp.float32,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "joint_mask": make_warp_joint_mask(config.num_joints, config.device),
    }


def gen_joint_armature_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_joint_armature_to_sim."""
    return {
        "armature": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32)
        * 0.1,
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


# --- Joint Friction Coefficient ---
def gen_joint_friction_coefficient_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for write_joint_friction_coefficient_to_sim."""
    return {
        "joint_friction_coeff": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.5,
            dtype=wp.float32,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "joint_mask": make_warp_joint_mask(config.num_joints, config.device),
    }


def gen_joint_friction_coefficient_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for write_joint_friction_coefficient_to_sim."""
    return {
        "joint_friction_coeff": torch.rand(
            config.num_instances, config.num_joints, device=config.device, dtype=torch.float32
        )
        * 0.5,
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


# --- Set Joint Position Target ---
def gen_set_joint_position_target_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for set_joint_position_target."""
    return {
        "target": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
            dtype=wp.float32,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "joint_mask": make_warp_joint_mask(config.num_joints, config.device),
    }


def gen_set_joint_position_target_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for set_joint_position_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


# --- Set Joint Velocity Target ---
def gen_set_joint_velocity_target_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for set_joint_velocity_target."""
    return {
        "target": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
            dtype=wp.float32,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "joint_mask": make_warp_joint_mask(config.num_joints, config.device),
    }


def gen_set_joint_velocity_target_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for set_joint_velocity_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


# --- Set Joint Effort Target ---
def gen_set_joint_effort_target_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for set_joint_effort_target."""
    return {
        "target": wp.from_torch(
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
            dtype=wp.float32,
        ),
        "env_mask": make_warp_env_mask(config.num_instances, config.device),
        "joint_mask": make_warp_joint_mask(config.num_joints, config.device),
    }


def gen_set_joint_effort_target_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for set_joint_effort_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
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
        input_generator_torch=gen_root_link_pose_torch,
        category="root_state",
    ),
    MethodBenchmark(
        name="write_root_com_pose_to_sim",
        method_name="write_root_com_pose_to_sim",
        input_generator_warp=gen_root_com_pose_warp,
        input_generator_torch=gen_root_com_pose_torch,
        category="root_state",
    ),
    MethodBenchmark(
        name="write_root_link_velocity_to_sim",
        method_name="write_root_link_velocity_to_sim",
        input_generator_warp=gen_root_link_velocity_warp,
        input_generator_torch=gen_root_link_velocity_torch,
        category="root_state",
    ),
    MethodBenchmark(
        name="write_root_com_velocity_to_sim",
        method_name="write_root_com_velocity_to_sim",
        input_generator_warp=gen_root_com_velocity_warp,
        input_generator_torch=gen_root_com_velocity_torch,
        category="root_state",
    ),
    # Root State Writers (Deprecated)
    MethodBenchmark(
        name="write_root_state_to_sim (deprecated)",
        method_name="write_root_state_to_sim",
        input_generator_warp=gen_root_state_warp,
        input_generator_torch=gen_root_state_torch,
        category="root_state_deprecated",
    ),
    MethodBenchmark(
        name="write_root_com_state_to_sim (deprecated)",
        method_name="write_root_com_state_to_sim",
        input_generator_warp=gen_root_com_state_warp,
        input_generator_torch=gen_root_com_state_torch,
        category="root_state_deprecated",
    ),
    MethodBenchmark(
        name="write_root_link_state_to_sim (deprecated)",
        method_name="write_root_link_state_to_sim",
        input_generator_warp=gen_root_link_state_warp,
        input_generator_torch=gen_root_link_state_torch,
        category="root_state_deprecated",
    ),
    # Joint State Writers
    MethodBenchmark(
        name="write_joint_state_to_sim",
        method_name="write_joint_state_to_sim",
        input_generator_warp=gen_joint_state_warp,
        input_generator_torch=gen_joint_state_torch,
        category="joint_state",
    ),
    MethodBenchmark(
        name="write_joint_position_to_sim",
        method_name="write_joint_position_to_sim",
        input_generator_warp=gen_joint_position_warp,
        input_generator_torch=gen_joint_position_torch,
        category="joint_state",
    ),
    MethodBenchmark(
        name="write_joint_velocity_to_sim",
        method_name="write_joint_velocity_to_sim",
        input_generator_warp=gen_joint_velocity_warp,
        input_generator_torch=gen_joint_velocity_torch,
        category="joint_state",
    ),
    # Joint Parameter Writers
    MethodBenchmark(
        name="write_joint_stiffness_to_sim",
        method_name="write_joint_stiffness_to_sim",
        input_generator_warp=gen_joint_stiffness_warp,
        input_generator_torch=gen_joint_stiffness_torch,
        category="joint_params",
    ),
    MethodBenchmark(
        name="write_joint_damping_to_sim",
        method_name="write_joint_damping_to_sim",
        input_generator_warp=gen_joint_damping_warp,
        input_generator_torch=gen_joint_damping_torch,
        category="joint_params",
    ),
    MethodBenchmark(
        name="write_joint_position_limit_to_sim",
        method_name="write_joint_position_limit_to_sim",
        input_generator_warp=gen_joint_position_limit_warp,
        input_generator_torch=gen_joint_position_limit_torch,
        category="joint_params",
    ),
    MethodBenchmark(
        name="write_joint_velocity_limit_to_sim",
        method_name="write_joint_velocity_limit_to_sim",
        input_generator_warp=gen_joint_velocity_limit_warp,
        input_generator_torch=gen_joint_velocity_limit_torch,
        category="joint_params",
    ),
    MethodBenchmark(
        name="write_joint_effort_limit_to_sim",
        method_name="write_joint_effort_limit_to_sim",
        input_generator_warp=gen_joint_effort_limit_warp,
        input_generator_torch=gen_joint_effort_limit_torch,
        category="joint_params",
    ),
    MethodBenchmark(
        name="write_joint_armature_to_sim",
        method_name="write_joint_armature_to_sim",
        input_generator_warp=gen_joint_armature_warp,
        input_generator_torch=gen_joint_armature_torch,
        category="joint_params",
    ),
    MethodBenchmark(
        name="write_joint_friction_coefficient_to_sim",
        method_name="write_joint_friction_coefficient_to_sim",
        input_generator_warp=gen_joint_friction_coefficient_warp,
        input_generator_torch=gen_joint_friction_coefficient_torch,
        category="joint_params",
    ),
    # Target Setters
    MethodBenchmark(
        name="set_joint_position_target",
        method_name="set_joint_position_target",
        input_generator_warp=gen_set_joint_position_target_warp,
        input_generator_torch=gen_set_joint_position_target_torch,
        category="targets",
    ),
    MethodBenchmark(
        name="set_joint_velocity_target",
        method_name="set_joint_velocity_target",
        input_generator_warp=gen_set_joint_velocity_target_warp,
        input_generator_torch=gen_set_joint_velocity_target_torch,
        category="targets",
    ),
    MethodBenchmark(
        name="set_joint_effort_target",
        method_name="set_joint_effort_target",
        input_generator_warp=gen_set_joint_effort_target_warp,
        input_generator_torch=gen_set_joint_effort_target_torch,
        category="targets",
    ),
]


def benchmark_method(
    articulation: Articulation,
    method_benchmark: MethodBenchmark,
    mode: InputMode,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark a single method of Articulation.

    Args:
        articulation: The Articulation instance.
        method_benchmark: The method benchmark definition.
        mode: Input mode (WARP or TORCH).
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    method_name = method_benchmark.method_name

    # Check if method exists
    if not hasattr(articulation, method_name):
        return BenchmarkResult(
            name=method_benchmark.name,
            mode=mode,
            mean_time_us=0.0,
            std_time_us=0.0,
            num_iterations=0,
            skipped=True,
            skip_reason="Method not found",
        )

    method = getattr(articulation, method_name)
    input_generator = (
        method_benchmark.input_generator_warp if mode == InputMode.WARP else method_benchmark.input_generator_torch
    )

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
        try:
            method(**inputs)
        except Exception:
            pass
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
    """Run all benchmarks for Articulation.

    Args:
        config: Benchmark configuration.

    Returns:
        Tuple of (List of BenchmarkResults, hardware_info dict).
    """
    results = []

    # Gather and print hardware info
    hardware_info = get_hardware_info()
    print_hardware_info(hardware_info)

    # Create articulation
    articulation, mock_view, _ = create_test_articulation(
        num_instances=config.num_instances,
        num_joints=config.num_joints,
        num_bodies=config.num_bodies,
        device=config.device,
    )

    # Determine modes to run
    modes = []
    if config.mode in ("both", "warp"):
        modes.append(InputMode.WARP)
    if config.mode in ("both", "torch"):
        modes.append(InputMode.TORCH)

    print(f"\nBenchmarking {len(METHOD_BENCHMARKS)} methods...")
    print(f"Config: {config.num_iterations} iterations, {config.warmup_steps} warmup steps")
    print(f"        {config.num_instances} instances, {config.num_bodies} bodies, {config.num_joints} joints")
    print(f"Modes:  {', '.join(m.value for m in modes)}")
    print("-" * 100)

    for i, method_benchmark in enumerate(METHOD_BENCHMARKS):
        for mode in modes:
            mode_str = f"[{mode.value.upper():5}]"
            print(f"[{i + 1}/{len(METHOD_BENCHMARKS)}] {mode_str} {method_benchmark.name}...", end=" ", flush=True)

            result = benchmark_method(articulation, method_benchmark, mode, config)
            results.append(result)

            if result.skipped:
                print(f"SKIPPED ({result.skip_reason})")
            else:
                print(f"{result.mean_time_us:.2f} ± {result.std_time_us:.2f} µs")

    return results, hardware_info


def print_results(results: list[BenchmarkResult]):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)

    # Separate by mode
    warp_results = [r for r in results if r.mode == InputMode.WARP and not r.skipped]
    torch_results = [r for r in results if r.mode == InputMode.TORCH and not r.skipped]
    skipped = [r for r in results if r.skipped]

    # Print comparison table
    if warp_results and torch_results:
        print("\n" + "-" * 100)
        print("COMPARISON: Warp (Best Case) vs Torch (Worst Case)")
        print("-" * 100)
        print(
            f"{'Method Name':<40} {'Warp (µs)':<15} {'Torch (µs)':<15} {'Overhead':<12} {'Slowdown':<10}"
        )
        print("-" * 100)

        warp_by_name = {r.name: r for r in warp_results}
        torch_by_name = {r.name: r for r in torch_results}

        for name in warp_by_name:
            if name in torch_by_name:
                warp_time = warp_by_name[name].mean_time_us
                torch_time = torch_by_name[name].mean_time_us
                overhead = torch_time - warp_time
                slowdown = torch_time / warp_time if warp_time > 0 else float("inf")
                print(
                    f"{name:<40} {warp_time:>12.2f}   {torch_time:>12.2f}   {overhead:>+9.2f}   {slowdown:>7.2f}x"
                )

    # Print individual results
    for mode_name, mode_results in [("WARP (Best Case)", warp_results), ("TORCH (Worst Case)", torch_results)]:
        if mode_results:
            print("\n" + "-" * 100)
            print(f"{mode_name}")
            print("-" * 100)

            # Sort by mean time (descending)
            mode_results_sorted = sorted(mode_results, key=lambda x: x.mean_time_us, reverse=True)

            print(f"{'Method Name':<45} {'Mean (µs)':<15} {'Std (µs)':<15} {'Iterations':<12}")
            print("-" * 87)

            for result in mode_results_sorted:
                print(
                    f"{result.name:<45} {result.mean_time_us:>12.2f}   {result.std_time_us:>12.2f}  "
                    f" {result.num_iterations:>10}"
                )

            # Summary stats
            mean_times = [r.mean_time_us for r in mode_results_sorted]
            print("-" * 87)
            print(f"  Fastest: {min(mean_times):.2f} µs ({mode_results_sorted[-1].name})")
            print(f"  Slowest: {max(mean_times):.2f} µs ({mode_results_sorted[0].name})")
            print(f"  Average: {np.mean(mean_times):.2f} µs")

    # Print skipped
    if skipped:
        print(f"\nSkipped Methods ({len(skipped)}):")
        for result in skipped:
            print(f"  - {result.name} [{result.mode.value}]: {result.skip_reason}")


def export_results_json(results: list[BenchmarkResult], config: BenchmarkConfig, hardware_info: dict, filename: str):
    """Export benchmark results to a JSON file."""
    import json
    from datetime import datetime

    completed = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]

    git_info = get_git_info()

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "repository": git_info,
            "config": {
                "num_iterations": config.num_iterations,
                "warmup_steps": config.warmup_steps,
                "num_instances": config.num_instances,
                "num_bodies": config.num_bodies,
                "num_joints": config.num_joints,
                "device": config.device,
                "mode": config.mode,
            },
            "hardware": hardware_info,
            "total_benchmarks": len(results),
            "completed_benchmarks": len(completed),
            "skipped_benchmarks": len(skipped),
        },
        "results": {
            "warp": [],
            "torch": [],
        },
        "comparison": [],
        "skipped": [],
    }

    # Add individual results
    for result in completed:
        result_entry = {
            "name": result.name,
            "mean_time_us": result.mean_time_us,
            "std_time_us": result.std_time_us,
            "num_iterations": result.num_iterations,
        }
        if result.mode == InputMode.WARP:
            output["results"]["warp"].append(result_entry)
        else:
            output["results"]["torch"].append(result_entry)

    # Add comparison data
    warp_by_name = {r.name: r for r in completed if r.mode == InputMode.WARP}
    torch_by_name = {r.name: r for r in completed if r.mode == InputMode.TORCH}

    for name in warp_by_name:
        if name in torch_by_name:
            warp_time = warp_by_name[name].mean_time_us
            torch_time = torch_by_name[name].mean_time_us
            output["comparison"].append({
                "name": name,
                "warp_time_us": warp_time,
                "torch_time_us": torch_time,
                "overhead_us": torch_time - warp_time,
                "slowdown_factor": torch_time / warp_time if warp_time > 0 else None,
            })

    # Add skipped
    for result in skipped:
        output["skipped"].append({
            "name": result.name,
            "mode": result.mode.value,
            "reason": result.skip_reason,
        })

    with open(filename, "w") as jsonfile:
        json.dump(output, jsonfile, indent=2)

    print(f"\nResults exported to {filename}")


def get_default_output_filename() -> str:
    """Generate default output filename with current date and time."""
    from datetime import datetime

    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"articulation_benchmark_{datetime_str}.json"


def main():
    """Main entry point for the benchmarking script."""
    parser = argparse.ArgumentParser(
        description="Micro-benchmarking framework for Articulation class.",
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
        help="Number of articulation instances.",
    )
    parser.add_argument(
        "--num_bodies",
        type=int,
        default=12,
        help="Number of bodies per articulation.",
    )
    parser.add_argument(
        "--num_joints",
        type=int,
        default=11,
        help="Number of joints per articulation.",
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
        choices=["warp", "torch", "both"],
        default="both",
        help="Benchmark mode: 'warp' (best case), 'torch' (worst case), or 'both'.",
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
        num_bodies=args.num_bodies,
        num_joints=args.num_joints,
        device=args.device,
        mode=args.mode,
    )

    # Run benchmarks
    results, hardware_info = run_benchmarks(config)

    # Print results
    print_results(results)

    # Export to JSON
    if not args.no_json:
        output_filename = args.output if args.output else get_default_output_filename()
        export_results_json(results, config, hardware_info, output_filename)


if __name__ == "__main__":
    main()

