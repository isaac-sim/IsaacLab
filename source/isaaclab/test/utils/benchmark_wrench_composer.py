# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for WrenchComposer class.

This module provides a benchmarking framework to measure the performance of the WrenchComposer
class methods. Each method is benchmarked under two scenarios:

1. **Best Case (Warp)**: Inputs are Warp arrays with masks - this is the optimal path that
   avoids any data conversion overhead.

2. **Worst Case (Torch)**: Inputs are PyTorch tensors with indices - this path requires
   conversion from Torch to Warp and from indices to masks.

Usage:
    python benchmark_wrench_composer.py [--num_iterations N] [--warmup_steps W] [--num_envs E] [--num_bodies B]

Example:
    python benchmark_wrench_composer.py --num_iterations 1000 --warmup_steps 10
    python benchmark_wrench_composer.py --mode warp  # Only run Warp benchmarks
    python benchmark_wrench_composer.py --mode torch  # Only run Torch benchmarks
"""

from __future__ import annotations

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import argparse
import contextlib
import numpy as np
import time
import torch
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import warp as wp

from isaaclab.utils.wrench_composer import WrenchComposer

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class InputMode(Enum):
    """Input mode for benchmarks."""

    WARP = "warp"
    TORCH = "torch"


# =============================================================================
# Mock Asset Classes
# =============================================================================


class MockAssetData:
    """Mock data class that provides body com poses (position + quaternion as transform)."""

    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        device: str,
    ):
        """Initialize mock asset data with random COM poses."""
        # Generate random COM poses
        com_pos = torch.rand((num_envs, num_bodies, 3), dtype=torch.float32, device=device)
        com_quat = torch.rand((num_envs, num_bodies, 4), dtype=torch.float32, device=device)
        # Normalize quaternions
        com_quat = com_quat / com_quat.norm(dim=-1, keepdim=True)

        # Create transform tensor: (num_envs, num_bodies, 7) -> (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w)
        com_pose = torch.cat([com_pos, com_quat[..., 1:], com_quat[..., 0:1]], dim=-1)
        self.body_com_pose_w = wp.from_torch(com_pose, dtype=wp.transformf)


class MockRigidObject:
    """Mock RigidObject that provides the minimal interface required by WrenchComposer."""

    def __init__(
        self,
        num_envs: int,
        num_bodies: int,
        device: str,
    ):
        """Initialize mock rigid object."""
        self.num_instances = num_envs
        self.num_bodies = num_bodies
        self.device = device
        self.data = MockAssetData(num_envs, num_bodies, device)


# =============================================================================
# Utility Functions
# =============================================================================


def get_git_info() -> dict:
    """Get git repository information."""
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
    """Gather hardware information for the benchmark."""
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


# =============================================================================
# Benchmark Configuration and Results
# =============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmarking framework."""

    num_iterations: int = 10000
    """Number of iterations to run each function."""

    warmup_steps: int = 10
    """Number of warmup steps before timing."""

    num_envs: int = 4096
    """Number of environments."""

    num_bodies: int = 1
    """Number of bodies per environment."""

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
    """Actual method name on the WrenchComposer class."""

    input_generator_warp: Callable
    """Function to generate Warp inputs."""

    input_generator_torch: Callable
    """Function to generate Torch inputs."""

    category: str = "general"
    """Category of the method."""

    reset_before: bool = True
    """Whether to reset the composer before each call."""


# =============================================================================
# Input Generators for Warp (Best Case)
# =============================================================================


def make_warp_env_mask(num_envs: int, device: str) -> wp.array:
    """Create an all-true environment mask."""
    return wp.ones((num_envs,), dtype=wp.bool, device=device)


def make_warp_body_mask(num_bodies: int, device: str) -> wp.array:
    """Create an all-true body mask."""
    return wp.ones((num_bodies,), dtype=wp.bool, device=device)


def gen_add_forces_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for add_forces_and_torques (forces only)."""
    return {
        "forces": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "env_mask": make_warp_env_mask(config.num_envs, config.device),
        "body_mask": make_warp_body_mask(config.num_bodies, config.device),
    }


def gen_add_torques_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for add_forces_and_torques (torques only)."""
    return {
        "torques": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "env_mask": make_warp_env_mask(config.num_envs, config.device),
        "body_mask": make_warp_body_mask(config.num_bodies, config.device),
    }


def gen_add_forces_and_torques_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for add_forces_and_torques (forces + torques)."""
    return {
        "forces": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "torques": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "env_mask": make_warp_env_mask(config.num_envs, config.device),
        "body_mask": make_warp_body_mask(config.num_bodies, config.device),
    }


def gen_add_forces_at_positions_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for add_forces_and_torques (forces + positions)."""
    return {
        "forces": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "positions": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "env_mask": make_warp_env_mask(config.num_envs, config.device),
        "body_mask": make_warp_body_mask(config.num_bodies, config.device),
    }


def gen_add_forces_and_torques_at_positions_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for add_forces_and_torques (forces + torques + positions)."""
    return {
        "forces": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "torques": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "positions": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "env_mask": make_warp_env_mask(config.num_envs, config.device),
        "body_mask": make_warp_body_mask(config.num_bodies, config.device),
    }


def gen_add_forces_global_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for add_forces_and_torques (forces, global frame)."""
    return {
        "forces": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "env_mask": make_warp_env_mask(config.num_envs, config.device),
        "body_mask": make_warp_body_mask(config.num_bodies, config.device),
        "is_global": True,
    }


def gen_add_forces_at_positions_global_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for add_forces_and_torques (forces + positions, global frame)."""
    return {
        "forces": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "positions": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "env_mask": make_warp_env_mask(config.num_envs, config.device),
        "body_mask": make_warp_body_mask(config.num_bodies, config.device),
        "is_global": True,
    }


def gen_set_forces_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for set_forces_and_torques (forces only)."""
    return {
        "forces": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "env_mask": make_warp_env_mask(config.num_envs, config.device),
        "body_mask": make_warp_body_mask(config.num_bodies, config.device),
    }


def gen_set_forces_and_torques_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for set_forces_and_torques (forces + torques)."""
    return {
        "forces": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "torques": wp.from_torch(
            torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
            dtype=wp.vec3f,
        ),
        "env_mask": make_warp_env_mask(config.num_envs, config.device),
        "body_mask": make_warp_body_mask(config.num_bodies, config.device),
    }


def gen_reset_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for reset (no env_ids, resets all)."""
    return {}


def gen_reset_partial_warp(config: BenchmarkConfig) -> dict:
    """Generate Warp inputs for reset (with env_mask, partial reset)."""
    # Reset half of the environments
    env_mask = torch.zeros(config.num_envs, dtype=torch.bool, device=config.device)
    env_mask[: config.num_envs // 2] = True
    return {
        "env_mask": wp.from_torch(env_mask, dtype=wp.bool),
    }


# =============================================================================
# Input Generators for Torch (Worst Case)
# =============================================================================


def gen_add_forces_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for add_forces_and_torques (forces only)."""
    return {
        "forces": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": torch.arange(config.num_envs, device=config.device),
        "body_ids": torch.arange(config.num_bodies, device=config.device),
    }


def gen_add_torques_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for add_forces_and_torques (torques only)."""
    return {
        "torques": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": torch.arange(config.num_envs, device=config.device),
        "body_ids": torch.arange(config.num_bodies, device=config.device),
    }


def gen_add_forces_and_torques_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for add_forces_and_torques (forces + torques)."""
    return {
        "forces": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": torch.arange(config.num_envs, device=config.device),
        "body_ids": torch.arange(config.num_bodies, device=config.device),
    }


def gen_add_forces_at_positions_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for add_forces_and_torques (forces + positions)."""
    return {
        "forces": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "positions": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": torch.arange(config.num_envs, device=config.device),
        "body_ids": torch.arange(config.num_bodies, device=config.device),
    }


def gen_add_forces_and_torques_at_positions_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for add_forces_and_torques (forces + torques + positions)."""
    return {
        "forces": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "positions": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": torch.arange(config.num_envs, device=config.device),
        "body_ids": torch.arange(config.num_bodies, device=config.device),
    }


def gen_add_forces_global_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for add_forces_and_torques (forces, global frame)."""
    return {
        "forces": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": torch.arange(config.num_envs, device=config.device),
        "body_ids": torch.arange(config.num_bodies, device=config.device),
        "is_global": True,
    }


def gen_add_forces_at_positions_global_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for add_forces_and_torques (forces + positions, global frame)."""
    return {
        "forces": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "positions": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": torch.arange(config.num_envs, device=config.device),
        "body_ids": torch.arange(config.num_bodies, device=config.device),
        "is_global": True,
    }


def gen_set_forces_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for set_forces_and_torques (forces only)."""
    return {
        "forces": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": torch.arange(config.num_envs, device=config.device),
        "body_ids": torch.arange(config.num_bodies, device=config.device),
    }


def gen_set_forces_and_torques_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for set_forces_and_torques (forces + torques)."""
    return {
        "forces": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_envs, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": torch.arange(config.num_envs, device=config.device),
        "body_ids": torch.arange(config.num_bodies, device=config.device),
    }


def gen_reset_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for reset (no env_ids, resets all)."""
    return {}


def gen_reset_partial_torch(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs for reset (with env_ids, partial reset)."""
    return {
        "env_ids": torch.arange(config.num_envs // 2, device=config.device),
    }


# =============================================================================
# Method Benchmark Definitions
# =============================================================================

METHOD_BENCHMARKS = [
    # --- add_forces_and_torques variants ---
    MethodBenchmark(
        name="add_forces (local)",
        method_name="add_forces_and_torques",
        input_generator_warp=gen_add_forces_warp,
        input_generator_torch=gen_add_forces_torch,
        category="add",
    ),
    MethodBenchmark(
        name="add_torques (local)",
        method_name="add_forces_and_torques",
        input_generator_warp=gen_add_torques_warp,
        input_generator_torch=gen_add_torques_torch,
        category="add",
    ),
    MethodBenchmark(
        name="add_forces_and_torques (local)",
        method_name="add_forces_and_torques",
        input_generator_warp=gen_add_forces_and_torques_warp,
        input_generator_torch=gen_add_forces_and_torques_torch,
        category="add",
    ),
    MethodBenchmark(
        name="add_forces_at_positions (local)",
        method_name="add_forces_and_torques",
        input_generator_warp=gen_add_forces_at_positions_warp,
        input_generator_torch=gen_add_forces_at_positions_torch,
        category="add",
    ),
    MethodBenchmark(
        name="add_forces_and_torques_at_positions (local)",
        method_name="add_forces_and_torques",
        input_generator_warp=gen_add_forces_and_torques_at_positions_warp,
        input_generator_torch=gen_add_forces_and_torques_at_positions_torch,
        category="add",
    ),
    MethodBenchmark(
        name="add_forces (global)",
        method_name="add_forces_and_torques",
        input_generator_warp=gen_add_forces_global_warp,
        input_generator_torch=gen_add_forces_global_torch,
        category="add_global",
    ),
    MethodBenchmark(
        name="add_forces_at_positions (global)",
        method_name="add_forces_and_torques",
        input_generator_warp=gen_add_forces_at_positions_global_warp,
        input_generator_torch=gen_add_forces_at_positions_global_torch,
        category="add_global",
    ),
    # --- set_forces_and_torques variants ---
    MethodBenchmark(
        name="set_forces (local)",
        method_name="set_forces_and_torques",
        input_generator_warp=gen_set_forces_warp,
        input_generator_torch=gen_set_forces_torch,
        category="set",
        reset_before=False,
    ),
    MethodBenchmark(
        name="set_forces_and_torques (local)",
        method_name="set_forces_and_torques",
        input_generator_warp=gen_set_forces_and_torques_warp,
        input_generator_torch=gen_set_forces_and_torques_torch,
        category="set",
        reset_before=False,
    ),
    # --- reset variants ---
    MethodBenchmark(
        name="reset (all)",
        method_name="reset",
        input_generator_warp=gen_reset_warp,
        input_generator_torch=gen_reset_torch,
        category="reset",
        reset_before=False,
    ),
    MethodBenchmark(
        name="reset (partial, half envs)",
        method_name="reset",
        input_generator_warp=gen_reset_partial_warp,
        input_generator_torch=gen_reset_partial_torch,
        category="reset",
        reset_before=False,
    ),
]


# =============================================================================
# Benchmarking Functions
# =============================================================================


def warmup_all_kernels(wrench_composer: WrenchComposer, config: BenchmarkConfig, modes: list[InputMode]):
    """Warm up all warp kernels to avoid compilation overhead during benchmarking.

    This function runs each method once for each mode to ensure all warp kernels
    are compiled before the actual benchmarking begins.

    Args:
        wrench_composer: The WrenchComposer instance.
        config: Benchmark configuration.
        modes: List of input modes to warm up.
    """
    print("\nWarming up kernels (compiling warp kernels)...", end=" ", flush=True)

    for method_benchmark in METHOD_BENCHMARKS:
        method_name = method_benchmark.method_name
        if not hasattr(wrench_composer, method_name):
            continue

        method = getattr(wrench_composer, method_name)

        for mode in modes:
            input_generator = (
                method_benchmark.input_generator_warp
                if mode == InputMode.WARP
                else method_benchmark.input_generator_torch
            )

            # Run multiple warmup iterations to ensure kernel is fully compiled
            for _ in range(3):
                try:
                    if method_benchmark.reset_before:
                        wrench_composer.reset()
                    inputs = input_generator(config)
                    method(**inputs)
                except Exception:
                    pass

    # Synchronize to ensure all kernels are compiled
    if config.device.startswith("cuda"):
        wp.synchronize()

    print("Done.")


def benchmark_method(
    wrench_composer: WrenchComposer,
    method_benchmark: MethodBenchmark,
    mode: InputMode,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Benchmark a single method of WrenchComposer.

    Args:
        wrench_composer: The WrenchComposer instance.
        method_benchmark: The method benchmark definition.
        mode: Input mode (WARP or TORCH).
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with timing statistics.
    """
    method_name = method_benchmark.method_name

    # Check if method exists
    if not hasattr(wrench_composer, method_name):
        return BenchmarkResult(
            name=method_benchmark.name,
            mode=mode,
            mean_time_us=0.0,
            std_time_us=0.0,
            num_iterations=0,
            skipped=True,
            skip_reason="Method not found",
        )

    method = getattr(wrench_composer, method_name)
    input_generator = (
        method_benchmark.input_generator_warp if mode == InputMode.WARP else method_benchmark.input_generator_torch
    )

    # Try to call the method once to check for errors
    try:
        if method_benchmark.reset_before:
            wrench_composer.reset()
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
        if method_benchmark.reset_before:
            wrench_composer.reset()
        inputs = input_generator(config)
        with contextlib.suppress(Exception):
            method(**inputs)
        if config.device.startswith("cuda"):
            wp.synchronize()

    # Timing phase
    times = []
    for _ in range(config.num_iterations):
        if method_benchmark.reset_before:
            wrench_composer.reset()
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
    """Run all benchmarks for WrenchComposer.

    Args:
        config: Benchmark configuration.

    Returns:
        Tuple of (List of BenchmarkResults, hardware_info dict).
    """
    results = []

    # Gather and print hardware info
    hardware_info = get_hardware_info()
    print_hardware_info(hardware_info)

    # Create mock asset and wrench composer
    mock_asset = MockRigidObject(config.num_envs, config.num_bodies, config.device)
    wrench_composer = WrenchComposer(mock_asset)

    # Determine modes to run
    modes = []
    if config.mode in ("both", "warp"):
        modes.append(InputMode.WARP)
    if config.mode in ("both", "torch"):
        modes.append(InputMode.TORCH)

    # Global warmup to compile all warp kernels before benchmarking
    warmup_all_kernels(wrench_composer, config, modes)

    print(f"\nBenchmarking {len(METHOD_BENCHMARKS)} methods...")
    print(f"Config: {config.num_iterations} iterations, {config.warmup_steps} warmup steps")
    print(f"        {config.num_envs} envs, {config.num_bodies} bodies")
    print(f"Modes:  {', '.join(m.value for m in modes)}")
    print("-" * 100)

    for i, method_benchmark in enumerate(METHOD_BENCHMARKS):
        for mode in modes:
            mode_str = f"[{mode.value.upper():5}]"
            print(f"[{i + 1}/{len(METHOD_BENCHMARKS)}] {mode_str} {method_benchmark.name}...", end=" ", flush=True)

            result = benchmark_method(wrench_composer, method_benchmark, mode, config)
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
        print(f"{'Method Name':<45} {'Warp (µs)':<15} {'Torch (µs)':<15} {'Overhead':<12} {'Slowdown':<10}")
        print("-" * 100)

        warp_by_name = {r.name: r for r in warp_results}
        torch_by_name = {r.name: r for r in torch_results}

        for name in warp_by_name:
            if name in torch_by_name:
                warp_time = warp_by_name[name].mean_time_us
                torch_time = torch_by_name[name].mean_time_us
                overhead = torch_time - warp_time
                slowdown = torch_time / warp_time if warp_time > 0 else float("inf")
                print(f"{name:<45} {warp_time:>12.2f}   {torch_time:>12.2f}   {overhead:>+9.2f}   {slowdown:>7.2f}x")

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
                "num_envs": config.num_envs,
                "num_bodies": config.num_bodies,
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
    return f"wrench_composer_benchmark_{datetime_str}.json"


def main():
    """Main entry point for the benchmarking script."""
    parser = argparse.ArgumentParser(
        description="Micro-benchmarking framework for WrenchComposer class.",
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
        "--num_envs",
        type=int,
        default=4096,
        help="Number of environments.",
    )
    parser.add_argument(
        "--num_bodies",
        type=int,
        default=1,
        help="Number of bodies per environment.",
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
        num_envs=args.num_envs,
        num_bodies=args.num_bodies,
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
