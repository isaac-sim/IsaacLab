# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for Articulation class (Newton backend).

This module provides a benchmarking framework to measure the performance of setter and writer
methods in the Articulation class. Each method is benchmarked under seven scenarios:

1. **Torch List**: Inputs are PyTorch tensors with list indices (via deprecated wrappers).
2. **Torch Tensor Int32**: Tensor indices use signed 32-bit integers.
3. **Torch Tensor Int64**: Tensor indices use signed 64-bit integers.
4. **Warp Int32**: Item IDs are raw Warp ``int32`` arrays.
5. **Warp Int64**: Item IDs are raw Warp ``int64`` arrays.
6. **Proxy Int32**: Item IDs are cached finder ``ProxyArray`` objects backed by Warp ``int32`` storage.
7. **Warp Mask**: Inputs are Warp arrays with boolean masks (via ``_mask`` methods).

Usage:
    python benchmark_articulation.py [--num_iterations N] [--warmup_steps W]
        [--num_instances I] [--num_bodies B] [--num_joints J]

Example:
    python benchmark_articulation.py --num_iterations 1000 --warmup_steps 10
    python benchmark_articulation.py --mode torch_tensor_int64  # Only run 64-bit tensor benchmarks
    python benchmark_articulation.py --mode warp_mask   # Only run warp mask benchmarks
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Benchmark Articulation methods (Newton backend).")
parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations")
parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
parser.add_argument("--num_instances", type=int, default=4096, help="Number of instances")
parser.add_argument("--num_bodies", type=int, default=12, help="Number of bodies")
parser.add_argument("--num_joints", type=int, default=11, help="Number of joints")
parser.add_argument(
    "--mode",
    type=str,
    default="all",
    help=(
        "Benchmark mode (all, torch_list, torch_tensor_int32, torch_tensor_int64, warp_int32, warp_int64, "
        "proxy_int32, warp_mask)"
    ),
)
parser.add_argument("--output_dir", type=str, default=".", help="Output directory for results")
parser.add_argument("--backend", type=str, default="json", choices=["json", "osmo", "omniperf"], help="Metrics backend")
parser.add_argument("--no_shape_checks", action="store_true", help="Disable shape/dtype assertions")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True, args=args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import time
import warnings

import numpy as np
import torch
import warp as wp
from isaaclab_newton.test.mock_interfaces import (
    MockNewtonArticulationView,
    MockWrenchComposer,
    create_mock_newton_manager,
)

from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.test.benchmark import (
    DictMeasurement,
    MethodBenchmarkDefinition,
    MethodBenchmarkRunner,
    MethodBenchmarkRunnerConfig,
)

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Also suppress logging warnings
logging.getLogger("isaaclab_newton").setLevel(logging.ERROR)
logging.getLogger("isaaclab").setLevel(logging.ERROR)


# =============================================================================
# Index Helpers
# =============================================================================


def make_tensor_env_ids(num_instances: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Create a tensor of environment IDs."""
    return torch.arange(num_instances, dtype=dtype, device=device)


def make_tensor_joint_ids(num_joints: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Create a tensor of joint IDs."""
    return torch.arange(num_joints, dtype=dtype, device=device)


def make_tensor_body_ids(num_bodies: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Create a tensor of body IDs."""
    return torch.arange(num_bodies, dtype=dtype, device=device)


# =============================================================================
# Test Articulation Factory
# =============================================================================


def create_test_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    device: str = "cuda:0",
):
    """Create a test Articulation instance with mocked dependencies."""
    from isaaclab_newton.assets.articulation.articulation import Articulation

    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]

    articulation = object.__new__(Articulation)

    articulation.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        soft_joint_pos_limit_factor=1.0,
        actuators={},
    )

    # Create Newton mock view
    mock_view = MockNewtonArticulationView(
        num_instances=num_instances,
        num_bodies=num_bodies,
        num_joints=num_joints,
        device=device,
        joint_names=joint_names,
        body_names=body_names,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    object.__setattr__(articulation, "_root_view", mock_view)
    object.__setattr__(articulation, "_device", device)
    object.__setattr__(articulation, "_check_shapes", not args.no_shape_checks)

    # Create ArticulationData instance (NewtonManager already mocked at call site)
    from isaaclab_newton.assets.articulation.articulation_data import ArticulationData

    data = ArticulationData(mock_view, device)
    object.__setattr__(articulation, "_data", data)

    # Create mock wrench composers
    mock_inst_wrench = MockWrenchComposer(articulation)
    mock_perm_wrench = MockWrenchComposer(articulation)
    object.__setattr__(articulation, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(articulation, "_permanent_wrench_composer", mock_perm_wrench)

    # Set up other required attributes
    object.__setattr__(articulation, "actuators", {})
    object.__setattr__(articulation, "_has_implicit_actuators", False)
    object.__setattr__(articulation, "_ALL_INDICES", wp.array(np.arange(num_instances, dtype=np.int32), device=device))
    object.__setattr__(
        articulation, "_ALL_BODY_INDICES", wp.array(np.arange(num_bodies, dtype=np.int32), device=device)
    )
    object.__setattr__(
        articulation, "_ALL_JOINT_INDICES", wp.array(np.arange(num_joints, dtype=np.int32), device=device)
    )
    object.__setattr__(articulation, "_ALL_ENV_MASK", wp.ones((num_instances,), dtype=wp.bool, device=device))
    object.__setattr__(articulation, "_ALL_JOINT_MASK", wp.ones((num_joints,), dtype=wp.bool, device=device))
    object.__setattr__(articulation, "_ALL_BODY_MASK", wp.ones((num_bodies,), dtype=wp.bool, device=device))
    object.__setattr__(articulation, "_ALL_FIXED_TENDON_INDICES", wp.array([], dtype=wp.int32, device=device))
    object.__setattr__(articulation, "_ALL_FIXED_TENDON_MASK", wp.zeros((0,), dtype=wp.bool, device=device))
    object.__setattr__(articulation, "_ALL_SPATIAL_TENDON_INDICES", wp.array([], dtype=wp.int32, device=device))
    object.__setattr__(articulation, "_ALL_SPATIAL_TENDON_MASK", wp.zeros((0,), dtype=wp.bool, device=device))

    # Initialize joint targets
    object.__setattr__(
        articulation, "_joint_pos_target_sim", wp.zeros((num_instances, num_joints), dtype=wp.float32, device=device)
    )
    object.__setattr__(
        articulation, "_joint_vel_target_sim", wp.zeros((num_instances, num_joints), dtype=wp.float32, device=device)
    )
    object.__setattr__(
        articulation,
        "_joint_effort_target_sim",
        wp.zeros((num_instances, num_joints), dtype=wp.float32, device=device),
    )

    return articulation, mock_view


# =============================================================================
# Input Generators (Torch-only for Newton backend)
# =============================================================================


# --- Root Link Pose ---
def gen_root_link_pose_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_link_pose_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# --- Root COM Pose ---
def gen_root_com_pose_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_com_pose_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# --- Root Link Velocity ---
def gen_root_link_velocity_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_link_velocity_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# --- Root COM Velocity ---
def gen_root_com_velocity_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_com_velocity_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# --- Root State (Deprecated) ---
def gen_root_state_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_state_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# --- Root COM State (Deprecated) ---
def gen_root_com_state_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_com_state_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# --- Root Link State (Deprecated) ---
def gen_root_link_state_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_link_state_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# --- Joint State ---
def gen_joint_state_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_state_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Position ---
def gen_joint_position_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_position_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Velocity ---
def gen_joint_velocity_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_velocity_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Stiffness ---
def gen_joint_stiffness_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "stiffness": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_stiffness_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "stiffness": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Damping ---
def gen_joint_damping_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "damping": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_damping_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "damping": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Position Limit ---
def gen_joint_position_limit_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    lower = torch.rand(config.num_instances, config.num_joints, 1, device=config.device, dtype=torch.float32) * -3.14
    upper = torch.rand(config.num_instances, config.num_joints, 1, device=config.device, dtype=torch.float32) * 3.14
    return {
        "limits": torch.cat([lower, upper], dim=-1),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_position_limit_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    lower = torch.rand(config.num_instances, config.num_joints, 1, device=config.device, dtype=torch.float32) * -3.14
    upper = torch.rand(config.num_instances, config.num_joints, 1, device=config.device, dtype=torch.float32) * 3.14
    return {
        "limits": torch.cat([lower, upper], dim=-1),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Velocity Limit ---
def gen_joint_velocity_limit_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "limits": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 10.0,
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_velocity_limit_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "limits": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 10.0,
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Effort Limit ---
def gen_joint_effort_limit_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "limits": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 100.0
        ),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_effort_limit_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "limits": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 100.0
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Armature ---
def gen_joint_armature_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "armature": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.1
        ),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_armature_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "armature": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.1
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Friction Coefficient ---
def gen_joint_friction_coefficient_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "joint_friction_coeff": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.5
        ),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_friction_coefficient_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "joint_friction_coeff": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.5
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Set Joint Position Target ---
def gen_set_joint_position_target_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_set_joint_position_target_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Set Joint Velocity Target ---
def gen_set_joint_velocity_target_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_set_joint_velocity_target_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Set Joint Effort Target ---
def gen_set_joint_effort_target_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_set_joint_effort_target_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device, torch.int32),
    }


# --- Set CoMs ---
def gen_set_coms_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_set_coms_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device, torch.int32),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device, torch.int32),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# =============================================================================
# Warp Mask Input Generators (for _mask methods)
# =============================================================================


def _env_mask(config: MethodBenchmarkRunnerConfig) -> wp.array:
    return wp.ones((config.num_instances,), dtype=wp.bool, device=config.device)


def _joint_mask(config: MethodBenchmarkRunnerConfig) -> wp.array:
    return wp.ones((config.num_joints,), dtype=wp.bool, device=config.device)


def _body_mask(config: MethodBenchmarkRunnerConfig) -> wp.array:
    return wp.ones((config.num_bodies,), dtype=wp.bool, device=config.device)


# --- Root Link Pose (mask) ---
def gen_root_link_pose_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_mask": _env_mask(config),
    }


# --- Root COM Pose (mask) ---
def gen_root_com_pose_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_mask": _env_mask(config),
    }


# --- Root Link Velocity (mask) ---
def gen_root_link_velocity_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_mask": _env_mask(config),
    }


# --- Root COM Velocity (mask) ---
def gen_root_com_velocity_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_mask": _env_mask(config),
    }


# --- Joint State (mask) ---
def gen_joint_state_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "joint_mask": _joint_mask(config),
        "env_mask": _env_mask(config),
    }


# --- Joint Position (mask) ---
def gen_joint_position_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "joint_mask": _joint_mask(config),
        "env_mask": _env_mask(config),
    }


# --- Joint Velocity (mask) ---
def gen_joint_velocity_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "joint_mask": _joint_mask(config),
        "env_mask": _env_mask(config),
    }


# --- Joint Stiffness (mask) ---
def gen_joint_stiffness_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "stiffness": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "joint_mask": _joint_mask(config),
        "env_mask": _env_mask(config),
    }


# --- Joint Damping (mask) ---
def gen_joint_damping_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "damping": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "joint_mask": _joint_mask(config),
        "env_mask": _env_mask(config),
    }


# --- Joint Position Limit (mask) ---
def gen_joint_position_limit_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    lower = torch.rand(config.num_instances, config.num_joints, 1, device=config.device, dtype=torch.float32) * -3.14
    upper = torch.rand(config.num_instances, config.num_joints, 1, device=config.device, dtype=torch.float32) * 3.14
    return {
        "limits": torch.cat([lower, upper], dim=-1),
        "joint_mask": _joint_mask(config),
        "env_mask": _env_mask(config),
    }


# --- Joint Velocity Limit (mask) ---
def gen_joint_velocity_limit_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "limits": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 10.0,
        "joint_mask": _joint_mask(config),
        "env_mask": _env_mask(config),
    }


# --- Joint Effort Limit (mask) ---
def gen_joint_effort_limit_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "limits": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 100.0
        ),
        "joint_mask": _joint_mask(config),
        "env_mask": _env_mask(config),
    }


# --- Joint Armature (mask) ---
def gen_joint_armature_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "armature": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.1
        ),
        "joint_mask": _joint_mask(config),
        "env_mask": _env_mask(config),
    }


# --- Joint Friction Coefficient (mask) ---
def gen_joint_friction_coefficient_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "joint_friction_coeff": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.5
        ),
        "joint_mask": _joint_mask(config),
        "env_mask": _env_mask(config),
    }


# --- Set Joint Position Target (mask) ---
def gen_set_joint_position_target_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "joint_mask": _joint_mask(config),
        "env_mask": _env_mask(config),
    }


# --- Set Joint Velocity Target (mask) ---
def gen_set_joint_velocity_target_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "joint_mask": _joint_mask(config),
        "env_mask": _env_mask(config),
    }


# --- Set Joint Effort Target (mask) ---
def gen_set_joint_effort_target_warp_mask(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "joint_mask": _joint_mask(config),
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
        "coms": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
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


def _make_tensor_dtype_generator(base_gen_fn, dtype: torch.dtype):
    """Create a Newton generator with all selectors of the requested dtype."""

    def generator(config):
        inputs = base_gen_fn(config)
        for key in ("env_ids", "joint_ids", "body_ids"):
            value = inputs.get(key)
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(dtype)
        return inputs

    return generator


ITEM_SELECTOR_MODES = (
    "torch_list",
    "torch_tensor_int32",
    "torch_tensor_int64",
    "warp_int32",
    "warp_int64",
    "proxy_int32",
)
"""Fair item-selector modes; every case shares one prebuilt ``torch.int32`` environment selector."""

ITEM_SELECTOR_KEYS = {
    "write_joint_state_to_sim": "joint_ids",
    "write_joint_position_to_sim": "joint_ids",
    "write_joint_velocity_to_sim": "joint_ids",
    "write_joint_stiffness_to_sim": "joint_ids",
    "write_joint_damping_to_sim": "joint_ids",
    "write_joint_position_limit_to_sim": "joint_ids",
    "write_joint_velocity_limit_to_sim": "joint_ids",
    "write_joint_effort_limit_to_sim": "joint_ids",
    "write_joint_armature_to_sim": "joint_ids",
    "write_joint_friction_coefficient_to_sim": "joint_ids",
    "set_joint_position_target": "joint_ids",
    "set_joint_velocity_target": "joint_ids",
    "set_joint_effort_target": "joint_ids",
    "set_masses": "body_ids",
    "set_coms": "body_ids",
    "set_inertias": "body_ids",
    "set_external_force_and_torque": "body_ids",
}
"""Actual articulation benchmarks that vary a joint/body selector representation."""


class _ItemSelectorInputFactory:
    """Prepare all item-selector representations once, outside writer timing."""

    def __init__(self, base_generator, item_key: str):
        self._base_generator = base_generator
        self._item_key = item_key
        self._inputs_by_mode = None
        self._proxy_selector = None
        self._setup_count = 0

    @property
    def setup_count(self) -> int:
        """Number of times selector setup ran."""
        return self._setup_count

    @property
    def proxy_selector(self):
        """Prepared cached proxy selector, or ``None`` before setup."""
        return self._proxy_selector

    def assert_proxy_unmaterialized(self) -> None:
        """Assert the prepared proxy still has no cached Torch view."""
        if self._proxy_selector is None:
            raise AssertionError("proxy selector setup has not run")
        if self._proxy_selector._torch_cache is not None:
            raise AssertionError("prepared proxy selector materialized a Torch view")

    def _prepare(self, config) -> None:
        base_inputs = self._base_generator(config)
        env_ids = base_inputs.get("env_ids")
        if not isinstance(env_ids, torch.Tensor) or env_ids.dtype != torch.int32:
            raise AssertionError("item-selector benchmarks require one prebuilt torch.int32 environment selector")
        item_ids = base_inputs.get(self._item_key)
        if not isinstance(item_ids, torch.Tensor) or item_ids.dtype != torch.int32:
            raise AssertionError(f"{self._item_key} setup must start from torch.int32")

        articulation = config.articulation
        finder_name = "find_joints" if self._item_key == "joint_ids" else "find_bodies"
        finder = getattr(articulation, finder_name)
        legacy, _ = finder(".*", as_proxy=False)
        proxy, _ = finder(".*", as_proxy=True)
        repeated_proxy, _ = finder(".*", as_proxy=True)
        if proxy is not repeated_proxy:
            raise AssertionError(f"{finder_name} did not reuse its cached ProxyArray")
        if proxy.dtype != wp.int32:
            raise AssertionError(f"{finder_name} proxy must use warp.int32 storage")

        item_values = item_ids.tolist()
        if legacy != item_values:
            raise AssertionError(f"{finder_name} full-range result differs from the benchmark workload")
        selectors = {
            "torch_list": legacy,
            "torch_tensor_int32": item_ids,
            "torch_tensor_int64": item_ids.to(torch.int64),
            "warp_int32": wp.array(item_values, dtype=wp.int32, device=config.device),
            "warp_int64": wp.array(item_values, dtype=wp.int64, device=config.device),
            "proxy_int32": proxy,
        }
        self._inputs_by_mode = {mode: {**base_inputs, self._item_key: selector} for mode, selector in selectors.items()}
        self._proxy_selector = proxy
        self._setup_count += 1
        self.assert_proxy_unmaterialized()

    def make_generator(self, mode: str):
        """Return a generator that reuses its prepared input dictionary."""

        def generator(config):
            if self._inputs_by_mode is None:
                self._prepare(config)
            return self._inputs_by_mode[mode]

        return generator


def _register_item_selector_modes(benchmark, item_key: str) -> _ItemSelectorInputFactory:
    """Replace legacy tensor generators with the complete fair representation grid."""
    base_generator = benchmark.input_generators["torch_tensor"]
    factory = _ItemSelectorInputFactory(base_generator, item_key)
    benchmark.input_generators = {mode: factory.make_generator(mode) for mode in ITEM_SELECTOR_MODES}
    return factory


def _register_benchmark_selector_modes(benchmarks) -> dict[str, _ItemSelectorInputFactory]:
    """Register item grids on actual item-bearing definitions and typed root modes elsewhere."""
    factories = {}
    for benchmark in benchmarks:
        item_key = ITEM_SELECTOR_KEYS.get(benchmark.name)
        if item_key is not None:
            factories[benchmark.name] = _register_item_selector_modes(benchmark, item_key)
        elif "torch_tensor" in benchmark.input_generators:
            base_generator = benchmark.input_generators.pop("torch_tensor")
            benchmark.input_generators.update(
                {
                    "torch_tensor_int32": _make_tensor_dtype_generator(base_generator, torch.int32),
                    "torch_tensor_int64": _make_tensor_dtype_generator(base_generator, torch.int64),
                }
            )
    return factories


def _measure_callable(callable_, num_iterations: int, warmup_steps: int, before_each=None, after_each=None) -> dict:
    """Measure a callable while keeping optional per-sample ownership setup outside timing."""
    for _ in range(warmup_steps):
        if before_each is not None:
            before_each()
        try:
            callable_()
        finally:
            if after_each is not None:
                after_each()

    samples = []
    for _ in range(num_iterations):
        if before_each is not None:
            before_each()
        try:
            start = time.perf_counter()
            callable_()
            end = time.perf_counter()
        finally:
            if after_each is not None:
                after_each()
        samples.append((end - start) * 1e6)
    return {
        "median": float(np.median(samples)),
        "iqr": float(np.percentile(samples, 75) - np.percentile(samples, 25)),
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "n": len(samples),
        "attempts": num_iterations,
        "failures": 0,
    }


def _measure_finder_paths(articulation, finder_name: str, num_iterations: int, warmup_steps: int) -> dict:
    """Measure cold allocation and cached lookup with explicit untimed selector ownership."""
    finder = getattr(articulation, finder_name)
    cold_selector = None

    def _clear_cache():
        clear_method = getattr(articulation, "_clear_selector_cache", None)
        if clear_method is not None:
            clear_method()
        else:
            selector_cache = getattr(articulation, "_selector_cache", None)
            if selector_cache is not None:
                selector_cache.clear()

    def prepare_cold_sample():
        nonlocal cold_selector
        cold_selector = None
        _clear_cache()
        wp.synchronize()

    def cold_allocation():
        nonlocal cold_selector
        cold_selector, _ = finder(".*", as_proxy=True)
        wp.synchronize()
        if cold_selector._torch_cache is not None:
            raise AssertionError("cold finder benchmark materialized a Torch view")

    def release_cold_sample():
        nonlocal cold_selector
        cold_selector = None
        _clear_cache()
        wp.synchronize()

    cold_stats = _measure_callable(
        cold_allocation,
        num_iterations,
        warmup_steps,
        before_each=prepare_cold_sample,
        after_each=release_cold_sample,
    )

    cached_selector, _ = finder(".*", as_proxy=True)
    wp.synchronize()

    def cached_lookup():
        selector, _ = finder(".*", as_proxy=True)
        if selector is not cached_selector:
            raise AssertionError("steady-state finder lookup missed the proxy cache")
        if selector._torch_cache is not None:
            raise AssertionError("cached finder benchmark materialized a Torch view")

    cached_stats = _measure_callable(cached_lookup, num_iterations, warmup_steps)
    if cached_selector._torch_cache is not None:
        raise AssertionError("cached finder benchmark materialized a Torch view")
    return {"cold_allocation": cold_stats, "cached_lookup": cached_stats}


def _summarize_writer_results(results: dict[str, dict], item_benchmark_names) -> dict:
    """Summarize only registered item benchmarks and add ratios against both P10 baselines."""
    grouped = {}
    item_benchmark_names = set(item_benchmark_names)
    for result_name, stats in results.items():
        for mode in ITEM_SELECTOR_MODES:
            suffix = f"_{mode}"
            if result_name.endswith(suffix):
                method_name = result_name[: -len(suffix)]
                if method_name not in item_benchmark_names:
                    break
                grouped.setdefault(method_name, {})[mode] = {
                    "median_us": stats["median"],
                    "iqr_us": stats["iqr"],
                    "n": stats["n"],
                    "attempts": stats["attempts"],
                    "failures": stats["failures"],
                }
                break
    for modes in grouped.values():
        tensor_stats = modes.get("torch_tensor_int32")
        list_stats = modes.get("torch_list")
        tensor_baseline = (
            tensor_stats["median_us"]
            if tensor_stats and tensor_stats["failures"] == 0 and tensor_stats["n"] == tensor_stats["attempts"]
            else None
        )
        list_baseline = (
            list_stats["median_us"]
            if list_stats and list_stats["failures"] == 0 and list_stats["n"] == list_stats["attempts"]
            else None
        )
        for stats in modes.values():
            median = stats["median_us"]
            valid = stats["failures"] == 0 and stats["n"] == stats["attempts"]
            stats["ratio_vs_torch_tensor_int32"] = median / tensor_baseline if valid and tensor_baseline else None
            stats["ratio_vs_torch_list"] = median / list_baseline if valid and list_baseline else None
    return grouped


class _SelectorBenchmarkRunner(MethodBenchmarkRunner):
    """Method runner that retains complete median/IQR samples for registered item benchmarks."""

    def __init__(self, *args, selector_factories=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._selector_factories = selector_factories or {}
        self._item_benchmark_names = set(self._selector_factories)
        self.selector_results = {}

    def _factory_for_result(self, method_name: str):
        for mode in ITEM_SELECTOR_MODES:
            suffix = f"_{mode}"
            if method_name.endswith(suffix):
                return self._selector_factories.get(method_name[: -len(suffix)])
        return None

    def _benchmark_method(self, method, method_name: str, generator, dependencies: list[str]) -> dict | None:
        """Benchmark only the prepared writer callable and fail on incomplete sampling."""
        if method is None:
            return None
        factory = self._factory_for_result(method_name)
        try:
            inputs = generator(self._config)
            method(**inputs)
        except NotImplementedError as error:
            if factory is not None:
                raise
            return {"skipped": True, "skip_reason": f"NotImplementedError: {error}"}
        except Exception as error:
            if factory is not None:
                raise
            return {"skipped": True, "skip_reason": f"Error: {type(error).__name__}: {error}"}
        if factory is not None:
            factory.assert_proxy_unmaterialized()

        for _ in range(self._config.warmup_steps):
            method(**inputs)
            if self._config.device.startswith("cuda"):
                self._sync_device()
        if factory is not None:
            factory.assert_proxy_unmaterialized()

        samples = []
        for _ in range(self._config.num_iterations):
            if self._config.device.startswith("cuda"):
                self._sync_device()
            start = time.perf_counter()
            method(**inputs)
            if self._config.device.startswith("cuda"):
                self._sync_device()
            samples.append((time.perf_counter() - start) * 1e6)
        if factory is not None:
            factory.assert_proxy_unmaterialized()

        result = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "median": float(np.median(samples)),
            "iqr": float(np.percentile(samples, 75) - np.percentile(samples, 25)),
            "n": len(samples),
            "attempts": self._config.num_iterations,
            "failures": 0,
        }
        if factory is not None:
            self.selector_results[method_name] = result
        return result


def _print_selector_summary(finder_results: dict, writer_summary: dict) -> None:
    """Print finder and writer selector statistics with sample completeness."""
    print("\n" + "=" * 80)
    print("Finder and item-selector representation summary (median / IQR, us)")
    print("=" * 80)
    for domain, paths in finder_results.items():
        for path, stats in paths.items():
            print(
                f"{domain:>5} {path:>15}: {stats['median']:.3f} / {stats['iqr']:.3f}"
                f"  n={stats['n']}/{stats['attempts']} failures={stats['failures']}"
            )
    for method_name, modes in writer_summary.items():
        print(method_name)
        for mode, stats in modes.items():
            tensor_ratio = stats["ratio_vs_torch_tensor_int32"]
            list_ratio = stats["ratio_vs_torch_list"]
            tensor_ratio_text = f"{tensor_ratio:.3f}" if tensor_ratio is not None else "n/a"
            list_ratio_text = f"{list_ratio:.3f}" if list_ratio is not None else "n/a"
            print(
                f"  {mode:>20}: {stats['median_us']:.3f} / {stats['iqr_us']:.3f}"
                f"  n={stats['n']}/{stats['attempts']} failures={stats['failures']}"
                f"  x torch.int32={tensor_ratio_text}  x list={list_ratio_text}"
            )


BENCHMARKS = [
    # --- Root State (Deprecated, no _mask equivalent) ---
    MethodBenchmarkDefinition(
        name="write_root_state_to_sim",
        method_name="write_root_state_to_sim",
        input_generators={
            "torch_list": gen_root_state_torch_list,
            "torch_tensor": gen_root_state_torch_tensor,
        },
        category="root_state",
    ),
    MethodBenchmarkDefinition(
        name="write_root_com_state_to_sim",
        method_name="write_root_com_state_to_sim",
        input_generators={
            "torch_list": gen_root_com_state_torch_list,
            "torch_tensor": gen_root_com_state_torch_tensor,
        },
        category="root_state",
    ),
    MethodBenchmarkDefinition(
        name="write_root_link_state_to_sim",
        method_name="write_root_link_state_to_sim",
        input_generators={
            "torch_list": gen_root_link_state_torch_list,
            "torch_tensor": gen_root_link_state_torch_tensor,
        },
        category="root_state",
    ),
    # --- Root Pose / Velocity ---
    MethodBenchmarkDefinition(
        name="write_root_link_pose_to_sim",
        method_name="write_root_link_pose_to_sim",
        input_generators={
            "torch_list": gen_root_link_pose_torch_list,
            "torch_tensor": gen_root_link_pose_torch_tensor,
        },
        category="root_pose",
    ),
    MethodBenchmarkDefinition(
        name="write_root_link_pose_to_sim_mask",
        method_name="write_root_link_pose_to_sim_mask",
        input_generators={"warp_mask": gen_root_link_pose_warp_mask},
        category="root_pose",
    ),
    MethodBenchmarkDefinition(
        name="write_root_com_pose_to_sim",
        method_name="write_root_com_pose_to_sim",
        input_generators={
            "torch_list": gen_root_com_pose_torch_list,
            "torch_tensor": gen_root_com_pose_torch_tensor,
        },
        category="root_pose",
    ),
    MethodBenchmarkDefinition(
        name="write_root_com_pose_to_sim_mask",
        method_name="write_root_com_pose_to_sim_mask",
        input_generators={"warp_mask": gen_root_com_pose_warp_mask},
        category="root_pose",
    ),
    MethodBenchmarkDefinition(
        name="write_root_link_velocity_to_sim",
        method_name="write_root_link_velocity_to_sim",
        input_generators={
            "torch_list": gen_root_link_velocity_torch_list,
            "torch_tensor": gen_root_link_velocity_torch_tensor,
        },
        category="root_velocity",
    ),
    MethodBenchmarkDefinition(
        name="write_root_link_velocity_to_sim_mask",
        method_name="write_root_link_velocity_to_sim_mask",
        input_generators={"warp_mask": gen_root_link_velocity_warp_mask},
        category="root_velocity",
    ),
    MethodBenchmarkDefinition(
        name="write_root_com_velocity_to_sim",
        method_name="write_root_com_velocity_to_sim",
        input_generators={
            "torch_list": gen_root_com_velocity_torch_list,
            "torch_tensor": gen_root_com_velocity_torch_tensor,
        },
        category="root_velocity",
    ),
    MethodBenchmarkDefinition(
        name="write_root_com_velocity_to_sim_mask",
        method_name="write_root_com_velocity_to_sim_mask",
        input_generators={"warp_mask": gen_root_com_velocity_warp_mask},
        category="root_velocity",
    ),
    # --- Joint State ---
    MethodBenchmarkDefinition(
        name="write_joint_state_to_sim",
        method_name="write_joint_state_to_sim",
        input_generators={
            "torch_list": gen_joint_state_torch_list,
            "torch_tensor": gen_joint_state_torch_tensor,
        },
        category="joint_state",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_state_to_sim_mask",
        method_name="write_joint_state_to_sim_mask",
        input_generators={"warp_mask": gen_joint_state_warp_mask},
        category="joint_state",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_position_to_sim",
        method_name="write_joint_position_to_sim",
        input_generators={
            "torch_list": gen_joint_position_torch_list,
            "torch_tensor": gen_joint_position_torch_tensor,
        },
        category="joint_state",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_position_to_sim_mask",
        method_name="write_joint_position_to_sim_mask",
        input_generators={"warp_mask": gen_joint_position_warp_mask},
        category="joint_state",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_velocity_to_sim",
        method_name="write_joint_velocity_to_sim",
        input_generators={
            "torch_list": gen_joint_velocity_torch_list,
            "torch_tensor": gen_joint_velocity_torch_tensor,
        },
        category="joint_state",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_velocity_to_sim_mask",
        method_name="write_joint_velocity_to_sim_mask",
        input_generators={"warp_mask": gen_joint_velocity_warp_mask},
        category="joint_state",
    ),
    # --- Joint Params ---
    MethodBenchmarkDefinition(
        name="write_joint_stiffness_to_sim",
        method_name="write_joint_stiffness_to_sim",
        input_generators={
            "torch_list": gen_joint_stiffness_torch_list,
            "torch_tensor": gen_joint_stiffness_torch_tensor,
        },
        category="joint_params",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_stiffness_to_sim_mask",
        method_name="write_joint_stiffness_to_sim_mask",
        input_generators={"warp_mask": gen_joint_stiffness_warp_mask},
        category="joint_params",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_damping_to_sim",
        method_name="write_joint_damping_to_sim",
        input_generators={
            "torch_list": gen_joint_damping_torch_list,
            "torch_tensor": gen_joint_damping_torch_tensor,
        },
        category="joint_params",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_damping_to_sim_mask",
        method_name="write_joint_damping_to_sim_mask",
        input_generators={"warp_mask": gen_joint_damping_warp_mask},
        category="joint_params",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_position_limit_to_sim",
        method_name="write_joint_position_limit_to_sim",
        input_generators={
            "torch_list": gen_joint_position_limit_torch_list,
            "torch_tensor": gen_joint_position_limit_torch_tensor,
        },
        category="joint_params",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_position_limit_to_sim_mask",
        method_name="write_joint_position_limit_to_sim_mask",
        input_generators={"warp_mask": gen_joint_position_limit_warp_mask},
        category="joint_params",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_velocity_limit_to_sim",
        method_name="write_joint_velocity_limit_to_sim",
        input_generators={
            "torch_list": gen_joint_velocity_limit_torch_list,
            "torch_tensor": gen_joint_velocity_limit_torch_tensor,
        },
        category="joint_params",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_velocity_limit_to_sim_mask",
        method_name="write_joint_velocity_limit_to_sim_mask",
        input_generators={"warp_mask": gen_joint_velocity_limit_warp_mask},
        category="joint_params",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_effort_limit_to_sim",
        method_name="write_joint_effort_limit_to_sim",
        input_generators={
            "torch_list": gen_joint_effort_limit_torch_list,
            "torch_tensor": gen_joint_effort_limit_torch_tensor,
        },
        category="joint_params",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_effort_limit_to_sim_mask",
        method_name="write_joint_effort_limit_to_sim_mask",
        input_generators={"warp_mask": gen_joint_effort_limit_warp_mask},
        category="joint_params",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_armature_to_sim",
        method_name="write_joint_armature_to_sim",
        input_generators={
            "torch_list": gen_joint_armature_torch_list,
            "torch_tensor": gen_joint_armature_torch_tensor,
        },
        category="joint_params",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_armature_to_sim_mask",
        method_name="write_joint_armature_to_sim_mask",
        input_generators={"warp_mask": gen_joint_armature_warp_mask},
        category="joint_params",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_friction_coefficient_to_sim",
        method_name="write_joint_friction_coefficient_to_sim",
        input_generators={
            "torch_list": gen_joint_friction_coefficient_torch_list,
            "torch_tensor": gen_joint_friction_coefficient_torch_tensor,
        },
        category="joint_params",
    ),
    MethodBenchmarkDefinition(
        name="write_joint_friction_coefficient_to_sim_mask",
        method_name="write_joint_friction_coefficient_to_sim_mask",
        input_generators={"warp_mask": gen_joint_friction_coefficient_warp_mask},
        category="joint_params",
    ),
    # --- Joint Targets ---
    MethodBenchmarkDefinition(
        name="set_joint_position_target",
        method_name="set_joint_position_target",
        input_generators={
            "torch_list": gen_set_joint_position_target_torch_list,
            "torch_tensor": gen_set_joint_position_target_torch_tensor,
        },
        category="joint_targets",
    ),
    MethodBenchmarkDefinition(
        name="set_joint_position_target_mask",
        method_name="set_joint_position_target_mask",
        input_generators={"warp_mask": gen_set_joint_position_target_warp_mask},
        category="joint_targets",
    ),
    MethodBenchmarkDefinition(
        name="set_joint_velocity_target",
        method_name="set_joint_velocity_target",
        input_generators={
            "torch_list": gen_set_joint_velocity_target_torch_list,
            "torch_tensor": gen_set_joint_velocity_target_torch_tensor,
        },
        category="joint_targets",
    ),
    MethodBenchmarkDefinition(
        name="set_joint_velocity_target_mask",
        method_name="set_joint_velocity_target_mask",
        input_generators={"warp_mask": gen_set_joint_velocity_target_warp_mask},
        category="joint_targets",
    ),
    MethodBenchmarkDefinition(
        name="set_joint_effort_target",
        method_name="set_joint_effort_target",
        input_generators={
            "torch_list": gen_set_joint_effort_target_torch_list,
            "torch_tensor": gen_set_joint_effort_target_torch_tensor,
        },
        category="joint_targets",
    ),
    MethodBenchmarkDefinition(
        name="set_joint_effort_target_mask",
        method_name="set_joint_effort_target_mask",
        input_generators={"warp_mask": gen_set_joint_effort_target_warp_mask},
        category="joint_targets",
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

ITEM_SELECTOR_FACTORIES = _register_benchmark_selector_modes(BENCHMARKS)


# =============================================================================
# Fill-Ratio Benchmarks (5%, 95%, 100% of env_ids filled)
# =============================================================================

FILL_RATIOS = {"5pct": 0.05, "95pct": 0.95, "100pct": 1.0}


def _make_fill_ratio_generator(base_gen_fn, fill_ratio):
    """Create a generator that subsets env_ids to a given fill ratio.

    Only env_ids are subsetted — joint_ids and body_ids remain full-range.
    Data tensors keyed on env count are sliced to match.
    """

    def generator(config):
        n = max(1, int(config.num_instances * fill_ratio))
        base_inputs = base_gen_fn(config)
        inputs = {}
        for key, value in base_inputs.items():
            if key == "env_ids":
                inputs[key] = (
                    torch.randperm(config.num_instances, device=config.device)[:n].sort().values.to(value.dtype)
                )
            elif isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == config.num_instances:
                inputs[key] = value[:n]
            else:
                inputs[key] = value
        return inputs

    return generator


def _make_fill_ratio_mask_generator(base_mask_gen_fn, fill_ratio):
    """Create a mask generator with a given fill ratio.

    Sets a random subset of the env_mask entries to True. Data stays full-sized (mask methods expect full data).
    """

    def generator(config):
        base_inputs = base_mask_gen_fn(config)
        n = max(1, int(config.num_instances * fill_ratio))
        # Create a mask with n random entries set to True
        perm = torch.randperm(config.num_instances, device=config.device)
        mask_tensor = torch.zeros(config.num_instances, dtype=torch.bool, device=config.device)
        mask_tensor[perm[:n]] = True
        base_inputs["env_mask"] = wp.from_torch(mask_tensor, dtype=wp.bool)
        return base_inputs

    return generator


def _build_fill_benchmarks():
    """Auto-generate fill-ratio benchmark definitions from existing generators."""
    fill_benchmarks = []
    for bm in BENCHMARKS:
        generators = {}
        # Add tensor fill variants for both selector dtypes.
        for mode in ("torch_tensor_int32", "torch_tensor_int64"):
            if mode in bm.input_generators:
                base_gen = bm.input_generators[mode]
                for suffix, ratio in FILL_RATIOS.items():
                    generators[f"{mode}_{suffix}"] = _make_fill_ratio_generator(base_gen, ratio)
        # Add mask fill variants from warp_mask generators
        if "warp_mask" in bm.input_generators:
            base_gen = bm.input_generators["warp_mask"]
            for suffix, ratio in FILL_RATIOS.items():
                generators[f"mask_{suffix}"] = _make_fill_ratio_mask_generator(base_gen, ratio)
        if generators:
            fill_benchmarks.append(
                MethodBenchmarkDefinition(
                    name=bm.name,
                    method_name=bm.method_name,
                    input_generators=generators,
                    category=f"{bm.category}_fill",
                )
            )
    return fill_benchmarks


FILL_BENCHMARKS = _build_fill_benchmarks()


def main():
    """Main entry point for the benchmarking script."""
    config = MethodBenchmarkRunnerConfig(
        num_iterations=args.num_iterations,
        warmup_steps=args.warmup_steps,
        num_instances=args.num_instances,
        num_bodies=args.num_bodies,
        num_joints=args.num_joints,
        device=args.device,
        mode=args.mode,
    )

    # Patch the NewtonManager for both articulation and articulation_data modules
    with (
        create_mock_newton_manager(
            "isaaclab_newton.assets.articulation.articulation_data.SimulationManager",
            gravity=(0.0, 0.0, -9.81),
            num_instances=config.num_instances,
            num_bodies=config.num_bodies,
            num_joints=config.num_joints,
        ),
        create_mock_newton_manager(
            "isaaclab_newton.assets.articulation.articulation.SimulationManager",
            gravity=(0.0, 0.0, -9.81),
            num_instances=config.num_instances,
            num_bodies=config.num_bodies,
            num_joints=config.num_joints,
        ),
    ):
        # Create the test articulation
        articulation, _ = create_test_articulation(
            num_instances=config.num_instances,
            num_bodies=config.num_bodies,
            num_joints=config.num_joints,
            device=config.device,
        )

        config.articulation = articulation
        print(
            f"Benchmarking Articulation (Newton) with {config.num_instances} instances, {config.num_bodies} bodies,"
            f" {config.num_joints} joints..."
        )

        # Create runner and run benchmarks
        runner = _SelectorBenchmarkRunner(
            benchmark_name="newton_articulation_benchmark",
            config=config,
            selector_factories=ITEM_SELECTOR_FACTORIES,
            backend_type=args.backend,
            output_path=args.output_dir,
            use_recorders=True,
        )

        runner.run_benchmarks(BENCHMARKS, articulation)

        finder_results = {
            "joint": _measure_finder_paths(articulation, "find_joints", config.num_iterations, config.warmup_steps),
            "body": _measure_finder_paths(articulation, "find_bodies", config.num_iterations, config.warmup_steps),
        }
        writer_summary = _summarize_writer_results(runner.selector_results, ITEM_SELECTOR_FACTORIES)
        selector_summary = {
            "config": {
                "num_iterations": config.num_iterations,
                "warmup_steps": config.warmup_steps,
                "num_instances": config.num_instances,
                "num_bodies": config.num_bodies,
                "num_joints": config.num_joints,
                "device": config.device,
            },
            "finder": finder_results,
            "writer": writer_summary,
        }
        runner.add_measurement(
            "selector_representation_summary",
            DictMeasurement(name="selector_representation_summary", value=selector_summary),
        )
        _print_selector_summary(finder_results, writer_summary)

        print("\n" + "=" * 80)
        print("Fill-Ratio Benchmarks (env_ids at 5%, 95%, 100% fill)")
        print("=" * 80)

        runner.run_benchmarks(FILL_BENCHMARKS, articulation)
        runner.finalize()

    # Close the simulation app
    simulation_app.close()


if __name__ == "__main__":
    main()
