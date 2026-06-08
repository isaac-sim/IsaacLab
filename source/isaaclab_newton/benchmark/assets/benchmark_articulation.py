# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for Articulation class (Newton backend).

This module provides a benchmarking framework to measure the performance of setter and writer
methods in the Articulation class. Each method is benchmarked under three scenarios:

1. **Torch List**: Inputs are PyTorch tensors with list indices (via deprecated wrappers).
2. **Torch Tensor**: Inputs are PyTorch tensors with tensor indices (via deprecated wrappers).
3. **Warp Mask**: Inputs are warp arrays with boolean masks (via ``_mask`` methods).

Usage:
    python benchmark_articulation.py [--num_iterations N] [--warmup_steps W]
        [--num_instances I] [--num_bodies B] [--num_joints J]

Example:
    python benchmark_articulation.py --num_iterations 1000 --warmup_steps 10
    python benchmark_articulation.py --mode torch_list  # Only run list-based benchmarks
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
parser.add_argument("--mode", type=str, default="all", help="Benchmark mode (all, torch_list, torch_tensor, warp_mask)")
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
from isaaclab.test.benchmark import MethodBenchmarkDefinition, MethodBenchmarkRunner, MethodBenchmarkRunnerConfig

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Also suppress logging warnings
logging.getLogger("isaaclab_newton").setLevel(logging.ERROR)
logging.getLogger("isaaclab").setLevel(logging.ERROR)


# =============================================================================
# Index Helpers
# =============================================================================


def make_tensor_env_ids(num_instances: int, device: str) -> torch.Tensor:
    """Create a tensor of environment IDs."""
    return torch.arange(num_instances, dtype=torch.int32, device=device)


def make_tensor_joint_ids(num_joints: int, device: str) -> torch.Tensor:
    """Create a tensor of joint IDs."""
    return torch.arange(num_joints, dtype=torch.int32, device=device)


def make_tensor_body_ids(num_bodies: int, device: str) -> torch.Tensor:
    """Create a tensor of body IDs."""
    return torch.arange(num_bodies, dtype=torch.int32, device=device)


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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "coms": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_set_coms_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
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
        for key, val in base_inputs.items():
            if key == "env_ids":
                inputs[key] = (
                    torch.randperm(config.num_instances, device=config.device)[:n].sort().values.to(torch.int32)
                )
            elif isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == config.num_instances:
                inputs[key] = val[:n]
            else:
                inputs[key] = val
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
        # Add tensor fill variants from torch_tensor generators
        if "torch_tensor" in bm.input_generators:
            base_gen = bm.input_generators["torch_tensor"]
            for suffix, ratio in FILL_RATIOS.items():
                generators[f"tensor_{suffix}"] = _make_fill_ratio_generator(base_gen, ratio)
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
        ),
        create_mock_newton_manager(
            "isaaclab_newton.assets.articulation.articulation.SimulationManager",
            gravity=(0.0, 0.0, -9.81),
        ),
    ):
        # Create the test articulation
        articulation, _ = create_test_articulation(
            num_instances=config.num_instances,
            num_bodies=config.num_bodies,
            num_joints=config.num_joints,
            device=config.device,
        )

        print(
            f"Benchmarking Articulation (Newton) with {config.num_instances} instances, {config.num_bodies} bodies,"
            f" {config.num_joints} joints..."
        )

        # Create runner and run benchmarks
        runner = MethodBenchmarkRunner(
            benchmark_name="newton_articulation_benchmark",
            config=config,
            backend_type=args.backend,
            output_path=args.output_dir,
            use_recorders=True,
        )

        runner.run_benchmarks(BENCHMARKS, articulation)

        print("\n" + "=" * 80)
        print("Fill-Ratio Benchmarks (env_ids at 5%, 95%, 100% fill)")
        print("=" * 80)

        runner.run_benchmarks(FILL_BENCHMARKS, articulation)
        runner.finalize()

    # Close the simulation app
    simulation_app.close()


if __name__ == "__main__":
    main()
