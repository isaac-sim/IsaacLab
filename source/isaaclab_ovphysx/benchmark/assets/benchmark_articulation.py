# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for Articulation class (OVPhysX backend).

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

import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Benchmark Articulation methods (OVPhysX backend).")
parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations")
parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
parser.add_argument("--num_instances", type=int, default=4096, help="Number of instances")
parser.add_argument("--num_bodies", type=int, default=12, help="Number of bodies")
parser.add_argument("--num_joints", type=int, default=11, help="Number of joints")
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

from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
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
    from isaaclab_ovphysx.assets.articulation.articulation import Articulation
    from isaaclab_ovphysx.assets.articulation.articulation_data import ArticulationData

    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]
    binding_set = MockOvPhysxBindingSet(
        num_instances=num_instances,
        num_joints=num_joints,
        num_bodies=num_bodies,
        joint_names=joint_names,
        body_names=body_names,
        benchmark_mode=True,
    )
    binding_set.set_random_data()
    mock_view = binding_set.view

    articulation = object.__new__(Articulation)
    articulation.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        soft_joint_pos_limit_factor=1.0,
        actuators={},
    )
    object.__setattr__(articulation, "_initialize_handle", None)
    object.__setattr__(articulation, "_invalidate_initialize_handle", None)
    object.__setattr__(articulation, "_prim_deletion_handle", None)
    object.__setattr__(articulation, "_debug_vis_handle", None)
    object.__setattr__(articulation, "_root_view", mock_view)
    object.__setattr__(articulation, "_device", device)
    object.__setattr__(articulation, "_check_shapes", not args.no_shape_checks)
    object.__setattr__(articulation, "_num_instances", num_instances)
    object.__setattr__(articulation, "_num_bodies", num_bodies)
    object.__setattr__(articulation, "_num_joints", num_joints)
    object.__setattr__(articulation, "_num_fixed_tendons", 0)
    object.__setattr__(articulation, "_num_spatial_tendons", 0)
    object.__setattr__(articulation, "_is_fixed_base", False)
    object.__setattr__(articulation, "_joint_names", joint_names)
    object.__setattr__(articulation, "_body_names", body_names)
    object.__setattr__(articulation, "_fixed_tendon_names", [])
    object.__setattr__(articulation, "_spatial_tendon_names", [])

    data = ArticulationData(mock_view, device)
    data._apply_ordering_maps_after_resolve()
    object.__setattr__(articulation, "_data", data)
    object.__setattr__(articulation, "actuators", {})
    object.__setattr__(articulation, "_has_implicit_actuators", False)
    articulation._create_buffers()

    return articulation, mock_view


# Input Generators (Torch-only for OVPhysX backend)
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

    # Create the test articulation
    articulation, _ = create_test_articulation(
        num_instances=config.num_instances,
        num_bodies=config.num_bodies,
        num_joints=config.num_joints,
        device=config.device,
    )

    print(
        f"Benchmarking Articulation (OVPhysX) with {config.num_instances} instances, {config.num_bodies} bodies,"
        f" {config.num_joints} joints..."
    )

    # Create runner and run benchmarks
    runner = MethodBenchmarkRunner(
        benchmark_name="ovphysx_articulation_benchmark",
        config=config,
        backend_type=args.benchmark_formatter,
        output_path=args.output_path,
        use_recorders=True,
    )

    runner.run_benchmarks(BENCHMARKS, articulation)
    runner.finalize()


if __name__ == "__main__":
    main()
