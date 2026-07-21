# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for Articulation class (PhysX backend).

This module provides a benchmarking framework to measure the performance of setter and writer
methods in the Articulation class. Each method is benchmarked under six scenarios:

1. **Torch List**: Inputs are PyTorch tensors with list indices.
2. **Torch Tensor Int32**: Item IDs are 32-bit PyTorch tensors.
3. **Torch Tensor Int64**: Item IDs are 64-bit PyTorch tensors.
4. **Warp Int32**: Item IDs are raw Warp ``int32`` arrays.
5. **Warp Int64**: Item IDs are raw Warp ``int64`` arrays.
6. **Proxy Int32**: Item IDs are cached finder ``ProxyArray`` objects backed by Warp ``int32`` storage.

Usage:
    python benchmark_articulation.py [--num_iterations N] [--warmup_steps W]
        [--num_instances I] [--num_bodies B] [--num_joints J]

Example:
    python benchmark_articulation.py --num_iterations 1000 --warmup_steps 10
    python benchmark_articulation.py --mode torch_list  # Only run list-based benchmarks
    python benchmark_articulation.py --mode torch_tensor_int64  # Only run 64-bit tensor benchmarks
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Benchmark Articulation methods (PhysX backend).")
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
        "Benchmark mode (all, torch_list, torch_tensor_int32, torch_tensor_int64, warp_int32, warp_int64, proxy_int32)"
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

import contextlib
import logging
import time
import warnings
from unittest.mock import MagicMock

import numpy as np
import torch

# Mock SimulationManager.get_physics_sim_view() to return a mock object with gravity
# This is needed because the Data classes call SimulationManager.get_physics_sim_view().get_gravity()
# but there's no actual physics scene when running benchmarks
_mock_physics_sim_view = MagicMock()
_mock_physics_sim_view.get_gravity.return_value = (0.0, 0.0, -9.81)

from isaaclab_physx.physics import PhysxManager as SimulationManager

SimulationManager.get_physics_sim_view = MagicMock(return_value=_mock_physics_sim_view)
import warp as wp
from isaaclab_physx.assets.articulation.articulation import Articulation
from isaaclab_physx.assets.articulation.articulation_data import ArticulationData
from isaaclab_physx.test.mock_interfaces.views import MockArticulationViewWarp

from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.test.benchmark import (
    DictMeasurement,
    MethodBenchmarkDefinition,
    MethodBenchmarkRunner,
    MethodBenchmarkRunnerConfig,
)
from isaaclab.test.mock_interfaces.utils import MockWrenchComposer

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Also suppress logging warnings (the body_acc_w deprecation warnings use logging)
logging.getLogger("isaaclab_physx").setLevel(logging.ERROR)
logging.getLogger("isaaclab").setLevel(logging.ERROR)


def create_test_articulation(
    num_instances: int = 2,
    num_joints: int = 6,
    num_bodies: int = 7,
    device: str = "cuda:0",
) -> tuple[Articulation, MockArticulationViewWarp, MagicMock]:
    """Create a test Articulation instance with mocked dependencies."""
    joint_names = [f"joint_{i}" for i in range(num_joints)]
    body_names = [f"body_{i}" for i in range(num_bodies)]

    articulation = object.__new__(Articulation)

    articulation.cfg = ArticulationCfg(
        prim_path="/World/Robot",
        soft_joint_pos_limit_factor=1.0,
        actuators={},
    )

    # Create PhysX mock view
    mock_view = MockArticulationViewWarp(
        count=num_instances,
        num_links=num_bodies,
        num_dofs=num_joints,
        device=device,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    # Set up the mock view's metatype for accessing names/counts
    mock_metatype = MagicMock()
    mock_metatype.fixed_base = False
    mock_metatype.dof_count = num_joints
    mock_metatype.link_count = num_bodies
    mock_metatype.dof_names = joint_names
    mock_metatype.link_names = body_names
    object.__setattr__(mock_view, "_shared_metatype", mock_metatype)

    object.__setattr__(articulation, "_root_view", mock_view)
    object.__setattr__(articulation, "_device", device)
    object.__setattr__(articulation, "_check_shapes", not args.no_shape_checks)

    # Create ArticulationData instance (SimulationManager already mocked at module level)
    data = ArticulationData(mock_view, device)
    object.__setattr__(articulation, "_data", data)

    # Create mock wrench composers (pass articulation which has num_instances, num_bodies, device properties)
    mock_inst_wrench = MockWrenchComposer(articulation)
    mock_perm_wrench = MockWrenchComposer(articulation)
    object.__setattr__(articulation, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(articulation, "_permanent_wrench_composer", mock_perm_wrench)

    # Set up other required attributes
    object.__setattr__(articulation, "actuators", {})
    object.__setattr__(articulation, "_has_implicit_actuators", False)

    # Use warp arrays for _ALL_* indices (matching real _create_buffers)
    import numpy as np

    all_indices_wp = wp.array(np.arange(num_instances, dtype=np.int32), device=device)
    all_joint_indices_wp = wp.array(np.arange(num_joints, dtype=np.int32), device=device)
    all_body_indices_wp = wp.array(np.arange(num_bodies, dtype=np.int32), device=device)
    object.__setattr__(articulation, "_ALL_INDICES", all_indices_wp)
    object.__setattr__(articulation, "_ALL_JOINT_INDICES", all_joint_indices_wp)
    object.__setattr__(articulation, "_ALL_BODY_INDICES", all_body_indices_wp)

    # Warp arrays for set_external_force_and_torque
    object.__setattr__(articulation, "_ALL_INDICES_WP", all_indices_wp)
    object.__setattr__(articulation, "_ALL_BODY_INDICES_WP", all_body_indices_wp)

    # Initialize joint targets
    object.__setattr__(articulation, "_joint_pos_target_sim", torch.zeros(num_instances, num_joints, device=device))
    object.__setattr__(articulation, "_joint_vel_target_sim", torch.zeros(num_instances, num_joints, device=device))
    object.__setattr__(articulation, "_joint_effort_target_sim", torch.zeros(num_instances, num_joints, device=device))

    # Cached .view() wrappers
    object.__setattr__(articulation, "_root_link_pose_w_f32", None)
    object.__setattr__(articulation, "_root_com_vel_w_f32", None)
    object.__setattr__(articulation, "_root_link_vel_w_f32", None)

    # Pre-allocated pinned CPU buffers for PhysX TensorAPI writes
    N, J, B = num_instances, num_joints, num_bodies
    object.__setattr__(articulation, "_cpu_env_ids_all", wp.zeros(N, dtype=wp.int32, device="cpu", pinned=True))
    wp.copy(articulation._cpu_env_ids_all, all_indices_wp)
    object.__setattr__(
        articulation, "_cpu_joint_stiffness", wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
    )
    object.__setattr__(
        articulation, "_cpu_joint_damping", wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
    )
    object.__setattr__(
        articulation, "_cpu_joint_pos_limits", wp.zeros((N, J, 2), dtype=wp.float32, device="cpu", pinned=True)
    )
    object.__setattr__(
        articulation, "_cpu_joint_vel_limits", wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
    )
    object.__setattr__(
        articulation, "_cpu_joint_effort_limits", wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
    )
    object.__setattr__(
        articulation, "_cpu_joint_armature", wp.zeros((N, J), dtype=wp.float32, device="cpu", pinned=True)
    )
    object.__setattr__(
        articulation, "_cpu_joint_friction_props", wp.zeros((N, J, 3), dtype=wp.float32, device="cpu", pinned=True)
    )
    object.__setattr__(articulation, "_cpu_body_mass", wp.zeros((N, B), dtype=wp.float32, device="cpu", pinned=True))
    object.__setattr__(articulation, "_cpu_body_coms", wp.zeros((N, B, 7), dtype=wp.float32, device="cpu", pinned=True))
    object.__setattr__(
        articulation, "_cpu_body_inertia", wp.zeros((N, B, 9), dtype=wp.float32, device="cpu", pinned=True)
    )

    return articulation, mock_view, None


# =============================================================================
# Input Generators (Torch-only for PhysX backend)
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


# --- Root Link Pose ---
def gen_root_link_pose_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_link_pose_to_sim."""
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_link_pose_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_link_pose_to_sim."""
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# --- Root COM Pose ---
def gen_root_com_pose_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_com_pose_to_sim."""
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_com_pose_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_com_pose_to_sim."""
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# --- Root Link Velocity ---
def gen_root_link_velocity_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_link_velocity_to_sim."""
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_link_velocity_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_link_velocity_to_sim."""
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# --- Root COM Velocity ---
def gen_root_com_velocity_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_com_velocity_to_sim."""
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_com_velocity_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_com_velocity_to_sim."""
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# --- Root State (Deprecated) ---
def gen_root_state_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_state_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# --- Root COM State (Deprecated) ---
def gen_root_com_state_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_com_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_com_state_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_com_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# --- Root Link State (Deprecated) ---
def gen_root_link_state_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_link_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_link_state_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_link_state_to_sim."""
    return {
        "root_state": torch.rand(config.num_instances, 13, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
    }


# --- Joint State ---
def gen_joint_state_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_state_to_sim."""
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_state_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_state_to_sim."""
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Position ---
def gen_joint_position_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_position_to_sim."""
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_position_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_position_to_sim."""
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Velocity ---
def gen_joint_velocity_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_velocity_to_sim."""
    return {
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_velocity_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_velocity_to_sim."""
    return {
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Stiffness ---
def gen_joint_stiffness_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_stiffness_to_sim."""
    return {
        "stiffness": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_stiffness_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_stiffness_to_sim."""
    return {
        "stiffness": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Damping ---
def gen_joint_damping_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_damping_to_sim."""
    return {
        "damping": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_damping_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_damping_to_sim."""
    return {
        "damping": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Position Limit ---
def gen_joint_position_limit_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_position_limit_to_sim."""
    # limits shape is (N, J, 2) where [:,: ,0] is lower and [:,:,1] is upper
    lower = torch.rand(config.num_instances, config.num_joints, 1, device=config.device, dtype=torch.float32) * -3.14
    upper = torch.rand(config.num_instances, config.num_joints, 1, device=config.device, dtype=torch.float32) * 3.14
    return {
        "limits": torch.cat([lower, upper], dim=-1),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_position_limit_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_position_limit_to_sim."""
    # limits shape is (N, J, 2) where [:,: ,0] is lower and [:,:,1] is upper
    lower = torch.rand(config.num_instances, config.num_joints, 1, device=config.device, dtype=torch.float32) * -3.14
    upper = torch.rand(config.num_instances, config.num_joints, 1, device=config.device, dtype=torch.float32) * 3.14
    return {
        "limits": torch.cat([lower, upper], dim=-1),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Velocity Limit ---
def gen_joint_velocity_limit_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_velocity_limit_to_sim."""
    return {
        "limits": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 10.0,
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_velocity_limit_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_velocity_limit_to_sim."""
    return {
        "limits": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 10.0,
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Effort Limit ---
def gen_joint_effort_limit_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_effort_limit_to_sim."""
    return {
        "limits": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 100.0
        ),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_effort_limit_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_effort_limit_to_sim."""
    return {
        "limits": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 100.0
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Armature ---
def gen_joint_armature_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_armature_to_sim."""
    return {
        "armature": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.1
        ),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_armature_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_armature_to_sim."""
    return {
        "armature": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.1
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Joint Friction Coefficient ---
def gen_joint_friction_coefficient_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_friction_coefficient_to_sim."""
    return {
        "joint_friction_coeff": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.5
        ),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_friction_coefficient_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_friction_coefficient_to_sim."""
    return {
        "joint_friction_coeff": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.5
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Set Joint Position Target ---
def gen_set_joint_position_target_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for set_joint_position_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_set_joint_position_target_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_joint_position_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Set Joint Velocity Target ---
def gen_set_joint_velocity_target_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for set_joint_velocity_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_set_joint_velocity_target_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_joint_velocity_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Set Joint Effort Target ---
def gen_set_joint_effort_target_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for set_joint_effort_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_set_joint_effort_target_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_joint_effort_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device, torch.int32),
    }


# --- Set Masses ---
def gen_set_masses_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for set_masses."""
    # Articulation masses shape is (N, B)
    return {
        "masses": torch.rand(config.num_instances, config.num_bodies, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_set_masses_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_masses."""
    # Articulation masses shape is (N, B)
    return {
        "masses": torch.rand(config.num_instances, config.num_bodies, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device, torch.int32),
    }


# --- Set CoMs ---
def gen_set_coms_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for set_coms."""
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_set_coms_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_coms."""
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device, torch.int32),
    }


# --- Set Inertias ---
def gen_set_inertias_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for set_inertias."""
    # Articulation inertias shape is (N, B, 9) - flattened 3x3 matrix
    return {
        "inertias": torch.rand(config.num_instances, config.num_bodies, 9, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_set_inertias_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_inertias."""
    # Articulation inertias shape is (N, B, 9) - flattened 3x3 matrix
    return {
        "inertias": torch.rand(config.num_instances, config.num_bodies, 9, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device, torch.int32),
    }


# --- Set External Force and Torque ---
def gen_set_external_force_and_torque_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for set_external_force_and_torque."""
    return {
        "forces": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_set_external_force_and_torque_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_external_force_and_torque."""
    return {
        "forces": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device, torch.int32),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device, torch.int32),
    }


# =============================================================================
# Benchmarks
# =============================================================================


def _make_tensor_dtype_generator(base_gen_fn, dtype: torch.dtype):
    """Create a PhysX generator with item selectors of the requested dtype."""

    def generator(config):
        inputs = base_gen_fn(config)
        for key in ("joint_ids", "body_ids"):
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


class _ItemSelectorInputFactory:
    """Prepare all item-selector representations once, outside writer timing."""

    def __init__(self, base_generator, item_key: str):
        self._base_generator = base_generator
        self._item_key = item_key
        self._inputs_by_mode = None
        self._setup_count = 0

    @property
    def setup_count(self) -> int:
        """Number of times selector setup ran."""
        return self._setup_count

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
        if proxy._torch_cache is not None:
            raise AssertionError("proxy benchmark setup must not materialize a Torch view")

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
        self._setup_count += 1

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


def _measure_callable(callable_, num_iterations: int, warmup_steps: int) -> dict:
    """Measure a prepared callable and return median and IQR dispersion in microseconds."""
    for _ in range(warmup_steps):
        callable_()
    samples = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        callable_()
        samples.append((time.perf_counter() - start) * 1e6)
    return {
        "median": float(np.median(samples)),
        "iqr": float(np.percentile(samples, 75) - np.percentile(samples, 25)),
        "mean": float(np.mean(samples)),
        "std": float(np.std(samples)),
        "n": len(samples),
    }


def _measure_finder_paths(articulation, finder_name: str, num_iterations: int, warmup_steps: int) -> dict:
    """Measure cold proxy allocation separately from steady-state cached finder lookup."""
    finder = getattr(articulation, finder_name)

    def _clear_cache():
        clear_method = getattr(articulation, "_clear_selector_cache", None)
        if clear_method is not None:
            clear_method()
        else:
            selector_cache = getattr(articulation, "_selector_cache", None)
            if selector_cache is not None:
                selector_cache.clear()

    def cold_allocation():
        _clear_cache()
        selector, _ = finder(".*", as_proxy=True)
        wp.synchronize()
        if selector._torch_cache is not None:
            raise AssertionError("cold finder benchmark materialized a Torch view")

    cold_stats = _measure_callable(cold_allocation, num_iterations, warmup_steps)
    _clear_cache()
    cached_selector, _ = finder(".*", as_proxy=True)

    def cached_lookup():
        selector, _ = finder(".*", as_proxy=True)
        if selector is not cached_selector:
            raise AssertionError("steady-state finder lookup missed the proxy cache")
        if selector._torch_cache is not None:
            raise AssertionError("cached finder benchmark materialized a Torch view")

    return {
        "cold_allocation": cold_stats,
        "cached_lookup": _measure_callable(cached_lookup, num_iterations, warmup_steps),
    }


def _summarize_writer_results(results: dict[str, dict]) -> dict:
    """Group writer statistics by mode and add ratios against both P10 baselines."""
    grouped = {}
    for result_name, stats in results.items():
        for mode in ITEM_SELECTOR_MODES:
            suffix = f"_{mode}"
            if result_name.endswith(suffix):
                method_name = result_name[: -len(suffix)]
                grouped.setdefault(method_name, {})[mode] = {
                    "median_us": stats["median"],
                    "iqr_us": stats["iqr"],
                }
                break
    for modes in grouped.values():
        tensor_baseline = modes.get("torch_tensor_int32", {}).get("median_us")
        list_baseline = modes.get("torch_list", {}).get("median_us")
        for stats in modes.values():
            median = stats["median_us"]
            stats["ratio_vs_torch_tensor_int32"] = median / tensor_baseline if tensor_baseline else None
            stats["ratio_vs_torch_list"] = median / list_baseline if list_baseline else None
    return grouped


class _SelectorBenchmarkRunner(MethodBenchmarkRunner):
    """Method runner that retains median/IQR samples for selector-mode comparisons."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector_results = {}

    def _benchmark_method(self, method, method_name: str, generator, dependencies: list[str]) -> dict | None:
        """Benchmark only the prepared writer callable; generator setup remains outside timing."""
        if method is None:
            return None
        try:
            inputs = generator(self._config)
            method(**inputs)
        except NotImplementedError as error:
            return {"skipped": True, "skip_reason": f"NotImplementedError: {error}"}
        except Exception as error:
            return {"skipped": True, "skip_reason": f"Error: {type(error).__name__}: {error}"}

        for _ in range(self._config.warmup_steps):
            with contextlib.suppress(Exception):
                method(**inputs)
            if self._config.device.startswith("cuda"):
                self._sync_device()

        samples = []
        for _ in range(self._config.num_iterations):
            if self._config.device.startswith("cuda"):
                self._sync_device()
            start = time.perf_counter()
            try:
                method(**inputs)
            except Exception:
                continue
            if self._config.device.startswith("cuda"):
                self._sync_device()
            samples.append((time.perf_counter() - start) * 1e6)
        if not samples:
            return {"skipped": True, "skip_reason": "No successful iterations"}

        result = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "median": float(np.median(samples)),
            "iqr": float(np.percentile(samples, 75) - np.percentile(samples, 25)),
            "n": len(samples),
        }
        if any(method_name.endswith(f"_{mode}") for mode in ITEM_SELECTOR_MODES):
            self.selector_results[method_name] = result
        return result


def _print_selector_summary(finder_results: dict, writer_summary: dict) -> None:
    """Print finder and writer selector statistics in a compact human-readable report."""
    print("\n" + "=" * 80)
    print("Finder and item-selector representation summary (median / IQR, us)")
    print("=" * 80)
    for domain, paths in finder_results.items():
        for path, stats in paths.items():
            print(f"{domain:>5} {path:>15}: {stats['median']:.3f} / {stats['iqr']:.3f}")
    for method_name, modes in writer_summary.items():
        print(method_name)
        for mode, stats in modes.items():
            tensor_ratio = stats["ratio_vs_torch_tensor_int32"]
            list_ratio = stats["ratio_vs_torch_list"]
            tensor_ratio_text = f"{tensor_ratio:.3f}" if tensor_ratio is not None else "n/a"
            list_ratio_text = f"{list_ratio:.3f}" if list_ratio is not None else "n/a"
            print(
                f"  {mode:>20}: {stats['median_us']:.3f} / {stats['iqr_us']:.3f}"
                f"  x torch.int32={tensor_ratio_text}  x list={list_ratio_text}"
            )


BENCHMARKS = [
    # --- Root State (Deprecated) ---
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
        name="write_root_com_pose_to_sim",
        method_name="write_root_com_pose_to_sim",
        input_generators={
            "torch_list": gen_root_com_pose_torch_list,
            "torch_tensor": gen_root_com_pose_torch_tensor,
        },
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
        name="write_root_com_velocity_to_sim",
        method_name="write_root_com_velocity_to_sim",
        input_generators={
            "torch_list": gen_root_com_velocity_torch_list,
            "torch_tensor": gen_root_com_velocity_torch_tensor,
        },
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
        name="write_joint_position_to_sim",
        method_name="write_joint_position_to_sim",
        input_generators={
            "torch_list": gen_joint_position_torch_list,
            "torch_tensor": gen_joint_position_torch_tensor,
        },
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
        name="write_joint_damping_to_sim",
        method_name="write_joint_damping_to_sim",
        input_generators={
            "torch_list": gen_joint_damping_torch_list,
            "torch_tensor": gen_joint_damping_torch_tensor,
        },
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
        name="write_joint_velocity_limit_to_sim",
        method_name="write_joint_velocity_limit_to_sim",
        input_generators={
            "torch_list": gen_joint_velocity_limit_torch_list,
            "torch_tensor": gen_joint_velocity_limit_torch_tensor,
        },
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
        name="write_joint_armature_to_sim",
        method_name="write_joint_armature_to_sim",
        input_generators={
            "torch_list": gen_joint_armature_torch_list,
            "torch_tensor": gen_joint_armature_torch_tensor,
        },
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
        name="set_joint_velocity_target",
        method_name="set_joint_velocity_target",
        input_generators={
            "torch_list": gen_set_joint_velocity_target_torch_list,
            "torch_tensor": gen_set_joint_velocity_target_torch_tensor,
        },
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
        name="set_coms",
        method_name="set_coms",
        input_generators={
            "torch_list": gen_set_coms_torch_list,
            "torch_tensor": gen_set_coms_torch_tensor,
        },
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
        name="set_external_force_and_torque",
        method_name="set_external_force_and_torque",
        input_generators={
            "torch_list": gen_set_external_force_and_torque_torch_list,
            "torch_tensor": gen_set_external_force_and_torque_torch_tensor,
        },
        category="external_wrench",
    ),
]

_BODY_ITEM_BENCHMARKS = {"set_masses", "set_coms", "set_inertias", "set_external_force_and_torque"}

for benchmark in BENCHMARKS:
    if benchmark.name.startswith(("write_joint_", "set_joint_")):
        _register_item_selector_modes(benchmark, "joint_ids")
    elif benchmark.name in _BODY_ITEM_BENCHMARKS:
        _register_item_selector_modes(benchmark, "body_ids")
    else:
        base_generator = benchmark.input_generators.pop("torch_tensor")
        benchmark.input_generators.update(
            {
                "torch_tensor_int32": _make_tensor_dtype_generator(base_generator, torch.int32),
                "torch_tensor_int64": _make_tensor_dtype_generator(base_generator, torch.int64),
            }
        )


# =============================================================================
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


def _build_fill_benchmarks():
    """Auto-generate fill-ratio benchmark definitions from the torch_tensor generators."""
    fill_benchmarks = []
    for bm in BENCHMARKS:
        generators = {}
        for mode in ("torch_tensor_int32", "torch_tensor_int64"):
            base_gen = bm.input_generators[mode]
            for suffix, ratio in FILL_RATIOS.items():
                generators[f"{mode}_{suffix}"] = _make_fill_ratio_generator(base_gen, ratio)
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

    # Create the test articulation
    articulation, _, _ = create_test_articulation(
        num_instances=config.num_instances,
        num_bodies=config.num_bodies,
        num_joints=config.num_joints,
        device=config.device,
    )

    config.articulation = articulation
    print(
        f"Benchmarking Articulation (PhysX) with {config.num_instances} instances, {config.num_bodies} bodies,"
        f" {config.num_joints} joints..."
    )

    # Create runner and run benchmarks
    runner = _SelectorBenchmarkRunner(
        benchmark_name="articulation_benchmark",
        config=config,
        backend_type=args.backend,
        output_path=args.output_dir,
        use_recorders=True,
    )

    runner.run_benchmarks(BENCHMARKS, articulation)

    finder_results = {
        "joint": _measure_finder_paths(articulation, "find_joints", config.num_iterations, config.warmup_steps),
        "body": _measure_finder_paths(articulation, "find_bodies", config.num_iterations, config.warmup_steps),
    }
    writer_summary = _summarize_writer_results(runner.selector_results)
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
