# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for Articulation class (PhysX backend).

This module provides a benchmarking framework to measure the performance of setter and writer
methods in the Articulation class. Each method is benchmarked under two scenarios:

1. **Torch List**: Inputs are PyTorch tensors with list indices.
2. **Torch Tensor**: Inputs are PyTorch tensors with tensor indices.

Usage:
    python benchmark_articulation.py [--num_iterations N] [--warmup_steps W]
        [--num_instances I] [--num_bodies B] [--num_joints J]

Example:
    python benchmark_articulation.py --num_iterations 1000 --warmup_steps 10
    python benchmark_articulation.py --mode torch_list  # Only run list-based benchmarks
    python benchmark_articulation.py --mode torch_tensor  # Only run tensor-based benchmarks
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
parser.add_argument("--mode", type=str, default="all", help="Benchmark mode (all, torch_list, torch_tensor)")
parser.add_argument("--output_dir", type=str, default=".", help="Output directory for results")
parser.add_argument("--backend", type=str, default="json", choices=["json", "osmo", "omniperf"], help="Metrics backend")

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
from unittest.mock import MagicMock

import torch

# Mock SimulationManager.get_physics_sim_view() to return a mock object with gravity
# This is needed because the Data classes call SimulationManager.get_physics_sim_view().get_gravity()
# but there's no actual physics scene when running benchmarks
_mock_physics_sim_view = MagicMock()
_mock_physics_sim_view.get_gravity.return_value = (0.0, 0.0, -9.81)

from isaacsim.core.simulation_manager import SimulationManager

SimulationManager.get_physics_sim_view = MagicMock(return_value=_mock_physics_sim_view)
import warp as wp
from isaaclab_physx.assets.articulation.articulation import Articulation
from isaaclab_physx.assets.articulation.articulation_data import ArticulationData
from isaaclab_physx.test.benchmark import make_tensor_body_ids, make_tensor_env_ids, make_tensor_joint_ids
from isaaclab_physx.test.mock_interfaces.views import MockArticulationView

from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.test.benchmark import MethodBenchmarkDefinition, MethodBenchmarkRunner, MethodBenchmarkRunnerConfig
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
) -> tuple[Articulation, MockArticulationView, MagicMock]:
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
    mock_view = MockArticulationView(
        count=num_instances,
        num_links=num_bodies,
        num_dofs=num_joints,
        device=device,
    )
    mock_view.set_random_mock_data()

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
    object.__setattr__(articulation, "_ALL_INDICES", torch.arange(num_instances, device=device))
    object.__setattr__(articulation, "_ALL_BODY_INDICES", torch.arange(num_bodies, device=device))
    object.__setattr__(articulation, "_ALL_JOINT_INDICES", torch.arange(num_joints, device=device))

    # Warp arrays for set_external_force_and_torque
    all_indices = torch.arange(num_instances, dtype=torch.int32, device=device)
    all_body_indices = torch.arange(num_bodies, dtype=torch.int32, device=device)
    object.__setattr__(articulation, "_ALL_INDICES_WP", wp.from_torch(all_indices, dtype=wp.int32))
    object.__setattr__(articulation, "_ALL_BODY_INDICES_WP", wp.from_torch(all_body_indices, dtype=wp.int32))

    # Initialize joint targets
    object.__setattr__(articulation, "_joint_pos_target_sim", torch.zeros(num_instances, num_joints, device=device))
    object.__setattr__(articulation, "_joint_vel_target_sim", torch.zeros(num_instances, num_joints, device=device))
    object.__setattr__(articulation, "_joint_effort_target_sim", torch.zeros(num_instances, num_joints, device=device))

    return articulation, mock_view, None


# =============================================================================
# Input Generators (Torch-only for PhysX backend)
# =============================================================================


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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
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
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- Set External Force and Torque ---
def gen_set_external_force_and_torque_torch_list(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with list ids for set_external_force_and_torque."""
    return {
        "forces": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_set_external_force_and_torque_torch_tensor(config: MethodBenchmarkRunnerConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_external_force_and_torque."""
    return {
        "forces": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


# =============================================================================
# Benchmarks
# =============================================================================

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

    print(
        f"Benchmarking Articulation (PhysX) with {config.num_instances} instances, {config.num_bodies} bodies,"
        f" {config.num_joints} joints..."
    )

    # Create runner and run benchmarks
    runner = MethodBenchmarkRunner(
        benchmark_name="articulation_benchmark",
        config=config,
        backend_type=args.backend,
        output_path=args.output_dir,
        use_recorders=True,
    )

    runner.run_benchmarks(BENCHMARKS, articulation)
    runner.finalize()

    # Close the simulation app
    simulation_app.close()


if __name__ == "__main__":
    main()
