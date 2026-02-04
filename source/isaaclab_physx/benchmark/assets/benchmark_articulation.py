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

import argparse
import sys
import warnings
from types import ModuleType
from unittest.mock import MagicMock

import torch
import warp as wp

# Initialize Warp
wp.init()


# =============================================================================
# Mock Setup - Must happen BEFORE importing Articulation
# =============================================================================


class MockPhysicsSimView:
    """Simple mock for the physics simulation view."""

    def get_gravity(self):
        return (0.0, 0.0, -9.81)

    def update_articulations_kinematic(self):
        pass


class MockSimulationManager:
    """Simple mock for SimulationManager."""

    @staticmethod
    def get_physics_sim_view():
        return MockPhysicsSimView()


# Mock isaacsim.core.simulation_manager
mock_sim_manager_module = ModuleType("isaacsim.core.simulation_manager")
mock_sim_manager_module.SimulationManager = MockSimulationManager
sys.modules["isaacsim"] = ModuleType("isaacsim")
sys.modules["isaacsim.core"] = ModuleType("isaacsim.core")
sys.modules["isaacsim.core.simulation_manager"] = mock_sim_manager_module

# Mock pxr (USD library)
sys.modules["pxr"] = MagicMock()
sys.modules["pxr.Usd"] = MagicMock()
sys.modules["pxr.UsdGeom"] = MagicMock()
sys.modules["pxr.UsdPhysics"] = MagicMock()
sys.modules["pxr.PhysxSchema"] = MagicMock()

# Mock omni module hierarchy (must be ModuleType for proper package behavior)
omni_mocks = [
    "omni",
    "omni.kit",
    "omni.kit.app",
    "omni.kit.commands",
    "omni.usd",
    "omni.client",
    "omni.timeline",
    "omni.physx",
    "omni.physx.scripts",
    "omni.physx.scripts.utils",
    "omni.physics",
    "omni.physics.tensors",
    "omni.physics.tensors.impl",
    "omni.physics.tensors.impl.api",
]
for mod_name in omni_mocks:
    mock = MagicMock()
    mock.__name__ = mod_name
    mock.__path__ = []
    mock.__package__ = mod_name
    sys.modules[mod_name] = mock

# Mock carb (needed by isaaclab.utils.assets)
mock_carb = MagicMock()
mock_carb.settings.get_settings.return_value.get.return_value = "/mock/path"
sys.modules["carb"] = mock_carb

# Mock isaaclab.sim module hierarchy (to avoid importing converters)
sim_mock = MagicMock()
sim_mock.find_first_matching_prim = MagicMock()
sim_mock.get_all_matching_child_prims = MagicMock(return_value=[])
sys.modules["isaaclab.sim"] = sim_mock
sys.modules["isaaclab.sim.utils"] = MagicMock()
sys.modules["isaaclab.sim.utils.stage"] = MagicMock()
sys.modules["isaaclab.sim.utils.queries"] = MagicMock()
sys.modules["isaaclab.sim.converters"] = MagicMock()

# Mock prettytable
sys.modules["prettytable"] = MagicMock()

# Mock WrenchComposer - import from mock_interfaces and patch into the module
from isaaclab.test.mock_interfaces.utils import MockWrenchComposer

mock_wrench_module = ModuleType("isaaclab.utils.wrench_composer")
mock_wrench_module.WrenchComposer = MockWrenchComposer
sys.modules["isaaclab.utils.wrench_composer"] = mock_wrench_module


# Mock base classes to avoid importing full isaaclab.assets package
class BaseArticulation:
    """Mock base class."""

    @property
    def device(self) -> str:
        return self._device


class BaseArticulationData:
    """Mock base class."""

    def __init__(self, root_view, device: str):
        self.device = device


mock_base_articulation = ModuleType("isaaclab.assets.articulation.base_articulation")
mock_base_articulation.BaseArticulation = BaseArticulation
sys.modules["isaaclab.assets.articulation.base_articulation"] = mock_base_articulation

mock_base_articulation_data = ModuleType("isaaclab.assets.articulation.base_articulation_data")
mock_base_articulation_data.BaseArticulationData = BaseArticulationData
sys.modules["isaaclab.assets.articulation.base_articulation_data"] = mock_base_articulation_data

# Mock ArticulationCfg
mock_cfg_module = ModuleType("isaaclab.assets.articulation.articulation_cfg")


class ArticulationCfg:
    """Mock ArticulationCfg for testing."""

    def __init__(
        self, prim_path: str = "/World/Robot", soft_joint_pos_limit_factor: float = 1.0, actuators: dict = None
    ):
        self.prim_path = prim_path
        self.soft_joint_pos_limit_factor = soft_joint_pos_limit_factor
        self.actuators = actuators or {}


mock_cfg_module.ArticulationCfg = ArticulationCfg
sys.modules["isaaclab.assets.articulation.articulation_cfg"] = mock_cfg_module

# Mock actuators module
mock_actuators = ModuleType("isaaclab.actuators")
mock_actuators.ActuatorBase = MagicMock()
mock_actuators.ActuatorBaseCfg = MagicMock()
mock_actuators.ImplicitActuator = MagicMock()
sys.modules["isaaclab.actuators"] = mock_actuators

# Mock utils modules - need to include all string functions used by isaaclab.utils.dict
mock_string_utils = ModuleType("isaaclab.utils.string")
mock_string_utils.resolve_matching_names = MagicMock(return_value=([], []))
mock_string_utils.callable_to_string = MagicMock(return_value="")
mock_string_utils.string_to_callable = MagicMock(return_value=None)
mock_string_utils.string_to_slice = MagicMock(return_value=slice(None))
sys.modules["isaaclab.utils.string"] = mock_string_utils

mock_types = ModuleType("isaaclab.utils.types")
mock_types.ArticulationActions = MagicMock()
sys.modules["isaaclab.utils.types"] = mock_types

mock_version = ModuleType("isaaclab.utils.version")


# Create a proper version tuple class that has .major, .minor, .patch attributes
class VersionTuple:
    def __init__(self, major, minor, patch):
        self.major = major
        self.minor = minor
        self.patch = patch

    def __iter__(self):
        return iter((self.major, self.minor, self.patch))


mock_version.get_isaac_sim_version = MagicMock(return_value=VersionTuple(4, 5, 0))
sys.modules["isaaclab.utils.version"] = mock_version

# Now import via importlib to bypass __init__.py
import importlib.util
from pathlib import Path

benchmark_dir = Path(__file__).resolve().parent

# Load ArticulationData
articulation_data_path = (
    benchmark_dir.parents[1] / "isaaclab_physx" / "assets" / "articulation" / "articulation_data.py"
)
spec = importlib.util.spec_from_file_location(
    "isaaclab_physx.assets.articulation.articulation_data", articulation_data_path
)
articulation_data_module = importlib.util.module_from_spec(spec)
sys.modules["isaaclab_physx.assets.articulation.articulation_data"] = articulation_data_module
spec.loader.exec_module(articulation_data_module)
ArticulationData = articulation_data_module.ArticulationData

# Load Articulation
articulation_path = benchmark_dir.parents[1] / "isaaclab_physx" / "assets" / "articulation" / "articulation.py"
spec = importlib.util.spec_from_file_location("isaaclab_physx.assets.articulation.articulation", articulation_path)
articulation_module = importlib.util.module_from_spec(spec)
sys.modules["isaaclab_physx.assets.articulation.articulation"] = articulation_module
spec.loader.exec_module(articulation_module)
Articulation = articulation_module.Articulation

# Import shared utilities from common module
# Import mock classes from PhysX test utilities
from isaaclab_physx.test.mock_interfaces.views import MockArticulationView

from isaaclab.test.benchmark import (
    BenchmarkConfig,
    MethodBenchmark,
    benchmark_method,
    export_results_csv,
    export_results_json,
    get_default_output_filename,
    get_hardware_info,
    make_tensor_body_ids,
    make_tensor_env_ids,
    make_tensor_joint_ids,
    print_hardware_info,
    print_results,
)

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Also suppress logging warnings (the body_acc_w deprecation warnings use logging)
import logging

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
def gen_root_link_pose_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_link_pose_to_sim."""
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_link_pose_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_link_pose_to_sim."""
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


# --- Root COM Pose ---
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


# --- Joint State ---
def gen_joint_state_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_state_to_sim."""
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_state_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_state_to_sim."""
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
    }


# --- Joint Position ---
def gen_joint_position_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_position_to_sim."""
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_position_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_position_to_sim."""
    return {
        "position": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
    }


# --- Joint Velocity ---
def gen_joint_velocity_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_velocity_to_sim."""
    return {
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_velocity_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_velocity_to_sim."""
    return {
        "velocity": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
    }


# --- Joint Stiffness ---
def gen_joint_stiffness_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_stiffness_to_sim."""
    return {
        "stiffness": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_stiffness_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_stiffness_to_sim."""
    return {
        "stiffness": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
    }


# --- Joint Damping ---
def gen_joint_damping_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_damping_to_sim."""
    return {
        "damping": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_damping_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_damping_to_sim."""
    return {
        "damping": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
    }


# --- Joint Position Limit ---
def gen_joint_position_limit_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_position_limit_to_sim."""
    # limits shape is (N, J, 2) where [:,: ,0] is lower and [:,:,1] is upper
    lower = torch.rand(config.num_instances, config.num_joints, 1, device=config.device, dtype=torch.float32) * -3.14
    upper = torch.rand(config.num_instances, config.num_joints, 1, device=config.device, dtype=torch.float32) * 3.14
    return {
        "limits": torch.cat([lower, upper], dim=-1),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_position_limit_torch_tensor(config: BenchmarkConfig) -> dict:
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
def gen_joint_velocity_limit_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_velocity_limit_to_sim."""
    return {
        "limits": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 10.0,
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_velocity_limit_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_velocity_limit_to_sim."""
    return {
        "limits": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 10.0,
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
    }


# --- Joint Effort Limit ---
def gen_joint_effort_limit_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_effort_limit_to_sim."""
    return {
        "limits": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 100.0
        ),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_effort_limit_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_effort_limit_to_sim."""
    return {
        "limits": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 100.0
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
    }


# --- Joint Armature ---
def gen_joint_armature_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_armature_to_sim."""
    return {
        "armature": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.1
        ),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_armature_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_armature_to_sim."""
    return {
        "armature": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.1
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
    }


# --- Joint Friction Coefficient ---
def gen_joint_friction_coefficient_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for write_joint_friction_coefficient_to_sim."""
    return {
        "joint_friction_coeff": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.5
        ),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_joint_friction_coefficient_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for write_joint_friction_coefficient_to_sim."""
    return {
        "joint_friction_coeff": (
            torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32) * 0.5
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
    }


# --- Set Joint Position Target ---
def gen_set_joint_position_target_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_joint_position_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_set_joint_position_target_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_joint_position_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
    }


# --- Set Joint Velocity Target ---
def gen_set_joint_velocity_target_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_joint_velocity_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_set_joint_velocity_target_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_joint_velocity_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
    }


# --- Set Joint Effort Target ---
def gen_set_joint_effort_target_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_joint_effort_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "joint_ids": list(range(config.num_joints)),
    }


def gen_set_joint_effort_target_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_joint_effort_target."""
    return {
        "target": torch.rand(config.num_instances, config.num_joints, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "joint_ids": make_tensor_joint_ids(config.num_joints, config.device),
    }


# --- Set Masses ---
def gen_set_masses_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_masses."""
    # Articulation masses shape is (N, B)
    return {
        "masses": torch.rand(config.num_instances, config.num_bodies, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_set_masses_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_masses."""
    # Articulation masses shape is (N, B)
    return {
        "masses": torch.rand(config.num_instances, config.num_bodies, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- Set CoMs ---
def gen_set_coms_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_coms."""
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_set_coms_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_coms."""
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- Set Inertias ---
def gen_set_inertias_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_inertias."""
    # Articulation inertias shape is (N, B, 9) - flattened 3x3 matrix
    return {
        "inertias": torch.rand(config.num_instances, config.num_bodies, 9, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_set_inertias_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_inertias."""
    # Articulation inertias shape is (N, B, 9) - flattened 3x3 matrix
    return {
        "inertias": torch.rand(config.num_instances, config.num_bodies, 9, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- Set External Force and Torque ---
def gen_set_external_force_and_torque_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_external_force_and_torque."""
    return {
        "forces": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_set_external_force_and_torque_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_external_force_and_torque."""
    return {
        "forces": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
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
            "torch_list": gen_root_state_torch_list,
            "torch_tensor": gen_root_state_torch_tensor,
        },
        category="root_state",
    ),
    MethodBenchmark(
        name="write_root_com_state_to_sim",
        method_name="write_root_com_state_to_sim",
        input_generators={
            "torch_list": gen_root_com_state_torch_list,
            "torch_tensor": gen_root_com_state_torch_tensor,
        },
        category="root_state",
    ),
    MethodBenchmark(
        name="write_root_link_state_to_sim",
        method_name="write_root_link_state_to_sim",
        input_generators={
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
            "torch_list": gen_root_link_pose_torch_list,
            "torch_tensor": gen_root_link_pose_torch_tensor,
        },
        category="root_pose",
    ),
    MethodBenchmark(
        name="write_root_com_pose_to_sim",
        method_name="write_root_com_pose_to_sim",
        input_generators={
            "torch_list": gen_root_com_pose_torch_list,
            "torch_tensor": gen_root_com_pose_torch_tensor,
        },
        category="root_pose",
    ),
    MethodBenchmark(
        name="write_root_link_velocity_to_sim",
        method_name="write_root_link_velocity_to_sim",
        input_generators={
            "torch_list": gen_root_link_velocity_torch_list,
            "torch_tensor": gen_root_link_velocity_torch_tensor,
        },
        category="root_velocity",
    ),
    MethodBenchmark(
        name="write_root_com_velocity_to_sim",
        method_name="write_root_com_velocity_to_sim",
        input_generators={
            "torch_list": gen_root_com_velocity_torch_list,
            "torch_tensor": gen_root_com_velocity_torch_tensor,
        },
        category="root_velocity",
    ),
    # --- Joint State ---
    MethodBenchmark(
        name="write_joint_state_to_sim",
        method_name="write_joint_state_to_sim",
        input_generators={
            "torch_list": gen_joint_state_torch_list,
            "torch_tensor": gen_joint_state_torch_tensor,
        },
        category="joint_state",
    ),
    MethodBenchmark(
        name="write_joint_position_to_sim",
        method_name="write_joint_position_to_sim",
        input_generators={
            "torch_list": gen_joint_position_torch_list,
            "torch_tensor": gen_joint_position_torch_tensor,
        },
        category="joint_state",
    ),
    MethodBenchmark(
        name="write_joint_velocity_to_sim",
        method_name="write_joint_velocity_to_sim",
        input_generators={
            "torch_list": gen_joint_velocity_torch_list,
            "torch_tensor": gen_joint_velocity_torch_tensor,
        },
        category="joint_state",
    ),
    # --- Joint Params ---
    MethodBenchmark(
        name="write_joint_stiffness_to_sim",
        method_name="write_joint_stiffness_to_sim",
        input_generators={
            "torch_list": gen_joint_stiffness_torch_list,
            "torch_tensor": gen_joint_stiffness_torch_tensor,
        },
        category="joint_params",
    ),
    MethodBenchmark(
        name="write_joint_damping_to_sim",
        method_name="write_joint_damping_to_sim",
        input_generators={
            "torch_list": gen_joint_damping_torch_list,
            "torch_tensor": gen_joint_damping_torch_tensor,
        },
        category="joint_params",
    ),
    MethodBenchmark(
        name="write_joint_position_limit_to_sim",
        method_name="write_joint_position_limit_to_sim",
        input_generators={
            "torch_list": gen_joint_position_limit_torch_list,
            "torch_tensor": gen_joint_position_limit_torch_tensor,
        },
        category="joint_params",
    ),
    MethodBenchmark(
        name="write_joint_velocity_limit_to_sim",
        method_name="write_joint_velocity_limit_to_sim",
        input_generators={
            "torch_list": gen_joint_velocity_limit_torch_list,
            "torch_tensor": gen_joint_velocity_limit_torch_tensor,
        },
        category="joint_params",
    ),
    MethodBenchmark(
        name="write_joint_effort_limit_to_sim",
        method_name="write_joint_effort_limit_to_sim",
        input_generators={
            "torch_list": gen_joint_effort_limit_torch_list,
            "torch_tensor": gen_joint_effort_limit_torch_tensor,
        },
        category="joint_params",
    ),
    MethodBenchmark(
        name="write_joint_armature_to_sim",
        method_name="write_joint_armature_to_sim",
        input_generators={
            "torch_list": gen_joint_armature_torch_list,
            "torch_tensor": gen_joint_armature_torch_tensor,
        },
        category="joint_params",
    ),
    MethodBenchmark(
        name="write_joint_friction_coefficient_to_sim",
        method_name="write_joint_friction_coefficient_to_sim",
        input_generators={
            "torch_list": gen_joint_friction_coefficient_torch_list,
            "torch_tensor": gen_joint_friction_coefficient_torch_tensor,
        },
        category="joint_params",
    ),
    # --- Joint Targets ---
    MethodBenchmark(
        name="set_joint_position_target",
        method_name="set_joint_position_target",
        input_generators={
            "torch_list": gen_set_joint_position_target_torch_list,
            "torch_tensor": gen_set_joint_position_target_torch_tensor,
        },
        category="joint_targets",
    ),
    MethodBenchmark(
        name="set_joint_velocity_target",
        method_name="set_joint_velocity_target",
        input_generators={
            "torch_list": gen_set_joint_velocity_target_torch_list,
            "torch_tensor": gen_set_joint_velocity_target_torch_tensor,
        },
        category="joint_targets",
    ),
    MethodBenchmark(
        name="set_joint_effort_target",
        method_name="set_joint_effort_target",
        input_generators={
            "torch_list": gen_set_joint_effort_target_torch_list,
            "torch_tensor": gen_set_joint_effort_target_torch_tensor,
        },
        category="joint_targets",
    ),
    # --- Body Properties ---
    MethodBenchmark(
        name="set_masses",
        method_name="set_masses",
        input_generators={
            "torch_list": gen_set_masses_torch_list,
            "torch_tensor": gen_set_masses_torch_tensor,
        },
        category="body_props",
    ),
    MethodBenchmark(
        name="set_coms",
        method_name="set_coms",
        input_generators={
            "torch_list": gen_set_coms_torch_list,
            "torch_tensor": gen_set_coms_torch_tensor,
        },
        category="body_props",
    ),
    MethodBenchmark(
        name="set_inertias",
        method_name="set_inertias",
        input_generators={
            "torch_list": gen_set_inertias_torch_list,
            "torch_tensor": gen_set_inertias_torch_tensor,
        },
        category="body_props",
    ),
    MethodBenchmark(
        name="set_external_force_and_torque",
        method_name="set_external_force_and_torque",
        input_generators={
            "torch_list": gen_set_external_force_and_torque_torch_list,
            "torch_tensor": gen_set_external_force_and_torque_torch_tensor,
        },
        category="external_wrench",
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
    print(f"Device: {config.device}")
    print(f"Iterations: {config.num_iterations}, Warmup: {config.warmup_steps}")
    print(f"Modes: {modes_to_run if modes_to_run else 'All available'}")

    print(f"\nBenchmarking {len(BENCHMARKS)} methods...")
    for i, benchmark in enumerate(BENCHMARKS):
        method = getattr(articulation, benchmark.method_name, None)

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
    parser = argparse.ArgumentParser(description="Benchmark Articulation methods (PhysX backend).")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--num_instances", type=int, default=4096, help="Number of instances")
    parser.add_argument("--num_bodies", type=int, default=12, help="Number of bodies")
    parser.add_argument("--num_joints", type=int, default=11, help="Number of joints")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--mode", type=str, default="all", help="Benchmark mode (all, torch_list, torch_tensor)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON filename")
    parser.add_argument("--no_csv", action="store_true", help="Disable CSV output")

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

    results = run_benchmark(config)

    hardware_info = get_hardware_info()
    print_hardware_info(hardware_info)
    print_results(results)

    if args.output:
        json_filename = args.output
    else:
        json_filename = get_default_output_filename("articulation_benchmark")

    export_results_json(results, config, hardware_info, json_filename)

    if not args.no_csv:
        csv_filename = json_filename.replace(".json", ".csv")
        export_results_csv(results, csv_filename)
