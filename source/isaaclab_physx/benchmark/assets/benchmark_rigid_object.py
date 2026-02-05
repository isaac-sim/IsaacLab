# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for RigidObject class (PhysX backend).

This module provides a benchmarking framework to measure the performance of setter and writer
methods in the RigidObject class. Each method is benchmarked under two scenarios:

1. **Torch List**: Inputs are PyTorch tensors with list indices.
2. **Torch Tensor**: Inputs are PyTorch tensors with tensor indices.

Usage:
    python benchmark_rigid_object.py [--num_iterations N] [--warmup_steps W] [--num_instances I]

Example:
    python benchmark_rigid_object.py --num_iterations 1000 --warmup_steps 10
    python benchmark_rigid_object.py --mode torch_list  # Only run list-based benchmarks
    python benchmark_rigid_object.py --mode torch_tensor  # Only run tensor-based benchmarks
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
# Mock Setup - Must happen BEFORE importing RigidObject
# =============================================================================


class MockPhysicsSimView:
    """Simple mock for the physics simulation view."""

    def get_gravity(self):
        return (0.0, 0.0, -9.81)


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
sys.modules["isaaclab.sim.converters"] = MagicMock()

# Mock WrenchComposer - import from mock_interfaces and patch into the module
from isaaclab.test.mock_interfaces.utils import MockWrenchComposer

mock_wrench_module = ModuleType("isaaclab.utils.wrench_composer")
mock_wrench_module.WrenchComposer = MockWrenchComposer
sys.modules["isaaclab.utils.wrench_composer"] = mock_wrench_module


# Mock base classes to avoid importing full isaaclab.assets package
class BaseRigidObject:
    """Mock base class."""

    @property
    def device(self) -> str:
        return self._device


class BaseRigidObjectData:
    """Mock base class."""

    def __init__(self, root_view, device: str):
        self.device = device


mock_base_rigid_object = ModuleType("isaaclab.assets.rigid_object.base_rigid_object")
mock_base_rigid_object.BaseRigidObject = BaseRigidObject
sys.modules["isaaclab.assets.rigid_object.base_rigid_object"] = mock_base_rigid_object

mock_base_rigid_object_data = ModuleType("isaaclab.assets.rigid_object.base_rigid_object_data")
mock_base_rigid_object_data.BaseRigidObjectData = BaseRigidObjectData
sys.modules["isaaclab.assets.rigid_object.base_rigid_object_data"] = mock_base_rigid_object_data

# Mock RigidObjectCfg
mock_cfg_module = ModuleType("isaaclab.assets.rigid_object.rigid_object_cfg")
mock_cfg_module.RigidObjectCfg = type("RigidObjectCfg", (), {"prim_path": None})
sys.modules["isaaclab.assets.rigid_object.rigid_object_cfg"] = mock_cfg_module

# Now import via importlib to bypass __init__.py
import importlib.util
from pathlib import Path

benchmark_dir = Path(__file__).resolve().parent

# Load RigidObjectData
rigid_object_data_path = (
    benchmark_dir.parents[1] / "isaaclab_physx" / "assets" / "rigid_object" / "rigid_object_data.py"
)
spec = importlib.util.spec_from_file_location(
    "isaaclab_physx.assets.rigid_object.rigid_object_data", rigid_object_data_path
)
rigid_object_data_module = importlib.util.module_from_spec(spec)
sys.modules["isaaclab_physx.assets.rigid_object.rigid_object_data"] = rigid_object_data_module
spec.loader.exec_module(rigid_object_data_module)
RigidObjectData = rigid_object_data_module.RigidObjectData

# Load RigidObject
rigid_object_path = benchmark_dir.parents[1] / "isaaclab_physx" / "assets" / "rigid_object" / "rigid_object.py"
spec = importlib.util.spec_from_file_location("isaaclab_physx.assets.rigid_object.rigid_object", rigid_object_path)
rigid_object_module = importlib.util.module_from_spec(spec)
sys.modules["isaaclab_physx.assets.rigid_object.rigid_object"] = rigid_object_module
spec.loader.exec_module(rigid_object_module)
RigidObject = rigid_object_module.RigidObject


# Simple RigidObjectCfg for testing
class RigidObjectCfg:
    def __init__(self, prim_path: str = "/World/Object"):
        self.prim_path = prim_path


# Import shared utilities from common module
# Import mock classes from PhysX test utilities
from isaaclab_physx.test.mock_interfaces.views import MockRigidBodyView

from isaaclab.test.benchmark import (
    BenchmarkConfig,
    MethodBenchmark,
    benchmark_method,
    export_results_csv,
    export_results_json,
    get_default_output_filename,
    get_hardware_info,
    make_tensor_env_ids,
    print_hardware_info,
    print_results,
)

# Suppress deprecation warnings during benchmarking
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress isaaclab logging (deprecation warnings)
import logging

logging.getLogger("isaaclab_physx").setLevel(logging.ERROR)
logging.getLogger("isaaclab").setLevel(logging.ERROR)


def create_test_rigid_object(
    num_instances: int = 2,
    num_bodies: int = 1,
    device: str = "cuda:0",
) -> tuple[RigidObject, MockRigidBodyView, MagicMock]:
    """Create a test RigidObject instance with mocked dependencies."""
    rigid_object = object.__new__(RigidObject)

    rigid_object.cfg = RigidObjectCfg(
        prim_path="/World/Object",
    )

    # Create PhysX mock view
    mock_view = MockRigidBodyView(
        count=num_instances,
        device=device,
    )
    mock_view.set_random_mock_data()

    # Set up attributes required before _create_buffers
    object.__setattr__(rigid_object, "_root_view", mock_view)
    object.__setattr__(rigid_object, "_device", device)

    # Create RigidObjectData instance (mocks already set up at module level)
    data = RigidObjectData(mock_view, device)
    object.__setattr__(rigid_object, "_data", data)

    # Call _create_buffers to set up all internal buffers and wrench composers
    rigid_object._create_buffers()

    return rigid_object, mock_view, None


# =============================================================================
# Input Generators (Torch-only for PhysX backend)
# =============================================================================


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


# --- Root Pose (Deprecated) ---
def gen_root_pose_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_pose_to_sim."""
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_pose_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_pose_to_sim."""
    return {
        "root_pose": torch.rand(config.num_instances, 7, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


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


# --- Root Velocity (Deprecated) ---
def gen_root_velocity_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list env_ids for write_root_velocity_to_sim."""
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_root_velocity_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor env_ids for write_root_velocity_to_sim."""
    return {
        "root_velocity": torch.rand(config.num_instances, 6, device=config.device, dtype=torch.float32),
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


# --- Masses ---
def gen_masses_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_masses."""
    # RigidObject has only 1 body, don't pass body_ids to avoid advanced indexing issues
    return {
        "masses": torch.rand(config.num_instances, config.num_bodies, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_masses_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_masses."""
    return {
        "masses": torch.rand(config.num_instances, config.num_bodies, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


# --- CoMs ---
def gen_coms_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_coms."""
    # RigidObject has only 1 body, don't pass body_ids to avoid advanced indexing issues
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_coms_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_coms."""
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


# --- Inertias ---
def gen_inertias_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_inertias."""
    # RigidObject has only 1 body, don't pass body_ids to avoid advanced indexing issues
    return {
        "inertias": torch.rand(
            config.num_instances, config.num_bodies, 3, 3, device=config.device, dtype=torch.float32
        ),
        "env_ids": list(range(config.num_instances)),
    }


def gen_inertias_torch_tensor(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with tensor ids for set_inertias."""
    return {
        "inertias": torch.rand(
            config.num_instances, config.num_bodies, 3, 3, device=config.device, dtype=torch.float32
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


# --- External Wrench ---
def gen_external_force_and_torque_torch_list(config: BenchmarkConfig) -> dict:
    """Generate Torch inputs with list ids for set_external_force_and_torque."""
    # Note: body_ids is not used by set_external_force_and_torque (it uses internal _ALL_BODY_INDICES_WP)
    return {
        "forces": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_external_force_and_torque_torch_tensor(config: BenchmarkConfig) -> dict:
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
        name="write_root_pose_to_sim",
        method_name="write_root_pose_to_sim",
        input_generators={
            "torch_list": gen_root_pose_torch_list,
            "torch_tensor": gen_root_pose_torch_tensor,
        },
        category="root_pose",
    ),
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
        name="write_root_velocity_to_sim",
        method_name="write_root_velocity_to_sim",
        input_generators={
            "torch_list": gen_root_velocity_torch_list,
            "torch_tensor": gen_root_velocity_torch_tensor,
        },
        category="root_velocity",
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
    # --- Body Properties ---
    MethodBenchmark(
        name="set_masses",
        method_name="set_masses",
        input_generators={
            "torch_list": gen_masses_torch_list,
            "torch_tensor": gen_masses_torch_tensor,
        },
        category="body_props",
    ),
    MethodBenchmark(
        name="set_coms",
        method_name="set_coms",
        input_generators={
            "torch_list": gen_coms_torch_list,
            "torch_tensor": gen_coms_torch_tensor,
        },
        category="body_props",
    ),
    MethodBenchmark(
        name="set_inertias",
        method_name="set_inertias",
        input_generators={
            "torch_list": gen_inertias_torch_list,
            "torch_tensor": gen_inertias_torch_tensor,
        },
        category="body_props",
    ),
    MethodBenchmark(
        name="set_external_force_and_torque",
        method_name="set_external_force_and_torque",
        input_generators={
            "torch_list": gen_external_force_and_torque_torch_list,
            "torch_tensor": gen_external_force_and_torque_torch_tensor,
        },
        category="body_props",
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

    print(f"Benchmarking RigidObject (PhysX) with {config.num_instances} instances, {config.num_bodies} bodies...")
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
    parser = argparse.ArgumentParser(description="Benchmark RigidObject methods (PhysX backend).")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--num_instances", type=int, default=4096, help="Number of instances")
    parser.add_argument("--num_bodies", type=int, default=1, help="Number of bodies")
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
        num_joints=0,
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
        export_results_csv(results, csv_filename)
