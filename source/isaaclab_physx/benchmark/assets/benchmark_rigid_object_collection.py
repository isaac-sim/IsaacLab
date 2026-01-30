# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for RigidObjectCollection class (PhysX backend).

This module provides a benchmarking framework to measure the performance of setter and writer
methods in the RigidObjectCollection class.

Usage:
    python benchmark_rigid_object_collection.py [--num_iterations N] [--num_instances I] [--num_bodies B]

Example:
    python benchmark_rigid_object_collection.py --num_iterations 1000 --num_instances 4096 --num_bodies 4
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
# Mock Setup - Must happen BEFORE importing RigidObjectCollection
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

# Mock omni module hierarchy
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

# Mock carb
mock_carb = MagicMock()
mock_carb.settings.get_settings.return_value.get.return_value = "/mock/path"
sys.modules["carb"] = mock_carb

# Mock isaaclab.sim module hierarchy
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


# Mock base classes
class BaseRigidObjectCollection:
    """Mock base class."""

    @property
    def device(self) -> str:
        return self._device


class BaseRigidObjectCollectionData:
    """Mock base class."""

    def __init__(self, root_view, num_bodies: int, device: str):
        self.device = device


mock_base_collection = ModuleType("isaaclab.assets.rigid_object_collection.base_rigid_object_collection")
mock_base_collection.BaseRigidObjectCollection = BaseRigidObjectCollection
sys.modules["isaaclab.assets.rigid_object_collection.base_rigid_object_collection"] = mock_base_collection

mock_base_collection_data = ModuleType("isaaclab.assets.rigid_object_collection.base_rigid_object_collection_data")
mock_base_collection_data.BaseRigidObjectCollectionData = BaseRigidObjectCollectionData
sys.modules["isaaclab.assets.rigid_object_collection.base_rigid_object_collection_data"] = mock_base_collection_data

# Now import via importlib
import importlib.util
from pathlib import Path

benchmark_dir = Path(__file__).resolve().parent

# Load RigidObjectCollectionData
data_path = (
    benchmark_dir.parents[1]
    / "isaaclab_physx"
    / "assets"
    / "rigid_object_collection"
    / "rigid_object_collection_data.py"
)
spec = importlib.util.spec_from_file_location(
    "isaaclab_physx.assets.rigid_object_collection.rigid_object_collection_data", data_path
)
data_module = importlib.util.module_from_spec(spec)
sys.modules["isaaclab_physx.assets.rigid_object_collection.rigid_object_collection_data"] = data_module
spec.loader.exec_module(data_module)
RigidObjectCollectionData = data_module.RigidObjectCollectionData

# Load RigidObjectCollection
collection_path = (
    benchmark_dir.parents[1] / "isaaclab_physx" / "assets" / "rigid_object_collection" / "rigid_object_collection.py"
)
spec = importlib.util.spec_from_file_location(
    "isaaclab_physx.assets.rigid_object_collection.rigid_object_collection", collection_path
)
collection_module = importlib.util.module_from_spec(spec)
sys.modules["isaaclab_physx.assets.rigid_object_collection.rigid_object_collection"] = collection_module
spec.loader.exec_module(collection_module)
RigidObjectCollection = collection_module.RigidObjectCollection

# Import shared utilities
# Import mock view
from isaaclab_physx.test.mock_interfaces.views import MockRigidBodyView

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
    print_hardware_info,
    print_results,
)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress isaaclab logging (deprecation warnings)
import logging

logging.getLogger("isaaclab_physx").setLevel(logging.ERROR)
logging.getLogger("isaaclab").setLevel(logging.ERROR)


# Simple config class
class RigidObjectCollectionCfg:
    def __init__(self, prim_path: str = "/World/Collection"):
        self.prim_path = prim_path


def create_test_collection(
    num_instances: int = 2,
    num_bodies: int = 4,
    device: str = "cuda:0",
) -> tuple[RigidObjectCollection, MockRigidBodyView]:
    """Create a test RigidObjectCollection instance with mocked dependencies."""
    object_names = [f"object_{i}" for i in range(num_bodies)]

    collection = object.__new__(RigidObjectCollection)

    collection.cfg = RigidObjectCollectionCfg(prim_path="/World/Collection")

    # Create PhysX mock view (total count = num_instances * num_bodies)
    total_count = num_instances * num_bodies
    mock_view = MockRigidBodyView(
        count=total_count,
        device=device,
    )
    mock_view.set_random_mock_data()

    # Set up attributes required before _create_buffers
    object.__setattr__(collection, "_root_view", mock_view)
    object.__setattr__(collection, "_device", device)
    object.__setattr__(collection, "_body_names_list", object_names)

    # Create RigidObjectCollectionData instance
    data = RigidObjectCollectionData(mock_view, num_bodies, device)
    data.object_names = object_names
    object.__setattr__(collection, "_data", data)

    # Call _create_buffers to set up all internal buffers and wrench composers
    collection._create_buffers()

    return collection, mock_view


# =============================================================================
# Input Generators
# =============================================================================


def gen_body_state_torch_list(config: BenchmarkConfig) -> dict:
    return {
        "body_states": torch.rand(
            config.num_instances, config.num_bodies, 13, device=config.device, dtype=torch.float32
        ),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_body_state_torch_tensor(config: BenchmarkConfig) -> dict:
    return {
        "body_states": torch.rand(
            config.num_instances, config.num_bodies, 13, device=config.device, dtype=torch.float32
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


def gen_body_pose_torch_list(config: BenchmarkConfig) -> dict:
    return {
        "body_poses": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_body_pose_torch_tensor(config: BenchmarkConfig) -> dict:
    return {
        "body_poses": torch.rand(config.num_instances, config.num_bodies, 7, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


def gen_body_velocity_torch_list(config: BenchmarkConfig) -> dict:
    return {
        "body_velocities": torch.rand(
            config.num_instances, config.num_bodies, 6, device=config.device, dtype=torch.float32
        ),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_body_velocity_torch_tensor(config: BenchmarkConfig) -> dict:
    return {
        "body_velocities": torch.rand(
            config.num_instances, config.num_bodies, 6, device=config.device, dtype=torch.float32
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- External Force and Torque ---
def gen_external_force_and_torque_torch_list(config: BenchmarkConfig) -> dict:
    return {
        "forces": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
    }


def gen_external_force_and_torque_torch_tensor(config: BenchmarkConfig) -> dict:
    return {
        "forces": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "torques": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
    }


# --- Masses ---
def gen_masses_torch_list(config: BenchmarkConfig) -> dict:
    return {
        "masses": torch.rand(config.num_instances, config.num_bodies, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_masses_torch_tensor(config: BenchmarkConfig) -> dict:
    return {
        "masses": torch.rand(config.num_instances, config.num_bodies, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- CoMs ---
def gen_coms_torch_list(config: BenchmarkConfig) -> dict:
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_coms_torch_tensor(config: BenchmarkConfig) -> dict:
    return {
        "coms": torch.rand(config.num_instances, config.num_bodies, 3, device=config.device, dtype=torch.float32),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# --- Inertias ---
def gen_inertias_torch_list(config: BenchmarkConfig) -> dict:
    return {
        "inertias": torch.rand(
            config.num_instances, config.num_bodies, 3, 3, device=config.device, dtype=torch.float32
        ),
        "env_ids": list(range(config.num_instances)),
        "body_ids": list(range(config.num_bodies)),
    }


def gen_inertias_torch_tensor(config: BenchmarkConfig) -> dict:
    return {
        "inertias": torch.rand(
            config.num_instances, config.num_bodies, 3, 3, device=config.device, dtype=torch.float32
        ),
        "env_ids": make_tensor_env_ids(config.num_instances, config.device),
        "body_ids": make_tensor_body_ids(config.num_bodies, config.device),
    }


# =============================================================================
# Benchmarks
# =============================================================================

BENCHMARK_DEPENDENCIES = {}

BENCHMARKS = [
    # --- Body State ---
    MethodBenchmark(
        name="write_body_state_to_sim",
        method_name="write_body_state_to_sim",
        input_generators={
            "torch_list": gen_body_state_torch_list,
            "torch_tensor": gen_body_state_torch_tensor,
        },
        category="body_state",
    ),
    MethodBenchmark(
        name="write_body_link_state_to_sim",
        method_name="write_body_link_state_to_sim",
        input_generators={
            "torch_list": gen_body_state_torch_list,
            "torch_tensor": gen_body_state_torch_tensor,
        },
        category="body_state",
    ),
    MethodBenchmark(
        name="write_body_com_state_to_sim",
        method_name="write_body_com_state_to_sim",
        input_generators={
            "torch_list": gen_body_state_torch_list,
            "torch_tensor": gen_body_state_torch_tensor,
        },
        category="body_state",
    ),
    # --- Body Pose ---
    MethodBenchmark(
        name="write_body_pose_to_sim",
        method_name="write_body_pose_to_sim",
        input_generators={
            "torch_list": gen_body_pose_torch_list,
            "torch_tensor": gen_body_pose_torch_tensor,
        },
        category="body_pose",
    ),
    MethodBenchmark(
        name="write_body_link_pose_to_sim",
        method_name="write_body_link_pose_to_sim",
        input_generators={
            "torch_list": gen_body_pose_torch_list,
            "torch_tensor": gen_body_pose_torch_tensor,
        },
        category="body_pose",
    ),
    MethodBenchmark(
        name="write_body_com_pose_to_sim",
        method_name="write_body_com_pose_to_sim",
        input_generators={
            "torch_list": gen_body_pose_torch_list,
            "torch_tensor": gen_body_pose_torch_tensor,
        },
        category="body_pose",
    ),
    # --- Body Velocity ---
    MethodBenchmark(
        name="write_body_velocity_to_sim",
        method_name="write_body_velocity_to_sim",
        input_generators={
            "torch_list": gen_body_velocity_torch_list,
            "torch_tensor": gen_body_velocity_torch_tensor,
        },
        category="body_velocity",
    ),
    MethodBenchmark(
        name="write_body_link_velocity_to_sim",
        method_name="write_body_link_velocity_to_sim",
        input_generators={
            "torch_list": gen_body_velocity_torch_list,
            "torch_tensor": gen_body_velocity_torch_tensor,
        },
        category="body_velocity",
    ),
    MethodBenchmark(
        name="write_body_com_velocity_to_sim",
        method_name="write_body_com_velocity_to_sim",
        input_generators={
            "torch_list": gen_body_velocity_torch_list,
            "torch_tensor": gen_body_velocity_torch_tensor,
        },
        category="body_velocity",
    ),
    # --- External Force and Torque ---
    MethodBenchmark(
        name="set_external_force_and_torque",
        method_name="set_external_force_and_torque",
        input_generators={
            "torch_list": gen_external_force_and_torque_torch_list,
            "torch_tensor": gen_external_force_and_torque_torch_tensor,
        },
        category="external_wrench",
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
]


def run_benchmark(config: BenchmarkConfig):
    """Run all benchmarks."""
    results = []

    modes_to_run = []
    if isinstance(config.mode, str):
        if config.mode == "all":
            modes_to_run = None
        else:
            modes_to_run = [config.mode]
    elif isinstance(config.mode, list):
        modes_to_run = config.mode

    collection, mock_view = create_test_collection(
        num_instances=config.num_instances,
        num_bodies=config.num_bodies,
        device=config.device,
    )

    print(
        f"Benchmarking RigidObjectCollection (PhysX) with {config.num_instances} instances, "
        f"{config.num_bodies} bodies..."
    )
    print(f"Device: {config.device}")
    print(f"Iterations: {config.num_iterations}, Warmup: {config.warmup_steps}")
    print(f"Modes: {modes_to_run if modes_to_run else 'All available'}")

    print(f"\nBenchmarking {len(BENCHMARKS)} methods...")
    for i, benchmark in enumerate(BENCHMARKS):
        method = getattr(collection, benchmark.method_name, None)

        available_modes = list(benchmark.input_generators.keys())
        current_modes = modes_to_run if modes_to_run is not None else available_modes
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
    parser = argparse.ArgumentParser(description="Benchmark RigidObjectCollection methods (PhysX backend).")
    parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--num_instances", type=int, default=4096, help="Number of instances")
    parser.add_argument("--num_bodies", type=int, default=4, help="Number of bodies per instance")
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
        json_filename = get_default_output_filename("rigid_object_collection_benchmark")

    export_results_json(results, config, hardware_info, json_filename)

    if not args.no_csv:
        csv_filename = json_filename.replace(".json", ".csv")
        export_results_csv(results, csv_filename)
