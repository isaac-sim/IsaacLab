# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Micro-benchmarking framework for RigidObject class (Newton backend).

This module provides a benchmarking framework to measure the performance of setter and writer
methods in the RigidObject class. Each method is benchmarked under three scenarios:

1. **Torch List**: Inputs are PyTorch tensors with list indices (via deprecated wrappers).
2. **Torch Tensor**: Inputs are PyTorch tensors with tensor indices (via deprecated wrappers).
3. **Warp Mask**: Inputs are warp arrays with boolean masks (via ``_mask`` methods).

Usage:
    python benchmark_rigid_object.py [--num_iterations N] [--warmup_steps W]
        [--num_instances I] [--num_bodies B]

Example:
    python benchmark_rigid_object.py --num_iterations 1000 --warmup_steps 10
    python benchmark_rigid_object.py --mode torch_list  # Only run list-based benchmarks
    python benchmark_rigid_object.py --mode warp_mask   # Only run warp mask benchmarks
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Benchmark RigidObject methods (Newton backend).")
parser.add_argument("--num_iterations", type=int, default=1000, help="Number of iterations")
parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup steps")
parser.add_argument("--num_instances", type=int, default=4096, help="Number of instances")
parser.add_argument("--num_bodies", type=int, default=1, help="Number of bodies")
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

from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
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


def make_tensor_body_ids(num_bodies: int, device: str) -> torch.Tensor:
    """Create a tensor of body IDs."""
    return torch.arange(num_bodies, dtype=torch.int32, device=device)


# =============================================================================
# Test RigidObject Factory
# =============================================================================


def create_test_rigid_object(
    num_instances: int = 2,
    num_bodies: int = 1,
    device: str = "cuda:0",
):
    """Create a test RigidObject instance with mocked dependencies."""
    from isaaclab_newton.assets.rigid_object.rigid_object import RigidObject

    body_names = [f"body_{i}" for i in range(num_bodies)]

    rigid_object = object.__new__(RigidObject)

    rigid_object.cfg = RigidObjectCfg(
        prim_path="/World/Object",
    )

    # Create Newton mock view
    mock_view = MockNewtonArticulationView(
        num_instances=num_instances,
        num_bodies=num_bodies,
        num_joints=0,
        device=device,
        joint_names=[],
        body_names=body_names,
    )
    mock_view.set_random_mock_data()
    mock_view._noop_setters = True

    object.__setattr__(rigid_object, "_root_view", mock_view)
    object.__setattr__(rigid_object, "_device", device)
    object.__setattr__(rigid_object, "_check_shapes", not args.no_shape_checks)

    # Create RigidObjectData instance (NewtonManager already mocked at call site)
    from isaaclab_newton.assets.rigid_object.rigid_object_data import RigidObjectData

    data = RigidObjectData(mock_view, device)
    object.__setattr__(rigid_object, "_data", data)

    # Create mock wrench composers
    mock_inst_wrench = MockWrenchComposer(rigid_object)
    mock_perm_wrench = MockWrenchComposer(rigid_object)
    object.__setattr__(rigid_object, "_instantaneous_wrench_composer", mock_inst_wrench)
    object.__setattr__(rigid_object, "_permanent_wrench_composer", mock_perm_wrench)

    # Set up other required attributes
    object.__setattr__(rigid_object, "actuators", {})
    object.__setattr__(rigid_object, "_ALL_INDICES", wp.array(np.arange(num_instances, dtype=np.int32), device=device))
    object.__setattr__(
        rigid_object, "_ALL_BODY_INDICES", wp.array(np.arange(num_bodies, dtype=np.int32), device=device)
    )
    object.__setattr__(rigid_object, "_ALL_ENV_MASK", wp.ones((num_instances,), dtype=wp.bool, device=device))
    object.__setattr__(rigid_object, "_ALL_BODY_MASK", wp.ones((num_bodies,), dtype=wp.bool, device=device))

    # set information about rigid body into data
    data.body_names = body_names

    return rigid_object, mock_view


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

    Only env_ids are subsetted — body_ids remain full-range.
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
        num_joints=0,
        device=args.device,
        mode=args.mode,
    )

    # Patch the NewtonManager for both rigid_object and rigid_object_data modules
    with (
        create_mock_newton_manager(
            "isaaclab_newton.assets.rigid_object.rigid_object_data.SimulationManager",
            gravity=(0.0, 0.0, -9.81),
        ),
        create_mock_newton_manager(
            "isaaclab_newton.assets.rigid_object.rigid_object.SimulationManager",
            gravity=(0.0, 0.0, -9.81),
        ),
    ):
        # Create the test rigid object
        rigid_object, _ = create_test_rigid_object(
            num_instances=config.num_instances,
            num_bodies=config.num_bodies,
            device=config.device,
        )

        print(f"Benchmarking RigidObject (Newton) with {config.num_instances} instances, {config.num_bodies} bodies...")

        # Create runner and run benchmarks
        runner = MethodBenchmarkRunner(
            benchmark_name="newton_rigid_object_benchmark",
            config=config,
            backend_type=args.backend,
            output_path=args.output_dir,
            use_recorders=True,
        )

        runner.run_benchmarks(BENCHMARKS, rigid_object)

        print("\n" + "=" * 80)
        print("Fill-Ratio Benchmarks (env_ids at 5%, 95%, 100% fill)")
        print("=" * 80)

        runner.run_benchmarks(FILL_BENCHMARKS, rigid_object)
        runner.finalize()

    # Close the simulation app
    simulation_app.close()


if __name__ == "__main__":
    main()
