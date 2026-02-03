# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common test utilities for isaaclab_newton tests.

This package provides shared mock classes and benchmarking utilities used across
different test modules (articulation, rigid_object, etc.).
"""

from .benchmark_core import (
    BenchmarkConfig,
    BenchmarkResult,
    MethodBenchmark,
    make_warp_body_mask,
    make_warp_env_mask,
    make_warp_joint_mask,
)
from .benchmark_io import (
    export_results_csv,
    export_results_json,
    get_default_output_filename,
    get_git_info,
    get_hardware_info,
    print_hardware_info,
    print_results,
)
from .mock_newton import MockNewtonArticulationView, MockNewtonModel, MockWrenchComposer, create_mock_newton_manager

__all__ = [
    # Mock classes
    "MockNewtonModel",
    "MockNewtonArticulationView",
    "MockWrenchComposer",
    "create_mock_newton_manager",
    # Benchmark core
    "BenchmarkConfig",
    "BenchmarkResult",
    "MethodBenchmark",
    "make_warp_env_mask",
    "make_warp_body_mask",
    "make_warp_joint_mask",
    # Benchmark I/O
    "get_git_info",
    "get_hardware_info",
    "print_hardware_info",
    "print_results",
    "export_results_json",
    "export_results_csv",
    "get_default_output_filename",
]
