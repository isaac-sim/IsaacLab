# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmarking utilities for IsaacLab.

This package provides benchmarking utilities used across different test modules.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .benchmark_core import BaseIsaacLabBenchmark, get_default_output_filename
    from .benchmark_monitor import BenchmarkMonitor
    from .method_benchmark import MethodBenchmarkDefinition, MethodBenchmarkRunner, MethodBenchmarkRunnerConfig
    from .measurements import BooleanMeasurement, DictMeasurement, DictMetadata, FloatMetadata, IntMetadata, ListMeasurement, Measurement, MetadataBase, SingleMeasurement, StatisticalMeasurement, StringMetadata, TestPhase

from isaaclab.utils.module import lazy_export

lazy_export(
    ("benchmark_core", ["BaseIsaacLabBenchmark", "get_default_output_filename"]),
    ("benchmark_monitor", "BenchmarkMonitor"),
    ("method_benchmark", [
        "MethodBenchmarkDefinition",
        "MethodBenchmarkRunner",
        "MethodBenchmarkRunnerConfig",
    ]),
    ("measurements", [
        "BooleanMeasurement",
        "DictMeasurement",
        "DictMetadata",
        "FloatMetadata",
        "IntMetadata",
        "ListMeasurement",
        "Measurement",
        "MetadataBase",
        "SingleMeasurement",
        "StatisticalMeasurement",
        "StringMetadata",
        "TestPhase",
    ]),
)
