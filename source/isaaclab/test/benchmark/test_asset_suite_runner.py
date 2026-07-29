# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for combined asset method/data micro-benchmark orchestration."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab.benchmark.asset_suites import AssetBenchmarkRequest, AssetBenchmarkTargets, run_asset_benchmark
from isaaclab.benchmark.method_benchmark import MethodBenchmarkRunnerConfig

pytestmark = pytest.mark.benchmark


class _FakeRunner:
    events: list[tuple] = []
    next_index = 0

    def __init__(self, benchmark_name, config, backend_type, output_path, use_recorders, physics_variant=None):
        self.index = _FakeRunner.next_index
        _FakeRunner.next_index += 1
        self.benchmark_name = benchmark_name
        _FakeRunner.events.append(("create", benchmark_name, config.num_bodies, physics_variant))

    def run_benchmarks(self, definitions, target):
        _FakeRunner.events.append(("methods", self.benchmark_name, tuple(d.method_name for d in definitions), target))

    def run_property_benchmarks(self, target_data, properties, gen_mock_data, dependencies, category):
        _FakeRunner.events.append(
            ("properties", self.benchmark_name, tuple(properties), dependencies, category, target_data)
        )

    def finalize(self):
        _FakeRunner.events.append(("finalize", self.benchmark_name))
        return (Path(f"{self.benchmark_name}.json"),)


class _FakeAdapter:
    physics = "physx"
    physics_variant = "physx"
    component = "rigid_object"
    method_benchmark_name = "rigid_object_benchmark"
    data_benchmark_name = "rigid_object_data_benchmark"
    capabilities = frozenset()
    generator_overrides = {}
    supported_properties = frozenset({"body_mass"})
    property_dependency_overrides = {}

    def method_config(self, request):
        return request.config

    def data_config(self, request):
        return MethodBenchmarkRunnerConfig(
            num_iterations=request.config.num_iterations,
            warmup_steps=request.config.warmup_steps,
            num_instances=request.config.num_instances,
            num_bodies=1,
            num_joints=0,
            device=request.config.device,
            mode=request.config.mode,
        )

    @contextmanager
    def open_targets(self, request, method_config, data_config):
        _FakeRunner.events.append(("enter", method_config.num_bodies, data_config.num_bodies))
        try:
            yield AssetBenchmarkTargets(
                method_target="method-target",
                data_target=SimpleNamespace(body_mass=object()),
                refresh_data=lambda _config: None,
            )
        finally:
            _FakeRunner.events.append(("exit",))


def test_asset_command_writes_method_and_data_artifacts_in_order(tmp_path) -> None:
    """One command should run methods then data while retaining both workflow names."""
    _FakeRunner.events = []
    _FakeRunner.next_index = 0
    request = AssetBenchmarkRequest(
        config=MethodBenchmarkRunnerConfig(
            num_iterations=1,
            warmup_steps=0,
            num_instances=2,
            num_bodies=4,
            num_joints=0,
            device="cpu",
        ),
        physics_variant="physx",
        formatter_type="json",
        output_path=tmp_path,
    )

    paths = run_asset_benchmark(request, _FakeAdapter(), runner_factory=_FakeRunner)

    assert paths == (
        Path("rigid_object_benchmark.json"),
        Path("rigid_object_data_benchmark.json"),
    )
    create_events = [event for event in _FakeRunner.events if event[0] == "create"]
    assert [event[3] for event in create_events] == ["physx", "physx"]
    assert [event[:2] for event in _FakeRunner.events] == [
        ("enter", 4),
        ("create", "rigid_object_benchmark"),
        ("methods", "rigid_object_benchmark"),
        ("finalize", "rigid_object_benchmark"),
        ("create", "rigid_object_data_benchmark"),
        ("properties", "rigid_object_data_benchmark"),
        ("finalize", "rigid_object_data_benchmark"),
        ("exit",),
    ]
    property_event = next(event for event in _FakeRunner.events if event[0] == "properties")
    assert property_event[2] == ("body_mass",)
    assert property_event[4] == "property"


def test_target_context_exits_when_method_benchmark_fails(tmp_path) -> None:
    """Backend application and patch contexts should close after a method failure."""

    class FailingRunner(_FakeRunner):
        def run_benchmarks(self, definitions, target):
            raise RuntimeError("boom")

    _FakeRunner.events = []
    _FakeRunner.next_index = 0
    request = AssetBenchmarkRequest(
        config=MethodBenchmarkRunnerConfig(num_iterations=1, warmup_steps=0, device="cpu"),
        physics_variant="physx",
        formatter_type="json",
        output_path=tmp_path,
    )

    with pytest.raises(RuntimeError, match="boom"):
        run_asset_benchmark(request, _FakeAdapter(), runner_factory=FailingRunner)

    assert _FakeRunner.events[-1] == ("exit",)
