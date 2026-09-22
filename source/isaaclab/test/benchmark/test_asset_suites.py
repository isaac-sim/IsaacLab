# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the declarative asset micro-benchmark suites, their adapters, and the script CLI."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp

import isaaclab.benchmark.asset_suites.cli as cli
from isaaclab.benchmark.asset_suites import (
    AssetBenchmarkRequest,
    AssetBenchmarkTargets,
    get_asset_benchmark_adapter,
    get_asset_benchmark_suite,
    resolve_method_benchmarks,
    run_asset_benchmark,
)
from isaaclab.benchmark.asset_suites import (
    dispatch as asset_dispatch,
)
from isaaclab.benchmark.asset_suites import (
    generators as asset_generators,
)
from isaaclab.benchmark.asset_suites.generators import build_fill_benchmarks
from isaaclab.benchmark.method_benchmark import MethodBenchmarkRunnerConfig

pytestmark = pytest.mark.benchmark

_CONFIG = MethodBenchmarkRunnerConfig(
    num_iterations=1, warmup_steps=0, num_instances=2, num_bodies=3, num_joints=4, device="cpu"
)


def _definitions(physics: str, component: str) -> list:
    adapter = get_asset_benchmark_adapter(physics, component)
    return resolve_method_benchmarks(get_asset_benchmark_suite(component), adapter)


def _definition(definitions: list, method_name: str):
    return next(definition for definition in definitions if definition.method_name == method_name)


# ---------------------------------------------------------------------------
# Suite definitions and generators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("component", "capabilities", "definition_count", "workload_count"),
    (
        ("articulation", frozenset({"tensor_fill"}), 30, 190),
        ("articulation", frozenset({"warp_mask", "tensor_fill", "mask_fill"}), 50, 210),
        ("rigid_object", frozenset({"physx_legacy_state", "tensor_fill"}), 13, 77),
        ("rigid_object", frozenset({"warp_mask", "tensor_fill", "mask_fill"}), 15, 59),
        ("rigid_object_collection", frozenset({"physx_legacy_state", "tensor_fill"}), 13, 113),
        ("rigid_object_collection", frozenset({"warp_mask"}), 15, 75),
    ),
)
def test_method_manifests_preserve_backend_workloads(component, capabilities, definition_count, workload_count) -> None:
    """Capability resolution should produce every declared method workload."""
    adapter = SimpleNamespace(physics="any", capabilities=capabilities, generator_overrides={})

    definitions = resolve_method_benchmarks(get_asset_benchmark_suite(component), adapter)

    assert len(definitions) == definition_count
    assert sum(len(definition.input_generators) for definition in definitions) == workload_count


def test_shared_generators_cover_selector_modes_and_shapes() -> None:
    adapter = SimpleNamespace(physics="physx", capabilities=frozenset(), generator_overrides={})
    definitions = resolve_method_benchmarks(get_asset_benchmark_suite("articulation"), adapter)

    joint_writer = _definition(definitions, "write_joint_state_to_sim")
    expected_dtypes = {
        "torch_tensor_int32": (torch.int32, torch.int32),
        "torch_tensor_int64": (torch.int64, torch.int64),
        "torch_tensor_int32_int64": (torch.int32, torch.int64),
        "torch_tensor_int64_int32": (torch.int64, torch.int32),
        "warp_int32": (wp.int32, wp.int32),
        "warp_int64": (wp.int64, wp.int64),
        "warp_int32_int64": (wp.int32, wp.int64),
        "warp_int64_int32": (wp.int64, wp.int32),
    }
    assert tuple(joint_writer.input_generators) == ("torch_list", *expected_dtypes)
    inputs_by_mode = {mode: generator(_CONFIG) for mode, generator in joint_writer.input_generators.items()}
    for inputs in inputs_by_mode.values():
        assert inputs["position"].shape == inputs["velocity"].shape == (2, 4)
    assert inputs_by_mode["torch_list"]["env_ids"] == [0, 1]
    assert inputs_by_mode["torch_list"]["joint_ids"] == [0, 1, 2, 3]
    for mode, (env_dtype, joint_dtype) in expected_dtypes.items():
        assert inputs_by_mode[mode]["env_ids"].dtype is env_dtype
        assert inputs_by_mode[mode]["joint_ids"].dtype is joint_dtype

    root_writer = _definition(definitions, "write_root_state_to_sim")
    assert tuple(root_writer.input_generators) == (
        "torch_list",
        "torch_tensor_int32",
        "torch_tensor_int64",
        "warp_int32",
        "warp_int64",
    )
    assert root_writer.input_generators["warp_int64"](_CONFIG)["env_ids"].dtype is wp.int64


def test_generator_override_changes_only_selected_workload() -> None:
    def newton_coms(config: MethodBenchmarkRunnerConfig) -> dict[str, object]:
        return {"coms": torch.zeros((config.num_instances, config.num_bodies, 3))}

    adapter = SimpleNamespace(
        physics="newton",
        capabilities=frozenset(),
        generator_overrides={("set_coms", "torch_tensor_int32"): newton_coms},
    )
    definition = _definition(resolve_method_benchmarks(get_asset_benchmark_suite("articulation"), adapter), "set_coms")

    assert definition.input_generators["torch_tensor_int32"](_CONFIG)["coms"].shape == (2, 3, 3)
    assert "env_ids" in definition.input_generators["torch_list"](_CONFIG)


@pytest.mark.parametrize("physics", ("physx", "newton_mjwarp", "ovphysx"))
@pytest.mark.parametrize("component", ("articulation", "rigid_object", "rigid_object_collection"))
def test_backend_adapters_shape_com_inputs_consistently(physics, component) -> None:
    """Every list, tensor, and mask workload should use the backend's centre-of-mass representation."""
    expected_width = 3 if physics.startswith("newton") else 7
    definitions = _definitions(physics, component)

    for definition in definitions:
        if definition.method_name in {"set_coms", "set_coms_mask"}:
            for generator in definition.input_generators.values():
                assert generator(_CONFIG)["coms"].shape == (2, 3, expected_width)


def test_physx_rigid_adapters_flatten_body_inputs() -> None:
    rigid = _definitions("physx", "rigid_object")
    for method_name in ("set_masses", "set_coms", "set_inertias"):
        for generator in _definition(rigid, method_name).input_generators.values():
            assert "body_ids" not in generator(_CONFIG)

    collection = _definitions("physx", "rigid_object_collection")
    for generator in _definition(collection, "set_inertias").input_generators.values():
        assert generator(_CONFIG)["inertias"].shape == (2, 3, 9)


def test_tensor_fill_preserves_item_selector_width() -> None:
    adapter = get_asset_benchmark_adapter("physx", "rigid_object_collection")
    definitions = resolve_method_benchmarks(get_asset_benchmark_suite("rigid_object_collection"), adapter)
    fills = build_fill_benchmarks(definitions, capabilities=adapter.capabilities)
    definition = _definition(fills, "write_body_state_to_sim")
    config = MethodBenchmarkRunnerConfig(
        num_iterations=1, warmup_steps=0, num_instances=4, num_bodies=4, num_joints=0, device="cpu"
    )

    inputs = definition.input_generators["tensor_5pct"](config)

    assert definition.category == "body_state_fill"
    assert inputs["body_states"].shape == (1, 4, 13)
    assert inputs["env_ids"].shape == (1,)
    assert inputs["body_ids"].shape == (4,)


@pytest.mark.parametrize("physics", ("physx", "newton_mjwarp", "ovphysx"))
def test_articulation_joint_parameter_generators_preserve_ranges(monkeypatch, physics) -> None:
    """Joint limits stay ordered and parameter magnitudes match the workload in every input mode."""
    monkeypatch.setattr(asset_generators.torch, "rand", lambda *shape, **kwargs: torch.ones(*shape, **kwargs))
    definitions = _definitions(physics, "articulation")
    expected_scales = {
        "write_joint_stiffness_to_sim": ("stiffness", 1.0),
        "write_joint_damping_to_sim": ("damping", 1.0),
        "write_joint_velocity_limit_to_sim": ("limits", 10.0),
        "write_joint_effort_limit_to_sim": ("limits", 100.0),
        "write_joint_armature_to_sim": ("armature", 0.1),
        "write_joint_friction_coefficient_to_sim": ("joint_friction_coeff", 0.5),
    }

    seen: set[str] = set()
    for definition in definitions:
        base_name = definition.method_name.removesuffix("_mask")
        for generator in definition.input_generators.values():
            inputs = generator(_CONFIG)
            if base_name == "write_joint_position_limit_to_sim":
                assert torch.all(inputs["limits"][..., 0] <= 0) and torch.all(inputs["limits"][..., 1] >= 0)
                seen.add(base_name)
            elif base_name in expected_scales:
                field_name, scale = expected_scales[base_name]
                assert torch.all(inputs[field_name] == scale)
                seen.add(base_name)
    assert seen == set(expected_scales) | {"write_joint_position_limit_to_sim"}


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------


def test_provider_lookup_is_lazy_and_preserves_exact_variant(monkeypatch) -> None:
    @dataclass(frozen=True)
    class FakeAdapter:
        component: str
        physics_variant: str = "newton_mjwarp"

    imports: list[str] = []

    def fake_import_module(module_name: str):
        imports.append(module_name)
        return SimpleNamespace(get_asset_benchmark_adapter=FakeAdapter)

    monkeypatch.setattr(asset_dispatch.importlib, "import_module", fake_import_module)

    adapter = get_asset_benchmark_adapter("newton_kamino", "articulation")

    assert (adapter.component, adapter.physics_variant) == ("articulation", "newton_kamino")
    assert imports == ["isaaclab_newton.benchmark.assets"]


@pytest.mark.parametrize(
    ("variant", "family"),
    (
        ("physx", "physx"),
        ("isaacsim_physx", "physx"),
        ("ovphysx", "ovphysx"),
        ("newton_mjwarp", "newton"),
        ("newton_kamino", "newton"),
    ),
)
def test_exact_physics_variant_maps_to_family(variant, family) -> None:
    adapter = get_asset_benchmark_adapter(variant, "articulation")
    assert (adapter.physics, adapter.physics_variant) == (family, variant)


@pytest.mark.parametrize(
    ("physics", "component", "message"),
    (
        ("unknown", "articulation", "Unsupported asset physics selector"),
        ("newton_unknown", "articulation", "Unsupported asset physics selector"),
        ("physx", "unknown", "Unsupported asset component"),
    ),
)
def test_provider_lookup_rejects_unknown_selection(physics, component, message) -> None:
    with pytest.raises(ValueError, match=message):
        get_asset_benchmark_adapter(physics, component)


def test_each_backend_retains_exactly_three_asset_scripts() -> None:
    root = Path(__file__).resolve().parents[4]
    expected = {"benchmark_articulation.py", "benchmark_rigid_object.py", "benchmark_rigid_object_collection.py"}

    for package in ("isaaclab_physx", "isaaclab_newton", "isaaclab_ov"):
        scripts = {path.name for path in (root / "source" / package / "benchmark" / "assets").glob("*.py")}
        assert scripts == expected, package


# ---------------------------------------------------------------------------
# Runner orchestration
# ---------------------------------------------------------------------------


class _FakeRunner:
    events: list[tuple] = []

    def __init__(self, benchmark_name, config, backend_type, output_path, use_recorders, physics_variant=None):
        self.benchmark_name = benchmark_name
        _FakeRunner.events.append(("create", benchmark_name, config.num_bodies, physics_variant))

    def run_benchmarks(self, definitions, target):
        _FakeRunner.events.append(("methods", self.benchmark_name, tuple(d.method_name for d in definitions), target))

    def run_property_benchmarks(self, target_data, properties, gen_mock_data, dependencies, category):
        _FakeRunner.events.append(("properties", self.benchmark_name, tuple(properties), dependencies, category))

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
        return MethodBenchmarkRunnerConfig(num_iterations=1, warmup_steps=0, num_bodies=1, num_joints=0, device="cpu")

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


def _request(tmp_path, **config) -> AssetBenchmarkRequest:
    return AssetBenchmarkRequest(
        config=MethodBenchmarkRunnerConfig(num_iterations=1, warmup_steps=0, num_joints=0, device="cpu", **config),
        physics_variant="physx",
        formatter_type="json",
        output_path=tmp_path,
    )


def test_asset_command_runs_methods_then_properties_inside_target_context(tmp_path) -> None:
    _FakeRunner.events = []

    paths = run_asset_benchmark(
        _request(tmp_path, num_instances=2, num_bodies=4), _FakeAdapter(), runner_factory=_FakeRunner
    )

    assert paths == (Path("rigid_object_benchmark.json"), Path("rigid_object_data_benchmark.json"))
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
    assert [event[3] for event in _FakeRunner.events if event[0] == "create"] == ["physx", "physx"]
    properties = next(event for event in _FakeRunner.events if event[0] == "properties")
    assert properties[2:] == (("body_mass",), {}, "property")


def test_target_context_exits_when_method_benchmark_fails(tmp_path) -> None:
    class FailingRunner(_FakeRunner):
        def run_benchmarks(self, definitions, target):
            raise RuntimeError("boom")

    _FakeRunner.events = []
    with pytest.raises(RuntimeError, match="boom"):
        run_asset_benchmark(_request(tmp_path), _FakeAdapter(), runner_factory=FailingRunner)

    assert _FakeRunner.events[-1] == ("exit",)


def test_run_asset_benchmark_rejects_mismatched_variant(tmp_path) -> None:
    adapter = _FakeAdapter()
    adapter.physics_variant = "isaacsim_physx"
    with pytest.raises(ValueError, match="does not match adapter variant"):
        run_asset_benchmark(_request(tmp_path), adapter)


# ---------------------------------------------------------------------------
# Script CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def captured_cli(monkeypatch):
    """Route the script CLI to a lightweight adapter and capture the dispatched request."""
    adapter = SimpleNamespace(
        default_num_bodies=4, default_num_joints=0, capabilities=frozenset(), generator_overrides={}
    )
    captured: dict = {"adapter_calls": []}
    monkeypatch.setattr(
        cli,
        "get_asset_benchmark_adapter",
        lambda physics, component: captured["adapter_calls"].append((physics, component)) or adapter,
    )
    monkeypatch.setattr(
        cli, "run_asset_benchmark", lambda request, selected: captured.update(request=request, adapter=selected) or ()
    )
    captured["stub"] = adapter
    return captured


def test_script_cli_builds_one_combined_request(captured_cli, tmp_path) -> None:
    result = cli.run_asset_benchmark_cli(
        "physx",
        "rigid_object_collection",
        [
            "--num_iterations",
            "3",
            "--warmup_steps",
            "1",
            "--num_instances",
            "8",
            "--output_dir",
            str(tmp_path),
            "--backend",
            "json",
            "--device",
            "cpu",
        ],
        include_app_launcher_args=False,
    )

    request = captured_cli["request"]
    assert result == ()
    assert captured_cli["adapter"] is captured_cli["stub"]
    assert (request.physics_variant, request.formatter_type, request.output_path) == ("physx", "json", Path(tmp_path))
    assert (request.config.num_iterations, request.config.num_instances) == (3, 8)
    assert (request.config.num_bodies, request.config.num_joints) == (4, 0)
    assert request.launcher_args is None and request.check_shapes


def test_script_cli_uses_explicit_exact_physics_variant(captured_cli, tmp_path) -> None:
    cli.run_asset_benchmark_cli(
        "newton_mjwarp",
        "articulation",
        ["--physics_variant", "newton_kamino", "--output_dir", str(tmp_path), "--device", "cpu", "--no_shape_checks"],
        include_app_launcher_args=False,
    )

    assert captured_cli["adapter_calls"] == [("newton_kamino", "articulation")]
    assert captured_cli["request"].physics_variant == "newton_kamino"
    assert not captured_cli["request"].check_shapes


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--num_iterations", "0", "must be greater than zero"),
        ("--warmup_steps", "-1", "must be non-negative"),
        ("--num_instances", "0", "must be greater than zero"),
        ("--num_bodies", "0", "must be greater than zero"),
        ("--num_joints", "-1", "must be non-negative"),
        ("--mode", "unknown", "invalid choice"),
    ],
)
def test_script_cli_reports_invalid_arguments(captured_cli, capsys, option, value, message) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.run_asset_benchmark_cli(
            "physx", "rigid_object", [option, value, "--device", "cpu"], include_app_launcher_args=False
        )

    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
