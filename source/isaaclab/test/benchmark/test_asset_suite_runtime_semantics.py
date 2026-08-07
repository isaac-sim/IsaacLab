# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for backend-specific asset data setup and refresh behavior."""

import importlib
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.benchmark.assets import runtime as newton_runtime
from isaaclab_ovphysx.benchmark.assets import runtime as ovphysx_runtime
from isaaclab_physx.benchmark.assets import runtime as physx_runtime

from isaaclab.benchmark.method_benchmark import MethodBenchmarkRunnerConfig

pytestmark = pytest.mark.benchmark


def _hide_app_launcher(monkeypatch) -> None:
    import isaaclab.app

    app = ModuleType("isaaclab.app")
    app.__path__ = isaaclab.app.__path__
    monkeypatch.setitem(sys.modules, "isaaclab.app", app)


@pytest.fixture(scope="module", autouse=True)
def _initialize_warp_runtime() -> None:
    """Initialize Warp so refresh tests also pass when run as a standalone file."""
    wp.init()


_CONFIG = MethodBenchmarkRunnerConfig(
    num_iterations=1,
    warmup_steps=0,
    num_instances=2,
    num_bodies=3,
    num_joints=4,
    device="cpu",
)


class _RecordingView:
    def __init__(self) -> None:
        self.calls: dict[str, tuple[int, ...]] = {}
        self._attributes: dict[str, object] = {}

    def _record(self, name: str, value) -> None:
        self.calls[name] = tuple(value.shape)

    def set_mock_transforms(self, value) -> None:
        self._record("transforms", value)

    def set_mock_velocities(self, value) -> None:
        self._record("velocities", value)

    def set_mock_accelerations(self, value) -> None:
        self._record("accelerations", value)

    def set_mock_coms(self, value) -> None:
        self._record("coms", value)

    def set_mock_root_transforms(self, value) -> None:
        self._record("root_transforms", value)

    def set_mock_root_velocities(self, value) -> None:
        self._record("root_velocities", value)

    def set_mock_link_transforms(self, value) -> None:
        self._record("link_transforms", value)

    def set_mock_link_velocities(self, value) -> None:
        self._record("link_velocities", value)

    def set_mock_inertias(self, value) -> None:
        self._record("inertias", value)

    def set_mock_masses(self, value) -> None:
        self._record("masses", value)


def test_physx_rigid_object_refresh_regenerates_all_source_buffers(monkeypatch) -> None:
    """PhysX rigid-object preparation should refresh transforms, velocities, accelerations, and CoMs."""
    view = _RecordingView()
    data = SimpleNamespace(_sim_timestamp=0.0)
    monkeypatch.setattr(physx_runtime, "torch", torch, raising=False)
    monkeypatch.setattr(physx_runtime, "wp", wp, raising=False)
    refresh = getattr(physx_runtime, "_refresh_rigid_object_data", None)

    assert refresh is not None
    refresh(view, data, _CONFIG)

    assert view.calls == {
        "transforms": (2, 7),
        "velocities": (2, 6),
        "accelerations": (2, 6),
        "coms": (2, 7),
    }
    assert data._sim_timestamp == 1.0


def test_physx_collection_refresh_uses_flattened_body_count(monkeypatch) -> None:
    """PhysX collection preparation should regenerate one source record per environment/body pair."""
    view = _RecordingView()
    data = SimpleNamespace(_sim_timestamp=0.0)
    monkeypatch.setattr(physx_runtime, "torch", torch, raising=False)
    monkeypatch.setattr(physx_runtime, "wp", wp, raising=False)
    refresh = getattr(physx_runtime, "_refresh_collection_data", None)

    assert refresh is not None
    refresh(view, data, _CONFIG)

    assert view.calls == {
        "transforms": (6, 7),
        "velocities": (6, 6),
        "accelerations": (6, 6),
        "coms": (6, 7),
    }
    assert data._sim_timestamp == 1.0


def test_newton_rigid_object_refresh_regenerates_kinematics_and_mass_properties(monkeypatch) -> None:
    """Newton rigid-object preparation should refresh every legacy source buffer."""
    view = _RecordingView()
    data = SimpleNamespace(_sim_timestamp=0.0)
    monkeypatch.setattr(newton_runtime, "np", np, raising=False)
    monkeypatch.setattr(newton_runtime, "wp", wp, raising=False)
    refresh = getattr(newton_runtime, "_refresh_rigid_object_data", None)

    assert refresh is not None
    refresh(view, data, _CONFIG)

    assert view.calls == {
        "root_transforms": (2, 1),
        "root_velocities": (2, 1),
        "link_transforms": (2, 1, 3),
        "link_velocities": (2, 1, 3),
        "coms": (2, 1, 3),
        "inertias": (2, 1, 3),
        "masses": (2, 1, 3),
    }
    assert data._sim_timestamp == 1.0


def test_newton_collection_refresh_regenerates_kinematics_and_mass_properties(monkeypatch) -> None:
    """Newton collection preparation should refresh every source buffer."""
    view = _RecordingView()
    data = SimpleNamespace(_sim_timestamp=0.0)
    monkeypatch.setattr(newton_runtime, "np", np, raising=False)
    monkeypatch.setattr(newton_runtime, "wp", wp, raising=False)
    refresh = getattr(newton_runtime, "_refresh_collection_data", None)

    assert refresh is not None
    refresh(view, data, _CONFIG)

    assert view.calls == {
        "root_transforms": (2, 3),
        "root_velocities": (2, 3),
        "coms": (2, 3, 1),
        "masses": (2, 3, 1),
        "inertias": (2, 3, 1),
    }
    assert data._sim_timestamp == 1.0


def test_ovphysx_data_targets_are_independent_and_properties_preflight_when_available() -> None:
    """OVPhysX property targets should be independent and expose every declared runtime property."""
    if not torch.cuda.is_available():
        pytest.skip("OVPhysX target construction requires CUDA pinned memory")

    from isaaclab_ovphysx.benchmark.assets import runtime as ovphysx_runtime

    from isaaclab.benchmark.asset_suites import (
        get_asset_benchmark_adapter,
        get_asset_benchmark_suite,
        resolve_property_benchmarks,
    )

    try:
        ovphysx_runtime._load_runtime_symbols()
    except ModuleNotFoundError as exc:
        if exc.name == "ovphysx":
            pytest.skip("optional ovphysx runtime is not installed")
        raise

    for component in ("articulation", "rigid_object", "rigid_object_collection"):
        config = replace(
            _CONFIG,
            num_bodies=1 if component == "rigid_object" else _CONFIG.num_bodies,
            num_joints=_CONFIG.num_joints if component == "articulation" else 0,
        )
        create_method_target = {
            "articulation": ovphysx_runtime.create_test_articulation,
            "rigid_object": ovphysx_runtime.create_test_rigid_object,
            "rigid_object_collection": ovphysx_runtime.create_test_collection,
        }[component]
        method_target, _ = create_method_target(
            num_instances=config.num_instances,
            num_bodies=config.num_bodies,
            **({"num_joints": config.num_joints} if component == "articulation" else {}),
            device=config.device,
        )
        data_target, refresh = ovphysx_runtime._create_data_target(component, config)

        assert data_target is not method_target._data
        adapter = get_asset_benchmark_adapter("ovphysx", component)
        suite = get_asset_benchmark_suite(component)
        properties, _ = resolve_property_benchmarks(suite, adapter, data_target)
        assert properties
        previous_timestamp = data_target._sim_timestamp
        refresh(config)
        assert data_target._sim_timestamp == previous_timestamp + 1.0


def test_newton_articulation_data_target_restores_model_dynamics_and_ordering(monkeypatch) -> None:
    """Dedicated Newton articulation data should allocate against configured model dimensions and apply ordering."""
    model = SimpleNamespace()
    view = SimpleNamespace(randomized=False)

    def set_random_mock_data() -> None:
        view.randomized = True

    view.set_random_mock_data = set_random_mock_data

    class FakeView:
        def __new__(cls, **kwargs):
            view.kwargs = kwargs
            return view

    class FakeData:
        def __init__(self, root_view, device):
            self.root_view = root_view
            self.device = device
            self.ordering_applied = False
            self._jacobian_body_user_to_backend = None
            self._sim_timestamp = 0.0
            self._fk_timestamp = 0.0

        def _apply_ordering_maps_after_resolve(self) -> None:
            self.ordering_applied = True
            self._jacobian_body_user_to_backend = object()

    monkeypatch.setattr(newton_runtime, "MockNewtonArticulationView", FakeView, raising=False)
    monkeypatch.setattr(newton_runtime, "NewtonArticulationData", FakeData, raising=False)
    monkeypatch.setattr(
        newton_runtime,
        "NewtonSimulationManager",
        SimpleNamespace(get_model=lambda: model),
        raising=False,
    )
    create_target = getattr(newton_runtime, "_create_articulation_data_target", None)

    assert create_target is not None
    data, refresh = create_target(_CONFIG)

    assert model.articulation_count == 2
    assert model.max_joints_per_articulation == 3
    assert model.max_dofs_per_articulation == 10
    assert model.joint_dof_count == 20
    assert model.body_count == 6
    assert view.randomized
    assert callable(view.eval_jacobian)
    assert callable(view.eval_mass_matrix)
    assert data.ordering_applied
    assert data._jacobian_body_user_to_backend is not None
    refresh(_CONFIG)
    assert data._sim_timestamp == 1.0
    assert data._fk_timestamp == 1.0


def test_newton_articulation_open_targets_constructs_real_method_and_data_targets(monkeypatch) -> None:
    """The combined Newton path should configure model dimensions before constructing either data target."""
    from isaaclab.benchmark.asset_suites import get_asset_benchmark_adapter

    _hide_app_launcher(monkeypatch)
    adapter = get_asset_benchmark_adapter("newton_mjwarp", "articulation")
    request = SimpleNamespace(launcher_args=None, check_shapes=True)

    with newton_runtime.open_asset_targets(adapter, request, _CONFIG, _CONFIG) as targets:
        assert targets.method_target._data._jacobian_buf.shape == (2, 3, 6, 10)
        assert targets.data_target._jacobian_buf.shape == (2, 3, 6, 10)


def test_physx_cpu_boundary_reuses_int32_scratch_for_int64_env_ids(monkeypatch) -> None:
    module_name = "isaaclab_physx.assets.articulation.articulation"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    physx_runtime._load_runtime_symbols()
    assert sys.modules[module_name].Articulation is physx_runtime.Articulation
    zeros = wp.zeros
    empty = wp.empty
    monkeypatch.setattr(wp, "zeros", lambda *args, **kwargs: zeros(*args, **(kwargs | {"pinned": False})))
    monkeypatch.setattr(wp, "empty", lambda *args, **kwargs: empty(*args, **(kwargs | {"pinned": False})))
    physx_runtime.PhysxManager.get_physics_sim_view.return_value.get_gravity.return_value = (0.0, 0.0, -9.81)
    target = physx_runtime.create_test_rigid_object(num_instances=2, num_bodies=1, device="cpu")[0]
    env_ids = wp.array([1, 0], dtype=wp.int64, device="cpu")
    sim_env_ids = wp.array([1, 0], dtype=wp.int32, device="cpu")

    assert target._get_cpu_env_ids(env_ids, sim_env_ids).ptr == sim_env_ids.ptr
    target.set_masses_index(masses=wp.ones((2, 1), dtype=wp.float32, device="cpu"), env_ids=env_ids)


def test_physx_collection_reuses_bounded_flat_view_id_scratch(monkeypatch) -> None:
    physx_runtime._load_runtime_symbols()
    physx_runtime.PhysxManager.get_physics_sim_view.return_value.get_gravity.return_value = (0.0, 0.0, -9.81)
    zeros = wp.zeros
    empty = wp.empty
    monkeypatch.setattr(wp, "zeros", lambda *args, **kwargs: zeros(*args, **(kwargs | {"pinned": False})))
    monkeypatch.setattr(wp, "empty", lambda *args, **kwargs: empty(*args, **(kwargs | {"pinned": False})))
    target = physx_runtime.create_test_collection(num_instances=2, num_bodies=2, device="cpu")[0]
    env_ids = wp.array([1, 0], dtype=wp.int64, device="cpu")
    body_ids = wp.array([0, 1], dtype=wp.int64, device="cpu")

    first = target._env_body_ids_to_view_ids(env_ids, body_ids, device="cpu")
    second = target._env_body_ids_to_view_ids(env_ids, body_ids, device="cpu")

    assert first.ptr == second.ptr
    np.testing.assert_array_equal(first.numpy(), [1, 0, 3, 2])

    with pytest.raises(AssertionError, match="env_ids"):
        target.set_masses_index(
            masses=wp.ones((3, 2), dtype=wp.float32, device="cpu"),
            env_ids=wp.array([0, 0, 1], dtype=wp.int64, device="cpu"),
            body_ids=body_ids,
        )

    target._device = "cuda:0"
    gpu_ids = SimpleNamespace(device="cuda:0", shape=(1,))
    sim_ids = object()
    cpu_ids = object()
    calls = []
    monkeypatch.setattr(target, "_sim_view_ids_view", lambda count: sim_ids)
    monkeypatch.setattr(target, "_cpu_view_ids_view", lambda count: cpu_ids)
    monkeypatch.setattr(
        importlib.import_module(target.__class__.__module__), "resolve_view_ids_kernel", lambda *_: None
    )
    monkeypatch.setattr(wp, "launch", lambda *args, **kwargs: None)
    monkeypatch.setattr(wp, "copy", lambda *args: calls.append("copy"))
    monkeypatch.setattr(wp, "synchronize_stream", lambda device: calls.append(("sync", device)))

    assert target._env_body_ids_to_view_ids(gpu_ids, gpu_ids, device="cpu") is cpu_ids
    assert calls == ["copy", ("sync", "cuda:0")]


@pytest.mark.parametrize("runtime", (physx_runtime, newton_runtime))
def test_open_targets_preserves_setup_errors(monkeypatch, runtime) -> None:
    _hide_app_launcher(monkeypatch)
    monkeypatch.setattr(runtime, "_load_runtime_symbols", lambda: (_ for _ in ()).throw(RuntimeError("setup")))
    request = SimpleNamespace(launcher_args=None, check_shapes=True)

    with pytest.raises(RuntimeError, match="setup"):
        with runtime.open_asset_targets(None, request, _CONFIG, _CONFIG):
            pass


@pytest.mark.parametrize(
    ("factory_name", "module_name", "class_name"),
    (
        (
            "create_test_articulation",
            "isaaclab_ovphysx.assets.articulation.articulation",
            "Articulation",
        ),
        (
            "create_test_rigid_object",
            "isaaclab_ovphysx.assets.rigid_object.rigid_object",
            "RigidObject",
        ),
        (
            "create_test_collection",
            "isaaclab_ovphysx.assets.rigid_object_collection.rigid_object_collection",
            "RigidObjectCollection",
        ),
        ("_create_data_target", None, None),
    ),
)
def test_ovphysx_factories_use_exported_binding_set_signature(
    monkeypatch, factory_name, module_name, class_name
) -> None:
    """Every OVPhysX target factory should call the exported binding constructor compatibly."""

    class ConstructorAccepted(Exception):
        pass

    class StrictBindingSet:
        def __init__(
            self,
            num_instances,
            num_joints,
            num_bodies,
            is_fixed_base=False,
            joint_names=None,
            body_names=None,
            num_fixed_tendons=0,
            num_spatial_tendons=0,
            *,
            asset_kind="articulation",
        ):
            raise ConstructorAccepted

    if module_name is not None:
        monkeypatch.setitem(sys.modules, module_name, SimpleNamespace(**{class_name: object}))
        monkeypatch.setitem(sys.modules, f"{module_name}_data", SimpleNamespace(**{f"{class_name}Data": object}))
    monkeypatch.setattr(ovphysx_runtime, "MockOvPhysxBindingSet", StrictBindingSet, raising=False)
    factory = getattr(ovphysx_runtime, factory_name)

    with pytest.raises(ConstructorAccepted):
        if factory_name == "_create_data_target":
            factory("articulation", _CONFIG)
        else:
            factory(
                num_instances=_CONFIG.num_instances,
                num_bodies=1 if factory_name == "create_test_rigid_object" else _CONFIG.num_bodies,
                **({"num_joints": _CONFIG.num_joints} if factory_name == "create_test_articulation" else {}),
                device=_CONFIG.device,
            )


@pytest.mark.parametrize("read_mode", ("numpy", "structured_warp", "view_preflight"))
def test_mock_ovphysx_binding_reads_latest_data_without_optional_runtime(monkeypatch, read_mode) -> None:
    """Repository mock reads should support every consumer mode and reflect subsequent writes."""
    monkeypatch.setitem(sys.modules, "isaaclab_ovphysx.tensor_types", SimpleNamespace())
    bindings_module = importlib.import_module("isaaclab_ovphysx.test.mock_interfaces.views.mock_ovphysx_bindings")
    binding = bindings_module.MockTensorBinding(tensor_type=0, shape=(2, 7), count=2)
    first = np.arange(14, dtype=np.float32).reshape(2, 7)
    latest = first + 100.0

    view = bindings_module.MockOvPhysxView({0: binding})

    def read_binding():
        if read_mode == "numpy":
            destination = np.zeros_like(first)
            binding.read(destination)
            return destination
        if read_mode == "structured_warp":
            destination = wp.zeros((2,), dtype=wp.transformf, device="cpu")
            binding.read(destination)
            return destination.view(wp.float32).numpy()
        return view.get_attribute(0).numpy()

    binding.write(first)
    np.testing.assert_array_equal(read_binding(), first)

    binding.write(latest)
    np.testing.assert_array_equal(read_binding(), latest)
