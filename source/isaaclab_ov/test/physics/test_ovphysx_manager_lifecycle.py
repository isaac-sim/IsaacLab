# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for OVPhysX 0.5.9 bootstrap and shutdown."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
import textwrap
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from isaaclab.test.utils import DeviceScope, test_devices

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

_CPU_DEVICES = test_devices(DeviceScope.CPU)
_CUDA_DEVICES = test_devices(DeviceScope.CUDA)


class _FakePhysXConfig:
    def __init__(self, num_threads=None, cooked_collider_cache_dir=None, carbonite_overrides=None):
        self.num_threads = num_threads
        self.cooked_collider_cache_dir = cooked_collider_cache_dir
        self.carbonite_overrides = carbonite_overrides or {}


class _FakePhysX:
    def __init__(self, active_cuda_gpus=None, config=None):
        self.active_cuda_gpus = active_cuda_gpus
        self.config = config


@pytest.fixture
def manager(monkeypatch):
    """Create a manager instance bound to a fake simulation and restore process-wide state after each test."""
    import isaaclab_ov.physics.ovphysx_manager as module
    from isaaclab_ov.physics import OvPhysxBackendCfg, OvPhysxCfg

    from isaaclab.sim import SimulationContext

    monkeypatch.setattr(module.ovphysx_compat, "OVPHYSX_USES_TASKS", False)
    cfg = OvPhysxBackendCfg(device="cpu")
    manager = module.OvPhysxManager(OvPhysxCfg())
    sim = SimpleNamespace(_backend_registry=[], backend_manager=manager)
    sim.get_or_create_backend = SimulationContext.get_or_create_backend.__get__(sim)
    sim.get_backends = SimulationContext.get_backends.__get__(sim)
    sim.close_backend = SimulationContext.close_backend.__get__(sim)
    with monkeypatch.context() as construction:
        construction.setattr(module, "ovphysx", _fake_ovphysx_module(lambda: None))
        backend = sim.get_or_create_backend(cfg)
    manager._sim = sim
    manager.backend = backend
    monkeypatch.setattr(SimulationContext, "_instance", sim)
    monkeypatch.setattr(module.atexit, "register", lambda callback: None)
    monkeypatch.setattr(module.OvPhysxManager, "_physx_schemas_registered", False)
    return manager


@pytest.fixture
def manager_module(manager):
    """Import the manager module after binding a manager instance to a fake simulation."""
    import isaaclab_ov.physics.ovphysx_manager as module

    return module


def _fake_ovphysx_module(bootstrap):
    module = ModuleType("ovphysx")
    module.bootstrap = bootstrap
    module.PhysX = _FakePhysX
    module.PhysXConfig = _FakePhysXConfig
    return module


def test_initialize_defers_native_resource_until_warmup(monkeypatch, manager_module, manager):
    # Physics must not manage rendering consumers when rebuilding a shared stage.
    imports = (
        node.module
        for node in ast.walk(ast.parse(inspect.getsource(manager_module)))
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert not any(module.startswith("isaaclab_ov.renderers") for module in imports)
    sim = SimpleNamespace(
        cfg=SimpleNamespace(physics=None, device="cpu", gravity=(0.0, 0.0, -9.81)),
        stage=object(),
        _backend_registry=[],
        clone_contexts={},
    )
    monkeypatch.setattr(manager, "_ensure_physx_schemas_registered", lambda: None)
    monkeypatch.setattr(manager, "backend", None)

    manager.initialize(sim)

    assert not sim._backend_registry
    assert manager.get_physx_instance() is None
    assert not any(hasattr(manager, name) for name in ("_backend", "_physx", "_ovstage"))


@pytest.mark.parametrize(
    ("registered_names", "expected_paths"),
    [
        (["physxSchema"], ["/schemas/OmniUsdPhysicsDeformableSchema/resources"]),
        (["PhysxSchema", "OmniUsdPhysicsDeformableSchema"], []),
    ],
)
def test_schema_registration_skips_providers_already_supplied_by_host(
    monkeypatch, manager_module, registered_names, expected_paths
):
    schema_paths = [
        Path("/schemas/PhysxSchema/resources"),
        Path("/schemas/OmniUsdPhysicsDeformableSchema/resources"),
    ]
    monkeypatch.setattr(manager_module.ovphysx_compat, "physx_schema_paths", lambda _: schema_paths)
    host_registrations, ovstage_registrations = [], []
    monkeypatch.setattr(manager_module.ovstage.population, "register_usd_schemas", ovstage_registrations.append)
    monkeypatch.setattr(
        manager_module.importlib.util, "find_spec", lambda name: SimpleNamespace(origin="/schemas/newton/__init__.py")
    )

    class FakeRegistry:
        def GetAllPlugins(self):
            return [SimpleNamespace(name=name) for name in registered_names]

        def RegisterPlugins(self, paths):
            host_registrations.append(list(paths))

    monkeypatch.setattr(manager_module.Plug, "Registry", FakeRegistry)
    manager_module.OvPhysxManager._ensure_physx_schemas_registered()
    manager_module.OvPhysxManager._ensure_physx_schemas_registered()

    assert ovstage_registrations == [[str(path) for path in schema_paths], "/schemas/newton"]
    assert host_registrations == ([expected_paths] if expected_paths else [])


def test_registry_shares_native_cfg_without_replacing_pxr(monkeypatch, manager_module):
    from isaaclab_ov.physics import OvPhysxBackendCfg

    from isaaclab.sim import SimulationContext

    host_pxr = ModuleType("pxr")
    host_usd = ModuleType("pxr.Usd")
    bootstrap_calls = []

    monkeypatch.setitem(sys.modules, "pxr", host_pxr)
    monkeypatch.setitem(sys.modules, "pxr.Usd", host_usd)

    def bootstrap():
        bootstrap_calls.append(None)
        assert sys.modules["pxr"] is host_pxr
        assert sys.modules["pxr.Usd"] is host_usd

    monkeypatch.setattr(manager_module, "ovphysx", _fake_ovphysx_module(bootstrap))

    sim = SimpleNamespace(_backend_registry=[])
    sim.get_or_create_backend = SimulationContext.get_or_create_backend.__get__(sim)
    cfg = OvPhysxBackendCfg(device="cpu")
    first = sim.get_or_create_backend(cfg)
    shared = sim.get_or_create_backend(OvPhysxBackendCfg(device="cpu"))

    assert sys.modules["pxr"] is host_pxr
    assert sys.modules["pxr.Usd"] is host_usd
    assert bootstrap_calls == [None]
    assert shared is first


@pytest.mark.parametrize("stop_fails", [False, True])
def test_close_releases_runtime_after_stop_even_on_listener_failure(monkeypatch, manager, stop_fails):
    import isaaclab_ov.physics.ovphysx_manager as module

    from isaaclab.physics import PhysicsManager

    events = []
    unregistered = []

    def stop(self):
        events.append("stop")
        if stop_fails:
            raise ValueError("listener failure")

    monkeypatch.setattr(PhysicsManager, "close", stop)
    monkeypatch.setattr(manager.backend, "close", lambda: events.append("release"))
    monkeypatch.setattr(module.atexit, "unregister", unregistered.append)
    manager._atexit_registered = True

    with pytest.raises(ValueError, match="listener failure") if stop_fails else nullcontext():
        manager.close()

    assert events == ["stop", "release"]
    # A released manager drops its exit hook so it is not retained until process exit.
    assert unregistered == [manager._close_at_exit]
    assert not manager._atexit_registered


def test_atexit_cleanup_noops_after_explicit_close(monkeypatch, manager):
    manager.backend.physx = None
    monkeypatch.setattr(manager, "close", lambda: pytest.fail("unexpected close"))
    monkeypatch.setattr(manager.backend, "close", lambda: pytest.fail("unexpected release"))

    manager._close_at_exit()


def test_atexit_cleanup_logs_and_swallows_active_close_failure(monkeypatch, manager, caplog):
    events = []
    manager.backend.physx = object()

    def fail_close():
        events.append("close")
        raise RuntimeError("failure")

    monkeypatch.setattr(manager, "close", fail_close)
    monkeypatch.setattr(manager.backend, "close", lambda: events.append("release"))

    manager._close_at_exit()

    assert events == ["close"]
    assert "Failed to close OVPhysX during process exit." in caplog.text


@pytest.mark.parametrize(("modern", "fails"), [(False, False), (True, False), (True, True)])
def test_stage_reuse_drains_bindings_before_reset(monkeypatch, manager_module, manager, modern, fails):
    events = []
    monkeypatch.setattr(manager_module.ovphysx_compat, "OVPHYSX_USES_TASKS", modern)

    class FakePhysX:
        def attach_ovstage(self, stage, *, read_ordinal):
            events.append(("attach", stage, read_ordinal))
            return receipt

        def reset_stage(self):
            events.append("reset")
            return 9

        def wait_op(self, operation):
            events.append(("wait", operation))

    class TaskPhysX(FakePhysX):
        def detach_ovstage(self):
            events.append("reset")
            return receipt

        def wait_task(self, task):
            assert task is receipt
            events.append(("wait", task))
            if fails:
                raise RuntimeError("detach failed")

    receipt = object() if modern else 9
    physx = TaskPhysX() if modern else FakePhysX()
    manager.backend.physx = physx
    monkeypatch.setattr(
        manager_module.OvPhysxView, "_close_all_for", lambda value: events.append(("close_views", value))
    )
    manager.backend.stage = object()
    manager.backend.rigid_body_view = SimpleNamespace(destroy=lambda: events.append("destroy_view"))
    scene = manager.backend.scene = SimpleNamespace(
        stage=object(), ordinal=2, reset=lambda: events.append("destroy_stage"),
        populate=lambda usda: events.append(("populate", usda)),
    )  # fmt: skip

    stage = manager.backend.stage
    with pytest.raises(RuntimeError, match="detach failed") if fails else nullcontext():
        manager.backend.load("#usda 1.0")

    assert events == [
        "destroy_view",
        ("close_views", physx),
        "reset",
        ("wait", receipt),
    ] + ([] if fails else ["destroy_stage", ("populate", "#usda 1.0"), ("attach", scene.stage, 1)]) + (
        [("wait", receipt)] if modern and not fails else []
    )
    assert manager.backend.rigid_body_view is None
    assert manager.backend.stage is (stage if fails else scene.stage)


@pytest.mark.parametrize(
    ("modern", "failure"), [(False, None), (False, "query"), (False, "write"), (True, None), (True, "update")]
)
def test_set_gravity_writes_and_releases_ovstage_control_resources(
    monkeypatch, manager_module, manager, modern, failure
):
    """Scene gravity updates must seal their ordinal and release resources on every path."""
    from isaaclab_ov.stage import OvstageBackend

    monkeypatch.setattr(manager_module.ovphysx_compat, "OVPHYSX_USES_TASKS", modern)
    calls = []
    task = object()

    class FakePathDictionary:
        def create_path_list_from_strings(self, paths):
            calls.append(("paths", paths))
            return "physics-scene-paths"

        def destroy_path_list(self, paths):
            calls.append(("destroy_paths", paths))

    class FakeQuery:
        def __enter__(self):
            return "physics-scene-query"

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append(("release_query", "physics-scene-query"))

    class FakeStage:
        def query_from_path_list(self, paths):
            calls.append(("query", paths))
            if failure == "query":
                raise RuntimeError("query failed")
            return FakeQuery()

        def write_attribute(self, query, attribute, ordinal, tensors, *, is_array):
            calls.append(("write", query, attribute, ordinal, tensors.tolist(), is_array))
            if failure == "write":
                raise RuntimeError("write failed")
            return SimpleNamespace(wait=lambda: None)

        def advance_write_floor(self, *, ordinal):
            calls.append(("seal", ordinal))
            return SimpleNamespace(wait=lambda: None)

    class FakePhysX:
        def update_from_ovstage(self, start_ordinal, end_ordinal):
            calls.append(("update", start_ordinal, end_ordinal))
            return task

        def wait_task(self, receipt):
            assert receipt is task
            calls.append(("wait", receipt))
            if failure == "update":
                raise RuntimeError("update failed")

    scene = OvstageBackend.__new__(OvstageBackend)
    scene.stage = manager.backend.stage = FakeStage()
    scene.paths = FakePathDictionary()
    scene.ordinal = 2
    manager.backend.scene = scene
    manager.backend.physx = FakePhysX()
    monkeypatch.setattr(
        manager,
        "_sim",
        SimpleNamespace(cfg=SimpleNamespace(physics_prim_path="/World/physicsScene", gravity=(0.0, 0.0, -9.81))),
    )
    monkeypatch.setattr(manager, "_gravity", (0.0, 0.0, -9.81))

    if failure is None:
        manager.set_gravity((0.0, 0.0, -3.72))
        expected_calls = [
            ("paths", ["/World/physicsScene"]),
            ("query", "physics-scene-paths"),
            ("write", "physics-scene-query", "physics:gravityDirection", 3, [[0.0, 0.0, -1.0]], False),
            ("write", "physics-scene-query", "physics:gravityMagnitude", 3, [pytest.approx(3.72)], False),
            ("seal", 3),
            ("update", 3, 3),
            ("wait", task),
            ("release_query", "physics-scene-query"),
            ("destroy_paths", "physics-scene-paths"),
        ]
    else:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            manager.set_gravity((0.0, 0.0, -3.72))
        expected_calls = [
            ("paths", ["/World/physicsScene"]),
            ("query", "physics-scene-paths"),
        ]
        if failure in ("write", "update"):
            expected_calls.append(
                ("write", "physics-scene-query", "physics:gravityDirection", 3, [[0.0, 0.0, -1.0]], False)
            )
            if failure == "update":
                expected_calls.extend(
                    [
                        ("write", "physics-scene-query", "physics:gravityMagnitude", 3, [pytest.approx(3.72)], False),
                        ("seal", 3),
                        ("update", 3, 3),
                        ("wait", task),
                    ]
                )
            expected_calls.append(("release_query", "physics-scene-query"))
        expected_calls.append(("destroy_paths", "physics-scene-paths"))

    if not modern:
        expected_calls = [call for call in expected_calls if call[0] != "wait"]
    assert calls == [("seal", 2), *expected_calls]
    # ``get_gravity`` reports the last successfully applied vector.
    expected_gravity = (0.0, 0.0, -3.72) if failure is None else (0.0, 0.0, -9.81)
    assert manager.get_gravity() == pytest.approx(expected_gravity)
    # ``cfg.gravity`` stays the nominal base that randomization terms resample from.
    assert manager._sim.cfg.gravity == (0.0, 0.0, -9.81)


def _retained_binding_script() -> str:
    return textwrap.dedent(
        """
        import atexit
        import gc
        import sys

        import warp as wp

        def report_unraisable(unraisable):
            print("ATEXIT_UNRAISABLE", repr(unraisable.exc_value), flush=True)

        sys.unraisablehook = report_unraisable
        atexit.register(lambda: print("NORMAL_ATEXIT", flush=True))
        wp.init()

        import isaaclab.sim as sim_utils
        from isaaclab.physics import PhysicsEvent
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab_ov.physics import OvPhysxCfg, OvPhysxManager
        from isaaclab_ov.sim.views import OvPhysxView

        sim = SimulationContext(SimulationCfg(physics=OvPhysxCfg(), device="cpu", dt=1.0 / 60.0))
        assert isinstance(sim.backend_manager, OvPhysxManager)
        sim.backend_manager.register_callback(
            lambda _payload: print("OVPHYSX_STOP", flush=True),
            PhysicsEvent.STOP,
            wrap_weak_ref=False,
        )
        cube_cfg = sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyBaseCfg(),
            collision_props=sim_utils.CollisionBaseCfg(),
        )
        cube_cfg.func("/World/Cube", cube_cfg, translation=(0.0, 0.0, 1.0))
        sim.reset()

        view = OvPhysxView(sim.backend_manager.get_physx_instance(), pattern="/World/Cube", device="cpu")
        binding = view.binding_for("rigid_body_pose")
        buffer = wp.zeros(tuple(binding.shape), dtype=wp.float32, device="cpu")
        binding.read(buffer)
        del view
        gc.collect()

        RETAINED = (sim, binding, buffer)
        """
    )


def _device_sequence_script(devices: tuple[str, ...]) -> str:
    return f"DEVICES = {devices!r}\n" + textwrap.dedent(
        """
        import torch
        from ovphysx.dlpack import DLDeviceType

        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim import SimulationCfg, build_simulation_context
        from isaaclab_ov import tensor_types as TT
        from isaaclab_ov.assets import RigidObject
        from isaaclab_ov.physics import OvPhysxCfg

        def drop_cube(device):
            sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, dt=1.0 / 60.0)
            with build_simulation_context(device=device, sim_cfg=sim_cfg) as sim:
                cube = RigidObject(
                    RigidObjectCfg(
                        prim_path="/World/Cube",
                        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
                        spawn=sim_utils.CuboidCfg(
                            size=(0.5, 0.5, 0.5),
                            rigid_props=sim_utils.RigidBodyBaseCfg(),
                            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                            collision_props=sim_utils.CollisionBaseCfg(),
                        ),
                    )
                )
                sim.reset()

                # A CPU scene after a CUDA scene must not inherit its DirectGPU state bindings.
                native_device = cube.root_view.binding_for(TT.RIGID_BODY_POSE).native_device
                expected_type = DLDeviceType.kDLCUDA if device.startswith("cuda") else DLDeviceType.kDLCPU
                assert native_device.device_type.value == expected_type, (device, str(native_device))

                root_pose = cube.data.root_link_pose_w.torch.clone()
                assert root_pose.device == torch.device(device)
                root_pose[:, 0] = 0.25
                cube.write_root_link_pose_to_sim_index(root_pose=root_pose)
                cube.update(sim.get_physics_dt())
                torch.testing.assert_close(cube.data.root_link_pose_w.torch, root_pose)

                heights = []
                for _ in range(10):
                    sim.step()
                    cube.update(sim.get_physics_dt())
                    heights.append(cube.data.root_link_pose_w.torch[0, 2].item())
                return heights

        trajectories = {}
        for device in DEVICES:
            heights = drop_cube(device)
            assert all(later < earlier for earlier, later in zip([2.0] + heights, heights)), (device, heights)
            # A device's scene is reproducible regardless of the scenes that ran before it.
            assert trajectories.setdefault(device, heights) == heights, (device, trajectories[device], heights)
        print("DEVICE_SEQUENCE_OK", flush=True)
        """
    )


def _run_child(script: str) -> tuple[subprocess.CompletedProcess[str], str]:
    completed = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    return completed, completed.stdout + completed.stderr


def _assert_no_atexit_errors(output: str) -> None:
    assert "ATEXIT_UNRAISABLE" not in output, output[-8000:]
    assert "Exception ignored in atexit callback" not in output, output[-8000:]
    assert "Error in atexit._run_exitfuncs" not in output, output[-8000:]


def test_retained_binding_preserves_uncaught_failure_exit_status():
    script = _retained_binding_script() + '\nraise RuntimeError("EXPECTED_CHILD_FAILURE")\n'

    completed, output = _run_child(script)

    assert completed.returncode == 1, output[-8000:]
    assert "EXPECTED_CHILD_FAILURE" in output, output[-8000:]
    assert "NORMAL_ATEXIT" in output, output[-8000:]
    assert "OVPHYSX_STOP" in output, output[-8000:]
    _assert_no_atexit_errors(output)


@pytest.mark.parametrize(("device", "rate"), [("cpu", 60), ("gpu", 120)])
@pytest.mark.parametrize(("override", "expected"), [(None, True), (False, False), (True, True)])
def test_scene_settings(monkeypatch, manager, device, rate, override, expected):
    """The simulation timestep, scene queries, and TGS force settings reach the native scene."""
    from isaaclab_ov.physics.ovphysx_manager_cfg import OvPhysxCfg

    from pxr import Usd

    cfg = OvPhysxCfg() if override is None else OvPhysxCfg(enable_external_forces_every_iteration=override)
    assert cfg.enable_external_forces_every_iteration is expected
    sim_cfg = SimpleNamespace(dt=1.0 / rate, enable_scene_query_support=device == "gpu")
    monkeypatch.setattr(manager, "_sim", SimpleNamespace(cfg=sim_cfg))
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World/PhysicsScene", "PhysicsScene")
    manager._configure_physx_scene_prim(prim, cfg, device)
    assert prim.GetAttribute("physxScene:enableExternalForcesEveryIteration").Get() is expected
    assert prim.GetAttribute("physxScene:timeStepsPerSecond").Get() == rate
    assert prim.GetAttribute("physxScene:enableSceneQuerySupport").Get() is sim_cfg.enable_scene_query_support


@pytest.mark.parametrize(("device", "gpu_dynamics", "broadphase"), [("cpu", False, "MBP"), ("gpu", True, "GPU")])
def test_scenes_author_device_dynamics_and_broadphase(monkeypatch, manager, device, gpu_dynamics, broadphase):
    """Every physics scene selects the simulation device, since CPU and GPU scenes share one process."""
    from isaaclab_ov.physics.ovphysx_manager_cfg import OvPhysxCfg

    from pxr import Sdf, Usd, UsdPhysics

    monkeypatch.setattr(
        manager, "_sim", SimpleNamespace(cfg=SimpleNamespace(dt=1.0 / 60.0, enable_scene_query_support=False))
    )
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    # An asset's own scene, authored for the other device.
    asset_scene = UsdPhysics.Scene.Define(stage, "/World/Asset/PhysicsScene").GetPrim()
    asset_scene.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(not gpu_dynamics)
    manager._configure_physics_scenes(stage, "/World/PhysicsScene", OvPhysxCfg(), device)
    for path in ("/World/PhysicsScene", "/World/Asset/PhysicsScene"):
        prim = stage.GetPrimAtPath(path)
        assert "PhysxSceneAPI" in prim.GetMetadata("apiSchemas").GetAddedOrExplicitItems()
        assert prim.GetAttribute("physxScene:enableGPUDynamics").Get() is gpu_dynamics
        assert prim.GetAttribute("physxScene:broadphaseType").Get() == broadphase


def test_construct_physx_forwards_cooked_collider_cache_dir(monkeypatch, manager_module, tmp_path):
    """Configured, default, and unset cache directories reach ``PhysXConfig`` unchanged."""
    from isaaclab_ov.physics.ovphysx_manager_cfg import DEFAULT_COOKED_COLLIDER_CACHE_DIR, OvPhysxBackendCfg, OvPhysxCfg

    monkeypatch.setattr(manager_module, "ovphysx", _fake_ovphysx_module(lambda: None))

    assert OvPhysxCfg().cooked_collider_cache_dir == DEFAULT_COOKED_COLLIDER_CACHE_DIR
    for cache_dir in (DEFAULT_COOKED_COLLIDER_CACHE_DIR, str(tmp_path / "configured_cache"), None):
        backend = manager_module.OvPhysxBackend(OvPhysxBackendCfg(device="cpu", cooked_collider_cache_dir=cache_dir))
        assert backend.physx.config.cooked_collider_cache_dir == cache_dir


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX ownership and mode semantics")
def test_default_cache_dir_is_created_owner_only(manager_module, tmp_path, monkeypatch):
    """The default directory is created ``0o700`` so another user cannot pre-own or read it."""
    import stat

    target = tmp_path / "ovphysx_derived_data_cache_1000"
    monkeypatch.setattr(manager_module, "DEFAULT_COOKED_COLLIDER_CACHE_DIR", str(target))

    assert manager_module._prepare_default_cache_dir(str(target)) == str(target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX ownership and mode semantics")
def test_default_cache_dir_rejects_a_planted_symlink(manager_module, tmp_path, monkeypatch):
    """A symlink planted at the predictable path is refused instead of written through."""
    victim = tmp_path / "victim"
    victim.mkdir()
    target = tmp_path / "ovphysx_derived_data_cache_1000"
    target.symlink_to(victim)
    monkeypatch.setattr(manager_module, "DEFAULT_COOKED_COLLIDER_CACHE_DIR", str(target))

    with pytest.raises(RuntimeError, match="symlink"):
        manager_module._prepare_default_cache_dir(str(target))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX ownership and mode semantics")
def test_default_cache_dir_rejects_a_directory_owned_by_another_user(manager_module, tmp_path, monkeypatch):
    """A pre-existing directory this user does not own is refused."""
    import os as _os

    target = tmp_path / "ovphysx_derived_data_cache_1000"
    target.mkdir(mode=0o700)
    monkeypatch.setattr(manager_module, "DEFAULT_COOKED_COLLIDER_CACHE_DIR", str(target))
    monkeypatch.setattr(_os, "getuid", lambda: _os.stat(target).st_uid + 1)

    with pytest.raises(RuntimeError, match="owned"):
        manager_module._prepare_default_cache_dir(str(target))


@pytest.mark.skipif(not (_CPU_DEVICES and _CUDA_DEVICES), reason="The device sequence requires a CPU and a CUDA device")
def test_cpu_and_cuda_scenes_run_sequentially_in_one_process():
    """A CPU scene must not prevent a later CUDA scene in the same process, or the reverse."""
    cpu, cuda = _CPU_DEVICES[0], _CUDA_DEVICES[0]
    completed, output = _run_child(_device_sequence_script((cpu, cuda, cpu)))

    assert completed.returncode == 0, output[-8000:]
    assert "DEVICE_SEQUENCE_OK" in output, output[-8000:]


@pytest.mark.parametrize("warmup, clone_fails", [("warmup_gpu", False), ("warmup", False), ("warmup", True)])
def test_warmup_collects_clone_receipt_before_model_init(monkeypatch, manager_module, manager, warmup, clone_fails):
    """Accepted clone failures prevent warmup and model-ready publication."""
    from pxr import Usd, UsdPhysics

    from isaaclab.physics import PhysicsEvent

    monkeypatch.setattr(manager_module.ovphysx_compat, "OVPHYSX_USES_TASKS", True)
    monkeypatch.setattr(manager_module, "OVPHYSX_LIFECYCLE_ENTRY_POINTS", {"warmup": warmup})
    events = []
    task = object()
    targets = ["/World/env_1", "/World/env_2"]
    transforms = [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)]
    env_ids = [4, 8]
    clones = [("/World/env_0", targets, transforms, env_ids, 0)]

    class FakePhysX:
        def warmup(self):
            events.append("warmup")

        def warmup_gpu(self):
            events.append("warmup_gpu")

        def clone(self, source, destinations, poses, *, env_ids):
            events.append(("clone", source, destinations, poses, env_ids))
            return task

        def wait_task(self, receipt):
            assert receipt is task
            events.append("wait")
            if clone_fails:
                raise RuntimeError("clone task failed")

    backend = SimpleNamespace(physx=FakePhysX(), stage=None, load=lambda usda: None)
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    sim = manager._sim
    sim.stage = stage
    sim.cfg = SimpleNamespace(
        physics_prim_path="/World/PhysicsScene",
        physics=SimpleNamespace(cooked_collider_cache_dir=None, legacy_joint_friction_replacements={}),
        dt=1.0 / 60.0,
        enable_scene_query_support=False,
    )
    sim.get_clone_plan = lambda: None
    sim.get_or_create_backend = lambda cfg, *, scene: backend
    monkeypatch.setattr(manager, "_device", "cuda:0")
    monkeypatch.setattr(manager, "_clone_recipes", clones)
    monkeypatch.setattr(manager_module, "_serialize_stage", lambda *args, **kwargs: ("#usda 1.0", clones))
    monkeypatch.setattr(
        manager, "_scene_data_backend", SimpleNamespace(_defer_setup=lambda *args: events.append("defer_bindings"))
    )
    monkeypatch.setattr(manager, "dispatch_event", lambda event, payload: events.append(event))

    with pytest.raises(RuntimeError, match="clone task failed") if clone_fails else nullcontext():
        manager._warmup_and_load()

    assert events == [("clone", "/World/env_0", targets, transforms, env_ids), "wait"] + (
        [] if clone_fails else [warmup, "defer_bindings", PhysicsEvent.MODEL_INIT]
    )
    assert manager._warmup_done is (not clone_fails)
