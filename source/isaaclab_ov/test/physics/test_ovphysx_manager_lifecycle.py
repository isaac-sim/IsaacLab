# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for OVPhysX 0.5.9 bootstrap and shutdown."""

from __future__ import annotations

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
def manager_module(monkeypatch):
    """Import the manager and restore its class-global state after each test."""
    import isaaclab_ov.physics.ovphysx_manager as module
    from isaaclab_ov.physics import OvPhysxBackendCfg

    from isaaclab.physics import PhysicsManager
    from isaaclab.sim import SimulationContext

    cfg = OvPhysxBackendCfg(device="cpu")
    sim = SimpleNamespace(_backend_registry=[], physics_manager=module.OvPhysxManager)
    sim.get_or_create_backend = SimulationContext.get_or_create_backend.__get__(sim)
    sim.close_backend = SimulationContext.close_backend.__get__(sim)
    with monkeypatch.context() as construction:
        construction.setattr(module, "import_ovphysx", lambda: _fake_ovphysx_module(lambda: None))
        backend = sim.get_or_create_backend(cfg)
    monkeypatch.setattr(PhysicsManager, "_sim", sim)
    monkeypatch.setattr(SimulationContext, "_instance", sim)
    monkeypatch.setattr(module.atexit, "register", lambda callback: None)
    manager = module.OvPhysxManager
    test_state = {
        "_cfg": None,
        "backend": backend,
        "_stage_usda": None,
        "_next_control_ordinal": 2,
        "_warmup_done": False,
        "_requires_full_stage": False,
        "_clone_recipes": [],
        "_atexit_registered": False,
        "_scene_data_backend": None,
        "_physx_schemas_registered": False,
        "_gravity": None,
    }
    for name, value in test_state.items():
        monkeypatch.setattr(manager, name, value)
    return module


def _fake_ovphysx_module(bootstrap):
    module = ModuleType("ovphysx")
    module.bootstrap = bootstrap
    module.PhysX = _FakePhysX
    module.PhysXConfig = _FakePhysXConfig
    return module


def test_initialize_defers_native_resource_until_warmup(monkeypatch, manager_module):
    from isaaclab.physics import PhysicsManager

    manager = manager_module.OvPhysxManager
    sim = SimpleNamespace(
        cfg=SimpleNamespace(physics=None, device="cpu", gravity=(0.0, 0.0, -9.81)),
        stage=object(),
        _backend_registry=[],
        clone_contexts={},
    )
    for name in ("_sim", "_cfg", "_device", "_sim_time"):
        monkeypatch.setattr(PhysicsManager, name, getattr(PhysicsManager, name))
    monkeypatch.setattr(manager, "_ensure_physx_schemas_registered", lambda: None)
    monkeypatch.setattr(manager, "backend", None)

    manager.initialize(sim)

    assert not sim._backend_registry
    assert manager.get_physx_instance() is None
    assert not any(hasattr(manager, name) for name in ("_backend", "_physx", "_ovstage"))


@pytest.mark.parametrize(
    ("registered_names", "expected_paths", "schema_root", "has_registration_api"),
    [
        (["physxSchema"], ["/schemas/OmniUsdPhysicsDeformableSchema/resources"], "/schemas", True),
        (["PhysxSchema", "OmniUsdPhysicsDeformableSchema"], [], "/schemas", True),
        (["PhysxSchema", "OmniUsdPhysicsDeformableSchema"], [], None, True),
        (["physxSchema"], ["/schemas/OmniUsdPhysicsDeformableSchema/resources"], "/schemas", False),
        pytest.param(
            ["physxSchema"],
            ["/schemas/OmniUsdPhysicsDeformableSchema/resources"],
            "/schemas",
            None,
            id="ovstage-import-unavailable",
        ),
    ],
)
def test_schema_registration_skips_providers_already_supplied_by_host(
    monkeypatch, manager_module, registered_names, expected_paths, schema_root, has_registration_api
):
    manager = manager_module.OvPhysxManager
    schema_paths = [
        Path("/schemas/PhysxSchema/resources"),
        Path("/schemas/OmniUsdPhysicsDeformableSchema/resources"),
    ]
    host_registrations = []
    ovstage_registrations = []

    fake_ovphysx = ModuleType("ovphysx")
    fake_ovphysx.codeless_schema_paths = lambda: schema_paths
    if schema_root is not None:
        fake_ovphysx.codeless_schema_root = lambda: Path(schema_root)

    fake_ovstage = ModuleType("ovstage")
    fake_ovstage.population = SimpleNamespace()
    if has_registration_api:
        fake_ovstage.population.register_usd_schemas = ovstage_registrations.append

    class FakeRegistry:
        def GetAllPlugins(self):
            return [SimpleNamespace(name=name) for name in registered_names]

        def RegisterPlugins(self, paths):
            host_registrations.append(list(paths))

    fake_pxr = ModuleType("pxr")
    fake_pxr.Plug = type("FakePlug", (), {"Registry": staticmethod(FakeRegistry)})
    monkeypatch.setitem(sys.modules, "ovphysx", fake_ovphysx)
    # A None entry makes importing OVStage raise ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, "ovstage", fake_ovstage if has_registration_api is not None else None)
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    newton_schema_root = "/schemas/newton"
    monkeypatch.setattr(manager_module, "_newton_schema_root", lambda: newton_schema_root, raising=False)

    manager._ensure_physx_schemas_registered()
    manager._ensure_physx_schemas_registered()

    expected_ovstage_registrations = []
    if has_registration_api:
        if schema_root is not None:
            expected_ovstage_registrations.append(schema_root)
        expected_ovstage_registrations.append(newton_schema_root)
    assert ovstage_registrations == expected_ovstage_registrations
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

    monkeypatch.setattr(manager_module, "import_ovphysx", lambda: _fake_ovphysx_module(bootstrap))

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
def test_close_releases_runtime_after_stop_even_on_listener_failure(monkeypatch, manager_module, stop_fails):
    from isaaclab.physics import PhysicsManager

    manager = manager_module.OvPhysxManager
    events = []

    def stop(cls):
        events.append("stop")
        if stop_fails:
            raise ValueError("listener failure")

    monkeypatch.setattr(PhysicsManager, "close", classmethod(stop))
    monkeypatch.setattr(manager.backend, "close", lambda: events.append("release"))

    with pytest.raises(ValueError, match="listener failure") if stop_fails else nullcontext():
        manager.close()

    assert events == ["stop", "release"]


def test_atexit_cleanup_noops_after_explicit_close(monkeypatch, manager_module):
    manager = manager_module.OvPhysxManager
    manager.backend.physx = None
    monkeypatch.setattr(manager, "close", classmethod(lambda cls: pytest.fail("unexpected close")))
    monkeypatch.setattr(manager.backend, "close", lambda: pytest.fail("unexpected release"))

    manager._close_at_exit()


def test_atexit_cleanup_releases_stale_runtime_without_clearing_active_backend(monkeypatch, manager_module):
    from isaaclab.physics import PhysicsManager

    manager = manager_module.OvPhysxManager
    events = []
    sentinel_callbacks = {17: object()}
    manager.backend.physx = object()
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(physics_manager=object()))
    monkeypatch.setattr(PhysicsManager, "_callbacks", sentinel_callbacks)

    def release():
        events.append("release")
        manager.backend.physx = None

    monkeypatch.setattr(manager.backend, "close", release)

    manager._close_at_exit()

    assert events == ["release"]
    assert PhysicsManager._callbacks is sentinel_callbacks


def test_atexit_cleanup_logs_and_swallows_active_close_failure(monkeypatch, manager_module, caplog):
    from isaaclab.physics import PhysicsManager

    manager = manager_module.OvPhysxManager
    events = []
    manager.backend.physx = object()
    lazy_manager = f"{manager.__module__}:{manager.__qualname__}"
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(physics_manager=lazy_manager))

    def fail_close(cls):
        events.append("close")
        raise RuntimeError("failure")

    monkeypatch.setattr(manager, "close", classmethod(fail_close))
    monkeypatch.setattr(manager.backend, "close", lambda: events.append("release"))

    manager._close_at_exit()

    assert events == ["close"]
    assert "Failed to close OVPhysX during process exit." in caplog.text


def test_stage_reuse_drains_bindings_before_reset(monkeypatch, manager_module):
    manager = manager_module.OvPhysxManager
    events = []

    class FakePhysX:
        def reset_stage(self):
            events.append("reset")
            return 9

        def wait_op(self, operation):
            events.append(("wait", operation))

    physx = FakePhysX()
    manager.backend.physx = physx
    monkeypatch.setattr(
        manager_module.OvPhysxView, "_close_all_for", lambda value: events.append(("close_views", value))
    )
    manager.backend.stage = SimpleNamespace(destroy=lambda: events.append("destroy_stage"))

    manager._prepare_physx_for_stage_reuse()

    assert events == [
        ("close_views", physx),
        "reset",
        ("wait", 9),
        "destroy_stage",
    ]


@pytest.mark.parametrize("failure", [None, "query", "write"])
def test_set_gravity_writes_and_releases_ovstage_control_resources(monkeypatch, manager_module, failure):
    """Scene gravity updates must seal their ordinal and release resources on every path."""
    manager = manager_module.OvPhysxManager
    calls = []

    class FakePathDictionary:
        def __init__(self, stage):
            self.stage = stage

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append(("destroy_dictionary",))

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

    fake_ovstage = ModuleType("ovstage")
    fake_ovstage.PathDictionary = FakePathDictionary
    monkeypatch.setitem(sys.modules, "ovstage", fake_ovstage)
    manager.backend.stage = FakeStage()
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
            ("write", "physics-scene-query", "physics:gravityDirection", 2, [[0.0, 0.0, -1.0]], False),
            ("write", "physics-scene-query", "physics:gravityMagnitude", 2, [pytest.approx(3.72)], False),
            ("seal", 2),
            ("update", 2, 2),
            ("release_query", "physics-scene-query"),
            ("destroy_paths", "physics-scene-paths"),
            ("destroy_dictionary",),
        ]
    else:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            manager.set_gravity((0.0, 0.0, -3.72))
        expected_calls = [
            ("paths", ["/World/physicsScene"]),
            ("query", "physics-scene-paths"),
        ]
        if failure == "write":
            expected_calls.extend(
                [
                    ("write", "physics-scene-query", "physics:gravityDirection", 2, [[0.0, 0.0, -1.0]], False),
                    ("release_query", "physics-scene-query"),
                ]
            )
        expected_calls.extend(
            [
                ("destroy_paths", "physics-scene-paths"),
                ("destroy_dictionary",),
            ]
        )

    assert calls == expected_calls
    # ``get_gravity`` must report what the scene is actually running with: the new vector
    # once the ordinal was applied, and the previous one when the update failed.
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
        assert sim.physics_manager == f"{OvPhysxManager.__module__}:{OvPhysxManager.__qualname__}"
        OvPhysxManager.register_callback(
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

        view = OvPhysxView(OvPhysxManager.get_physx_instance(), pattern="/World/Cube", device="cpu")
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
def test_scene_settings(monkeypatch, manager_module, device, rate, override, expected):
    """The simulation timestep, scene queries, and TGS force settings reach the native scene."""
    from isaaclab_ov.physics.ovphysx_manager_cfg import OvPhysxCfg

    from pxr import Usd

    from isaaclab.physics import PhysicsManager

    cfg = OvPhysxCfg() if override is None else OvPhysxCfg(enable_external_forces_every_iteration=override)
    assert cfg.enable_external_forces_every_iteration is expected
    sim_cfg = SimpleNamespace(dt=1.0 / rate, enable_scene_query_support=device == "gpu")
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(cfg=sim_cfg))
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World/PhysicsScene", "PhysicsScene")
    manager_module.OvPhysxManager._configure_physx_scene_prim(prim, cfg, device)
    assert prim.GetAttribute("physxScene:enableExternalForcesEveryIteration").Get() is expected
    assert prim.GetAttribute("physxScene:timeStepsPerSecond").Get() == rate
    assert prim.GetAttribute("physxScene:enableSceneQuerySupport").Get() is sim_cfg.enable_scene_query_support


@pytest.mark.parametrize(("device", "gpu_dynamics", "broadphase"), [("cpu", False, "MBP"), ("gpu", True, "GPU")])
def test_scenes_author_device_dynamics_and_broadphase(monkeypatch, manager_module, device, gpu_dynamics, broadphase):
    """Every physics scene selects the simulation device, since CPU and GPU scenes share one process."""
    from isaaclab_ov.physics.ovphysx_manager_cfg import OvPhysxCfg

    from pxr import Sdf, Usd, UsdPhysics

    from isaaclab.physics import PhysicsManager

    monkeypatch.setattr(
        PhysicsManager, "_sim", SimpleNamespace(cfg=SimpleNamespace(dt=1.0 / 60.0, enable_scene_query_support=False))
    )
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    # An asset's own scene, authored for the other device.
    asset_scene = UsdPhysics.Scene.Define(stage, "/World/Asset/PhysicsScene").GetPrim()
    asset_scene.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(not gpu_dynamics)
    manager_module.OvPhysxManager._configure_physics_scenes(stage, "/World/PhysicsScene", OvPhysxCfg(), device)
    for path in ("/World/PhysicsScene", "/World/Asset/PhysicsScene"):
        prim = stage.GetPrimAtPath(path)
        assert "PhysxSceneAPI" in prim.GetMetadata("apiSchemas").GetAddedOrExplicitItems()
        assert prim.GetAttribute("physxScene:enableGPUDynamics").Get() is gpu_dynamics
        assert prim.GetAttribute("physxScene:broadphaseType").Get() == broadphase


def test_construct_physx_forwards_cooked_collider_cache_dir(monkeypatch, manager_module, tmp_path):
    """Configured, default, and unset cache directories reach ``PhysXConfig`` unchanged."""
    from isaaclab_ov.physics.ovphysx_manager_cfg import DEFAULT_COOKED_COLLIDER_CACHE_DIR, OvPhysxBackendCfg, OvPhysxCfg

    monkeypatch.setattr(manager_module, "import_ovphysx", lambda: _fake_ovphysx_module(lambda: None))

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
