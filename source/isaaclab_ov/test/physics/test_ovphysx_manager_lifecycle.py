# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for OVPhysX 0.5.9 bootstrap and shutdown."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")


class _FakePhysXConfig:
    def __init__(self, num_threads=None, carbonite_overrides=None):
        self.num_threads = num_threads
        self.carbonite_overrides = carbonite_overrides or {}


class _FakePhysX:
    @classmethod
    def set_cpu_mode(cls, enabled):
        pass

    def __init__(self, active_cuda_gpus=None, config=None):
        self.active_cuda_gpus = active_cuda_gpus
        self.config = config


@pytest.fixture
def manager_module(monkeypatch):
    """Import the manager and restore its class-global state after each test."""
    import isaaclab_ov.physics.ovphysx_manager as module

    monkeypatch.setattr(module.atexit, "register", lambda callback: None)
    manager = module.OvPhysxManager
    test_state = {
        "_cfg": None,
        "_physx": None,
        "_ovstage": None,
        "_stage_usda": None,
        "_next_control_ordinal": 2,
        "_warmup_done": False,
        "_requires_full_stage": False,
        "_locked_device": None,
        "_active_clone_recipes": [],
        "_pending_clones": [],
        "_atexit_registered": False,
        "_scene_data_backend": None,
        "_physx_schemas_registered": False,
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
    manager = manager_module.OvPhysxManager
    schema_paths = [
        Path("/schemas/PhysxSchema/resources"),
        Path("/schemas/OmniUsdPhysicsDeformableSchema/resources"),
    ]
    registrations = []

    fake_ovphysx = ModuleType("ovphysx")
    fake_ovphysx.codeless_schema_paths = lambda: schema_paths

    class FakeRegistry:
        def GetAllPlugins(self):
            return [SimpleNamespace(name=name) for name in registered_names]

        def RegisterPlugins(self, paths):
            registrations.append(list(paths))

    fake_pxr = ModuleType("pxr")
    fake_pxr.Plug = type("FakePlug", (), {"Registry": staticmethod(FakeRegistry)})
    monkeypatch.setitem(sys.modules, "ovphysx", fake_ovphysx)
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)

    manager._ensure_physx_schemas_registered()
    manager._ensure_physx_schemas_registered()

    assert registrations == ([expected_paths] if expected_paths else [])


def test_construct_physx_bootstraps_each_runtime_without_replacing_pxr(monkeypatch, manager_module):
    manager = manager_module.OvPhysxManager
    host_pxr = ModuleType("pxr")
    host_usd = ModuleType("pxr.Usd")
    bootstrap_calls = []
    registrations = []

    monkeypatch.setitem(sys.modules, "pxr", host_pxr)
    monkeypatch.setitem(sys.modules, "pxr.Usd", host_usd)
    monkeypatch.setattr(manager_module.atexit, "register", registrations.append)

    def bootstrap():
        bootstrap_calls.append(None)
        assert sys.modules["pxr"] is host_pxr
        assert sys.modules["pxr.Usd"] is host_usd

    monkeypatch.setattr(manager_module, "import_ovphysx", lambda: _fake_ovphysx_module(bootstrap))

    manager._construct_physx("cpu", 0)
    manager._physx = None
    manager._construct_physx("cpu", 0)

    assert sys.modules["pxr"] is host_pxr
    assert sys.modules["pxr.Usd"] is host_usd
    assert bootstrap_calls == [None, None]
    assert registrations == [manager._close_at_exit]


def test_close_dispatches_stop_before_runtime_release(monkeypatch, manager_module):
    from isaaclab.physics import PhysicsManager

    manager = manager_module.OvPhysxManager
    events = []
    monkeypatch.setattr(PhysicsManager, "close", classmethod(lambda cls: events.append("stop")))
    monkeypatch.setattr(manager, "_release_physx", classmethod(lambda cls: events.append("release")))

    manager.close()

    assert events == ["stop", "release"]


def test_close_releases_runtime_after_stop_listener_failure(monkeypatch, manager_module):
    from isaaclab.physics import PhysicsManager

    manager = manager_module.OvPhysxManager
    events = []

    def fail_stop(cls):
        raise ValueError("listener failure")

    monkeypatch.setattr(PhysicsManager, "close", classmethod(fail_stop))
    monkeypatch.setattr(manager, "_release_physx", classmethod(lambda cls: events.append("release")))

    with pytest.raises(ValueError, match="listener failure"):
        manager.close()

    assert events == ["release"]


def test_atexit_cleanup_noops_after_explicit_close(monkeypatch, manager_module):
    manager = manager_module.OvPhysxManager
    monkeypatch.setattr(manager, "_physx", None)
    monkeypatch.setattr(manager, "close", classmethod(lambda cls: pytest.fail("unexpected close")))
    monkeypatch.setattr(manager, "_release_physx", classmethod(lambda cls: pytest.fail("unexpected release")))

    manager._close_at_exit()


def test_atexit_cleanup_releases_stale_runtime_without_clearing_active_backend(monkeypatch, manager_module):
    from isaaclab.physics import PhysicsManager

    manager = manager_module.OvPhysxManager
    events = []
    sentinel_callbacks = {17: object()}
    monkeypatch.setattr(manager, "_physx", object())
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(physics_manager=object()))
    monkeypatch.setattr(PhysicsManager, "_callbacks", sentinel_callbacks)

    def release(cls):
        events.append("release")
        cls._physx = None

    monkeypatch.setattr(manager, "_release_physx", classmethod(release))

    manager._close_at_exit()

    assert events == ["release"]
    assert PhysicsManager._callbacks is sentinel_callbacks


def test_atexit_cleanup_logs_and_swallows_active_close_failure(monkeypatch, manager_module, caplog):
    from isaaclab.physics import PhysicsManager

    manager = manager_module.OvPhysxManager
    events = []
    monkeypatch.setattr(manager, "_physx", object())
    lazy_manager = f"{manager.__module__}:{manager.__qualname__}"
    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(physics_manager=lazy_manager))

    def fail_close(cls):
        events.append("close")
        raise RuntimeError("failure")

    monkeypatch.setattr(manager, "close", classmethod(fail_close))
    monkeypatch.setattr(manager, "_release_physx", classmethod(lambda cls: events.append("release")))

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
    manager._physx = physx
    monkeypatch.setattr(
        manager, "_close_physx_views", staticmethod(lambda value: events.append(("close_views", value)))
    )
    monkeypatch.setattr(manager, "_destroy_ovstage", classmethod(lambda cls: events.append("destroy_stage")))

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
    monkeypatch.setattr(manager, "_ovstage", FakeStage())
    monkeypatch.setattr(manager, "_physx", FakePhysX())
    monkeypatch.setattr(manager, "_sim", SimpleNamespace(cfg=SimpleNamespace(physics_prim_path="/World/physicsScene")))

    if failure is None:
        manager.set_gravity((0.0, 0.0, -9.81))
        expected_calls = [
            ("paths", ["/World/physicsScene"]),
            ("query", "physics-scene-paths"),
            ("write", "physics-scene-query", "physics:gravityDirection", 2, [[0.0, 0.0, -1.0]], False),
            ("write", "physics-scene-query", "physics:gravityMagnitude", 2, [pytest.approx(9.81)], False),
            ("seal", 2),
            ("update", 2, 2),
            ("release_query", "physics-scene-query"),
            ("destroy_paths", "physics-scene-paths"),
            ("destroy_dictionary",),
        ]
    else:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            manager.set_gravity((0.0, 0.0, -9.81))
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
