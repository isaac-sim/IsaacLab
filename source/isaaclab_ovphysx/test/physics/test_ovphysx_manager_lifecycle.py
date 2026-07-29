# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for OVPhysX 0.5.9 bootstrap and shutdown."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from types import ModuleType

import pytest

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

_MANAGER_STATE_FIELDS = (
    "_cfg",
    "_physx",
    "_ovstage",
    "_stage_usda",
    "_warmup_done",
    "_locked_device",
    "_pending_clones",
    "_atexit_registered",
    "_scene_data_backend",
    "_physx_schemas_registered",
)


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
    import isaaclab_ovphysx.physics.ovphysx_manager as module

    monkeypatch.setattr(module.atexit, "register", lambda callback: None)
    manager = module.OvPhysxManager
    saved = {
        name: list(getattr(manager, name)) if name == "_pending_clones" else getattr(manager, name)
        for name in _MANAGER_STATE_FIELDS
    }
    for name in _MANAGER_STATE_FIELDS:
        setattr(manager, name, [] if name == "_pending_clones" else False if name.endswith("registered") else None)
    manager._warmup_done = False
    manager._atexit_registered = False
    manager._physx_schemas_registered = False
    try:
        yield module
    finally:
        for name, value in saved.items():
            setattr(manager, name, value)


def _fake_ovphysx_module(bootstrap):
    module = ModuleType("ovphysx")
    module.bootstrap = bootstrap
    module.PhysX = _FakePhysX
    module.PhysXConfig = _FakePhysXConfig
    return module


def test_schema_registration_uses_public_codeless_paths(monkeypatch, manager_module):
    manager = manager_module.OvPhysxManager
    schema_paths = ["/schemas/deformable", "/schemas/physx"]
    registrations = []

    fake_ovphysx = ModuleType("ovphysx")
    fake_ovphysx.codeless_schema_paths = lambda: schema_paths

    class FakeRegistry:
        def RegisterPlugins(self, paths):
            registrations.append(list(paths))

    fake_pxr = ModuleType("pxr")
    fake_pxr.Plug = type("FakePlug", (), {"Registry": staticmethod(FakeRegistry)})
    monkeypatch.setitem(sys.modules, "ovphysx", fake_ovphysx)
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)

    manager._ensure_physx_schemas_registered()
    manager._ensure_physx_schemas_registered()

    assert registrations == [schema_paths]


def test_bootstrap_preserves_pxr_and_registers_normal_cleanup(monkeypatch, manager_module):
    manager = manager_module.OvPhysxManager
    host_pxr = ModuleType("pxr")
    host_usd = ModuleType("pxr.Usd")
    registrations = []

    monkeypatch.setitem(sys.modules, "pxr", host_pxr)
    monkeypatch.setitem(sys.modules, "pxr.Usd", host_usd)
    monkeypatch.setattr(manager_module.atexit, "register", registrations.append)

    def bootstrap():
        assert sys.modules["pxr"] is host_pxr
        assert sys.modules["pxr.Usd"] is host_usd

    monkeypatch.setattr(manager_module, "import_ovphysx", lambda: _fake_ovphysx_module(bootstrap))

    manager._construct_physx("cpu", 0)

    assert sys.modules["pxr"] is host_pxr
    assert sys.modules["pxr.Usd"] is host_usd
    assert registrations == [manager.close]


def test_release_destroys_bindings_before_runtime_and_stage(monkeypatch, manager_module):
    manager = manager_module.OvPhysxManager
    events = []

    class FakePhysX:
        def reset_stage(self):
            events.append("reset")
            return 7

        def wait_op(self, operation):
            events.append(("wait", operation))

        def release(self):
            events.append("release")

    class FakeStage:
        def destroy(self):
            events.append("destroy_stage")

    physx = FakePhysX()
    manager._physx = physx
    manager._ovstage = FakeStage()
    monkeypatch.setattr(
        manager, "_close_physx_views", staticmethod(lambda value: events.append(("close_views", value)))
    )

    manager._release_physx()

    assert events == [("close_views", physx), "reset", ("wait", 7), "release", "destroy_stage"]
    assert manager._physx is None
    assert manager._ovstage is None


def test_close_dispatches_stop_before_runtime_release(monkeypatch, manager_module):
    from isaaclab.physics import PhysicsManager

    manager = manager_module.OvPhysxManager
    events = []
    monkeypatch.setattr(PhysicsManager, "close", classmethod(lambda cls: events.append("stop")))
    monkeypatch.setattr(manager, "_release_physx", classmethod(lambda cls: events.append("release")))

    manager.close()

    assert events == ["stop", "release"]


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


def test_retained_binding_exits_through_normal_atexit():
    script = textwrap.dedent(
        """
        import atexit
        import gc

        import warp as wp

        atexit.register(lambda: print("NORMAL_ATEXIT", flush=True))
        wp.init()

        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab_ovphysx.assets import RigidObject
        from isaaclab_ovphysx.physics import OvPhysxCfg, OvPhysxManager
        from isaaclab_ovphysx.sim.views import OvPhysxView

        sim = SimulationContext(SimulationCfg(physics=OvPhysxCfg(), device="cpu", dt=1.0 / 60.0))
        obj = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.5, 0.5),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
            )
        )
        sim.reset()

        view = OvPhysxView(OvPhysxManager.get_physx_instance(), pattern="/World/Cube", device="cpu")
        binding = view.binding_for("rigid_body_pose")
        buffer = wp.zeros(tuple(binding.shape), dtype=wp.float32, device="cpu")
        binding.read(buffer)
        del view
        gc.collect()

        RETAINED = (sim, obj, binding, buffer)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output[-8000:]
    assert "NORMAL_ATEXIT" in output, output[-8000:]
