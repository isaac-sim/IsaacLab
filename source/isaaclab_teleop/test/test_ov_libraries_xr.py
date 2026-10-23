# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the Isaac Lab host side of external OVXR presentation."""

from __future__ import annotations

import ast
import builtins
import contextlib
import importlib
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class _FakeLease:
    def __init__(self):
        self.ovrtx_renderer = object()
        self.ovstage = object()
        self.partition_calls = []
        self.validated_ordinals = []
        self.owner_lane_entries = 0
        self.auxiliary_render_product_path = None

    @contextmanager
    def ovstage_owner_lane(self):
        self.owner_lane_entries += 1
        yield

    def configure_external_camera_scene_partitions(self, environment_root_path, camera_paths):
        self.partition_calls.append((environment_root_path, camera_paths))

    def validate_external_scene_partitions(self, ordinal):
        self.validated_ordinals.append(ordinal)


class _FakePublisher:
    def __init__(self):
        self.calls = []
        self.fail = None
        self.required = True
        self.closed = False

    def publication_required(self, _publication_id):
        return self.required

    def write_complete_scene(self, ordinal, publication_id):
        self.calls.append((ordinal, publication_id, threading.get_native_id()))
        if self.fail is not None:
            raise self.fail

    def close(self):
        self.closed = True


class _FakeOrdinalLedger:
    def __init__(self, stage):
        self.stage = stage
        self.next_ordinal = 11
        self.reserved = []
        self.published = []
        self.abandoned = []
        self.publish_error = None
        self.destroyed = False
        self.external_resolutions = []

    def reserve(self):
        ordinal = self.next_ordinal
        self.next_ordinal += 1
        self.reserved.append(ordinal)
        return ordinal

    def publish(self, ordinal):
        if self.publish_error is not None:
            raise self.publish_error
        self.published.append(ordinal)

    def abandon(self, ordinal):
        self.abandoned.append(ordinal)

    def resolve_external(self, ordinal, *, expected_published):
        self.external_resolutions.append((ordinal, expected_published))
        return True

    def destroy(self):
        self.destroyed = True


class _FakeBootstrap:
    def __init__(self):
        self.lease = _FakeLease()
        self.publisher = _FakePublisher()
        self.events = []
        self.closed = False
        self.aborted = False
        self.auxiliary_camera_cfgs = []

    def capture_scene_publisher(self):
        self.events.append("capture")
        return self.publisher

    def offer_render_owner_transfer(self):
        self.events.append("offer")
        return "capability"

    def claim_render_owner_transfer(self, capability):
        self.events.append(("claim", capability))

    def bind_environment(self, env):
        self.events.append(("bind", env))
        return self.lease

    def configure_auxiliary_camera(self, camera_cfg, *, max_fps=30.0):
        self.auxiliary_camera_cfgs.append(camera_cfg)

    def get_target_frame_T_world(self, path):
        return ("target", path)

    def close(self):
        self.closed = True

    def abort(self):
        self.aborted = True


@pytest.fixture
def xr_module(monkeypatch):
    """Import the adapter with native renderer dependencies stubbed."""
    isaaclab_ov = ModuleType("isaaclab_ov")
    isaaclab_ov.__path__ = []
    ovxr_session = ModuleType("isaaclab_ov.ovxr_session")
    ovxr_session.OvxrSessionBootstrap = object
    # Execute the actual pure eligibility function with host-owned endpoints.
    from isaaclab.sim import SimulationContext

    source = Path(__file__).parents[2] / "isaaclab_ov/isaaclab_ov/ovxr_session.py"
    tree = ast.parse(source.read_text())
    check = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "validate_ovxr_session_preparation"
    )
    namespace = {
        "SimulationContext": SimulationContext,
        "builtins": builtins,
        "_TERMINAL_MODE_SENTINEL": "ISAAC_LAUNCHED_FROM_TERMINAL",
    }
    exec(compile(ast.Module(body=[check], type_ignores=[]), str(source), "exec"), namespace)
    ovxr_session.validate_ovxr_session_preparation = namespace[check.name]
    renderers = ModuleType("isaaclab_ov.renderers")
    renderers.OVRTXRendererCfg = object
    monkeypatch.setitem(sys.modules, "isaaclab_ov", isaaclab_ov)
    monkeypatch.setitem(sys.modules, "isaaclab_ov.ovxr_session", ovxr_session)
    monkeypatch.setitem(sys.modules, "isaaclab_ov.renderers", renderers)
    sys.modules.pop("isaaclab_teleop.ov_libraries_xr", None)
    module = importlib.import_module("isaaclab_teleop.ov_libraries_xr")
    module._test_declare_ovxr_runtime_prims = module._declare_ovxr_runtime_prims
    monkeypatch.setattr(module, "_import_ovxr", _fake_ovxr)
    return module


def _fake_ovxr():
    return SimpleNamespace(apply_renderer_settings=lambda _settings: None)


def test_stage_ordinal_ledger_tracks_one_application_reservation(xr_module):
    class ImmediateQuery:
        def __init__(self, fetch):
            self._fetch = fetch

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

        def wait(self):
            pass

        def fetch(self):
            return self._fetch()

    class ImmediateAdvance:
        def wait(self):
            pass

    class Stage:
        def __init__(self):
            self.floor = 10
            self.advances = []
            self.queries = 0

        def get_attribute_write_floor(self):
            self.queries += 1
            return ImmediateQuery(lambda: self.floor)

        def advance_write_floor(self, *, ordinal):
            self.advances.append(ordinal)
            self.floor = ordinal
            return ImmediateAdvance()

    stage = Stage()
    ledger = xr_module._StageOrdinalLedger(stage)
    assert stage.queries == 1

    assert ledger.reserve() == 11
    with pytest.raises(RuntimeError, match="still unresolved"):
        ledger.reserve()
    ledger.abandon(11)

    assert ledger.reserve() == 11
    ledger.publish(11)
    assert stage.advances == [11]
    assert stage.queries == 2  # Only the earlier abandon needs reconciliation.

    assert ledger.reserve() == 12
    stage.floor = 12  # The explicit-ordinal OVXR call sealed its own write.
    assert ledger.resolve_external(12, expected_published=True)
    assert stage.queries == 2

    assert ledger.reserve() == 13
    assert not ledger.resolve_external(13, expected_published=None)
    assert stage.queries == 3
    assert ledger.reserve() == 13
    stage.floor = 13
    with pytest.raises(RuntimeError, match="expected unsealed"):
        ledger.resolve_external(13, expected_published=False)
    assert stage.queries == 4
    # Even a contradicted response must not permit reuse of an observed sealed ordinal.
    assert ledger.reserve() == 14
    ledger.abandon(14)
    ledger.destroy()
    with pytest.raises(RuntimeError, match="closed"):
        ledger.reserve()


def test_runtime_render_prims_are_type_only_and_precede_attach(xr_module):
    class FakePrim:
        def __init__(self, type_name=""):
            self.type_name = type_name

        def __bool__(self):
            return bool(self.type_name)

        def IsValid(self):
            return bool(self.type_name)

        def GetTypeName(self):
            return self.type_name

    class FakeStage:
        def __init__(self):
            self.prims = {}
            self.definitions = []

        def GetPrimAtPath(self, path):
            return self.prims.get(path, FakePrim())

        def DefinePrim(self, path, type_name):
            self.definitions.append((path, type_name))
            self.prims[path] = FakePrim(type_name)

    stage = FakeStage()
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=SimpleNamespace(stage=stage)))
    declared = xr_module._test_declare_ovxr_runtime_prims(env, xr_module.OvLibrariesXrConfig())

    assert len(declared) == 7
    assert stage.definitions == list(declared)
    assert all(len(item) == 2 for item in declared)


def test_runtime_render_prim_declaration_preserves_disabled_and_fails_on_conflict(xr_module):
    class ExistingPrim:
        def IsValid(self):
            return True

        def GetTypeName(self):
            return "Xform"

    stage = SimpleNamespace(GetPrimAtPath=lambda _path: ExistingPrim(), DefinePrim=lambda *_args: None)
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=SimpleNamespace(stage=stage)))

    assert (
        xr_module._test_declare_ovxr_runtime_prims(env, xr_module.OvLibrariesXrConfig(create_render_products=False))
        == ()
    )
    with pytest.raises(RuntimeError, match="incompatible type"):
        xr_module._test_declare_ovxr_runtime_prims(env, xr_module.OvLibrariesXrConfig())


def test_hidden_visual_prims_fullmatch_imageables_and_fail_closed(xr_module, monkeypatch):
    class FakePrim:
        def __init__(self, path):
            self.path = path
            self.visible = True

        def GetPath(self):
            return self.path

        def IsA(self, _schema):
            return True

    class FakeImageable:
        def __init__(self, prim):
            self.prim = prim

        def MakeInvisible(self):
            self.prim.visible = False

    pxr = ModuleType("pxr")
    pxr.UsdGeom = SimpleNamespace(Imageable=FakeImageable)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    head = FakePrim("/World/envs/env_0/Robot/head_yaw_link")
    body = FakePrim("/World/envs/env_0/Robot/base_link")
    stage = SimpleNamespace(Traverse=lambda: iter((head, body)))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=SimpleNamespace(stage=stage)))

    assert xr_module._apply_hidden_visual_prims(env, ("/World/envs/env_.*/Robot/head_.*",)) == (head.path,)
    assert not head.visible and body.visible
    with pytest.raises(RuntimeError, match="matched no imageable prims"):
        xr_module._apply_hidden_visual_prims(env, ("/World/missing/.*",))


def _make_host(xr_module, monkeypatch):
    bootstrap = _FakeBootstrap()
    env = SimpleNamespace(close=lambda: bootstrap.events.append("environment-close"))
    monkeypatch.setattr(xr_module, "_import_ovxr", _fake_ovxr)
    monkeypatch.setattr(xr_module, "_StageOrdinalLedger", _FakeOrdinalLedger)
    monkeypatch.setattr(
        xr_module,
        "_apply_hidden_visual_prims",
        lambda _env, _patterns: bootstrap.events.append("hide"),
    )
    monkeypatch.setattr(xr_module, "_declare_ovxr_runtime_prims", lambda _env, _cfg: bootstrap.events.append("declare"))
    host = xr_module._IsaacLabOvxrApplicationInterface(
        bootstrap,
        env,
        xr_module.OvLibrariesXrConfig(),
        ("/hidden",),
    )
    return host, bootstrap, env


def test_host_lends_exact_pair_and_keeps_application_publication(xr_module, monkeypatch):
    host, bootstrap, env = _make_host(xr_module, monkeypatch)

    capability = host.offer_render_owner_transfer()
    binding = host.attach_on_render_thread(capability)
    with host.publication_guard():
        ordinal = host.publish_complete_scene("step-1")

    assert binding.renderer is bootstrap.lease.ovrtx_renderer
    assert binding.stage is bootstrap.lease.ovstage
    assert binding.ordinal_ledger.stage is bootstrap.lease.ovstage
    assert bootstrap.events == ["capture", "offer", ("claim", "capability"), "hide", "declare", ("bind", env)]
    assert ordinal == 11
    assert bootstrap.publisher.calls[0][:2] == (11, "step-1")
    host.validate_first_published_scene(ordinal)
    assert bootstrap.lease.validated_ordinals == [11]
    with host.owner_lane():
        pass
    assert bootstrap.lease.owner_lane_entries == 3
    assert host.get_target_frame_T_world("/World/target") == ("target", "/World/target")

    host.close_publication()
    host.close_on_render_thread()
    host.close_environment_on_render_thread()
    assert bootstrap.publisher.closed and bootstrap.closed
    assert binding.ordinal_ledger.destroyed
    assert bootstrap.events[-1] == "environment-close"


def test_host_abandons_failed_write_and_failed_publish(xr_module, monkeypatch):
    host, bootstrap, _env = _make_host(xr_module, monkeypatch)
    host.attach_on_render_thread(host.offer_render_owner_transfer())
    bootstrap.publisher.fail = RuntimeError("write failed")
    with host.publication_guard(), pytest.raises(RuntimeError, match="write failed"):
        host.publish_complete_scene("step-1")
    assert host._ordinal_ledger.abandoned == [11]

    bootstrap.publisher.fail = None
    host._ordinal_ledger.publish_error = RuntimeError("publish failed")
    with host.publication_guard(), pytest.raises(RuntimeError, match="publish failed"):
        host.publish_complete_scene("step-2")
    assert host._ordinal_ledger.abandoned == [11, 12]


@pytest.mark.parametrize("sealed", [False, True])
def test_host_reconciles_failed_publication_before_reserving_again(xr_module, monkeypatch, sealed):
    ledger_type = xr_module._StageOrdinalLedger
    host, _bootstrap, _env = _make_host(xr_module, monkeypatch)
    host.attach_on_render_thread(host.offer_render_owner_transfer())
    failure = RuntimeError("native publication wait failed")
    stage = SimpleNamespace(floor=10)
    stage.get_attribute_write_floor = lambda: contextlib.nullcontext(
        SimpleNamespace(wait=lambda: None, fetch=lambda: stage.floor)
    )

    def failed_advance(*, ordinal):
        if sealed:
            stage.floor = ordinal

        def wait():
            raise failure

        return SimpleNamespace(wait=wait)

    stage.advance_write_floor = failed_advance
    ledger = ledger_type(stage)
    host._ordinal_ledger = ledger
    with host.publication_guard(), pytest.raises(RuntimeError) as caught:
        host.publish_complete_scene("step-1")
    assert caught.value is failure
    next_ordinal = ledger.reserve()
    assert next_ordinal == (12 if sealed else 11)
    ledger.abandon(next_ordinal)
    ledger.destroy()


def test_failed_publication_with_unknown_floor_retains_reservation_and_first_error(xr_module, monkeypatch):
    ledger_type = xr_module._StageOrdinalLedger
    host, _bootstrap, _env = _make_host(xr_module, monkeypatch)
    host.attach_on_render_thread(host.offer_render_owner_transfer())
    stage = SimpleNamespace(unknown=False)
    failure = RuntimeError("native publication wait failed")

    def fetch():
        if stage.unknown:
            raise RuntimeError("floor read unavailable")
        return 10

    stage.get_attribute_write_floor = lambda: contextlib.nullcontext(SimpleNamespace(wait=lambda: None, fetch=fetch))

    def failed_advance(*, ordinal):
        stage.unknown = True

        def wait():
            raise failure

        return SimpleNamespace(wait=wait)

    stage.advance_write_floor = failed_advance
    ledger = ledger_type(stage)
    host._ordinal_ledger = ledger
    with host.publication_guard(), pytest.raises(RuntimeError) as caught:
        host.publish_complete_scene("step-1")
    assert caught.value is failure
    assert any("floor read unavailable" in note for note in getattr(caught.value, "__notes__", []))
    with pytest.raises(RuntimeError, match="still unresolved"):
        ledger.reserve()


@pytest.mark.parametrize("failure", ["write", "publish"])
def test_publication_retains_primary_cause_when_abandon_fails(xr_module, monkeypatch, failure):
    host, bootstrap, _env = _make_host(xr_module, monkeypatch)
    host.attach_on_render_thread(host.offer_render_owner_transfer())
    original_cause = ValueError("original publication cause")
    first_error = RuntimeError("publication failed")
    first_error.__cause__ = original_cause
    if failure == "write":
        bootstrap.publisher.fail = first_error
    else:
        host._ordinal_ledger.publish_error = first_error

    def fail_abandon(_ordinal):
        raise RuntimeError("abandon failed")

    monkeypatch.setattr(host._ordinal_ledger, "abandon", fail_abandon)
    with host.publication_guard(), pytest.raises(RuntimeError) as caught:
        host.publish_complete_scene("step-1")
    assert caught.value is first_error
    assert caught.value.__cause__ is original_cause
    assert any("abandon failed" in note for note in caught.value.__notes__)


def test_session_delegates_lifecycle_to_isaacteleop_owner(xr_module, monkeypatch):
    calls = []

    class FakeOwner:
        owns_environment = True
        render_worker_bootstrap_ready = True
        shutdown_requested = False
        desktop_mirror_status = SimpleNamespace(main_thread_fps=30.0, render_thread_fps=40.0)

        def __init__(self, host, config, **kwargs):
            calls.append(("init", host, config, kwargs))

        def start(self, timeout_s):
            calls.append(("start", timeout_s))
            return True

        def external_oxr_handles(self):
            return "handles"

        def request_render(self, identity, *, force=False):
            calls.append(("request", identity, force))
            return 17

        def set_desktop_mirror_status_text(self, text):
            calls.append(("status", text))

        def request_shutdown(self):
            self.shutdown_requested = True
            calls.append(("request_shutdown",))

        def get_target_frame_T_world(self, path):
            return ("target", path)

        def close(self):
            calls.append(("close",))

    external = importlib.import_module("isaaccapture.ovxr_session_manager.ovxr_session")
    monkeypatch.setattr(external, "OvxrSession", FakeOwner)
    bootstrap = _FakeBootstrap()
    assert xr_module.OvLibrariesXrConfig().extra_openxr_extensions == ()
    session = xr_module.OvLibrariesXrSession(
        bootstrap, xr_module.OvLibrariesXrConfig(extra_openxr_extensions=("XR_EXT_debug_utils",))
    )
    env = SimpleNamespace()
    assert session.start_render_worker(env, required_openxr_extensions=("XR_EXT_hand_tracking",), timeout_s=4.0)
    assert calls[0][0] == "init"
    assert calls[0][2].session_mode == "ar"
    assert calls[0][3]["required_openxr_extensions"] == ("XR_EXT_hand_tracking",)
    assert calls[0][2].extra_openxr_extensions == ("XR_EXT_debug_utils",)
    assert calls[0][2].host_wait_render_output is True
    assert calls[1] == ("start", 4.0)
    assert session.external_oxr_handles() == "handles"
    assert session.request_render("step") == 17
    assert session.request_render("reset", force=True) == 17
    assert calls[-2:] == [("request", "step", False), ("request", "reset", True)]
    assert session.get_target_frame_T_world("/target") == ("target", "/target")
    assert session.async_worker_owns_environment and session.render_worker_bootstrap_ready
    assert not session.shutdown_requested
    session.request_shutdown()
    assert calls[-1] == ("request_shutdown",)
    assert session.shutdown_requested
    session._owner.shutdown_requested = True
    assert session.shutdown_requested
    session.close()
    assert calls[-1] == ("close",)


def test_session_abort_before_owner_delegates_to_bootstrap(xr_module):
    bootstrap = _FakeBootstrap()
    session = xr_module.OvLibrariesXrSession(bootstrap, xr_module.OvLibrariesXrConfig())
    session.abort()
    assert bootstrap.aborted


@pytest.mark.parametrize(
    "settings", ["", "--/rtx/rendermode=MinimalRendering", "--/rtx/rendering/perSensorTickTlas=true"]
)
def test_caller_renderer_settings_precede_renderer_creation(xr_module, monkeypatch, settings):
    events = []
    bootstrap = _FakeBootstrap()
    monkeypatch.setattr(
        xr_module, "_import_ovxr", lambda: SimpleNamespace(apply_renderer_settings=lambda value: events.append(value))
    )

    def create_bootstrap(*_args):
        assert len(events) == 1
        assert events[0] == settings.strip()
        return bootstrap

    monkeypatch.setattr(xr_module, "OvxrSessionBootstrap", create_bootstrap)
    xr_module.OvLibrariesXrSession.prepare(object(), xr_module.OvLibrariesXrConfig(renderer_settings=settings))


def test_config_rejects_unknown_session_mode(xr_module):
    with pytest.raises(ValueError, match="session_mode must be 'ar' or 'vr'"):
        xr_module.OvLibrariesXrConfig(session_mode="automatic")


def test_public_xr_exports_resolve_to_implemented_types(xr_module):
    import isaaclab_teleop

    assert isaaclab_teleop.OvLibrariesXrConfig is xr_module.OvLibrariesXrConfig
    assert isaaclab_teleop.OvLibrariesXrSession is xr_module.OvLibrariesXrSession
    assert "OvLibrariesXrFrame" not in isaaclab_teleop.__all__
