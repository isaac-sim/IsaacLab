# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for the Isaac Lab host side of external OVXR presentation."""

from __future__ import annotations

import builtins
import contextlib
import importlib
import math
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
        self.auxiliary_render_product_paths = ()

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
def real_lease_type():
    from isaaclab_ov.renderers.render_lease import OvLibrariesRenderLease

    return OvLibrariesRenderLease


@pytest.fixture
def xr_module(monkeypatch, real_lease_type):
    """Import the adapter with native renderer dependencies stubbed."""
    isaaclab_ov = ModuleType("isaaclab_ov")
    isaaclab_ov.__path__ = []
    renderers = ModuleType("isaaclab_ov.renderers")
    renderers.OVRTXRendererCfg = object
    renderers.OVRTXRenderer = object
    renderers.OvLibrariesRenderLease = object
    monkeypatch.setitem(sys.modules, "isaaclab_ov", isaaclab_ov)
    monkeypatch.setitem(sys.modules, "isaaclab_ov.renderers", renderers)
    source = Path(__file__).parents[1] / "isaaclab_teleop/ovxr_session.py"
    spec = importlib.util.spec_from_file_location("isaaclab_teleop.ovxr_session", source)
    ovxr_session = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, ovxr_session)
    spec.loader.exec_module(ovxr_session)
    sys.modules.pop("isaaclab_teleop.ov_libraries_xr", None)
    module = importlib.import_module("isaaclab_teleop.ov_libraries_xr")
    monkeypatch.setitem(sys.modules, "ovxr", _fake_ovxr())
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


def test_runtime_render_prims_have_the_declared_paths_and_types(xr_module):
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
    xr_module._declare_ovxr_runtime_prims(env, xr_module.OvLibrariesXrConfig())

    assert stage.definitions == [
        ("/OVXRGeneratedCameras/OVXR_LeftEyeCamera", "Camera"),
        ("/OVXRGeneratedCameras/OVXR_RightEyeCamera", "Camera"),
        ("/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0_OVXR_Left", "RenderProduct"),
        ("/Render/OmniverseKit/HydraTextures/omni_kit_widget_viewport_ViewportTexture_0_OVXR_Right", "RenderProduct"),
        ("/Render/Vars/LdrColor", "RenderVar"),
        ("/Render/Vars/XRDepth", "RenderVar"),
        ("/Render/OmniverseGlobalRenderSettings", "RenderSettings"),
    ]


def test_runtime_render_prim_declaration_preserves_disabled_and_fails_on_conflict(xr_module):
    class ExistingPrim:
        def IsValid(self):
            return True

        def GetTypeName(self):
            return "Xform"

    stage = SimpleNamespace(GetPrimAtPath=lambda _path: ExistingPrim(), DefinePrim=lambda *_args: None)
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=SimpleNamespace(stage=stage)))

    xr_module._declare_ovxr_runtime_prims(env, xr_module.OvLibrariesXrConfig(create_render_products=False))
    with pytest.raises(RuntimeError, match="incompatible type"):
        xr_module._declare_ovxr_runtime_prims(env, xr_module.OvLibrariesXrConfig())


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
    env = SimpleNamespace(scene=SimpleNamespace(stage=None), close=lambda: bootstrap.events.append("environment-close"))
    monkeypatch.setitem(sys.modules, "ovxr", _fake_ovxr())
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


def test_host_smooths_committed_anchor_on_xr_clock_and_reseeds_reset(xr_module, monkeypatch):
    """Use the real pose policy; substitute only frame time, scene I/O, and native sealing."""
    import numpy as np
    from isaaclab_teleop import xr_anchor_manager
    from isaaclab_teleop.xr_cfg import XrAnchorRotationMode, XrCfg
    from scipy.spatial.transform import Rotation

    from pxr import Usd, UsdGeom

    from isaaclab.sim.utils.stage import use_stage

    monkeypatch.setattr(xr_anchor_manager, "XRCore", None)
    cfg = XrCfg(
        anchor_prim_path="/World/Robot",
        anchor_rotation_mode=XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED,
        anchor_rotation_smoothing_time=0.1,
    )
    with use_stage(Usd.Stage.CreateInMemory()):
        UsdGeom.Xform.Define(xr_anchor_manager.get_current_stage(), "/World/Robot")
        manager = xr_anchor_manager.XrAnchorManager(cfg, ovxr_session=True)

    bootstrap = _FakeBootstrap()
    reference = np.eye(4)
    bootstrap.get_world_transform = lambda _path: reference.copy()
    env = SimpleNamespace(sim=SimpleNamespace(get_rendering_dt=lambda: 0.25))
    host = xr_module._IsaacLabOvxrApplicationInterface(
        bootstrap, env, xr_module.OvLibrariesXrConfig(), (), external_anchor_state=manager
    )
    host._scene_publisher = bootstrap.publisher
    host._ordinal_ledger = _FakeOrdinalLedger(None)
    writes = []
    monkeypatch.setattr(
        host, "_write_anchor_matrices", lambda matrices, ordinal: writes.append((matrices.copy(), ordinal))
    )
    frame_times = iter((10.0, 10.01, 10.02, 10.05, 20.0, 20.1, 20.11))
    monkeypatch.setattr(xr_module.time, "monotonic", lambda: next(frame_times))

    def committed_yaw():
        return Rotation.from_matrix(writes[-1][0][0, :3, :3]).as_euler("xyz")[2]

    seed = host.publish_complete_scene("initial")
    assert writes[-1][1] == seed == host._ordinal_ledger.published[-1]
    host.prepare_frame_scene()
    reference[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    host.prepare_frame_scene()
    assert committed_yaw() == pytest.approx(0.0)  # Unpublished physics must not move the view.
    host.publish_complete_scene("moved")
    host.prepare_frame_scene()
    assert committed_yaw() == pytest.approx(math.pi / 2 * (1.0 - math.exp(-0.01 / 0.1)))
    host.prepare_frame_scene()
    assert committed_yaw() == pytest.approx(math.pi / 2 * (1.0 - math.exp(-0.04 / 0.1)))

    manager.reset()
    write_count = len(writes)
    host.prepare_frame_scene()
    assert len(writes) == write_count  # An old reference cannot seed the new epoch.
    reference[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    seed = host.publish_complete_scene("reset")
    assert writes[-1][1] == seed == host._ordinal_ledger.published[-1]
    assert committed_yaw() == pytest.approx(0.0)
    assert np.isfinite(manager.get_world_matrix()).all()  # Input is usable before the next XR frame.
    host.prepare_frame_scene()
    reference[:3, :3] = Rotation.from_euler("z", 135, degrees=True).as_matrix()
    host.publish_complete_scene("after-reset")
    host.prepare_frame_scene()
    assert committed_yaw() == pytest.approx(math.pi / 2 * (1.0 - math.exp(-0.01 / 0.1)))


def test_host_lends_the_selected_auxiliary_product_without_rendering_it(xr_module, monkeypatch):
    bootstrap = _FakeBootstrap()
    bootstrap.lease.auxiliary_render_product_paths = ("/IsaacLabAuxiliaryRender/RenderProduct_robot_pov_cam",)
    camera = object()
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(scene=SimpleNamespace(stage=None, sensors={"robot_pov_cam": camera}))
    )
    monkeypatch.setattr(xr_module, "_StageOrdinalLedger", _FakeOrdinalLedger)
    monkeypatch.setattr(xr_module, "_apply_hidden_visual_prims", lambda *_args: None)
    monkeypatch.setattr(xr_module, "_declare_ovxr_runtime_prims", lambda *_args: None)
    host = xr_module._IsaacLabOvxrApplicationInterface(
        bootstrap,
        env,
        xr_module.OvLibrariesXrConfig(
            auxiliary_cameras=(
                xr_module.OvLibrariesXrCameraConfig("robot_pov_cam", SimpleNamespace(width=720, height=450)),
            ),
        ),
        (),
    )

    binding = host.attach_on_render_thread(host.offer_render_owner_transfer())

    assert binding.frame_consumers[0].render_product_paths == bootstrap.lease.auxiliary_render_product_paths
    assert not hasattr(binding, "auxiliary_frame_source")


def test_host_rejects_a_missing_selected_auxiliary_camera(xr_module, monkeypatch):
    bootstrap = _FakeBootstrap()
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=SimpleNamespace(stage=None, sensors={})))
    monkeypatch.setattr(xr_module, "_apply_hidden_visual_prims", lambda *_args: None)
    monkeypatch.setattr(xr_module, "_declare_ovxr_runtime_prims", lambda *_args: None)
    host = xr_module._IsaacLabOvxrApplicationInterface(
        bootstrap,
        env,
        xr_module.OvLibrariesXrConfig(
            auxiliary_cameras=(
                xr_module.OvLibrariesXrCameraConfig("robot_pov_cam", SimpleNamespace(width=720, height=450)),
            ),
        ),
        (),
    )

    with pytest.raises(RuntimeError, match="not present"):
        host.attach_on_render_thread(host.offer_render_owner_transfer())


def test_selected_camera_poses_follow_motion_and_reset_before_publication(xr_module, monkeypatch):
    host, bootstrap, env = _make_host(xr_module, monkeypatch)
    names = ("head", "left_wrist", "right_wrist")
    events = []

    class PoseOnlyCamera:
        live = 0
        cached = -1
        fail = False

        def _update_poses(self):
            if self.fail:
                raise RuntimeError("pose unavailable")
            self.cached = self.live
            events.append(self)

        @property
        def data(self):
            pytest.fail("Publication must not fetch/render camera images")

    cameras = {name: PoseOnlyCamera() for name in (*names, "unselected")}
    env.scene = SimpleNamespace(stage=None, sensors=cameras)
    host._auxiliary_cameras = tuple(
        xr_module.OvLibrariesXrCameraConfig(name, SimpleNamespace(width=720, height=450)) for name in names
    )
    bootstrap.lease.auxiliary_render_product_paths = tuple(f"/Render/{name}" for name in names)
    binding = host.attach_on_render_thread(host.offer_render_owner_transfer())
    published = []
    bootstrap.publisher.write_complete_scene = lambda *_args: published.append(
        tuple(cameras[name].cached for name in names)
    )

    for identity, positions in ((1, (0, 2, 3)), (("reset", 1, 1), (0, -2, -3)), (2, (0, 4, 5))):
        for name, value in zip(names, positions, strict=True):
            cameras[name].live = value
        with host.publication_guard():
            host.publish_complete_scene(identity)
        assert published[-1] == positions
    assert events == [cameras[name] for name in names] * 3
    assert cameras["unselected"].cached == -1

    bootstrap.publisher.required = False
    with host.publication_guard():
        host.publish_complete_scene(2)
    assert len(events) == 9  # Reused scene/cadence gaps do not resample poses.

    bootstrap.publisher.required = True
    cameras["left_wrist"].fail = True
    reserved = list(binding.ordinal_ledger.reserved)
    with host.publication_guard(), pytest.raises(RuntimeError, match="pose unavailable"):
        host.publish_complete_scene(3)
    assert binding.ordinal_ledger.reserved == reserved
    assert len(published) == 3


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
def test_host_reconciles_failed_publication_without_reusing_generation(xr_module, monkeypatch, real_lease_type, sealed):
    ledger_type = xr_module._StageOrdinalLedger
    host, _bootstrap, _env = _make_host(xr_module, monkeypatch)
    host.attach_on_render_thread(host.offer_render_owner_transfer())
    failure = RuntimeError("native publication wait failed")
    stage = SimpleNamespace(floor=10)
    stage.get_attribute_write_floor = lambda: contextlib.nullcontext(
        SimpleNamespace(wait=lambda: None, fetch=lambda: stage.floor)
    )
    # Use the real publication identity owner; a permanently-dirty fake cannot
    # detect a retry returning the older ordinal after a failed seal.
    controller_type = sys.modules[real_lease_type.__module__].RenderLeaseController
    renderer = SimpleNamespace(
        _closed=False,
        _initialized_scene=False,
        backend=None,
        scene=None,
        _geometry_publication=None,
        _update_transforms=lambda: None,
        _update_geometries=lambda: None,
        _visual_material_writer_ref=None,
        _publish_visual_materials=contextlib.nullcontext,
    )
    monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    monkeypatch.delenv("ISAAC_LAB_SHARE_OVSTAGE", raising=False)
    controller = controller_type(renderer)
    lease = controller.acquire()
    host._scene_publisher = lease.capture_scene_publisher()
    renderer.scene = SimpleNamespace(stage=stage, ordinal=10)
    lease._bind(object(), stage, 1)
    stage.advance_write_floor = lambda *, ordinal: SimpleNamespace(wait=lambda: setattr(stage, "floor", ordinal))
    ledger = ledger_type(stage)
    host._ordinal_ledger = ledger
    with host.publication_guard():
        assert host.publish_complete_scene("previous") == 11

    def failed_advance(*, ordinal):
        if sealed:
            stage.floor = ordinal

        def wait():
            raise failure

        return SimpleNamespace(wait=wait)

    stage.advance_write_floor = failed_advance
    with host.publication_guard(), pytest.raises(RuntimeError) as caught:
        host.publish_complete_scene("changed")
    assert caught.value is failure
    with host.publication_guard(), pytest.raises(RuntimeError, match="publication is not available"):
        host.publish_complete_scene("changed")
    next_ordinal = ledger.reserve()
    assert next_ordinal == (13 if sealed else 12)
    ledger.abandon(next_ordinal)
    host.close_on_render_thread()
    lease.release()


def test_failed_bootstrap_retirement_retains_environment(xr_module, monkeypatch):
    """Capture's independent environment callback cannot bypass failed lease retirement."""
    host, bootstrap, _env = _make_host(xr_module, monkeypatch)
    host.attach_on_render_thread(host.offer_render_owner_transfer())

    def fail_close():
        raise RuntimeError("producer drain failed")

    monkeypatch.setattr(bootstrap, "close", fail_close)
    with pytest.raises(RuntimeError, match="producer drain failed"):
        host.close_on_render_thread()
    with pytest.raises(RuntimeError, match="lease retirement is unproved"):
        host.close_environment_on_render_thread()
    assert "environment-close" not in bootstrap.events


def test_failed_publication_with_unknown_floor_retains_reservation_and_first_error(xr_module, monkeypatch):
    ledger_type = xr_module._StageOrdinalLedger
    host, bootstrap, _env = _make_host(xr_module, monkeypatch)
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

    # Even repeated cleanup cannot retire queries/Stage while the write is unresolved.
    anchor_resources_closed = []
    host._anchor_resources.callback(lambda: anchor_resources_closed.append(True))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="unresolved OVStage ordinal"):
            host.close_on_render_thread()
        assert not bootstrap.closed
        assert not anchor_resources_closed
        with pytest.raises(RuntimeError, match="write retirement is unproved"):
            host.close_environment_on_render_thread()
        assert "environment-close" not in bootstrap.events


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


@pytest.mark.parametrize("camera_enabled", [False, True])
def test_session_delegates_lifecycle_to_isaacteleop_owner(xr_module, monkeypatch, camera_enabled):
    calls = []

    class FakeOwner:
        owns_environment = True
        shutdown_requested = False

        def __init__(self, host, config, **kwargs):
            calls.append(("init", host, config, kwargs))
            self.host = host

        def start(self, timeout_s):
            calls.append(("start", timeout_s))
            self.binding = self.host.attach_on_render_thread(self.host.offer_render_owner_transfer())
            return True

        def external_oxr_handles(self):
            return "handles"

        def request_render(self, identity, *, force=False):
            calls.append(("request", identity, force))
            return 17

        def request_shutdown(self):
            self.shutdown_requested = True
            calls.append(("request_shutdown",))

        def get_target_frame_T_world(self, path):
            return ("target", path)

        def close(self):
            calls.append(("close",))

    class FakeMirror:
        status = SimpleNamespace(main_thread_fps=30.0, render_thread_fps=40.0)

        def __init__(self, config, *, camera_products=()):
            self.config = config

        def record_host_frame(self):
            calls.append(("timing",))

        def set_status_text(self, text):
            calls.append(("status", text))

    mirror_module = importlib.import_module("isaaccapture.ovxr_session_manager.owned_mirror")
    monkeypatch.setattr(mirror_module, "OwnedDesktopMirror", FakeMirror)
    external = importlib.import_module("isaaccapture.ovxr_session_manager.ovxr_session")
    monkeypatch.setattr(external, "OvxrSession", FakeOwner)
    bootstrap = _FakeBootstrap()
    assert xr_module.OvLibrariesXrConfig().extra_openxr_extensions == ()
    session = xr_module.OvLibrariesXrSession(
        bootstrap,
        xr_module.OvLibrariesXrConfig(
            extra_openxr_extensions=("XR_EXT_debug_utils",),
            desktop_mirror_enabled=True,
            desktop_mirror_max_fps=20.0,
            auxiliary_cameras=(
                xr_module.OvLibrariesXrCameraConfig("robot_pov_cam", SimpleNamespace(width=720, height=450)),
            )
            if camera_enabled
            else (),
        ),
    )
    env = SimpleNamespace(scene=SimpleNamespace(stage=None, sensors={"robot_pov_cam": object()}))
    bootstrap.lease.auxiliary_render_product_paths = ("/Render/robot_pov_cam",)
    monkeypatch.setattr(xr_module, "_StageOrdinalLedger", _FakeOrdinalLedger)
    monkeypatch.setattr(xr_module, "_apply_hidden_visual_prims", lambda *_args: None)
    monkeypatch.setattr(xr_module, "_declare_ovxr_runtime_prims", lambda *_args: None)

    assert session.start_render_worker(
        env,
        required_openxr_extensions=("XR_EXT_hand_tracking",),
        max_blank_frames=0 if camera_enabled else 1,
        timeout_s=4.0,
    )

    assert calls[0][0] == "init"
    assert calls[0][2].session_mode == "ar"
    assert calls[0][3]["required_openxr_extensions"] == ("XR_EXT_hand_tracking",)
    assert calls[0][3]["max_blank_frames"] == (0 if camera_enabled else 1)
    assert calls[0][2].extra_openxr_extensions == ("XR_EXT_debug_utils",)
    assert calls[0][2].host_wait_render_output is True
    consumers = session._owner.binding.frame_consumers
    consumer = consumers[-1]
    assert consumer.config.max_fps == 20.0
    assert consumer.config.initial_status_text == "Waiting for headset"
    if camera_enabled:
        assert consumers[0].panels[0].label == "robot_pov_cam"
    else:
        assert len(consumers) == 1
    assert calls[1] == ("start", 4.0)
    assert session.external_oxr_handles() == "handles"
    assert session.request_render("step") == 17
    assert session.request_render("reset", force=True) == 17
    assert calls[-2:] == [("request", "step", False), ("request", "reset", True)]
    monkeypatch.setattr(session._physics_rate, "sample", lambda _count: 114.7)
    session.report_host_timing(42, 1.0 / 120.0)
    assert calls[-2:] == [
        ("timing",),
        ("status", "Main  30.0 FPS   Physics 114.7/120 Hz (0.96× RT)   Render  40.0 FPS"),
    ]
    assert session.get_target_frame_T_world("/target") == ("target", "/target")
    assert session.async_worker_owns_environment
    assert not session.shutdown_requested
    session.request_shutdown()
    assert calls[-1] == ("request_shutdown",)
    assert session.shutdown_requested
    session._owner.shutdown_requested = True
    assert session.shutdown_requested
    session.close()
    assert calls[-1] == ("close",)


@pytest.fixture
def bootstrap_context(xr_module, monkeypatch):
    """Use the real bootstrap with only simulation/native construction replaced."""
    module = sys.modules["isaaclab_teleop.ovxr_session"]
    events = []

    class Context:
        current = None
        failure = None
        cleanup_failure = None

        def __init__(self, _cfg):
            type(self).current = self

        @classmethod
        def instance(cls):
            return cls.current

        @classmethod
        def clear_instance(cls):
            events.append("context-close")
            cls.current = None
            if cls.cleanup_failure is not None:
                raise cls.cleanup_failure

        def get_or_create_backend(self, _cfg):
            if self.failure is not None:
                raise self.failure
            lease = SimpleNamespace(release=lambda: events.append("lease-release"))
            return SimpleNamespace(acquire_ov_libraries_render_lease=lambda: lease)

    monkeypatch.setattr(module, "SimulationContext", Context)
    monkeypatch.delattr(builtins, module._TERMINAL_MODE_SENTINEL, raising=False)
    return module, Context, events


@pytest.mark.parametrize("route", ["before_owner", "before_worker"])
def test_failed_start_releases_lease_before_environment_and_context(xr_module, bootstrap_context, route):
    """An existing environment retains its context until the caller closes it."""
    module, context, events = bootstrap_context
    bootstrap = module.OvxrSessionBootstrap(object())
    if route == "before_owner":
        session = xr_module.OvLibrariesXrSession(bootstrap, xr_module.OvLibrariesXrConfig())
        session.close()
    else:
        # Capture calls host.abort when validation fails before its worker exists.
        host = xr_module._IsaacLabOvxrApplicationInterface.__new__(xr_module._IsaacLabOvxrApplicationInterface)
        host._bootstrap = bootstrap
        host.abort()
    assert context.current is not None
    events.append("environment-close")
    context.clear_instance()
    assert events == ["lease-release", "environment-close", "context-close"]
    assert not hasattr(builtins, module._TERMINAL_MODE_SENTINEL)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_bootstrap_construction_failure_restores_terminal_mode(bootstrap_context, error_type, cleanup_fails):
    """Failed or interrupted construction restores process state and preserves its cause."""
    module, context, events = bootstrap_context
    error = context.failure = error_type("renderer construction failed")
    if cleanup_fails:
        context.cleanup_failure = RuntimeError("context cleanup failed")
    with pytest.raises(error_type) as caught:
        module.OvxrSessionBootstrap(object())
    assert caught.value is error
    assert events == ["context-close"]
    assert context.current is None
    assert not hasattr(builtins, module._TERMINAL_MODE_SENTINEL)
    if cleanup_fails:
        assert any("context cleanup failed" in note for note in error.__notes__)


def test_desktop_mirror_rate_must_be_positive_and_finite(xr_module):
    with pytest.raises(ValueError, match="desktop_mirror_max_fps"):
        xr_module.OvLibrariesXrConfig(desktop_mirror_max_fps=0.0)
    with pytest.raises(ValueError, match="desktop_mirror_max_fps"):
        xr_module.OvLibrariesXrConfig(desktop_mirror_max_fps=float("nan"))


def test_auxiliary_camera_config_is_preconfigured(xr_module, monkeypatch):
    with pytest.raises(ValueError, match="name and configuration"):
        xr_module.OvLibrariesXrCameraConfig("", object())
    with pytest.raises(ValueError, match="name and configuration"):
        xr_module.OvLibrariesXrCameraConfig("robot_pov_cam", None)
    with pytest.raises(ValueError, match="max_fps"):
        xr_module.OvLibrariesXrCameraConfig("robot_pov_cam", object(), max_fps=0.0)

    bootstrap = _FakeBootstrap()
    camera_cfg = SimpleNamespace(width=720, height=450)
    monkeypatch.setattr(xr_module, "OvxrSessionBootstrap", lambda *_args: bootstrap)
    camera = xr_module.OvLibrariesXrCameraConfig("robot_pov_cam", camera_cfg)
    session = xr_module.OvLibrariesXrSession.prepare(
        object(), xr_module.OvLibrariesXrConfig(auxiliary_cameras=(camera,))
    )
    assert isinstance(session, xr_module.OvLibrariesXrSession)
    assert bootstrap.auxiliary_camera_cfgs == [camera_cfg]


def test_caller_renderer_settings_precede_renderer_creation(xr_module, monkeypatch):
    settings = "  --/rtx/rendermode=MinimalRendering  "
    events = []
    bootstrap = _FakeBootstrap()
    monkeypatch.setitem(
        sys.modules, "ovxr", SimpleNamespace(apply_renderer_settings=lambda value: events.append(value))
    )

    def create_bootstrap(*_args):
        assert len(events) == 1
        assert events[0] == settings.strip()
        return bootstrap

    monkeypatch.setattr(xr_module, "OvxrSessionBootstrap", create_bootstrap)
    xr_module.OvLibrariesXrSession.prepare(object(), xr_module.OvLibrariesXrConfig(renderer_settings=settings))


def test_prepare_rejects_existing_context_before_global_settings(xr_module, monkeypatch):
    from isaaclab.sim import SimulationContext

    monkeypatch.setattr(SimulationContext, "instance", lambda: object())
    monkeypatch.setitem(
        sys.modules,
        "ovxr",
        SimpleNamespace(apply_renderer_settings=lambda _: pytest.fail("changed existing renderer settings")),
    )
    monkeypatch.setattr(xr_module, "OvxrSessionBootstrap", lambda *_: pytest.fail("allocated second bootstrap"))
    with pytest.raises(RuntimeError, match="must create the SimulationContext"):
        xr_module.OvLibrariesXrSession.prepare(object(), xr_module.OvLibrariesXrConfig())


@pytest.mark.parametrize("terminal_mode", [False, None, 1])
def test_prepare_rejects_extension_mode_before_global_settings(xr_module, monkeypatch, terminal_mode):
    monkeypatch.setattr(builtins, "ISAAC_LAUNCHED_FROM_TERMINAL", terminal_mode, raising=False)
    monkeypatch.setitem(
        sys.modules, "ovxr", SimpleNamespace(apply_renderer_settings=lambda _: pytest.fail("changed renderer settings"))
    )
    monkeypatch.setattr(xr_module, "OvxrSessionBootstrap", lambda *_: pytest.fail("allocated bootstrap"))
    with pytest.raises(RuntimeError, match="extension mode"):
        xr_module.OvLibrariesXrSession.prepare(object(), xr_module.OvLibrariesXrConfig())


def test_counter_rate_uses_application_counter_and_resets_cleanly(xr_module):
    rate = xr_module._CounterRate()
    assert rate.sample(10, now_ns=0) == 0.0
    assert rate.sample(15, now_ns=500_000_000) == pytest.approx(10.0)
    assert rate.sample(20, now_ns=1_000_000_000) == pytest.approx(10.0)
    assert rate.sample(0, now_ns=1_100_000_000) == 0.0


def test_disabled_desktop_mirror_does_not_collect_or_format_status(xr_module):
    session = xr_module.OvLibrariesXrSession(
        _FakeBootstrap(),
        xr_module.OvLibrariesXrConfig(desktop_mirror_enabled=False),
    )
    session._owner = SimpleNamespace(report_host_timing=lambda: pytest.fail("unexpected mirror work"))

    session.report_host_timing(42, 1.0 / 120.0)


def test_config_rejects_unknown_session_mode(xr_module):
    with pytest.raises(ValueError, match="session_mode must be 'ar' or 'vr'"):
        xr_module.OvLibrariesXrConfig(session_mode="automatic")


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_config_rejects_invalid_scene_scale(xr_module, scale):
    with pytest.raises(ValueError, match="meters_per_unit must be finite and positive"):
        xr_module.OvLibrariesXrConfig(meters_per_unit=scale)


def test_prepare_revalidates_changed_scale_before_global_settings(xr_module, monkeypatch):
    cfg = xr_module.OvLibrariesXrConfig()
    cfg.meters_per_unit = float("nan")
    monkeypatch.setitem(
        sys.modules, "ovxr", SimpleNamespace(apply_renderer_settings=lambda _: pytest.fail("changed renderer settings"))
    )
    monkeypatch.setattr(xr_module, "OvxrSessionBootstrap", lambda *_: pytest.fail("allocated bootstrap"))
    with pytest.raises(ValueError, match="meters_per_unit must be finite and positive"):
        xr_module.OvLibrariesXrSession.prepare(object(), cfg)


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_start_rejects_invalid_bootstrap_count_before_owner_creation(xr_module, monkeypatch, count):
    external = importlib.import_module("isaaccapture.ovxr_session_manager.ovxr_session")
    monkeypatch.setattr(external, "OvxrSession", lambda *_args, **_kwargs: pytest.fail("created an XR owner"))
    bootstrap = _FakeBootstrap()
    session = xr_module.OvLibrariesXrSession(bootstrap, xr_module.OvLibrariesXrConfig())
    with pytest.raises(ValueError, match="max_blank_frames must be a non-negative integer"):
        session.start_render_worker(SimpleNamespace(), max_blank_frames=count)
    assert bootstrap.events == []


def test_three_auxiliary_camera_configs_prepare_the_same_bootstrap(xr_module, monkeypatch):
    bootstrap = _FakeBootstrap()
    monkeypatch.setattr(xr_module, "OvxrSessionBootstrap", lambda *_: bootstrap)
    cameras = tuple(
        xr_module.OvLibrariesXrCameraConfig(
            name=name, camera_cfg=SimpleNamespace(width=640, height=480), max_fps=hz, space="head_locked"
        )
        for name, hz in zip(("robot_pov_cam", "left_wrist_cam", "right_wrist_cam"), (20, 10, 10), strict=True)
    )
    session = xr_module.OvLibrariesXrSession.prepare(object(), xr_module.OvLibrariesXrConfig(auxiliary_cameras=cameras))
    assert bootstrap.auxiliary_camera_cfgs == [camera.camera_cfg for camera in cameras]
    assert session is not None
    extra = xr_module.OvLibrariesXrCameraConfig("overview", SimpleNamespace(width=640, height=480))
    with pytest.raises(ValueError, match="at most three"):
        xr_module.OvLibrariesXrConfig(auxiliary_cameras=(*cameras, extra))
    with pytest.raises(ValueError, match="exactly two"):
        xr_module.OvLibrariesXrConfig(camera_paths=("/Camera/mono",))


def test_prepare_snapshots_camera_selection_without_mutating_caller(xr_module, monkeypatch):
    bootstrap = _FakeBootstrap()
    monkeypatch.setattr(xr_module, "OvxrSessionBootstrap", lambda *_: bootstrap)
    camera_cfg = SimpleNamespace(width=640, height=480)
    camera = xr_module.OvLibrariesXrCameraConfig("robot_pov_cam", camera_cfg)
    cfg = xr_module.OvLibrariesXrConfig(auxiliary_cameras=(camera,))
    session = xr_module.OvLibrariesXrSession.prepare(object(), cfg)
    cfg.auxiliary_cameras = ()
    assert session._cfg.auxiliary_cameras == (camera,)
    assert session._cfg.auxiliary_cameras[0].camera_cfg is bootstrap.auxiliary_camera_cfgs[0]


@pytest.mark.parametrize("fail_during", ["camera", "session"])
@pytest.mark.parametrize("abort_fails", [False, True])
def test_prepare_aborts_partial_bootstrap_and_preserves_first_error(xr_module, monkeypatch, fail_during, abort_fails):
    bootstrap = _FakeBootstrap()
    first_error = RuntimeError("preparation failed")
    original_cause = ValueError("invalid camera input")
    first_error.add_note("Original preparation diagnostic")
    abort_error = RuntimeError("abort failed")
    monkeypatch.setitem(sys.modules, "ovxr", _fake_ovxr())
    monkeypatch.setattr(xr_module, "OvxrSessionBootstrap", lambda *_: bootstrap)

    def fail(*_args, **_kwargs):
        raise first_error from original_cause

    def abort():
        bootstrap.aborted = True
        if abort_fails:
            raise abort_error

    monkeypatch.setattr(bootstrap, "abort", abort)
    if fail_during == "camera":
        monkeypatch.setattr(bootstrap, "configure_auxiliary_camera", fail)
    else:
        monkeypatch.setattr(xr_module.OvLibrariesXrSession, "__init__", fail)

    cfg = xr_module.OvLibrariesXrConfig(
        auxiliary_cameras=(
            xr_module.OvLibrariesXrCameraConfig("robot_pov_cam", SimpleNamespace(width=640, height=480)),
        ),
    )
    with pytest.raises(RuntimeError) as caught:
        xr_module.OvLibrariesXrSession.prepare(object(), cfg)

    assert bootstrap.aborted
    assert caught.value is first_error
    assert caught.value.__cause__ is original_cause
    assert caught.value.__notes__ == ["Original preparation diagnostic"] + (
        [f"Cleanup after failed XR preparation also failed: {abort_error!r}"] if abort_fails else []
    )


@pytest.mark.parametrize("space", ["viewer_start", "world"])
def test_prepare_rejects_unsupported_quad_placement_before_native_preparation(xr_module, monkeypatch, space):
    monkeypatch.setitem(
        sys.modules,
        "ovxr",
        SimpleNamespace(apply_renderer_settings=lambda _: pytest.fail("renderer settings changed before validation")),
    )
    monkeypatch.setattr(xr_module, "OvxrSessionBootstrap", lambda *_: pytest.fail("allocated before validation"))
    with pytest.raises(ValueError, match="head_locked"):
        camera = xr_module.OvLibrariesXrCameraConfig(
            name="robot_pov_cam", camera_cfg=SimpleNamespace(width=640, height=480), space=space
        )
        xr_module.OvLibrariesXrSession.prepare(object(), xr_module.OvLibrariesXrConfig(auxiliary_cameras=(camera,)))


def test_public_xr_exports_resolve_to_implemented_types(xr_module):
    import isaaclab_teleop

    assert isaaclab_teleop.OvLibrariesXrConfig is xr_module.OvLibrariesXrConfig
    assert isaaclab_teleop.OvLibrariesXrCameraConfig is xr_module.OvLibrariesXrCameraConfig
    assert isaaclab_teleop.OvLibrariesXrSession is xr_module.OvLibrariesXrSession
