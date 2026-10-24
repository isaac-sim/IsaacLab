# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused tests for kitless XR anchor construction."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest
from isaaclab_teleop import xr_anchor_manager
from isaaclab_teleop.xr_cfg import XrCfg

from pxr import Usd, UsdGeom

from isaaclab.sim.utils.stage import use_stage


def test_dynamic_external_anchor_commits_pose_before_input_and_preserves_reset(monkeypatch):
    from isaaclab_teleop.xr_cfg import XrAnchorRotationMode
    from scipy.spatial.transform import Rotation

    monkeypatch.setattr(xr_anchor_manager, "XRCore", None)
    cfg = XrCfg(
        anchor_prim_path="/World/Robot",
        anchor_pos=(1.0, 2.0, -0.95),
        fixed_anchor_height=True,
        anchor_rotation_mode=XrAnchorRotationMode.FOLLOW_PRIM,
    )
    with use_stage(Usd.Stage.CreateInMemory()):
        UsdGeom.Xform.Define(xr_anchor_manager.get_current_stage(), "/World/Robot")
        manager = xr_anchor_manager.XrAnchorManager(cfg, ovxr_session=True)
    with pytest.raises(RuntimeError, match="committed post-reset"):
        manager.get_world_matrix()
    reference = np.eye(4)
    reference[:3, 3] = (3, 4, 5)
    committed = []
    epoch, seed = manager.external_epoch
    assert seed
    assert manager.commit_external_pose(reference, 0.02, epoch, lambda m: committed.append(m.copy()))
    np.testing.assert_allclose(committed[-1][0, :3, 3], (4, 6, 4.05))
    reference[:3, 3] = (8, 9, 12)
    reference[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    manager.commit_external_pose(reference, 0.02, epoch, lambda m: committed.append(m.copy()))
    expected = committed[-1][0].copy()
    expected[:3, :3] = expected[:3, :3] @ manager._OXR_TO_USD_ROTATION
    np.testing.assert_allclose(manager.get_world_matrix(), expected, atol=1e-7)
    np.testing.assert_allclose(expected[:3, 3], (9, 11, 4.05))
    before = manager.get_world_matrix()
    np.testing.assert_array_equal(manager.get_world_matrix(), before)
    manager.toggle_anchor_rotation()
    reference[:3, :3] = np.eye(3)
    manager.commit_external_pose(reference, 0.02, epoch, lambda m: committed.append(m.copy()))
    np.testing.assert_allclose(manager.get_world_matrix()[:3, :3], before[:3, :3], atol=1e-7)
    manager.reset()
    assert not manager.commit_external_pose(reference, 0.02, epoch, lambda _: pytest.fail("old epoch sealed"))
    with pytest.raises(RuntimeError, match="committed post-reset"):
        manager.get_world_matrix()
    manager.commit_external_pose(reference, 0.02, manager.external_epoch[0], lambda m: committed.append(m.copy()))
    np.testing.assert_allclose(manager.get_world_matrix()[:3, 3], (9, 11, 11.05))
    with pytest.raises(RuntimeError, match="write failed"):
        manager.commit_external_pose(
            reference, 0.02, manager.external_epoch[0], lambda _: (_ for _ in ()).throw(RuntimeError("write failed"))
        )
    with pytest.raises(RuntimeError, match="publication failed"):
        manager.get_world_matrix()

    manager.reset()
    with pytest.raises(RuntimeError, match="publication failed"):
        manager.commit_external_pose(reference, 0.02, epoch, lambda _: pytest.fail("failed pair revived"))


def test_standalone_replay_without_kit_xr_preserves_anchor_fallback(monkeypatch):
    monkeypatch.setattr(xr_anchor_manager, "XRCore", None)
    monkeypatch.setattr(xr_anchor_manager, "_xr_anchor_prim_exists", lambda _path: True)
    cfg = XrCfg(anchor_prim_path="/World/Robot", anchor_pos=(1.0, 2.0, 3.0))
    manager = xr_anchor_manager.XrAnchorManager(cfg)
    np.testing.assert_allclose(manager.get_world_matrix()[:3, 3], cfg.anchor_pos)


def test_ovxr_session_does_not_borrow_an_available_kit_xr_core(monkeypatch):
    core = MagicMock()
    monkeypatch.setattr(xr_anchor_manager, "XRCore", core)
    with use_stage(Usd.Stage.CreateInMemory()):
        manager = xr_anchor_manager.XrAnchorManager(XrCfg(), ovxr_session=True)
    assert manager.xr_core is None
    core.get_singleton.assert_not_called()


@pytest.mark.parametrize("capability", ["none", "live-handles", "recorded-input-viewer"])
def test_public_device_exposes_dynamic_anchor_only_for_ovxr_session(monkeypatch, capability):
    from isaaclab_teleop import isaac_teleop_device
    from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg

    monkeypatch.setattr(xr_anchor_manager, "XRCore", None)
    monkeypatch.setattr(isaac_teleop_device, "CommandHandler", MagicMock())
    lifecycle = MagicMock()
    monkeypatch.setattr(isaac_teleop_device, "TeleopSessionLifecycle", lifecycle)
    cfg = IsaacTeleopCfg(
        pipeline_builder=lambda: None,
        xr_cfg=XrCfg(anchor_prim_path="/World/Robot"),
    )
    kwargs = dict(use_kit_xr_bridge=False)
    if capability == "live-handles":
        kwargs["external_oxr_handles_provider"] = lambda: None
    elif capability == "recorded-input-viewer":
        kwargs.update(ovxr_session=True, mcap_replay_path="recording.mcap")
    with use_stage(Usd.Stage.CreateInMemory()):
        device = isaac_teleop_device.IsaacTeleopDevice(cfg, **kwargs)
    assert device.xr_anchor_path == "/World/Robot/XRAnchor"
    assert (device.external_anchor_state is not None) == (capability != "none")
    lifecycle.assert_called_once()


def test_no_xr_core_does_not_acquire_carb_settings(monkeypatch):
    settings = ModuleType("carb.settings")
    settings.get_settings = MagicMock(side_effect=AssertionError("kitless anchor touched carb settings"))
    carb = ModuleType("carb")
    carb.settings = settings
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "carb.settings", settings)
    monkeypatch.setattr(xr_anchor_manager, "XRCore", None)
    monkeypatch.setattr(xr_anchor_manager, "_xr_anchor_prim_exists", lambda _path: True)
    cfg = XrCfg(anchor_pos=(1.0, 2.0, 3.0), anchor_rot=(0.0, 0.0, 0.0, 1.0))

    manager = xr_anchor_manager.XrAnchorManager(cfg)

    settings.get_settings.assert_not_called()
    np.testing.assert_allclose(manager.get_world_matrix()[:3, 3], cfg.anchor_pos)


def test_render_anchor_does_not_offset_tracked_device_poses(monkeypatch):
    created_prims = []
    monkeypatch.setattr(xr_anchor_manager, "XRCore", None)
    monkeypatch.setattr(xr_anchor_manager, "_xr_anchor_prim_exists", lambda _path: False)
    monkeypatch.setattr(
        xr_anchor_manager,
        "_create_prim",
        lambda path, **kwargs: created_prims.append((path, kwargs)),
    )
    cfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0),
        render_anchor_pos=(0.0, -2.5, 0.0),
    )

    manager = xr_anchor_manager.XrAnchorManager(cfg)

    assert manager.anchor_headset_path == "/World/XRAnchor"
    assert manager.render_anchor_path == "/World/XRRenderAnchor"
    np.testing.assert_allclose(manager.get_world_matrix()[:3, 3], cfg.anchor_pos)
    assert [path for path, _kwargs in created_prims] == ["/World/XRAnchor", "/World/XRRenderAnchor"]
    np.testing.assert_allclose(created_prims[1][1]["position"], cfg.render_anchor_pos)


def test_replay_reuses_real_stage_scoped_anchors_without_kit(monkeypatch, caplog):
    """A second device must not recreate or overwrite the same USD anchors."""
    stage = Usd.Stage.CreateInMemory()
    cfg = XrCfg(anchor_pos=(1.0, 2.0, 3.0), render_anchor_pos=(0.0, -2.5, 0.0))
    monkeypatch.setattr(xr_anchor_manager, "XRCore", None)
    with use_stage(stage):
        first = xr_anchor_manager.XrAnchorManager(cfg)
        first.cleanup()
        second = xr_anchor_manager.XrAnchorManager(cfg)
        second.cleanup()
    assert "Failed to create XR" not in caplog.text
    for path, position in (("/World/XRAnchor", cfg.anchor_pos), ("/World/XRRenderAnchor", cfg.render_anchor_pos)):
        prim = stage.GetPrimAtPath(path)
        assert prim.IsValid()
        np.testing.assert_allclose(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation(), position
        )


def test_replay_refreshes_configured_static_anchors_on_the_same_stage(monkeypatch):
    from isaaclab.sim.utils.transforms import resolve_prim_pose

    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    world.AddTranslateOp().Set((2.0, -1.0, 0.5))
    world.AddRotateZOp().Set(90.0)
    monkeypatch.setattr(xr_anchor_manager, "XRCore", None)
    first_cfg = XrCfg(anchor_pos=(1.0, 2.0, 3.0), render_anchor_pos=(0.0, -2.5, 0.0))
    next_cfg = XrCfg(
        anchor_pos=(-1.0, 0.5, 2.0),
        anchor_rot=(0.0, 0.0, 1.0, 0.0),
        render_anchor_pos=(3.0, -4.0, 0.25),
        render_anchor_rot=(0.0, 0.0, 0.0, 1.0),
    )
    with use_stage(stage):
        first = xr_anchor_manager.XrAnchorManager(first_cfg, ovxr_session=True)
        for path, position in (
            (first.anchor_headset_path, first_cfg.anchor_pos),
            (first.render_anchor_path, first_cfg.render_anchor_pos),
        ):
            observed_pos, observed_rot = resolve_prim_pose(stage.GetPrimAtPath(path))
            np.testing.assert_allclose(observed_pos, position, atol=1e-7)
            assert abs(np.dot(observed_rot, first_cfg.anchor_rot)) == pytest.approx(1.0)
        first.cleanup()
        second = xr_anchor_manager.XrAnchorManager(next_cfg, ovxr_session=True)
        for path, position, rotation in (
            (second.anchor_headset_path, next_cfg.anchor_pos, next_cfg.anchor_rot),
            (second.render_anchor_path, next_cfg.render_anchor_pos, next_cfg.render_anchor_rot),
        ):
            observed_pos, observed_rot = resolve_prim_pose(stage.GetPrimAtPath(path))
            np.testing.assert_allclose(observed_pos, position, atol=1e-7)
            assert abs(np.dot(observed_rot, rotation)) == pytest.approx(1.0)
        np.testing.assert_allclose(second.get_world_matrix()[:3, 3], next_cfg.anchor_pos)
        second.cleanup()


@pytest.mark.parametrize("anchor_path", ["/World/XRAnchor", "/World/XRRenderAnchor"])
def test_ovxr_session_rejects_an_unusable_stage_anchor(monkeypatch, anchor_path):
    """An invalid anchor must not silently misalign tracking and rendering."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Scope.Define(stage, anchor_path)
    monkeypatch.setattr(xr_anchor_manager, "XRCore", None)
    cfg = XrCfg(render_anchor_pos=(0.0, -2.5, 0.0))
    with use_stage(stage), pytest.raises(ValueError, match="is not transformable"):
        xr_anchor_manager.XrAnchorManager(cfg, ovxr_session=True)


def test_standalone_replay_keeps_warning_for_an_unusable_anchor(monkeypatch, caplog):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Scope.Define(stage, "/World/XRAnchor")
    monkeypatch.setattr(xr_anchor_manager, "XRCore", None)
    cfg = XrCfg(anchor_pos=(1.0, 2.0, 3.0))
    with use_stage(stage):
        manager = xr_anchor_manager.XrAnchorManager(cfg)
        np.testing.assert_allclose(manager.get_world_matrix()[:3, 3], cfg.anchor_pos)
        manager.cleanup()
    assert "is not transformable" in caplog.text


def test_kit_anchor_clock_is_lazy_until_subsequent_smoothed_update(monkeypatch):
    from isaaclab_teleop import xr_anchor_utils
    from isaaclab_teleop.xr_cfg import XrAnchorRotationMode

    cfg = XrCfg(anchor_prim_path=None)
    core = MagicMock()
    clock = MagicMock()
    clock.instance.side_effect = AssertionError("unused simulation clock")
    monkeypatch.setattr(xr_anchor_utils, "SimulationContext", clock)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/XRAnchor")
    with use_stage(stage):
        synchronizer = xr_anchor_utils.XrAnchorSynchronizer(core, cfg, "/XRAnchor")
    synchronizer.sync_headset_to_anchor()
    core.set_world_transform_matrix.assert_called_once()
    clock.instance.assert_not_called()

    cfg.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED
    policy = xr_anchor_utils._XrAnchorPosePolicy(cfg)
    dt = MagicMock(return_value=0.02)
    policy.update(np.eye(4), dt)
    dt.assert_not_called()
    moved = np.eye(4)
    moved[0, 3] = 2.0
    policy.update(moved, dt)
    dt.assert_called_once_with()
