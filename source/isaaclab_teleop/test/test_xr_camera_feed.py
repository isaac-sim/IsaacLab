# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import isaaclab_teleop.camera_feed as camera_feed
import pytest
import torch
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_teleop import IsaacTeleopCfg, XrCameraFeedCfg, XrCameraFeedLayoutCfg, XrCameraFeedSession

from isaaclab.sensors import CameraCfg


class _FakeImage:
    def __init__(self, height=8, width=12, *, device="cuda:0", data_ptr=100):
        self.dtype = torch.uint8
        self.ndim = 3
        self.shape = (height, width, 4)
        self.device = torch.device(device)
        self._data_ptr = data_ptr

    def data_ptr(self):
        return self._data_ptr


class _FakeBatch:
    def __init__(self, image):
        self.image = image
        self.ndim = 4
        self.shape = (1, *image.shape)

    def __getitem__(self, index):
        assert index == 0
        return self.image


class _FakeCamera:
    def __init__(self, image):
        self.output = {"rgba": SimpleNamespace(torch=_FakeBatch(image))}
        self.update_calls = []
        self.image_on_update = None

    @property
    def data(self):
        return SimpleNamespace(output=self.output)

    def update(self, dt, force_recompute=False):
        self.update_calls.append((dt, force_recompute))
        if self.image_on_update is not None:
            self.output["rgba"].torch = _FakeBatch(self.image_on_update)


class _FakePanel:
    def __init__(self, descriptor, width, height):
        self.descriptor = descriptor
        self.width = width
        self.height = height
        self.uploads = []
        self.closed = False

    def upload(self, image):
        self.uploads.append(image)

    def close(self):
        self.closed = True


class _FakeSubscription:
    def __init__(self, callback):
        self.callback = callback
        self.closed = False

    def publish(self):
        if not self.closed:
            self.callback(None)

    def close(self):
        self.closed = True


class _FakePresenter:
    def __init__(self):
        self.panels = []
        self.subscription = None
        self.staged = []

    def prepare_upload_image(self, _name, image, previous_source=None, previous_upload=None):
        del previous_source, previous_upload
        return image

    def create_panel(self, descriptor, width, height):
        panel = _FakePanel(descriptor, width, height)
        self.panels.append(panel)
        return panel

    def stage_upload_image(self, image, upload_image):
        self.staged.append((image, upload_image))

    def subscribe_to_frame_updates(self, callback):
        self.subscription = _FakeSubscription(callback)
        return self.subscription


def _camera_cfg(renderer_cfg=None):
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        height=8,
        width=12,
        data_types=["rgb"],
        renderer_cfg=renderer_cfg,
    )


def _teleop_env_cfg(
    feeds,
    *,
    num_envs=1,
    camera=None,
    layout=None,
):
    scene = SimpleNamespace(num_envs=num_envs)
    if camera is not None:
        scene.robot_pov_cam = camera
    return SimpleNamespace(
        scene=scene,
        isaac_teleop=SimpleNamespace(
            xr_camera_feeds=feeds,
            xr_camera_feed_layout=layout or XrCameraFeedLayoutCfg(),
        ),
    )


def _manager(monkeypatch, cfgs, images, layout=None):
    cameras = {name: _FakeCamera(image) for name, image in images.items()}
    monkeypatch.setattr(camera_feed, "_camera_type", lambda: _FakeCamera)
    presenter = _FakePresenter()
    env = SimpleNamespace(scene=SimpleNamespace(sensors=cameras))
    manager = camera_feed._XrCameraFeedManager(env, cfgs, layout or XrCameraFeedLayoutCfg(), presenter)
    return manager, presenter, cameras


def test_pip_rejects_multiple_environments_before_camera_creation(monkeypatch):
    env_cfg = _teleop_env_cfg([XrCameraFeedCfg(camera_name="robot_pov_cam")], num_envs=2, camera=_camera_cfg())
    load_presenter = Mock(return_value=_FakePresenter())
    monkeypatch.setattr(camera_feed, "_load_kit_scene_ui_presenter", load_presenter)

    with pytest.raises(ValueError, match="exactly one environment"):
        XrCameraFeedSession.prepare(env_cfg, enabled=True, camera_rendering_enabled=True)

    load_presenter.assert_called_once_with()


def test_xr_without_pip_preserves_multiple_environments(monkeypatch):
    env_cfg = _teleop_env_cfg([], num_envs=2)
    load_presenter = Mock()
    monkeypatch.setattr(camera_feed, "_load_kit_scene_ui_presenter", load_presenter)

    session = XrCameraFeedSession.prepare(
        env_cfg,
        enabled=True,
        camera_rendering_enabled=True,
    )

    assert not session.enabled
    load_presenter.assert_not_called()
    assert env_cfg.scene.num_envs == 2


def test_kitless_xr_with_configured_pip_preserves_multiple_environments(monkeypatch):
    env_cfg = _teleop_env_cfg(
        [XrCameraFeedCfg(camera_name="robot_pov_cam")],
        num_envs=2,
        camera=_camera_cfg(),
    )
    monkeypatch.setattr(camera_feed, "_load_kit_scene_ui_presenter", lambda: None)

    session = XrCameraFeedSession.prepare(
        env_cfg,
        enabled=True,
        camera_rendering_enabled=True,
    )

    assert not session.enabled
    assert env_cfg.scene.num_envs == 2


def test_empty_camera_feed_selection_skips_pip(monkeypatch):
    env_cfg = _teleop_env_cfg([])
    load_presenter = Mock()
    monkeypatch.setattr(camera_feed, "_load_kit_scene_ui_presenter", load_presenter)

    session = XrCameraFeedSession.prepare(env_cfg, enabled=True, camera_rendering_enabled=True)

    assert not session.enabled
    load_presenter.assert_not_called()
    assert vars(env_cfg.scene) == {"num_envs": 1}


def test_existing_camera_is_selected_without_replacement(monkeypatch):
    selected = _camera_cfg(IsaacRtxRendererCfg(enable_dlss_ray_reconstruction=True))
    env_cfg = _teleop_env_cfg([XrCameraFeedCfg(camera_name="robot_pov_cam")], camera=selected)
    monkeypatch.setattr(camera_feed, "_load_kit_scene_ui_presenter", _FakePresenter)

    session = XrCameraFeedSession.prepare(env_cfg, enabled=True, camera_rendering_enabled=True)

    assert session.enabled
    assert env_cfg.scene.robot_pov_cam is selected
    assert session.requires_responsive_denoising


def test_session_refresh_publishes_buffer_refreshed_by_env_reset():
    events = []
    session = XrCameraFeedSession([], None, None, requires_responsive_denoising=False)

    class _Manager:
        def refresh(self, *, publish=True):
            events.append("publish" if publish else "request")

    session._manager = _Manager()
    session._bound = True

    session.refresh()

    assert events == ["publish"]


def test_camera_rendering_switch_disables_pip_before_presenter_load(monkeypatch):
    env_cfg = _teleop_env_cfg(
        [XrCameraFeedCfg(camera_name="robot_pov_cam")],
        num_envs=2,
        camera=_camera_cfg(),
    )
    load_presenter = Mock()
    monkeypatch.setattr(camera_feed, "_load_kit_scene_ui_presenter", load_presenter)

    session = XrCameraFeedSession.prepare(env_cfg, enabled=True, camera_rendering_enabled=False)

    assert not session.enabled
    load_presenter.assert_not_called()


def test_feed_selection_requires_existing_rgb_camera(monkeypatch):
    monkeypatch.setattr(camera_feed, "_load_kit_scene_ui_presenter", _FakePresenter)
    missing = _teleop_env_cfg([XrCameraFeedCfg(camera_name="missing")])
    with pytest.raises(ValueError, match="not present"):
        XrCameraFeedSession.prepare(missing, enabled=True, camera_rendering_enabled=True)

    depth = _camera_cfg()
    depth.data_types = ["distance_to_image_plane"]
    env_cfg = _teleop_env_cfg([XrCameraFeedCfg(camera_name="robot_pov_cam")], camera=depth)
    with pytest.raises(ValueError, match="RGB or RGBA"):
        XrCameraFeedSession.prepare(env_cfg, enabled=True, camera_rendering_enabled=True)


def test_manual_layout_preserves_per_feed_transforms():
    cfgs = [
        XrCameraFeedCfg(camera_name="left", offset_m=(-0.4, 0.1), distance_m=0.7),
        XrCameraFeedCfg(camera_name="right", offset_m=(0.5, -0.2), distance_m=1.1),
    ]

    resolved = camera_feed._layout_feed_cfgs(cfgs, [(100, 50), (100, 50)], XrCameraFeedLayoutCfg())

    assert [(cfg.offset_m, cfg.distance_m) for cfg in resolved] == [
        ((-0.4, 0.1), 0.7),
        ((0.5, -0.2), 1.1),
    ]
    assert resolved is not cfgs


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        (
            XrCameraFeedLayoutCfg(mode="horizontal", panel_gap_m=0.1),
            [(-0.25, 0.0), (0.25, 0.0)],
        ),
        (
            XrCameraFeedLayoutCfg(mode="vertical", panel_gap_m=0.1),
            [(0.0, 0.15), (0.0, -0.15)],
        ),
        (
            XrCameraFeedLayoutCfg(mode="grid", panel_gap_m=0.1, max_columns=2),
            [(-0.25, 0.15), (0.25, 0.15), (0.0, -0.15)],
        ),
    ],
)
def test_automatic_layouts_preserve_order(layout, expected):
    count = len(expected)
    cfgs = [XrCameraFeedCfg(camera_name=str(index), panel_width_m=0.4) for index in range(count)]

    resolved = camera_feed._layout_feed_cfgs(cfgs, [(100, 50)] * count, layout)

    for cfg, expected_offset in zip(resolved, expected):
        assert cfg.offset_m == pytest.approx(expected_offset)


@pytest.mark.parametrize(
    "layout",
    [
        XrCameraFeedLayoutCfg(mode="diagonal"),
        XrCameraFeedLayoutCfg(distance_m=0.0),
        XrCameraFeedLayoutCfg(placement="world"),
        XrCameraFeedLayoutCfg(placement="world", world_position_m=(0.0, 0.0, 0.0), world_orientation_xyzw=(0, 0, 0, 0)),
    ],
)
def test_invalid_layouts_fail_before_panel_creation(layout):
    with pytest.raises(ValueError):
        camera_feed._validate_layout_cfg(layout)


def test_manager_publishes_on_kit_frame_and_closes(monkeypatch):
    cfg = XrCameraFeedCfg(camera_name="robot_pov_cam", max_update_hz=0.0)
    image = _FakeImage()
    manager, presenter, _ = _manager(monkeypatch, [cfg], {"robot_pov_cam": image})

    presenter.subscription.publish()
    manager.close()

    assert presenter.panels[0].uploads == [image]
    assert presenter.subscription.closed
    assert presenter.panels[0].closed


def test_manager_refresh_rebinds_reset_camera_output(monkeypatch):
    cfg = XrCameraFeedCfg(camera_name="robot_pov_cam", max_update_hz=0.0)
    before = _FakeImage(data_ptr=100)
    after = _FakeImage(data_ptr=200)
    manager, presenter, cameras = _manager(monkeypatch, [cfg], {"robot_pov_cam": before})
    cameras["robot_pov_cam"].image_on_update = after

    manager.refresh()

    assert cameras["robot_pov_cam"].update_calls == [(0.0, True)]
    assert presenter.panels[0].uploads == [after]


def test_manager_can_refresh_reset_camera_without_publishing(monkeypatch):
    cfg = XrCameraFeedCfg(camera_name="robot_pov_cam", max_update_hz=0.0)
    before = _FakeImage(data_ptr=100)
    after = _FakeImage(data_ptr=200)
    manager, presenter, cameras = _manager(monkeypatch, [cfg], {"robot_pov_cam": before})
    cameras["robot_pov_cam"].image_on_update = after

    manager.refresh(publish=False)

    assert cameras["robot_pov_cam"].update_calls == [(0.0, True)]
    assert presenter.panels[0].uploads == []


def test_manager_recreates_panel_when_resolution_changes(monkeypatch):
    cfg = XrCameraFeedCfg(camera_name="robot_pov_cam", max_update_hz=0.0)
    before = _FakeImage(height=8, width=12)
    after = _FakeImage(height=10, width=16, data_ptr=200)
    manager, presenter, cameras = _manager(monkeypatch, [cfg], {"robot_pov_cam": before})
    cameras["robot_pov_cam"].output["rgba"].torch = _FakeBatch(after)

    manager.update()

    assert len(presenter.panels) == 2
    assert presenter.panels[0].closed
    assert presenter.panels[1].uploads == [after]


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_manager_rejects_invalid_update_rate(monkeypatch, value):
    cfg = XrCameraFeedCfg(camera_name="robot_pov_cam", max_update_hz=value)
    with pytest.raises(ValueError, match="max_update_hz"):
        _manager(monkeypatch, [cfg], {"robot_pov_cam": _FakeImage()})


def test_public_api_exports_camera_feed_types():
    import isaaclab_teleop

    assert isaaclab_teleop.XrCameraFeedCfg is XrCameraFeedCfg
    assert isaaclab_teleop.XrCameraFeedLayoutCfg is XrCameraFeedLayoutCfg
    assert isaaclab_teleop.XrCameraFeedSession is XrCameraFeedSession
    for removed_name in (
        "XrCameraFeedManager",
        "XrCameraFeedPresentationBackend",
        "XrCameraFeedPresentationCfg",
    ):
        assert not hasattr(isaaclab_teleop, removed_name)


def test_isaac_teleop_default_has_no_camera_feeds():
    cfg = IsaacTeleopCfg(pipeline_builder=lambda: None)

    assert cfg.xr_camera_feeds == []
