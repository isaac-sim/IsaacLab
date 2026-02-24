# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import Any, cast

import pytest

pytest.importorskip("newton.viewer")

import isaaclab.visualizers.viser_visualizer as viser_visualizer
from isaaclab.visualizers.viser_visualizer_cfg import ViserVisualizerCfg


class _DummySceneDataProvider:
    def __init__(self):
        self._metadata = {"num_envs": 4}
        self.state_calls: list[list[int] | None] = []

    def get_metadata(self) -> dict:
        return self._metadata

    def get_newton_model(self):
        return "dummy-model"

    def get_newton_state(self, env_ids: list[int] | None):
        self.state_calls.append(env_ids)
        return {"state_call": len(self.state_calls), "env_ids": env_ids}


class _DummyViewer:
    def __init__(self):
        self.calls = []

    def begin_frame(self, sim_time: float) -> None:
        self.calls.append(("begin_frame", sim_time))

    def log_state(self, state) -> None:
        self.calls.append(("log_state", state))

    def end_frame(self) -> None:
        self.calls.append(("end_frame",))

    def close(self) -> None:
        self.calls.append(("close",))

    def is_running(self) -> bool:
        return True


def test_newton_viewer_viser_forwards_init_args_and_metadata(monkeypatch: pytest.MonkeyPatch):
    disable_calls = []
    monkeypatch.setattr(
        viser_visualizer, "_disable_viser_runtime_client_rebuild_if_bundled", lambda: disable_calls.append(True)
    )

    captured = {}

    def _fake_base_init(self, *, port: int, label: str, verbose: bool, share: bool, record_to_viser: str | None):
        captured.update(
            {"port": port, "label": label, "verbose": verbose, "share": share, "record_to_viser": record_to_viser}
        )

    monkeypatch.setattr(viser_visualizer.ViewerViser, "__init__", _fake_base_init)

    metadata = {"num_envs": 16}
    viewer = viser_visualizer.NewtonViewerViser(
        port=9090,
        label="unit-test",
        verbose=False,
        share=True,
        record_to_viser="record.viser",
        metadata=metadata,
    )

    assert disable_calls == [True]
    assert captured == {
        "port": 9090,
        "label": "unit-test",
        "verbose": False,
        "share": True,
        "record_to_viser": "record.viser",
    }
    assert viewer._metadata is metadata


def test_viser_visualizer_initialize_and_step_uses_provider_state(monkeypatch: pytest.MonkeyPatch):
    provider = _DummySceneDataProvider()
    viewer = _DummyViewer()

    def _fake_create_viewer(self, record_to_viser: str | None, metadata: dict | None = None):
        assert record_to_viser is None
        assert metadata == provider.get_metadata()
        self._viewer = viewer

    monkeypatch.setattr(viser_visualizer.ViserVisualizer, "_create_viewer", _fake_create_viewer)

    visualizer = viser_visualizer.ViserVisualizer(ViserVisualizerCfg())
    visualizer.initialize(cast(Any, provider))
    visualizer.step(0.25)

    assert visualizer.is_initialized
    assert provider.state_calls == [None, None]
    assert visualizer._sim_time == pytest.approx(0.25)
    assert viewer.calls[0][0] == "begin_frame"
    assert viewer.calls[0][1] == pytest.approx(0.25)
    assert viewer.calls[1] == ("log_state", {"state_call": 2, "env_ids": None})
    assert viewer.calls[2] == ("end_frame",)


@pytest.mark.parametrize(
    ("cfg_max_worlds", "expected_max_worlds"),
    [
        (None, None),
        (0, None),
        (3, 3),
    ],
)
def test_viser_visualizer_create_viewer_forwards_max_worlds(
    monkeypatch: pytest.MonkeyPatch, cfg_max_worlds: int | None, expected_max_worlds: int | None
):
    captured = {}

    class _FakeNewtonViewerViser:
        def __init__(
            self,
            *,
            port: int,
            label: str | None,
            verbose: bool,
            share: bool,
            record_to_viser: str | None,
            metadata: dict | None = None,
        ):
            captured["init"] = {
                "port": port,
                "label": label,
                "verbose": verbose,
                "share": share,
                "record_to_viser": record_to_viser,
                "metadata": metadata,
            }

        def set_model(self, model: Any, max_worlds: int | None) -> None:
            captured["set_model"] = {"model": model, "max_worlds": max_worlds}

    monkeypatch.setattr(viser_visualizer, "NewtonViewerViser", _FakeNewtonViewerViser)
    monkeypatch.setattr(
        viser_visualizer.ViserVisualizer,
        "_resolve_initial_camera_pose",
        lambda self: ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0)),
    )
    monkeypatch.setattr(viser_visualizer.ViserVisualizer, "_set_viser_camera_view", lambda self, pose: None)

    cfg = ViserVisualizerCfg(max_worlds=cfg_max_worlds)
    visualizer = viser_visualizer.ViserVisualizer(cfg)
    visualizer._model = "dummy-model"
    visualizer._create_viewer(record_to_viser="record.viser", metadata={"num_envs": 8})

    assert captured["set_model"] == {"model": "dummy-model", "max_worlds": expected_max_worlds}


def test_viser_visualizer_close_uses_recording_flag(monkeypatch: pytest.MonkeyPatch):
    cfg = ViserVisualizerCfg(record_to_viser="record.viser")
    visualizer = viser_visualizer.ViserVisualizer(cfg)
    visualizer._is_initialized = True
    visualizer._viewer = cast(Any, object())
    visualizer._active_record_path = "record.viser"
    visualizer._pending_camera_pose = ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

    close_call = {}

    def _fake_close_viewer(self, finalize_viser: bool = False):
        close_call["finalize_viser"] = finalize_viser

    monkeypatch.setattr(viser_visualizer.ViserVisualizer, "_close_viewer", _fake_close_viewer)

    visualizer.close()

    assert close_call["finalize_viser"] is True
    assert visualizer._viewer is None
    assert visualizer._is_initialized is False
    assert visualizer._is_closed is True
    assert visualizer._active_record_path is None
    assert visualizer._pending_camera_pose is None
