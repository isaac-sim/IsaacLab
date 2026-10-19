# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Viser visualizer renders for a recording even when no browser is connected."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from isaaclab_visualizers.viser import ViserVisualizer

pytestmark = [pytest.mark.unit]


def _visualizer(clients: int, record_path: str | None) -> tuple[ViserVisualizer, list[str]]:
    """Visualizer with a fake viewer that records the frames it is asked to draw."""
    calls: list[str] = []
    viewer = SimpleNamespace(
        _server=SimpleNamespace(get_clients=lambda: [object()] * clients),
        begin_frame=lambda time: calls.append("begin_frame"),
        log_state=lambda state: calls.append("log_state"),
        end_frame=lambda: calls.append("end_frame"),
    )
    visualizer = object.__new__(ViserVisualizer)
    visualizer.__dict__.update(
        _is_initialized=True,
        _viewer=viewer,
        _pending_camera_pose=None,
        _sim=SimpleNamespace(get_scene_data_provider=lambda: SimpleNamespace(get_transforms=lambda *a, **k: False)),
        _transform_mapping=None,
        backend=SimpleNamespace(
            model=SimpleNamespace(num_envs=1, body_count=1), geometry_offsets=None, state_0=object()
        ),
        cfg=SimpleNamespace(streaming_view=False, enable_markers=False),
        _sim_time=0.0,
        _paused_rendering=False,
        _active_record_path=record_path,
    )
    visualizer._render_live_plots = lambda: None
    visualizer._push_streaming_frame = lambda: None
    return visualizer, calls


@pytest.mark.parametrize(
    ("clients", "record_path", "draws"),
    [(0, None, False), (0, "run.viser", True), (1, None, True)],
    ids=["idle-without-browser", "recording-without-browser", "browser-connected"],
)
def test_step_draws_only_when_a_browser_or_a_recording_needs_the_frame(
    clients: int, record_path: str | None, draws: bool
) -> None:
    visualizer, calls = _visualizer(clients, record_path)

    visualizer.step(0.02)

    assert (calls == ["begin_frame", "log_state", "end_frame"]) == draws
    assert bool(calls) == draws
