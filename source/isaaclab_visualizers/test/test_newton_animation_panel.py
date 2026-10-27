# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One recorded animation is drawn as a tinted ghost over the simulation and plays with the Playback pause button."""

from types import SimpleNamespace

import numpy as np
import pytest
from isaaclab_visualizers.newton.newton_animation_panel import GHOST_COLOR, NewtonAnimationPanel
from isaaclab_visualizers.newton.newton_replay import NewtonReplayPlayer, ReplayShape, ViserRecording

pytestmark = [pytest.mark.unit]


class _Viewer:
    """Records what the players publish, in place of a Newton viewer."""

    device = "cpu"

    def __init__(self):
        self.objects = {}
        self._paused_training = False
        self.instances = {}

    def log_mesh(self, name, *args, **kwargs):
        self.objects[name] = kwargs

    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False, opacities=None):
        self.objects[name] = {}
        self.instances[name] = (colors.numpy().copy(), hidden)


def _recording(path="clip.viser", static=False):
    xforms = np.zeros((3, 2, 7), np.float32)
    xforms[..., 6] = 1.0
    if not static:
        xforms[:, :, 0] = np.arange(3)[:, None]
    shape = ReplayShape(
        name="shape",
        vertices=np.zeros((3, 3), np.float32),
        indices=np.array([0, 1, 2], np.int32),
        positions=xforms[..., :3],
        wxyzs=np.zeros((3, 2, 4), np.float32),
        scales=np.ones((2, 3), np.float32),
        colors=np.full((2, 3), 0.2, np.float32),
        xforms=xforms,
    )
    return ViserRecording(path=path, times=np.array([0.0, 0.1, 0.2]), shapes=[shape])


@pytest.fixture
def panel(monkeypatch):
    monkeypatch.setattr(NewtonReplayPlayer, "load", lambda self, path: None)  # The tests hand over decoded recordings.
    viewer = _Viewer()
    owner = SimpleNamespace(_viewer=viewer, simulation_hidden=False, rtx_rendering=False)
    return NewtonAnimationPanel(owner, lambda: "scene"), viewer


def _open(panel, recording):
    panel.open(recording.path)
    panel.player._pending = recording
    panel.update()  # The first update installs the decoded recording.
    return panel.player


def test_the_animation_is_tinted_with_the_ghost_color_and_can_be_closed(panel):
    panel, viewer = panel
    player = _open(panel, _recording())
    assert player.color == GHOST_COLOR and player.tint == 1.0
    player.tint = 0.5
    panel.update()
    colors, hidden = viewer.instances["/replay/instances_0"]
    np.testing.assert_allclose(colors[0], 0.2 + (np.asarray(GHOST_COLOR) - 0.2) * 0.5, atol=1e-6)
    assert not hidden and panel.active
    panel.close()
    assert not viewer.objects and not panel.active


def test_opening_another_animation_replaces_the_current_one_and_keeps_the_settings(panel):
    panel, viewer = panel
    player = _open(panel, _recording("a.viser"))
    player.opacity = 0.3
    player._pending = _recording("b.viser")
    panel.update()
    assert panel.player.recording.path == "b.viser" and panel.player.opacity == 0.3
    assert {key.rsplit("/", 1)[0] for key in viewer.objects} == {"/replay"}


def test_playback_follows_the_pause_button_and_static_shapes_follow_the_live_scene(panel):
    panel, viewer = panel
    player = _open(panel, _recording(static=True))
    panel.update()
    assert player.playing and player.hide_static
    assert viewer.instances["/replay/instances_0"][1]  # Static shapes are hidden over the live scene.
    viewer._paused_training = True
    panel._owner.simulation_hidden = True
    panel.update()
    assert not player.playing and not player.hide_static


def test_the_timeline_can_be_dragged_only_without_live_checkpoint_ghosts(panel):
    panel, _ = panel
    assert panel.can_scrub  # No Policy Ghosts at all.
    slots = [SimpleNamespace(active=True), SimpleNamespace(active=False)]
    panel._owner._policy_ghost_panel = SimpleNamespace(layout=SimpleNamespace(slots=slots))
    assert panel.can_scrub  # Only the reference is loaded.
    slots[1].active = True
    assert not panel.can_scrub
