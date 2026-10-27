# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One recorded animation is drawn as a tinted ghost over the simulation and plays with the Playback pause button."""

from types import SimpleNamespace

import numpy as np
import pytest
from isaaclab_visualizers.newton.newton_animation_panel import GHOST_COLOR, NewtonAnimationPanel
from isaaclab_visualizers.newton.newton_replay import (
    NewtonReplayPlayer,
    RecordedSlot,
    ReplayShape,
    ViserRecording,
    _attach_slots,
    write_slot_sidecar,
)

pytestmark = [pytest.mark.unit]


class _Viewer:
    """Records what the players publish, in place of a Newton viewer."""

    device = "cpu"

    def __init__(self):
        self.objects = {}
        self._paused_training = False
        self.instances = {}
        self.instance_scales = {}

    def log_mesh(self, name, *args, **kwargs):
        self.objects[name] = kwargs

    def log_instances(self, name, mesh, xforms, scales, colors, materials, hidden=False, opacities=None):
        self.objects[name] = {}
        self.instances[name] = (colors.numpy().copy(), hidden)
        self.instance_scales[name] = scales.numpy().copy()


def _recording(path="clip.viser", static=False):
    xforms = np.zeros((3, 2, 7), np.float32)
    xforms[..., 6] = 1.0
    if not static:
        xforms[:, :, 0] = np.arange(3)[:, None]
    shape = ReplayShape(
        name="/model/shapes/shape_0",
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
    owner.set_simulation_hidden = lambda hidden: setattr(owner, "simulation_hidden", hidden)
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


def test_the_animation_plays_on_its_own_clock_and_static_shapes_follow_the_live_scene(panel):
    panel, viewer = panel
    player = _open(panel, _recording(static=True))
    assert player.playing  # A new animation starts playing.
    panel.update()
    assert player.hide_static
    assert viewer.instances["/replay/instances_0"][1]  # Static shapes are hidden over the live scene.
    viewer._paused_training = True  # Pausing the simulation does not pause the animation.
    panel._owner.simulation_hidden = True
    panel.update()
    assert player.playing and not player.hide_static


def _ghost_session(tmp_path):
    """A recording of two slots of one environment each: the reference and a ghost."""
    recording = _recording(str(tmp_path / "ghosts.viser"))
    recording.shapes[0].worlds = np.array([0, 1], np.int32)
    recording.slots = [RecordedSlot(0, "model_100.pt", None), RecordedSlot(1, "model_50.pt", (1.0, 0.0, 0.0))]
    recording.slice_size = 1
    return recording


def test_a_ghost_session_tints_only_the_ghost_slots_and_hides_slots_on_request(panel, tmp_path):
    panel, viewer = panel
    recording = _ghost_session(tmp_path)
    player = _open(panel, recording)
    colors, _ = viewer.instances["/replay/instances_0"]
    np.testing.assert_allclose(colors[0], [0.2] * 3, atol=1e-6)  # The reference keeps its colors.
    np.testing.assert_allclose(colors[1], [1.0, 0.0, 0.0], atol=1e-6)
    assert panel._owner.simulation_hidden  # The recording already holds the reference.
    player.slot_visible[1] = False
    panel.update()
    scales = viewer.instance_scales["/replay/instances_0"]
    assert scales[0].tolist() == [1.0] * 3 and scales[1].tolist() == [0.0] * 3


def test_the_slot_description_is_read_back_next_to_the_recording(tmp_path):
    recording = _ghost_session(tmp_path)
    recording.slots, recording.slice_size, recording.shapes[0].worlds = [], 0, None
    write_slot_sidecar(
        recording.path, 1, [RecordedSlot(0, "a.pt", None), RecordedSlot(1, "b.pt", (1.0, 0.0, 0.0))],
        {"/model/shapes/shape_0": [0, 1]},
    )  # fmt: skip
    _attach_slots(recording)
    assert [slot.label for slot in recording.slots] == ["a.pt", "b.pt"] and recording.slice_size == 1
    assert recording.shapes[0].worlds.tolist() == [0, 1]


def test_markers_of_a_ghost_session_start_hidden_and_can_be_shown(panel, tmp_path):
    panel, viewer = panel
    recording = _ghost_session(tmp_path)
    arrows = _recording()
    arrows.shapes[0].name = "/Visuals/Command/velocity_goal_1/arrow"
    recording.shapes.append(arrows.shapes[0])
    player = _open(panel, recording)
    assert not player.show_markers
    assert viewer.instances["/replay/instances_1"][1]  # Hidden, while the robots stay.
    assert not viewer.instances["/replay/instances_0"][1]
    player.show_markers = True
    panel.update()
    assert not viewer.instances["/replay/instances_1"][1]


def test_markers_are_matched_to_slots_by_the_recorded_environments(tmp_path):
    recording = _ghost_session(tmp_path)
    recording.slots, recording.slice_size, recording.shapes[0].worlds = [], 0, None
    recording.shapes[0].name = "/Visuals/Command/velocity_goal_1/arrow"
    write_slot_sidecar(
        recording.path, 1, [RecordedSlot(0, "a.pt", None), RecordedSlot(1, "b.pt", (1.0, 0.0, 0.0))], {}, [0, 1]
    )
    _attach_slots(recording)
    assert recording.shapes[0].worlds.tolist() == [0, 1]
