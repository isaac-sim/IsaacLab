# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Several recorded animations are drawn as tinted ghosts that share one timeline."""

from types import SimpleNamespace

import numpy as np
import pytest
from isaaclab_visualizers.newton.newton_animation_ghosts import MAX_ANIMATIONS, AnimationGhostSet
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
def ghosts(monkeypatch):
    monkeypatch.setattr(NewtonReplayPlayer, "load", lambda self, path: None)  # The tests hand over decoded recordings.
    viewer = _Viewer()
    owner = SimpleNamespace(_viewer=viewer, simulation_hidden=False)
    return AnimationGhostSet(owner, lambda: "scene"), viewer


def _open(ghosts_set, recording):
    ghosts_set.add(recording.path)
    player = ghosts_set.players[-1]
    player._pending = recording
    return player


def test_each_animation_has_its_own_shapes_and_is_removed_alone(ghosts):
    animations, viewer = ghosts
    first, second = _open(animations, _recording("a.viser")), _open(animations, _recording("b.viser"))
    animations.update()
    assert {key.rsplit("/", 1)[0] for key in viewer.objects} == {first.prefix, second.prefix}
    animations.remove(0)
    assert all(key.startswith(second.prefix) for key in viewer.objects)
    assert len(animations) == 1


def test_a_ghost_is_tinted_toward_its_color_and_hides_with_the_checkbox(ghosts):
    animations, viewer = ghosts
    player = _open(animations, _recording())
    player.color, player.tint = (1.0, 0.0, 0.0), 0.5
    animations.update()
    colors, hidden = viewer.instances[f"{player.prefix}/instances_0"]
    np.testing.assert_allclose(colors[0], [0.6, 0.1, 0.1], atol=1e-6)
    assert not hidden
    player.visible = False
    animations.update()
    assert viewer.instances[f"{player.prefix}/instances_0"][1]


def test_playback_follows_the_pause_button_and_static_shapes_follow_the_live_scene(ghosts):
    animations, viewer = ghosts
    player = _open(animations, _recording(static=True))
    animations.update()  # The first update installs the decoded recording.
    animations.update()
    assert player.playing and player.hide_static
    assert viewer.instances[f"{player.prefix}/instances_0"][1]  # Static shapes are hidden over the live scene.
    viewer._paused_training = True
    animations._owner.simulation_hidden = True
    animations.update()
    assert not player.playing and not player.hide_static


def test_new_animations_take_unused_colors_and_the_set_is_limited(ghosts):
    animations, _ = ghosts
    for index in range(MAX_ANIMATIONS):
        animations.add(f"{index}.viser")
    assert len({player.color for player in animations.players}) == min(MAX_ANIMATIONS, 7)
    animations.add("extra.viser")
    assert len(animations) == MAX_ANIMATIONS and "At most" in animations.message


def test_a_recording_that_fails_to_load_is_dropped_with_its_message(ghosts):
    animations, _ = ghosts
    animations.add("broken.viser")
    animations.players[0]._pending = RuntimeError("bad file")
    animations.update()
    assert not animations.players and "bad file" in animations.message


def test_seeking_moves_every_animation(ghosts):
    animations, _ = ghosts
    first, second = _open(animations, _recording("a.viser")), _open(animations, _recording("b.viser"))
    animations.update()
    animations.seek(0.15)
    assert first.time == second.time == pytest.approx(0.15)
    assert animations.duration == pytest.approx(0.2) and isinstance(first, NewtonReplayPlayer)


def test_the_timeline_can_be_dragged_only_without_live_checkpoint_ghosts(ghosts):
    animations, _ = ghosts
    assert animations.can_scrub  # No Policy Ghosts at all.
    slots = [SimpleNamespace(active=True), SimpleNamespace(active=False)]
    animations._owner._policy_ghost_panel = SimpleNamespace(layout=SimpleNamespace(slots=slots))
    assert animations.can_scrub  # Only the reference is loaded.
    slots[1].active = True
    assert not animations.can_scrub
