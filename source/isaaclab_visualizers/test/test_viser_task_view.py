# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Viser task view: a toggle that is off by default and an environment selector."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch
from isaaclab_visualizers.viser import ViserVisualizer, ViserVisualizerCfg

from isaaclab.visualizers import PerspectiveCameraCfg


class _Scene:
    num_envs = 3
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])

    def __getitem__(self, name):
        return SimpleNamespace(is_initialized=False)


class _Element:
    """A GUI element that records its initial value and its update callback."""

    def __init__(self, initial_value):
        self.initial_value, self.callback = initial_value, None

    def on_update(self, fn):
        self.callback = fn
        return fn


class _Gui:
    def __init__(self):
        self.elements = {}

    def add_folder(self, label, expand_by_default=True):
        return contextlib.nullcontext()

    def add_checkbox(self, label, initial_value):
        return self.elements.setdefault(label, _Element(initial_value))

    def add_number(self, label, initial_value, min=None, max=None, step=None):
        element = self.elements.setdefault(label, _Element(initial_value))
        element.bounds = (min, max, step)
        return element


def _visualizer(cameras):
    viz = ViserVisualizer(ViserVisualizerCfg())
    scene = _Scene()
    viz._sim = SimpleNamespace(
        get_scene_data_provider=lambda: SimpleNamespace(get_interactive_scene=lambda: scene, num_envs=scene.num_envs)
    )
    viz._bind_task_view(cameras)
    return viz


def test_the_task_view_is_off_by_default_and_the_environment_defaults_to_zero():
    viz = _visualizer([PerspectiveCameraCfg(), PerspectiveCameraCfg(env_index=1, eye=(1.0, 2.0, 3.0))])
    server = SimpleNamespace(gui=_Gui())
    viz._add_task_view_controls(server)
    toggle, env = server.gui.elements["Follow task view"], server.gui.elements["Environment"]
    assert toggle.initial_value is False and viz._following_task_view is False
    assert env.initial_value == 1 and env.bounds == (0, 2, 1)  # the environment the view declares
    assert _visualizer([PerspectiveCameraCfg(track_path="robot")])._task_follower.env == 0  # else 0


def test_toggling_follows_the_task_view_and_the_selector_changes_its_environment():
    viz = _visualizer(
        [PerspectiveCameraCfg(env_index=0, eye=(1.0, 0.0, 0.0), lookat=(0.0, 0.0, 0.0), focal_length=30.0)]
    )
    server = SimpleNamespace(gui=_Gui())
    viz._add_task_view_controls(server)
    poses = []
    viz.set_camera_view = lambda eye, target, focal_length=None: poses.append((eye, focal_length))
    viz._is_initialized, viz._sim_time = True, 0.0
    viz._viewer = SimpleNamespace(_server=SimpleNamespace(get_clients=lambda: {}))
    viz._pending_camera_pose = None
    viz.step(0.1)
    assert poses == []  # off: the browser camera is left alone

    server.gui.elements["Follow task view"].callback(SimpleNamespace(target=SimpleNamespace(value=True)))
    server.gui.elements["Environment"].callback(SimpleNamespace(target=SimpleNamespace(value=2)))
    viz.step(0.1)
    assert poses == [((21.0, 0.0, 0.0), 30.0)]

    server.gui.elements["Environment"].callback(SimpleNamespace(target=SimpleNamespace(value=9)))  # out of range
    assert viz._task_follower.env == 2


def test_a_visualizer_without_a_task_view_adds_no_controls():
    viz = _visualizer([PerspectiveCameraCfg()])
    assert viz._task_follower is None and viz._task_view is None
    with pytest.raises(AttributeError):
        viz._add_task_view_controls(SimpleNamespace(gui=_Gui()))
