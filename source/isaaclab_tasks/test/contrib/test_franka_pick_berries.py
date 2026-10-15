# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Franka berry-picking task's punnet layout, tissue resolution and gamepad, without assets or a GPU."""

import os
import struct
from itertools import combinations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from isaaclab_tasks.contrib.franka_pick_berries.control import teleop_devices
from isaaclab_tasks.contrib.franka_pick_berries.control.teleop_devices import BerryGamepad, BerryGamepadCfg
from isaaclab_tasks.contrib.franka_pick_berries.physics.tissue import _half_resolution, random_punnet_poses
from isaaclab_tasks.contrib.franka_pick_berries.scene.tableware import PUNNET


def _berry_points(spacing: float = 0.001) -> np.ndarray:
    """Return a raspberry-sized ellipsoid sampled on a cubic lattice [m], resting on z = 0."""
    axis = np.arange(-0.011, 0.0111, spacing)
    xyz = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    xyz = xyz[((xyz / [0.009, 0.009, 0.011]) ** 2).sum(1) <= 1]
    return xyz - [0, 0, xyz[:, 2].min()]


@pytest.mark.parametrize("count", [2, 3])
def test_random_layout_places_separated_berries_on_the_punnet_floor(count):
    """Random layouts keep each berry inside the punnet, apart from the others and on the floor; a seed replays one."""
    berry = _berry_points()
    rotations, shifts = random_punnet_poses(berry, berry, count, 7, PUNNET)
    placed = [berry @ rotation.T + shift for rotation, shift in zip(rotations, shifts)]
    half_width, half_depth, _ = PUNNET.floor_outline
    for points in placed:
        assert np.all(np.abs(points[:, 0]) < half_width) and np.all(np.abs(points[:, 1]) < half_depth)
        assert points[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    for first, second in combinations(placed, 2):
        # The gripper needs room between the berries.
        assert cKDTree(first).query(second)[0].min() > 0.01
    replayed = random_punnet_poses(berry, berry, count, 7, PUNNET)
    np.testing.assert_array_equal(replayed[0], rotations)
    np.testing.assert_array_equal(replayed[1], shifts)


def test_half_resolution_keeps_the_tissue_mass_and_needs_a_lattice():
    """Half of the lattice's particles carry the whole volume; scattered particles are rejected."""
    xyz = _berry_points()
    particles = {
        "xyz": xyz,
        "regions": np.zeros(len(xyz), np.int32),
        "interface": np.zeros(len(xyz), bool),
        "spacing": np.float32(0.001),
        "particle_volume": np.float64(1e-9),
    }
    half = _half_resolution(particles)
    assert len(half["xyz"]) == pytest.approx(len(xyz) / 2, rel=0.02)
    assert len(half["xyz"]) * half["particle_volume"] == pytest.approx(len(xyz) * 1e-9)
    assert half["spacing"] ** 3 == pytest.approx(half["particle_volume"], rel=1e-5)
    scattered = dict(particles, xyz=xyz + np.random.default_rng(0).uniform(-3e-4, 3e-4, xyz.shape))
    with pytest.raises(ValueError, match="regular-lattice"):
        _half_resolution(scattered)


# A Linux joystick with the xpad layout: axes [LX, LY, LT, RX, RY, RT, HAT X, HAT Y]; LB is button 4 and Menu button 7.
_AXIS_CODES = bytes([0, 1, 2, 3, 4, 5, 16, 17])
_BUTTON_CODES = {4: 310, 7: 315}
_EVENT = struct.Struct("=IhBB")
_BUTTON, _AXIS, _INITIAL = 0x01, 0x02, 0x80


def _joystick_ioctl(fd, request, buffer):
    """Answer the joystick driver's ioctls for the xpad layout."""
    if request == BerryGamepad._JSIOCGNAME:
        buffer[:4] = b"Pad\0"
    elif request == BerryGamepad._JSIOCGAXES:
        buffer[0] = len(_AXIS_CODES)
    elif request == BerryGamepad._JSIOCGAXMAP:
        buffer[: len(_AXIS_CODES)] = _AXIS_CODES
    elif request == BerryGamepad._JSIOCGBTNMAP:
        buffer[:] = struct.pack("=512H", *(_BUTTON_CODES.get(i, 0) for i in range(512)))


class _Joystick:
    """The driver side of a joystick node: a FIFO the gamepad reads, fed with joystick events."""

    def __init__(self, path):
        self._fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)

    def button(self, number, pressed, initial=False):
        os.write(self._fd, _EVENT.pack(0, int(pressed), _BUTTON | (_INITIAL if initial else 0), number))

    def axis(self, number, value):
        os.write(self._fd, _EVENT.pack(0, value, _AXIS, number))

    def unplug(self):
        """Close the node's writing end: the gamepad then reads end of file, as from a disconnected device."""
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


# The gamepad reads Linux joystick devices, simulated here with a FIFO.
linux_joystick = pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Linux joystick devices need a POSIX FIFO")


@pytest.fixture
def gamepad_and_joystick(monkeypatch, tmp_path):
    monkeypatch.setattr(teleop_devices, "ioctl", _joystick_ioctl)
    node = tmp_path / "js0"
    os.mkfifo(node)
    gamepad = BerryGamepad(BerryGamepadCfg(device=str(node)))
    joystick = _Joystick(node)
    yield gamepad, joystick
    joystick.unplug()
    gamepad.close()


@linux_joystick
def test_gamepad_moves_and_grips_only_while_lb_is_held(gamepad_and_joystick):
    """LB enables motion and the triggers; released, the command stops and the gripper holds its opening."""
    gamepad, joystick = gamepad_and_joystick
    joystick.axis(1, -32767)  # Left stick fully forward.
    joystick.axis(5, 32767)  # RT fully pressed: close.
    command = gamepad.advance()
    assert not command[:6].any() and command[6] == 1.0
    joystick.button(4, True)
    command = gamepad.advance()
    assert command[0] > 0 and not command[1:6].any()
    assert command[6] < 1.0
    closing = command[6]
    joystick.button(4, False)
    command = gamepad.advance()
    assert not command[:6].any() and command[6] == closing
    resets = []
    gamepad.add_callback("R", lambda: resets.append(True))
    joystick.button(7, True)
    gamepad.advance()
    assert resets == [True]


@linux_joystick
def test_gamepad_ignores_lb_held_at_start_and_stops_when_unplugged(gamepad_and_joystick):
    """LB already down when the device opens must be pressed again; a lost device stops all motion."""
    gamepad, joystick = gamepad_and_joystick
    joystick.button(4, True, initial=True)
    joystick.axis(1, -32767)
    assert not gamepad.advance()[:6].any()
    joystick.button(4, False)
    joystick.button(4, True)
    assert gamepad.advance()[0] > 0
    joystick.unplug()
    assert not gamepad.advance()[:6].any()
