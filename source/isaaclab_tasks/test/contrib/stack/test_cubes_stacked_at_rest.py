# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the at-rest requirement in :func:`cubes_stacked`.

The success check describes an instantaneous configuration -- each cube within ``xy_threshold`` of
the one below and one cube-height above it. A cube released above its target passes through that
configuration on the way down, so without an at-rest requirement a drop scores as a completed
stack. These tests drive the function with hand-built scene objects: no gym.make, USD, or physics.
"""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.stack.mdp.terminations import cubes_stacked

CUBE = 0.0468
"""Cube edge length used by the stack tasks, and therefore the expected vertical gap."""


class _Arr:
    """Mimics the ``.torch`` accessor the asset data classes expose."""

    def __init__(self, tensor: torch.Tensor):
        self.torch = tensor


class _Scene(dict):
    """A scene without ``surface_grippers`` so the parallel-gripper branch is taken."""


def _make_env(cube_pos: dict, cube_vel: dict, jaw: float = 0.04) -> SimpleNamespace:
    n = next(iter(cube_pos.values())).shape[0]
    scene = _Scene()
    for name in cube_pos:
        scene[name] = SimpleNamespace(
            data=SimpleNamespace(root_pos_w=_Arr(cube_pos[name]), root_lin_vel_w=_Arr(cube_vel[name]))
        )
    joint_pos = torch.zeros(n, 9)
    joint_pos[:, 7:9] = jaw
    scene["robot"] = SimpleNamespace(
        find_joints=lambda names: ([7, 8], ["panda_finger_joint1", "panda_finger_joint2"]),
        data=SimpleNamespace(joint_pos=_Arr(joint_pos)),
    )
    cfg = SimpleNamespace(gripper_joint_names=["panda_finger_.*"], gripper_open_val=0.04)
    return SimpleNamespace(scene=scene, cfg=cfg, device="cpu")


def _stacked(n: int = 1, top_speed: float = 0.0, mid_speed: float = 0.0, bottom_speed: float = 0.0):
    """A perfect three-cube stack, with optional velocity on any cube."""
    base = torch.tensor([0.5, 0.0, 0.0203]).repeat(n, 1)
    pos = {
        "cube_1": base.clone(),
        "cube_2": base + torch.tensor([0.0, 0.0, CUBE]),
        "cube_3": base + torch.tensor([0.0, 0.0, 2 * CUBE]),
    }
    vel = {
        "cube_1": torch.tensor([bottom_speed, 0.0, 0.0]).repeat(n, 1),
        "cube_2": torch.tensor([0.0, mid_speed, 0.0]).repeat(n, 1),
        "cube_3": torch.tensor([0.0, 0.0, -top_speed]).repeat(n, 1),
    }
    return pos, vel


def test_resting_stack_is_success():
    env = _make_env(*_stacked())
    assert bool(cubes_stacked(env)[0])


def test_falling_cube_is_not_success():
    """The defect: the stacked configuration is satisfied mid-fall, and must not count."""
    env = _make_env(*_stacked(top_speed=0.10))
    assert not bool(cubes_stacked(env)[0])


def test_opt_out_restores_position_only_check():
    env = _make_env(*_stacked(top_speed=0.10))
    assert bool(cubes_stacked(env, max_lin_vel=None)[0])


@pytest.mark.parametrize("kwargs", [{"mid_speed": 0.10}, {"bottom_speed": 0.10}])
def test_every_cube_in_the_stack_must_be_at_rest(kwargs):
    env = _make_env(*_stacked(**kwargs))
    assert not bool(cubes_stacked(env)[0])


@pytest.mark.parametrize(
    ("speed", "expected"),
    [
        (0.030, True),  # residual contact-solver jitter of a resting cube stays under the default
        (0.049, True),
        (0.051, False),
    ],
)
def test_default_threshold_edges(speed, expected):
    env = _make_env(*_stacked(top_speed=speed))
    assert bool(cubes_stacked(env)[0]) is expected


def test_two_cube_variant_ignores_the_third_object():
    """With ``cube_3_cfg=None`` only the two named cubes are checked, in position and in speed."""
    env = _make_env(*_stacked(top_speed=0.5))
    two = dict(cube_1_cfg=SceneEntityCfg("cube_1"), cube_2_cfg=SceneEntityCfg("cube_2"), cube_3_cfg=None)
    assert bool(cubes_stacked(env, **two)[0])

    env = _make_env(*_stacked(mid_speed=0.5))
    assert not bool(cubes_stacked(env, **two)[0])


def test_remapped_roles_are_checked_as_passed():
    """The Franka variants pass e.g. ``cube_1_cfg=cube_2, cube_2_cfg=cube_3``; the at-rest check follows."""
    env = _make_env(*_stacked(top_speed=0.10))
    remap = dict(cube_1_cfg=SceneEntityCfg("cube_2"), cube_2_cfg=SceneEntityCfg("cube_3"), cube_3_cfg=None)
    assert not bool(cubes_stacked(env, **remap)[0])


def test_batched_verdicts_are_independent():
    pos, vel = _stacked(n=4)
    vel["cube_3"][1, 2] = -0.2
    vel["cube_2"][3, 0] = 0.2
    assert cubes_stacked(_make_env(pos, vel)).tolist() == [True, False, True, False]


def test_gripper_still_gates_success():
    env = _make_env(*_stacked(), jaw=0.02)
    assert not bool(cubes_stacked(env)[0])
