# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that :func:`cubes_stacked` does not report success for a cube still in flight."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.contrib.stack.mdp.terminations import cubes_stacked

CUBE = 0.0468


def _env(moving: str | None = None, speed: float = 0.1) -> SimpleNamespace:
    """A perfect three-cube stack with the gripper open; ``moving`` names a cube still falling."""
    scene = {}
    for i, name in enumerate(["cube_1", "cube_2", "cube_3"]):
        pos = SimpleNamespace(torch=torch.tensor([[0.5, 0.0, 0.0203 + i * CUBE]]))
        vel = SimpleNamespace(torch=torch.tensor([[0.0, 0.0, -speed if name == moving else 0.0]]))
        scene[name] = SimpleNamespace(data=SimpleNamespace(root_pos_w=pos, root_lin_vel_w=vel))
    joint_pos = SimpleNamespace(torch=torch.full((1, 9), 0.04))
    scene["robot"] = SimpleNamespace(find_joints=lambda names: ([7, 8], []), data=SimpleNamespace(joint_pos=joint_pos))
    cfg = SimpleNamespace(gripper_joint_names=["panda_finger_.*"], gripper_open_val=0.04)
    return SimpleNamespace(scene=scene, cfg=cfg, device="cpu")


def test_resting_stack_is_success():
    assert bool(cubes_stacked(_env())[0])


@pytest.mark.parametrize("moving", ["cube_1", "cube_2", "cube_3"])
def test_moving_cube_is_not_success(moving):
    assert not bool(cubes_stacked(_env(moving))[0])


def test_opt_out_keeps_position_only_check():
    assert bool(cubes_stacked(_env("cube_3"), max_lin_vel=None)[0])
