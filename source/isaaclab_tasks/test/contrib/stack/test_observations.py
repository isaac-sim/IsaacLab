# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for stack-task observations."""

from types import SimpleNamespace

import torch

from isaaclab_tasks.contrib.stack.mdp import observations


class _Scene(dict):
    """Dictionary-backed scene exposing the surface-gripper collection."""

    surface_grippers: dict


def test_surface_gripper_subtask_observations_preserve_environment_batch(monkeypatch) -> None:
    """Surface-gripper predicates return one value per environment without cross-environment broadcasting."""
    num_envs = 3
    lower_pos = torch.zeros(num_envs, 3)
    upper_pos = torch.tensor(
        [
            [0.0, 0.0, 0.0468],
            [0.0, 0.0, 0.0468],
            [0.1, 0.0, 0.0468],
        ]
    )
    scene = _Scene(
        robot=SimpleNamespace(),
        ee_frame=SimpleNamespace(
            data=SimpleNamespace(
                target_pos_w=SimpleNamespace(torch=torch.zeros(num_envs, 1, 3)),
            )
        ),
        cube_1=SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=SimpleNamespace(torch=lower_pos),
            )
        ),
        cube_2=SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=SimpleNamespace(torch=upper_pos),
            )
        ),
    )
    scene.surface_grippers = {
        "surface_gripper": SimpleNamespace(state=torch.tensor([1.0, -1.0, 0.0])),
    }
    env = SimpleNamespace(scene=scene)
    monkeypatch.setattr(observations.wp, "to_torch", lambda state: state)

    grasped = observations.object_grasped(
        env,
        robot_cfg=SimpleNamespace(name="robot"),
        ee_frame_cfg=SimpleNamespace(name="ee_frame"),
        object_cfg=SimpleNamespace(name="cube_1"),
    )
    stacked = observations.object_stacked(
        env,
        robot_cfg=SimpleNamespace(name="robot"),
        upper_object_cfg=SimpleNamespace(name="cube_2"),
        lower_object_cfg=SimpleNamespace(name="cube_1"),
    )

    assert grasped.shape == (num_envs,)
    assert torch.equal(grasped, torch.tensor([True, False, False]))
    assert stacked.shape == (num_envs,)
    assert torch.equal(stacked, torch.tensor([False, True, False]))
