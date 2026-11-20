# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect body-name alignment, velocity reference points, and source immutability."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_tasks.core.motion_tracking.config.micro_duck.tracking_env_cfg import (
    MicroDuckMotionCommand,
    MicroDuckMotionCommandCfg,
)


@pytest.mark.parametrize("velocity_point", ["link", "com"])
def test_source_motion_adaptation(tmp_path, velocity_point):
    """A rotating, translated link has a different COM velocity but the same pose."""
    path = tmp_path / "motion.npz"
    np.savez(
        path,
        fps=50.0,
        body_names=["neck_pitch", "trunk_base"],
        joint_names=["right", "left"],
        joint_pos=[[0.1, 0.2], [0.3, 0.4]],
        joint_vel=[[1.0, 2.0], [3.0, 4.0]],
        body_pos_w=[[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]] * 2,
        body_quat_w=[[[1.0, 0.0, 0.0, 0.0], [2**-0.5, 0.0, 0.0, 2**-0.5]]] * 2,
        body_lin_vel_w=[[[2.0, 3.0, 4.0], [1.0, 0.0, 0.0]]] * 2,
        body_ang_vel_w=[[[0.0, 0.0, 3.0], [0.0, 0.0, 2.0]]] * 2,
        source_contact=[[True, False], [False, True]],
    )
    original_bytes = path.read_bytes()
    command = SimpleNamespace(
        cfg=MicroDuckMotionCommandCfg(
            motion_file=str(path), body_names=["trunk_base", "neck_pitch_1"], input_velocity_point=velocity_point
        ),
        robot=SimpleNamespace(
            joint_names=["left", "right"],
            data=SimpleNamespace(
                body_com_pos_b=SimpleNamespace(torch=torch.tensor([[[0.0, 0.3, 0.0], [0.2, 0.0, 0.0]]]))
            ),
        ),
        body_indices=[1, 0],
        device="cpu",
    )
    motion = MicroDuckMotionCommand._load_motion(command)
    expected_velocities = (
        [[0.6, 0.0, 0.0], [1.1, 3.0, 4.0]] if velocity_point == "link" else [[1.0, 0.0, 0.0], [2.0, 3.0, 4.0]]
    )
    torch.testing.assert_close(motion.body_lin_vel_w, torch.tensor([expected_velocities] * 2))
    torch.testing.assert_close(motion.body_pos_w, torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]] * 2))
    torch.testing.assert_close(
        motion.body_quat_w, torch.tensor([[[0.0, 0.0, 2**-0.5, 2**-0.5], [0.0, 0.0, 0.0, 1.0]]] * 2)
    )
    torch.testing.assert_close(motion.body_ang_vel_w, torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]] * 2))
    torch.testing.assert_close(motion.joint_pos, torch.tensor([[0.2, 0.1], [0.4, 0.3]]))
    torch.testing.assert_close(motion.joint_vel, torch.tensor([[2.0, 1.0], [4.0, 3.0]]))
    assert motion.fps == 50.0 and motion.time_step_total == 2
    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "velocity_point, offsets, message",
    [
        ("unspecified", torch.zeros(1, 3), "Unknown input velocity point"),
        ("link", torch.zeros(2, 3), "Expected finite body COM offsets"),
        ("link", torch.full((1, 3), float("nan")), "Expected finite body COM offsets"),
    ],
)
def test_invalid_velocity_contract(tmp_path, velocity_point, offsets, message):
    """Reject ambiguous velocity conventions and invalid offsets before loading."""
    command = SimpleNamespace(
        cfg=MicroDuckMotionCommandCfg(
            motion_file=str(tmp_path / "not_loaded.npz"), body_names=["trunk_base"], input_velocity_point=velocity_point
        ),
        robot=SimpleNamespace(
            joint_names=["joint"], data=SimpleNamespace(body_com_pos_b=SimpleNamespace(torch=offsets.unsqueeze(0)))
        ),
        body_indices=slice(None),
        device="cpu",
    )
    with pytest.raises(ValueError, match=message):
        MicroDuckMotionCommand._load_motion(command)
