# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion shards must cover each complete clip once without changing its frames."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_tasks.core.motion_tracking.mdp.model_based.commands import _sample_frames
from isaaclab_tasks.core.motion_tracking.mdp.model_based.reference import MotionReference


@pytest.mark.parametrize("frame_count", (1_000_000, 47_400_000))
def test_last_timeline_bin_stays_inside_reference(frame_count):
    """Float32 rounding must not turn a valid random sample into an exclusive end."""
    distribution = torch.zeros(frame_count // 50 + 1)
    distribution[-1] = 1.0
    torch.manual_seed(73)
    frames = _sample_frames(distribution, frame_count, 8192)
    assert frames.min() >= frame_count - 64
    assert frames.max() < frame_count


def test_reference_shards_preserve_unequal_length_clips(tmp_path):
    model = tmp_path / "robot.xml"
    model.write_text(
        '<mujoco><worldbody><body name="base"><freejoint/><geom size="0.1"/>'
        '<body name="link"><joint name="joint"/><geom size="0.1"/></body>'
        "</body></worldbody></mujoco>"
    )
    expected = {}
    for index, length in enumerate((2, 7, 3, 5, 4)):
        fields = {
            name: np.full((length, width), index, dtype=np.float32)
            for name, width in zip(
                ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w"),
                (1, 1, 3, 4, 3, 3),
                strict=True,
            )
        }
        name = f"clip_{index}"
        np.savez(tmp_path / f"{name}.npz", **fields, fps=50)
        expected[name] = fields
    seen = set()
    for rank in range(2):
        cfg = SimpleNamespace(
            motion_dir=str(tmp_path),
            kinematics_path=str(model),
            joint_names=["joint"],
            tracked_bodies=["base", "link"],
            motion_shard_count=2,
            motion_shard_index=rank,
        )
        reference = MotionReference(cfg, "cpu")
        assert not seen.intersection(reference.names)
        seen.update(reference.names)
        for name, start, end in zip(reference.names, reference.motion_starts, reference.motion_ends, strict=True):
            for field, values in expected[name].items():
                torch.testing.assert_close(getattr(reference, field)[start:end], torch.from_numpy(values))
    assert seen == set(expected)
