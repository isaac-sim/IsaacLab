# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapt named MicroDuck references without rewriting their source NPZ files."""

from collections.abc import Sequence
from typing import Literal

import torch

from isaaclab.utils.math import quat_apply

from ...data.motion_loader import MotionLoader


def load_microduck_motion(
    motion_file: str,
    body_names: Sequence[str],
    joint_names: Sequence[str],
    device: str,
    body_com_pos_b: torch.Tensor,
    input_velocity_point: Literal["link", "com"],
) -> MotionLoader:
    """Load source body names and normalize world linear velocities to each body's COM.

    ``body_com_pos_b`` contains nominal target-asset COM offsets in link coordinates,
    in ``body_names`` order, with shape (bodies, 3). Input poses remain link-origin
    poses. COM input must already refer to these same inertial offsets.
    The source body ``neck_pitch`` corresponds to the USD body ``neck_pitch_1``;
    the joint name is unchanged. Conversion runs once, after the shared loader's
    name alignment and WXYZ-to-XYZW conversion. No source file is written.
    """
    if input_velocity_point not in ("link", "com"):
        raise ValueError(f"Unknown input velocity point: {input_velocity_point!r}; expected 'link' or 'com'.")
    if body_com_pos_b.shape != (len(body_names), 3) or not torch.isfinite(body_com_pos_b).all():
        raise ValueError("Expected finite body COM offsets with shape (tracked bodies, 3).")
    source_body_names = ["neck_pitch" if name == "neck_pitch_1" else name for name in body_names]
    motion = MotionLoader(motion_file, body_names=source_body_names, joint_names=joint_names, device=device)
    if input_velocity_point == "link":
        offsets = body_com_pos_b.to(device=device, dtype=motion.body_pos_w.dtype)
        offsets_w = quat_apply(motion.body_quat_w, offsets.expand_as(motion.body_pos_w))
        motion.body_lin_vel_w = motion.body_lin_vel_w + torch.cross(motion.body_ang_vel_w, offsets_w, dim=-1)
    return motion
