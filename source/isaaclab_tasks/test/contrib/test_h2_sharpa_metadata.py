# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the joint orders and action offsets the H2 + Sharpa tasks rest on."""

from isaaclab_tasks.contrib.h2_sharpa.metadata import (
    ACTION_DIM,
    H2_ACTION_JOINT_ORDER,
    H2_DEFAULT_JOINT_POS,
    H2_PNP_APPLE_CUSTOM_JOINT_POS,
    POLICY_58_ORDER,
)
from isaaclab_tasks.contrib.h2_sharpa.robot_config import h2_body_joint_offsets


def test_policy_order_is_a_subset_of_the_action_order() -> None:
    """The GR00T action converter scatters policy outputs into the action vector by name."""
    assert len(POLICY_58_ORDER) == ACTION_DIM
    assert len(set(POLICY_58_ORDER)) == len(POLICY_58_ORDER)
    assert len(set(H2_ACTION_JOINT_ORDER)) == len(H2_ACTION_JOINT_ORDER)
    assert set(POLICY_58_ORDER) <= set(H2_ACTION_JOINT_ORDER)


def test_unpredicted_joints_have_a_default_pose() -> None:
    """Joints the policy leaves out are held at their default through the action term's offset.

    Without an entry here the offset would be zero and the head, which defaults to a 0.6 rad pitch,
    would be driven upright with the front camera looking past the table.
    """
    unpredicted = [name for name in H2_ACTION_JOINT_ORDER if name not in POLICY_58_ORDER]

    assert unpredicted, "the policy is expected to predict a subset of the articulation"
    assert all(name in H2_DEFAULT_JOINT_POS for name in unpredicted)
    assert "head_pitch_joint" in unpredicted


def test_body_joint_offsets_cover_every_unpredicted_joint() -> None:
    """The action offset holds the unpredicted joints at the pose the task resets them to.

    The GR00T converter zero-fills those entries, so a joint missing from the offset is commanded
    0 rad: the head would tilt up from its 0.6 rad task pose and the front camera would miss the table.
    """
    offsets = h2_body_joint_offsets(H2_PNP_APPLE_CUSTOM_JOINT_POS)

    assert set(offsets) == {name for name in H2_ACTION_JOINT_ORDER if name not in POLICY_58_ORDER}
    assert offsets["head_pitch_joint"] == H2_PNP_APPLE_CUSTOM_JOINT_POS["head_pitch_joint"]
    assert offsets["left_knee_joint"] == H2_DEFAULT_JOINT_POS["left_knee_joint"]
