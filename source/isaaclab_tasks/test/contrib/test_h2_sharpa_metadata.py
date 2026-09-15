# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the joint orders and action offsets the H2 + Sharpa tasks rest on."""

import pytest

from isaaclab_tasks.contrib.pack_agx.config import metadata as pack_agx_metadata
from isaaclab_tasks.contrib.pack_agx.config.robot_config import h2_body_joint_offsets as pack_agx_offsets
from isaaclab_tasks.contrib.pick_and_place_apple.config import env_config as apple_env_config
from isaaclab_tasks.contrib.pick_and_place_apple.config import metadata as apple_metadata
from isaaclab_tasks.contrib.pick_and_place_apple.config.robot_config import h2_body_joint_offsets as apple_offsets

TASK_METADATA = {"pick_and_place_apple": apple_metadata, "pack_agx": pack_agx_metadata}


@pytest.mark.parametrize("metadata", TASK_METADATA.values(), ids=TASK_METADATA.keys())
def test_policy_order_is_a_subset_of_the_action_order(metadata) -> None:
    """The GR00T action converter scatters policy outputs into the action vector by name."""
    assert len(metadata.POLICY_58_ORDER) == metadata.ACTION_DIM
    assert len(set(metadata.POLICY_58_ORDER)) == len(metadata.POLICY_58_ORDER)
    assert len(set(metadata.H2_ACTION_JOINT_ORDER)) == len(metadata.H2_ACTION_JOINT_ORDER)
    assert set(metadata.POLICY_58_ORDER) <= set(metadata.H2_ACTION_JOINT_ORDER)


@pytest.mark.parametrize("metadata", TASK_METADATA.values(), ids=TASK_METADATA.keys())
def test_unpredicted_joints_have_a_default_pose(metadata) -> None:
    """Joints the policy leaves out are held at their default through the action term's offset.

    Without an entry here the offset would be zero and the head, which defaults to a 0.6 rad pitch,
    would be driven upright with the front camera looking past the table.
    """
    unpredicted = [name for name in metadata.H2_ACTION_JOINT_ORDER if name not in metadata.POLICY_58_ORDER]

    assert unpredicted, "the policy is expected to predict a subset of the articulation"
    assert all(name in metadata.H2_DEFAULT_JOINT_POS for name in unpredicted)
    assert "head_pitch_joint" in unpredicted


def test_body_joint_offsets_cover_every_unpredicted_joint() -> None:
    """The action offset holds the unpredicted joints at the pose the task resets them to.

    The GR00T converter zero-fills those entries, so a joint missing from the offset is commanded
    0 rad: the head would tilt up from its 0.6 rad task pose and the front camera would miss the table.
    """
    offsets = apple_offsets(apple_env_config.CUSTOM_JOINT_POS)
    unpredicted = {name for name in apple_metadata.H2_ACTION_JOINT_ORDER if name not in apple_metadata.POLICY_58_ORDER}

    assert set(offsets) == unpredicted
    assert offsets["head_pitch_joint"] == apple_env_config.CUSTOM_JOINT_POS["head_pitch_joint"]
    assert offsets["left_knee_joint"] == apple_metadata.H2_DEFAULT_JOINT_POS["left_knee_joint"]


def test_the_tasks_embodiment_copies_have_not_drifted() -> None:
    """Each task carries its own copy of the embodiment metadata; the copies must still agree.

    The joint orders index into one shared checkpoint and one shared robot USD, so a name added to
    or reordered in only one copy would silently mis-wire that task's actions.
    """
    for name in ("ACTION_DIM", "H2_ACTION_JOINT_ORDER", "POLICY_58_ORDER", "H2_DEFAULT_JOINT_POS"):
        assert getattr(apple_metadata, name) == getattr(pack_agx_metadata, name), name
    assert apple_offsets() == pack_agx_offsets()
