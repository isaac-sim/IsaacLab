# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared utility functions in isaaclab_newton.assets.utils.shared.

These tests validate the find_joints and find_bodies functions work correctly,
particularly when joint_subset is provided.
"""

from __future__ import annotations

import numpy as np

import pytest
import warp as wp
from isaaclab_newton.assets.utils.shared import find_bodies, find_joints

# Initialize Warp
wp.init()


class TestFindBodies:
    """Tests for find_bodies function."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_bodies_basic(self, device: str):
        """Test that find_bodies works with basic name matching."""
        body_names = ["body_0", "body_1", "body_2", "body_3"]
        mask, names, indices = find_bodies(body_names, ["body_0", "body_2"], device=device)

        assert names == ["body_0", "body_2"]
        assert indices == [0, 2]

        mask_np = mask.numpy()
        expected_mask = np.array([True, False, True, False])
        np.testing.assert_array_equal(mask_np, expected_mask)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_bodies_regex(self, device: str):
        """Test that find_bodies works with regex patterns."""
        body_names = ["arm_link", "arm_joint", "leg_link", "leg_joint"]
        mask, names, indices = find_bodies(body_names, ".*_link", device=device)

        assert names == ["arm_link", "leg_link"]
        assert indices == [0, 2]

        mask_np = mask.numpy()
        expected_mask = np.array([True, False, True, False])
        np.testing.assert_array_equal(mask_np, expected_mask)


class TestFindBodiesWithSubset:
    """Tests for find_bodies function when body_subset is provided.

    These tests mirror the TestFindJointsWithSubset tests to ensure
    consistent behavior between find_bodies and find_joints.
    """

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_bodies_with_subset_basic(self, device: str):
        """Test find_bodies with body_subset returns correct global indices."""
        body_names = [
            "torso",
            "head",
            "left_arm",
            "left_hand",
            "right_arm",
            "right_hand",
            "left_leg",
            "right_leg",
        ]
        body_subset = ["torso", "head", "left_leg", "right_leg"]

        mask, names, indices = find_bodies(body_names, ".*", body_subset=body_subset, device=device)

        assert names == ["torso", "head", "left_leg", "right_leg"]
        # Critical: indices should be GLOBAL indices [0, 1, 6, 7], not local indices [0, 1, 2, 3]
        assert indices == [0, 1, 6, 7], f"Expected global indices [0, 1, 6, 7], got {indices}"

        mask_np = mask.numpy()
        expected_mask = np.array([True, True, False, False, False, False, True, True])
        np.testing.assert_array_equal(mask_np, expected_mask)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_bodies_with_subset_partial_match(self, device: str):
        """Test find_bodies with subset when only some bodies in subset match."""
        body_names = ["base", "link_1", "link_2", "link_3", "link_4", "end_effector"]
        body_subset = ["base", "link_2", "link_4", "end_effector"]

        mask, names, indices = find_bodies(body_names, "link_.*", body_subset=body_subset, device=device)

        assert names == ["link_2", "link_4"]
        assert indices == [2, 4]

        mask_np = mask.numpy()
        expected_mask = np.array([False, False, True, False, True, False])
        np.testing.assert_array_equal(mask_np, expected_mask)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_bodies_with_subset_non_contiguous(self, device: str):
        """Test find_bodies with non-contiguous subset returns correct global indices."""
        body_names = ["a", "b", "c", "d", "e", "f", "g", "h"]
        body_subset = ["b", "d", "f", "h"]

        mask, names, indices = find_bodies(body_names, ".*", body_subset=body_subset, device=device)

        assert names == ["b", "d", "f", "h"]
        assert indices == [1, 3, 5, 7]

        mask_np = mask.numpy()
        expected_mask = np.array([False, True, False, True, False, True, False, True])
        np.testing.assert_array_equal(mask_np, expected_mask)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_bodies_with_subset_preserve_order(self, device: str):
        """Test find_bodies with subset and preserve_order."""
        body_names = ["body_0", "body_1", "body_2", "body_3", "body_4", "body_5"]
        body_subset = ["body_1", "body_3", "body_5"]

        mask, names, indices = find_bodies(
            body_names, ["body_5", "body_1"], body_subset=body_subset, preserve_order=True, device=device
        )

        assert names == ["body_5", "body_1"]
        assert indices == [5, 1]

        mask_np = mask.numpy()
        expected_mask = np.array([False, True, False, False, False, True])
        np.testing.assert_array_equal(mask_np, expected_mask)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_bodies_without_subset_unchanged(self, device: str):
        """Test that find_bodies without subset still works correctly (no regression)."""
        body_names = ["body_0", "body_1", "body_2", "body_3"]

        mask, names, indices = find_bodies(body_names, ["body_1", "body_3"], device=device)

        assert names == ["body_1", "body_3"]
        assert indices == [1, 3]

        mask_np = mask.numpy()
        expected_mask = np.array([False, True, False, True])
        np.testing.assert_array_equal(mask_np, expected_mask)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_bodies_subset_with_single_match(self, device: str):
        """Test find_bodies with subset when only one body matches."""
        body_names = ["body_a", "body_b", "body_c", "body_d"]
        body_subset = ["body_b", "body_d"]

        mask, names, indices = find_bodies(body_names, "body_d", body_subset=body_subset, device=device)

        assert names == ["body_d"]
        assert indices == [3]

        mask_np = mask.numpy()
        expected_mask = np.array([False, False, False, True])
        np.testing.assert_array_equal(mask_np, expected_mask)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_bodies_subset_order_differs_from_global(self, device: str):
        """Test find_bodies when subset order differs from global order."""
        body_names = ["a", "b", "c", "d", "e"]
        body_subset = ["e", "c", "a"]

        mask, names, indices = find_bodies(body_names, ".*", body_subset=body_subset, device=device)

        assert names == ["e", "c", "a"]
        assert indices == [4, 2, 0]

        mask_np = mask.numpy()
        expected_mask = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(mask_np, expected_mask)


class TestFindJoints:
    """Tests for find_joints function."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_joints_basic(self, device: str):
        """Test that find_joints works with basic name matching."""
        joint_names = ["joint_0", "joint_1", "joint_2", "joint_3"]
        mask, names, indices = find_joints(joint_names, ["joint_1", "joint_3"], device=device)

        assert names == ["joint_1", "joint_3"]
        assert indices == [1, 3]

        mask_np = mask.numpy()
        expected_mask = np.array([False, True, False, True])
        np.testing.assert_array_equal(mask_np, expected_mask)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_joints_regex(self, device: str):
        """Test that find_joints works with regex patterns."""
        joint_names = ["hip_left", "knee_left", "hip_right", "knee_right"]
        mask, names, indices = find_joints(joint_names, "hip_.*", device=device)

        assert names == ["hip_left", "hip_right"]
        assert indices == [0, 2]

        mask_np = mask.numpy()
        expected_mask = np.array([True, False, True, False])
        np.testing.assert_array_equal(mask_np, expected_mask)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_joints_preserve_order(self, device: str):
        """Test that find_joints respects preserve_order parameter."""
        joint_names = ["joint_0", "joint_1", "joint_2", "joint_3"]
        mask, names, indices = find_joints(joint_names, ["joint_3", "joint_1"], preserve_order=True, device=device)

        assert names == ["joint_3", "joint_1"]
        assert indices == [3, 1]

        mask_np = mask.numpy()
        expected_mask = np.array([False, True, False, True])
        np.testing.assert_array_equal(mask_np, expected_mask)


class TestFindJointsWithSubset:
    """Tests for find_joints function when joint_subset is provided.

    These tests specifically validate the fix for GitHub issue #4439.
    The bug was that when joint_subset was provided, the returned indices
    were relative to the subset instead of the global joint_names list.
    """

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_joints_with_subset_basic(self, device: str):
        """Test find_joints with joint_subset returns correct global indices.

        This is the core test case from GitHub issue #4439.
        Ant joint_names: ['front_left_leg', 'front_left_foot', 'front_right_leg',
                          'front_right_foot', 'left_back_leg', 'left_back_foot',
                          'right_back_leg', 'right_back_foot']
        Target subset: ['front_left_leg', 'front_left_foot', 'left_back_leg', 'left_back_foot']
        Expected indices: [0, 1, 4, 5] (global indices)
        Bug would return: [0, 1, 2, 3] (local indices of subset)
        """
        joint_names = [
            "front_left_leg",
            "front_left_foot",
            "front_right_leg",
            "front_right_foot",
            "left_back_leg",
            "left_back_foot",
            "right_back_leg",
            "right_back_foot",
        ]
        joint_subset = ["front_left_leg", "front_left_foot", "left_back_leg", "left_back_foot"]

        mask, names, indices = find_joints(joint_names, ".*", joint_subset=joint_subset, device=device)

        # Expected: all joints in the subset should match ".*"
        assert names == ["front_left_leg", "front_left_foot", "left_back_leg", "left_back_foot"]
        # Critical: indices should be GLOBAL indices, not local indices within subset
        assert indices == [0, 1, 4, 5], f"Expected global indices [0, 1, 4, 5], got {indices}"

        # Mask should be True at global positions 0, 1, 4, 5
        mask_np = mask.numpy()
        expected_mask = np.array([True, True, False, False, True, True, False, False])
        np.testing.assert_array_equal(mask_np, expected_mask, err_msg="Mask does not match expected global positions")

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_joints_with_subset_partial_match(self, device: str):
        """Test find_joints with subset when only some joints in subset match."""
        joint_names = ["hip_left", "knee_left", "ankle_left", "hip_right", "knee_right", "ankle_right"]
        joint_subset = ["hip_left", "knee_left", "hip_right", "knee_right"]

        # Match only hip joints within the subset
        mask, names, indices = find_joints(joint_names, "hip_.*", joint_subset=joint_subset, device=device)

        assert names == ["hip_left", "hip_right"]
        # Global indices for hip_left (0) and hip_right (3)
        assert indices == [0, 3]

        mask_np = mask.numpy()
        expected_mask = np.array([True, False, False, True, False, False])
        np.testing.assert_array_equal(mask_np, expected_mask)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_joints_with_subset_non_contiguous(self, device: str):
        """Test find_joints with non-contiguous subset returns correct global indices."""
        joint_names = ["a", "b", "c", "d", "e", "f", "g", "h"]
        # Subset is non-contiguous: indices 1, 3, 5, 7 in global list
        joint_subset = ["b", "d", "f", "h"]

        mask, names, indices = find_joints(joint_names, ".*", joint_subset=joint_subset, device=device)

        assert names == ["b", "d", "f", "h"]
        # Global indices should be [1, 3, 5, 7], not [0, 1, 2, 3]
        assert indices == [1, 3, 5, 7]

        mask_np = mask.numpy()
        expected_mask = np.array([False, True, False, True, False, True, False, True])
        np.testing.assert_array_equal(mask_np, expected_mask)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_joints_with_subset_preserve_order(self, device: str):
        """Test find_joints with subset and preserve_order."""
        joint_names = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"]
        joint_subset = ["joint_1", "joint_3", "joint_5"]

        # Request in specific order with preserve_order=True
        mask, names, indices = find_joints(
            joint_names, ["joint_5", "joint_1"], joint_subset=joint_subset, preserve_order=True, device=device
        )

        assert names == ["joint_5", "joint_1"]
        # Indices should follow the requested order with global indices
        assert indices == [5, 1]

        mask_np = mask.numpy()
        expected_mask = np.array([False, True, False, False, False, True])
        np.testing.assert_array_equal(mask_np, expected_mask)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_joints_without_subset_unchanged(self, device: str):
        """Test that find_joints without subset still works correctly (no regression)."""
        joint_names = ["joint_0", "joint_1", "joint_2", "joint_3"]

        mask, names, indices = find_joints(joint_names, ["joint_1", "joint_3"], device=device)

        assert names == ["joint_1", "joint_3"]
        assert indices == [1, 3]

        mask_np = mask.numpy()
        expected_mask = np.array([False, True, False, True])
        np.testing.assert_array_equal(mask_np, expected_mask)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_joints_subset_with_single_match(self, device: str):
        """Test find_joints with subset when only one joint matches."""
        joint_names = ["joint_a", "joint_b", "joint_c", "joint_d"]
        joint_subset = ["joint_b", "joint_d"]

        mask, names, indices = find_joints(joint_names, "joint_d", joint_subset=joint_subset, device=device)

        assert names == ["joint_d"]
        assert indices == [3]  # Global index of joint_d

        mask_np = mask.numpy()
        expected_mask = np.array([False, False, False, True])
        np.testing.assert_array_equal(mask_np, expected_mask)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_find_joints_subset_order_differs_from_global(self, device: str):
        """Test find_joints when subset order differs from global order."""
        joint_names = ["a", "b", "c", "d", "e"]
        # Subset in different order than they appear in joint_names
        joint_subset = ["e", "c", "a"]

        mask, names, indices = find_joints(joint_names, ".*", joint_subset=joint_subset, device=device)

        # Names should be in subset order (default preserve_order=False means subset order)
        assert names == ["e", "c", "a"]
        # Indices should be global indices
        assert indices == [4, 2, 0]

        mask_np = mask.numpy()
        expected_mask = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(mask_np, expected_mask)
