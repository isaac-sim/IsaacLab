# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for warp-first action MDP terms."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import warp as wp

wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

from isaaclab_experimental.envs.mdp.actions import (
    JointEffortAction,
    JointEffortActionCfg,
    JointPositionAction,
    JointPositionActionCfg,
)
from parity_helpers import MockArticulation, MockArticulationData, MockScene, copy_np_to_wp

NUM_ENVS = 32
NUM_JOINTS = 6
NUM_BODIES = 3
DEVICE = "cuda:0"
ATOL = 1e-5
RTOL = 1e-5
JOINT_NAMES = [f"joint_{i}" for i in range(NUM_JOINTS)]


# ============================================================================
# Mock infrastructure
# ============================================================================


class MockEnv:
    def __init__(self, asset):
        self.scene = MockScene({"robot": asset}, env_origins=None)
        self.num_envs = NUM_ENVS
        self.device = DEVICE


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def art_data():
    data = MockArticulationData(num_envs=NUM_ENVS, num_joints=NUM_JOINTS, num_bodies=NUM_BODIES)
    # Override defaults with specific per-joint values for action tests
    copy_np_to_wp(
        data.default_joint_pos,
        np.tile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], (NUM_ENVS, 1)).astype(np.float32),
    )
    # Body quaternion for NonHolonomicAction (identity = [0,0,0,1] in xyzw)
    quat_np = np.zeros((NUM_ENVS, NUM_BODIES, 4), dtype=np.float32)
    quat_np[:, :, 3] = 1.0
    data.body_quat_w = wp.array(quat_np, dtype=wp.quatf, device=DEVICE)
    data._num_joints = NUM_JOINTS
    return data


@pytest.fixture()
def asset(art_data):
    return MockArticulation(art_data, num_bodies=NUM_BODIES, num_joints=NUM_JOINTS)


@pytest.fixture()
def env(asset):
    return MockEnv(asset)


@pytest.fixture()
def actions_wp():
    rng = np.random.RandomState(99)
    return wp.array(rng.randn(NUM_ENVS, NUM_JOINTS).astype(np.float32), device=DEVICE)


# ============================================================================
# Helpers
# ============================================================================


def assert_close(actual, expected, atol=ATOL, rtol=RTOL):
    if isinstance(actual, wp.array):
        actual = wp.to_torch(actual)
    if isinstance(expected, wp.array):
        expected = wp.to_torch(expected)
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


# ============================================================================
# Joint action tests (JointPosition, JointEffort)
# ============================================================================


class TestJointActions:
    """Test JointAction subclasses: process, apply, reset."""

    def test_joint_effort_process_apply(self, env, asset, actions_wp):
        cfg = JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=2.5)
        term = JointEffortAction(cfg, env)

        term.process_actions(actions_wp, action_offset=0)
        term.apply_actions()

        # Processed = raw * scale(2.5) + offset(0.0), and that buffer is what reaches the asset
        assert_close(term.processed_actions, wp.to_torch(actions_wp) * 2.5)
        assert asset.last_effort_target is term.processed_actions

    def test_position_scale_and_offset(self, env, asset, art_data, actions_wp):
        """scale=2, use_default_offset=True -> processed == raw * 2 + defaults[0], applied as the target.

        Same affine formula as the stable ``JointAction.process_actions``."""
        cfg = JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=2.0, use_default_offset=True)
        term = JointPositionAction(cfg, env)
        term.process_actions(actions_wp, action_offset=0)
        term.apply_actions()

        raw = wp.to_torch(actions_wp)
        defaults = art_data.default_joint_pos.torch[0]
        expected = raw * 2.0 + defaults.unsqueeze(0)
        assert_close(term.processed_actions, expected)
        assert asset.last_pos_target is term.processed_actions

    def test_joint_action_reset(self, env, asset, actions_wp):
        cfg = JointEffortActionCfg(asset_name="robot", joint_names=[".*"])
        term = JointEffortAction(cfg, env)

        # Process some actions
        term.process_actions(actions_wp, action_offset=0)
        assert wp.to_torch(term.raw_actions).abs().sum() > 0

        # Reset all
        term.reset(env_mask=None)
        assert_close(term.raw_actions, wp.zeros_like(term.raw_actions))

    def test_joint_action_reset_masked(self, env, asset, actions_wp):
        cfg = JointEffortActionCfg(asset_name="robot", joint_names=[".*"])
        term = JointEffortAction(cfg, env)

        term.process_actions(actions_wp, action_offset=0)
        raw_before = wp.to_torch(term.raw_actions).clone()

        # Reset only first half
        mask_np = [i < NUM_ENVS // 2 for i in range(NUM_ENVS)]
        mask = wp.array(mask_np, dtype=wp.bool, device=DEVICE)
        term.reset(env_mask=mask)

        raw_after = wp.to_torch(term.raw_actions)
        # First half zeroed
        assert_close(raw_after[: NUM_ENVS // 2], torch.zeros(NUM_ENVS // 2, NUM_JOINTS, device=DEVICE))
        # Second half unchanged
        assert_close(raw_after[NUM_ENVS // 2 :], raw_before[NUM_ENVS // 2 :])
