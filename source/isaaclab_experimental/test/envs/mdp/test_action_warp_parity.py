# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for Warp-first action term classes.

Tests all 10 experimental action classes: process_actions, apply_actions, reset.

Usage::
    python -m pytest test_action_warp_parity.py -v
"""

from __future__ import annotations

import numpy as np
import torch

import pytest
import warp as wp

wp.init()
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA device required")

from isaaclab_experimental.envs.mdp.actions import (
    AbsBinaryJointPositionAction,
    AbsBinaryJointPositionActionCfg,
    BinaryJointPositionAction,
    BinaryJointPositionActionCfg,
    BinaryJointVelocityAction,
    BinaryJointVelocityActionCfg,
    EMAJointPositionToLimitsAction,
    EMAJointPositionToLimitsActionCfg,
    JointEffortAction,
    JointEffortActionCfg,
    JointPositionAction,
    JointPositionActionCfg,
    JointPositionToLimitsAction,
    JointPositionToLimitsActionCfg,
    JointVelocityAction,
    JointVelocityActionCfg,
    NonHolonomicAction,
    NonHolonomicActionCfg,
    RelativeJointPositionAction,
    RelativeJointPositionActionCfg,
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
# Joint action tests (JointPosition, JointVelocity, JointEffort, Relative)
# ============================================================================


class TestJointActions:
    """Test JointAction subclasses: process, apply, reset."""

    def test_joint_effort_process_apply(self, env, asset, actions_wp):
        cfg = JointEffortActionCfg(asset_name="robot", joint_names=[".*"])
        term = JointEffortAction(cfg, env)

        term.process_actions(actions_wp, action_offset=0)
        term.apply_actions()

        # Processed = raw * scale(1.0) + offset(0.0) = raw
        assert_close(term.processed_actions, actions_wp)
        assert asset.last_effort_target is not None

    def test_joint_position_default_offset(self, env, asset, art_data, actions_wp):
        cfg = JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)
        term = JointPositionAction(cfg, env)

        term.process_actions(actions_wp, action_offset=0)
        term.apply_actions()

        # Processed = raw * 1.0 + default_joint_pos[0]
        defaults = wp.to_torch(art_data.default_joint_pos)[0]
        raw = wp.to_torch(actions_wp)
        expected = raw + defaults.unsqueeze(0)
        assert_close(term.processed_actions, expected)
        assert asset.last_pos_target is not None

    def test_joint_velocity_default_offset(self, env, asset, actions_wp):
        cfg = JointVelocityActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)
        term = JointVelocityAction(cfg, env)

        term.process_actions(actions_wp, action_offset=0)
        term.apply_actions()

        # Default vel is all zeros, so processed = raw
        assert_close(term.processed_actions, actions_wp)
        assert asset.last_vel_target is not None

    def test_relative_joint_position(self, env, asset, art_data, actions_wp):
        cfg = RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_zero_offset=True)
        term = RelativeJointPositionAction(cfg, env)

        term.process_actions(actions_wp, action_offset=0)
        term.apply_actions()

        # Applied = processed(=raw) + current joint_pos
        raw = wp.to_torch(actions_wp)
        current_pos = wp.to_torch(art_data.joint_pos)
        expected = raw + current_pos
        assert_close(asset.last_pos_target, expected)

    def test_joint_action_reset(self, env, asset, actions_wp):
        cfg = JointEffortActionCfg(asset_name="robot", joint_names=[".*"])
        term = JointEffortAction(cfg, env)

        # Process some actions
        term.process_actions(actions_wp, action_offset=0)
        assert wp.to_torch(term.raw_actions).abs().sum() > 0

        # Reset all
        term.reset(mask=None)
        assert_close(term.raw_actions, wp.zeros_like(term.raw_actions))

    def test_joint_action_reset_masked(self, env, asset, actions_wp):
        cfg = JointEffortActionCfg(asset_name="robot", joint_names=[".*"])
        term = JointEffortAction(cfg, env)

        term.process_actions(actions_wp, action_offset=0)
        raw_before = wp.to_torch(term.raw_actions).clone()

        # Reset only first half
        mask_np = [i < NUM_ENVS // 2 for i in range(NUM_ENVS)]
        mask = wp.array(mask_np, dtype=wp.bool, device=DEVICE)
        term.reset(mask=mask)

        raw_after = wp.to_torch(term.raw_actions)
        # First half zeroed
        assert_close(raw_after[: NUM_ENVS // 2], torch.zeros(NUM_ENVS // 2, NUM_JOINTS, device=DEVICE))
        # Second half unchanged
        assert_close(raw_after[NUM_ENVS // 2 :], raw_before[NUM_ENVS // 2 :])

    def test_joint_action_with_scale(self, env, asset, actions_wp):
        cfg = JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=2.5)
        term = JointEffortAction(cfg, env)

        term.process_actions(actions_wp, action_offset=0)

        raw = wp.to_torch(actions_wp)
        expected = raw * 2.5
        assert_close(term.processed_actions, expected)


# ============================================================================
# Binary joint action tests
# ============================================================================


class TestBinaryJointActions:
    """Test BinaryJointAction subclasses."""

    def _make_binary_cfg(self, cls):
        return cls(
            asset_name="robot",
            joint_names=[".*"],
            open_command_expr={f"joint_{i}": 0.04 for i in range(NUM_JOINTS)},
            close_command_expr={f"joint_{i}": 0.0 for i in range(NUM_JOINTS)},
        )

    def test_binary_position_open(self, env, asset):
        cfg = self._make_binary_cfg(BinaryJointPositionActionCfg)
        term = BinaryJointPositionAction(cfg, env)

        # Positive action → open
        actions = wp.array(np.full((NUM_ENVS, NUM_JOINTS + 10), 1.0, dtype=np.float32), device=DEVICE)
        term.process_actions(actions, action_offset=0)
        term.apply_actions()

        processed = wp.to_torch(term.processed_actions)
        expected_open = torch.full((NUM_ENVS, NUM_JOINTS), 0.04, device=DEVICE)
        assert_close(processed, expected_open)

    def test_binary_position_close(self, env, asset):
        cfg = self._make_binary_cfg(BinaryJointPositionActionCfg)
        term = BinaryJointPositionAction(cfg, env)

        # Negative action → close
        actions = wp.array(np.full((NUM_ENVS, NUM_JOINTS + 10), -1.0, dtype=np.float32), device=DEVICE)
        term.process_actions(actions, action_offset=0)

        processed = wp.to_torch(term.processed_actions)
        expected_close = torch.zeros(NUM_ENVS, NUM_JOINTS, device=DEVICE)
        assert_close(processed, expected_close)

    def test_binary_velocity(self, env, asset):
        cfg = self._make_binary_cfg(BinaryJointVelocityActionCfg)
        term = BinaryJointVelocityAction(cfg, env)

        actions = wp.array(np.full((NUM_ENVS, 20), 1.0, dtype=np.float32), device=DEVICE)
        term.process_actions(actions, action_offset=0)
        term.apply_actions()
        assert asset.last_vel_target is not None

    def test_abs_binary_threshold(self, env, asset):
        cfg = AbsBinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            open_command_expr={f"joint_{i}": 0.04 for i in range(NUM_JOINTS)},
            close_command_expr={f"joint_{i}": 0.0 for i in range(NUM_JOINTS)},
            threshold=0.5,
            positive_threshold=True,
        )
        term = AbsBinaryJointPositionAction(cfg, env)

        # Action > threshold → open
        actions = wp.array(np.full((NUM_ENVS, 20), 0.8, dtype=np.float32), device=DEVICE)
        term.process_actions(actions, action_offset=0)
        processed = wp.to_torch(term.processed_actions)
        assert_close(processed, torch.full((NUM_ENVS, NUM_JOINTS), 0.04, device=DEVICE))

        # Action < threshold → close
        actions2 = wp.array(np.full((NUM_ENVS, 20), 0.2, dtype=np.float32), device=DEVICE)
        term.process_actions(actions2, action_offset=0)
        processed2 = wp.to_torch(term.processed_actions)
        assert_close(processed2, torch.zeros(NUM_ENVS, NUM_JOINTS, device=DEVICE))


# ============================================================================
# Joint position to limits tests
# ============================================================================


class TestJointPositionToLimitsActions:
    """Test JointPositionToLimitsAction and EMA variant."""

    def test_rescale_to_limits(self, env, asset):
        cfg = JointPositionToLimitsActionCfg(asset_name="robot", joint_names=[".*"], rescale_to_limits=True, scale=1.0)
        term = JointPositionToLimitsAction(cfg, env)

        # Input +1.0 → should map to upper limit (3.14)
        actions = wp.array(np.full((NUM_ENVS, NUM_JOINTS), 1.0, dtype=np.float32), device=DEVICE)
        term.process_actions(actions, action_offset=0)

        processed = wp.to_torch(term.processed_actions)
        expected = torch.full((NUM_ENVS, NUM_JOINTS), 3.14, device=DEVICE)
        assert_close(processed, expected)

    def test_rescale_negative_one(self, env, asset):
        cfg = JointPositionToLimitsActionCfg(asset_name="robot", joint_names=[".*"], rescale_to_limits=True, scale=1.0)
        term = JointPositionToLimitsAction(cfg, env)

        # Input -1.0 → should map to lower limit (-3.14)
        actions = wp.array(np.full((NUM_ENVS, NUM_JOINTS), -1.0, dtype=np.float32), device=DEVICE)
        term.process_actions(actions, action_offset=0)

        processed = wp.to_torch(term.processed_actions)
        expected = torch.full((NUM_ENVS, NUM_JOINTS), -3.14, device=DEVICE)
        assert_close(processed, expected)

    def test_rescale_zero(self, env, asset):
        cfg = JointPositionToLimitsActionCfg(asset_name="robot", joint_names=[".*"], rescale_to_limits=True, scale=1.0)
        term = JointPositionToLimitsAction(cfg, env)

        # Input 0.0 → should map to midpoint (0.0 for symmetric limits)
        actions = wp.array(np.zeros((NUM_ENVS, NUM_JOINTS), dtype=np.float32), device=DEVICE)
        term.process_actions(actions, action_offset=0)

        processed = wp.to_torch(term.processed_actions)
        expected = torch.zeros(NUM_ENVS, NUM_JOINTS, device=DEVICE)
        assert_close(processed, expected)

    def test_ema_alpha_one(self, env, asset):
        """alpha=1.0 means no smoothing — should behave like parent."""
        cfg = EMAJointPositionToLimitsActionCfg(
            asset_name="robot", joint_names=[".*"], rescale_to_limits=True, scale=1.0, alpha=1.0
        )
        term = EMAJointPositionToLimitsAction(cfg, env)
        term.reset(mask=None)

        actions = wp.array(np.full((NUM_ENVS, NUM_JOINTS), 0.5, dtype=np.float32), device=DEVICE)
        term.process_actions(actions, action_offset=0)

        # With alpha=1.0, EMA = 1.0 * processed + 0.0 * prev = processed
        # 0.5 rescaled: (0.5+1)/2 * 6.28 + (-3.14) = 0.75*6.28 - 3.14 = 4.71 - 3.14 = 1.57
        processed = wp.to_torch(term.processed_actions)
        expected = torch.full((NUM_ENVS, NUM_JOINTS), 1.57, device=DEVICE)
        assert_close(processed, expected, atol=0.01, rtol=0.01)

    def test_ema_reset_to_current_pos(self, env, asset, art_data):
        """After reset, prev_applied should match current joint positions."""
        cfg = EMAJointPositionToLimitsActionCfg(
            asset_name="robot", joint_names=[".*"], rescale_to_limits=True, alpha=0.5
        )
        term = EMAJointPositionToLimitsAction(cfg, env)
        term.reset(mask=None)

        prev = wp.to_torch(term._prev_applied_actions)
        current_pos = wp.to_torch(art_data.joint_pos)
        assert_close(prev, current_pos)


# ============================================================================
# Non-holonomic action tests
# ============================================================================


class TestNonHolonomicAction:
    """Test NonHolonomicAction."""

    def test_identity_orientation(self, env, asset):
        """With identity quaternion (yaw=0), forward velocity maps to x only."""
        cfg = NonHolonomicActionCfg(
            asset_name="robot",
            body_name="base",
            x_joint_name="joint_0",
            y_joint_name="joint_1",
            yaw_joint_name="joint_2",
            scale=(1.0, 1.0),
            offset=(0.0, 0.0),
        )
        term = NonHolonomicAction(cfg, env)

        # Forward velocity = 1.0, yaw rate = 0.0
        actions = wp.zeros((NUM_ENVS, NUM_JOINTS), dtype=wp.float32, device=DEVICE)
        # Set first 2 columns (action_dim=2) to [1.0, 0.0]
        act_np = np.zeros((NUM_ENVS, NUM_JOINTS), dtype=np.float32)
        act_np[:, 0] = 1.0  # forward vel
        act_np[:, 1] = 0.0  # yaw rate
        actions = wp.array(act_np, device=DEVICE)

        term.process_actions(actions, action_offset=0)
        term.apply_actions()

        vel_cmd = wp.to_torch(term._joint_vel_command)
        # With identity quat (yaw=0): vx = cos(0)*1.0 = 1.0, vy = sin(0)*1.0 = 0.0, omega = 0.0
        expected = torch.zeros(NUM_ENVS, 3, device=DEVICE)
        expected[:, 0] = 1.0
        assert_close(vel_cmd, expected, atol=1e-4, rtol=1e-4)

    def test_pure_yaw(self, env, asset):
        """Pure yaw rate input."""
        cfg = NonHolonomicActionCfg(
            asset_name="robot",
            body_name="base",
            x_joint_name="joint_0",
            y_joint_name="joint_1",
            yaw_joint_name="joint_2",
        )
        term = NonHolonomicAction(cfg, env)

        act_np = np.zeros((NUM_ENVS, NUM_JOINTS), dtype=np.float32)
        act_np[:, 0] = 0.0  # no forward vel
        act_np[:, 1] = 0.5  # yaw rate
        actions = wp.array(act_np, device=DEVICE)

        term.process_actions(actions, action_offset=0)
        term.apply_actions()

        vel_cmd = wp.to_torch(term._joint_vel_command)
        # vx = vy = 0 (no forward vel), omega = 0.5
        expected = torch.zeros(NUM_ENVS, 3, device=DEVICE)
        expected[:, 2] = 0.5
        assert_close(vel_cmd, expected, atol=1e-4, rtol=1e-4)

    def test_reset(self, env, asset):
        cfg = NonHolonomicActionCfg(
            asset_name="robot",
            body_name="base",
            x_joint_name="joint_0",
            y_joint_name="joint_1",
            yaw_joint_name="joint_2",
        )
        term = NonHolonomicAction(cfg, env)

        act_np = np.ones((NUM_ENVS, NUM_JOINTS), dtype=np.float32)
        term.process_actions(wp.array(act_np, device=DEVICE), action_offset=0)
        assert wp.to_torch(term.raw_actions).abs().sum() > 0

        term.reset(mask=None)
        assert_close(term.raw_actions, wp.zeros((NUM_ENVS, 2), dtype=wp.float32, device=DEVICE))
