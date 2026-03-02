# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the governance wrapper module.

These tests verify governance policy enforcement, provenance logging,
and hash chain integrity without requiring Isaac Sim or GPU access.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from isaaclab_contrib.governance import GovernancePolicy, GovernedEnvWrapper, ProvenanceLogger


class TestGovernancePolicy(unittest.TestCase):
    """Tests for GovernancePolicy validation."""

    def test_default_policy(self):
        """Default policy should be permissive (no constraints)."""
        policy = GovernancePolicy()
        self.assertIsNone(policy.action_bounds)
        self.assertIsNone(policy.max_rate_of_change)
        self.assertTrue(policy.fail_closed)

    def test_valid_bounds(self):
        policy = GovernancePolicy(action_bounds=(-1.0, 1.0))
        self.assertEqual(policy.action_bounds, (-1.0, 1.0))

    def test_invalid_bounds_raises(self):
        with self.assertRaises(ValueError):
            GovernancePolicy(action_bounds=(1.0, -1.0))

    def test_invalid_rate_raises(self):
        with self.assertRaises(ValueError):
            GovernancePolicy(max_rate_of_change=-0.5)

    def test_gateway_requires_url(self):
        with self.assertRaises(ValueError):
            GovernancePolicy(require_external_verification=True)

    def test_gateway_with_url(self):
        policy = GovernancePolicy(
            require_external_verification=True,
            gateway_url="https://governance.example.com",
        )
        self.assertEqual(policy.gateway_url, "https://governance.example.com")


class TestProvenanceLogger(unittest.TestCase):
    """Tests for hash-chained provenance logging."""

    def test_empty_chain_is_valid(self):
        logger = ProvenanceLogger()
        is_valid, last_epoch = logger.verify_chain()
        self.assertTrue(is_valid)
        self.assertEqual(last_epoch, -1)

    def test_single_record(self):
        logger = ProvenanceLogger()
        record = logger.log(
            step=0,
            agent_id="agent-0",
            decision="ALLOW",
            proposed_action=[0.5, 0.3],
            applied_action=[0.5, 0.3],
            violations=[],
        )
        self.assertEqual(record["decision"], "ALLOW")
        self.assertEqual(record["epoch"], 0)
        self.assertEqual(record["hash_prev"], "0" * 64)
        self.assertNotEqual(record["hash_curr"], "0" * 64)

    def test_chain_integrity(self):
        logger = ProvenanceLogger()
        for i in range(10):
            logger.log(
                step=i,
                agent_id="agent-0",
                decision="ALLOW" if i % 3 == 0 else "CLAMP",
                proposed_action=[float(i)],
                applied_action=[float(i)],
                violations=[] if i % 3 == 0 else ["action_bounds"],
            )
        is_valid, last_epoch = logger.verify_chain()
        self.assertTrue(is_valid)
        self.assertEqual(last_epoch, 9)

    def test_tamper_detection(self):
        logger = ProvenanceLogger()
        for i in range(5):
            logger.log(
                step=i, agent_id="0", decision="ALLOW",
                proposed_action=[0.0], applied_action=[0.0], violations=[],
            )
        # Tamper with a record
        logger.records[2]["decision"] = "DENY"
        is_valid, first_bad = logger.verify_chain()
        self.assertFalse(is_valid)
        self.assertEqual(first_bad, 2)

    def test_persistent_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "provenance.jsonl"
            logger = ProvenanceLogger(log_path=log_path)
            logger.log(
                step=0, agent_id="0", decision="ALLOW",
                proposed_action=[1.0], applied_action=[1.0], violations=[],
            )
            logger.log(
                step=1, agent_id="0", decision="CLAMP",
                proposed_action=[2.0], applied_action=[1.0],
                violations=["action_bounds"],
            )
            # Verify file was written
            lines = log_path.read_text().strip().split("\n")
            self.assertEqual(len(lines), 2)
            record = json.loads(lines[0])
            self.assertEqual(record["decision"], "ALLOW")

    def test_summary(self):
        logger = ProvenanceLogger()
        logger.log(step=0, agent_id="0", decision="ALLOW",
                   proposed_action=[0.0], applied_action=[0.0], violations=[])
        logger.log(step=1, agent_id="0", decision="CLAMP",
                   proposed_action=[0.0], applied_action=[0.0], violations=["bounds"])
        logger.log(step=2, agent_id="0", decision="DENY",
                   proposed_action=[0.0], applied_action=[0.0], violations=["spatial"])
        summary = logger.summary()
        self.assertEqual(summary, {"ALLOW": 1, "CLAMP": 1, "DENY": 1})


class MockEnv:
    """Minimal mock environment for testing the wrapper."""

    def __init__(self, obs_shape=(4,)):
        self.obs_shape = obs_shape
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        return np.zeros(self.obs_shape), {}

    def step(self, actions):
        self.step_count += 1
        obs = np.random.randn(*self.obs_shape)
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


class TestGovernedEnvWrapper(unittest.TestCase):
    """Tests for the governance environment wrapper."""

    def test_allow_passthrough(self):
        """Actions within bounds should pass through unchanged."""
        env = MockEnv()
        policy = GovernancePolicy(action_bounds=(-1.0, 1.0))
        governed = GovernedEnvWrapper(env, governance_policy=policy)
        governed.reset()
        actions = np.array([0.5, -0.3, 0.0, 0.8])
        obs, reward, terminated, truncated, info = governed.step(actions)
        self.assertEqual(info["governance"]["decision"], "ALLOW")
        self.assertEqual(info["governance"]["violations"], [])

    def test_action_clamping(self):
        """Actions exceeding bounds should be clamped."""
        env = MockEnv()
        policy = GovernancePolicy(action_bounds=(-1.0, 1.0))
        governed = GovernedEnvWrapper(env, governance_policy=policy)
        governed.reset()
        actions = np.array([2.0, -3.0, 0.5, 1.5])
        obs, reward, terminated, truncated, info = governed.step(actions)
        self.assertEqual(info["governance"]["decision"], "CLAMP")
        self.assertIn("action_bounds", info["governance"]["violations"])

    def test_rate_limiting(self):
        """Large action changes should be rate-limited."""
        env = MockEnv()
        policy = GovernancePolicy(max_rate_of_change=0.1)
        governed = GovernedEnvWrapper(env, governance_policy=policy)
        governed.reset()
        # First step: no rate limit (no previous action)
        governed.step(np.array([0.0, 0.0, 0.0, 0.0]))
        # Second step: large jump should be clamped
        obs, reward, terminated, truncated, info = governed.step(
            np.array([1.0, 1.0, 1.0, 1.0])
        )
        self.assertEqual(info["governance"]["decision"], "CLAMP")
        self.assertIn("rate_of_change", info["governance"]["violations"])

    def test_provenance_chain_after_episode(self):
        """Provenance chain should be intact after a full episode."""
        env = MockEnv()
        policy = GovernancePolicy(action_bounds=(-1.0, 1.0))
        governed = GovernedEnvWrapper(env, governance_policy=policy)
        governed.reset()
        for _ in range(50):
            actions = np.random.uniform(-2.0, 2.0, size=(4,))
            governed.step(actions)
        report = governed.governance_report()
        self.assertTrue(report["chain_intact"])
        self.assertEqual(report["chain_length"], 50)
        self.assertEqual(report["total_steps"], 50)

    def test_governance_report_structure(self):
        env = MockEnv()
        policy = GovernancePolicy(action_bounds=(-1.0, 1.0))
        governed = GovernedEnvWrapper(env, governance_policy=policy)
        governed.reset()
        governed.step(np.array([0.5, 0.5, 0.5, 0.5]))
        report = governed.governance_report()
        self.assertIn("total_steps", report)
        self.assertIn("decisions", report)
        self.assertIn("chain_intact", report)
        self.assertIn("policy", report)

    def test_multi_agent_governance(self):
        """Multi-agent actions (dict) should be governed per-agent."""
        env = MockEnv()
        # Mock the env to accept dict actions
        env.step = MagicMock(return_value=(
            np.zeros(4), 1.0, False, False, {}
        ))
        policy = GovernancePolicy(action_bounds=(-1.0, 1.0))
        governed = GovernedEnvWrapper(env, governance_policy=policy)
        governed.reset()
        actions = {
            "robot_0": np.array([0.5, 0.3]),
            "robot_1": np.array([2.0, -3.0]),  # will be clamped
        }
        obs, reward, terminated, truncated, info = governed.step(actions)
        self.assertEqual(info["governance"]["robot_0"]["decision"], "ALLOW")
        self.assertEqual(info["governance"]["robot_1"]["decision"], "CLAMP")

    def test_delegation_to_wrapped_env(self):
        """Attributes not on wrapper should delegate to wrapped env."""
        env = MockEnv()
        env.custom_attr = "test_value"
        governed = GovernedEnvWrapper(env, governance_policy=GovernancePolicy())
        self.assertEqual(governed.custom_attr, "test_value")

    def test_no_constraints_all_allow(self):
        """With no constraints, all actions should be ALLOW."""
        env = MockEnv()
        policy = GovernancePolicy()  # no constraints
        governed = GovernedEnvWrapper(env, governance_policy=policy)
        governed.reset()
        for _ in range(20):
            actions = np.random.uniform(-100, 100, size=(4,))
            obs, reward, terminated, truncated, info = governed.step(actions)
            self.assertEqual(info["governance"]["decision"], "ALLOW")


if __name__ == "__main__":
    unittest.main()
