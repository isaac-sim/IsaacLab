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


class TestSpatialBoundsEnforcement(unittest.TestCase):
    """Tests for spatial bounds checking (hard violation)."""

    @staticmethod
    def _make_position_extractor(positions: dict[str, dict]):
        """Create a position extractor from a static position dict."""
        def extractor(env, agent_id):
            return positions.get(agent_id)
        return extractor

    def test_spatial_bounds_allow(self):
        """Agent inside spatial bounds should be ALLOW."""
        env = MockEnv()
        positions = {"0": {"x": 0.0, "y": 0.0, "z": 1.0}}
        policy = GovernancePolicy(
            spatial_bounds={"x": (-5.0, 5.0), "y": (-5.0, 5.0), "z": (0.0, 3.0)},
        )
        governed = GovernedEnvWrapper(
            env, governance_policy=policy,
            position_extractor=self._make_position_extractor(positions),
        )
        governed.reset()
        _, _, _, _, info = governed.step(np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(info["governance"]["decision"], "ALLOW")

    def test_spatial_bounds_deny(self):
        """Agent outside spatial bounds should be DENY (fail_closed)."""
        env = MockEnv()
        positions = {"0": {"x": 10.0, "y": 0.0, "z": 1.0}}  # x=10 > 5
        policy = GovernancePolicy(
            spatial_bounds={"x": (-5.0, 5.0), "y": (-5.0, 5.0), "z": (0.0, 3.0)},
            fail_closed=True,
        )
        governed = GovernedEnvWrapper(
            env, governance_policy=policy,
            position_extractor=self._make_position_extractor(positions),
        )
        governed.reset()
        _, _, _, _, info = governed.step(np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(info["governance"]["decision"], "DENY")
        self.assertIn("spatial_bounds", info["governance"]["violations"])

    def test_spatial_bounds_no_extractor_warns_and_allows(self):
        """Missing position extractor should warn and skip spatial check."""
        env = MockEnv()
        policy = GovernancePolicy(
            spatial_bounds={"x": (-5.0, 5.0)},
        )
        governed = GovernedEnvWrapper(env, governance_policy=policy)
        governed.reset()
        _, _, _, _, info = governed.step(np.array([0.1, 0.2, 0.3, 0.4]))
        # Without extractor, spatial check is skipped — action passes
        self.assertEqual(info["governance"]["decision"], "ALLOW")


class TestGeofenceEnforcement(unittest.TestCase):
    """Tests for geofence polygon containment (hard violation)."""

    @staticmethod
    def _make_position_extractor(positions: dict[str, dict]):
        def extractor(env, agent_id):
            return positions.get(agent_id)
        return extractor

    def test_inside_geofence_allow(self):
        """Agent inside geofence polygon should be ALLOW."""
        env = MockEnv()
        # Square geofence: (0,0), (10,0), (10,10), (0,10)
        polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        positions = {"0": {"x": 5.0, "y": 5.0, "z": 0.0}}
        policy = GovernancePolicy(geofence_polygons=[polygon])
        governed = GovernedEnvWrapper(
            env, governance_policy=policy,
            position_extractor=self._make_position_extractor(positions),
        )
        governed.reset()
        _, _, _, _, info = governed.step(np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(info["governance"]["decision"], "ALLOW")

    def test_outside_geofence_deny(self):
        """Agent outside all geofence polygons should be DENY."""
        env = MockEnv()
        polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        positions = {"0": {"x": 50.0, "y": 50.0, "z": 0.0}}  # far outside
        policy = GovernancePolicy(geofence_polygons=[polygon], fail_closed=True)
        governed = GovernedEnvWrapper(
            env, governance_policy=policy,
            position_extractor=self._make_position_extractor(positions),
        )
        governed.reset()
        _, _, _, _, info = governed.step(np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(info["governance"]["decision"], "DENY")
        self.assertIn("geofence", info["governance"]["violations"])

    def test_multiple_geofences_any_match(self):
        """Agent inside any one of multiple geofences should be ALLOW."""
        env = MockEnv()
        poly1 = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]
        poly2 = [(100.0, 100.0), (110.0, 100.0), (110.0, 110.0), (100.0, 110.0)]
        positions = {"0": {"x": 105.0, "y": 105.0, "z": 0.0}}  # inside poly2
        policy = GovernancePolicy(geofence_polygons=[poly1, poly2])
        governed = GovernedEnvWrapper(
            env, governance_policy=policy,
            position_extractor=self._make_position_extractor(positions),
        )
        governed.reset()
        _, _, _, _, info = governed.step(np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(info["governance"]["decision"], "ALLOW")


class TestMinSeparationEnforcement(unittest.TestCase):
    """Tests for minimum inter-agent separation (hard violation)."""

    @staticmethod
    def _make_position_extractor(positions: dict[str, dict]):
        def extractor(env, agent_id):
            return positions.get(agent_id)
        return extractor

    def test_agents_far_apart_allow(self):
        """Agents with sufficient separation should be ALLOW."""
        env = MockEnv()
        env.step = MagicMock(return_value=(np.zeros(4), 1.0, False, False, {}))
        positions = {
            "robot_0": {"x": 0.0, "y": 0.0, "z": 0.0},
            "robot_1": {"x": 10.0, "y": 0.0, "z": 0.0},
        }
        policy = GovernancePolicy(min_separation=2.0)
        governed = GovernedEnvWrapper(
            env, governance_policy=policy,
            position_extractor=self._make_position_extractor(positions),
        )
        governed.reset()
        actions = {
            "robot_0": np.array([0.1, 0.2]),
            "robot_1": np.array([0.3, 0.4]),
        }
        _, _, _, _, info = governed.step(actions)
        self.assertEqual(info["governance"]["robot_0"]["decision"], "ALLOW")
        self.assertEqual(info["governance"]["robot_1"]["decision"], "ALLOW")

    def test_agents_too_close_deny(self):
        """Agents violating min_separation should be DENY."""
        env = MockEnv()
        env.step = MagicMock(return_value=(np.zeros(4), 1.0, False, False, {}))
        positions = {
            "robot_0": {"x": 0.0, "y": 0.0, "z": 0.0},
            "robot_1": {"x": 0.5, "y": 0.0, "z": 0.0},  # 0.5m apart < 2.0m
        }
        policy = GovernancePolicy(min_separation=2.0, fail_closed=True)
        governed = GovernedEnvWrapper(
            env, governance_policy=policy,
            position_extractor=self._make_position_extractor(positions),
        )
        governed.reset()
        actions = {
            "robot_0": np.array([0.1, 0.2]),
            "robot_1": np.array([0.3, 0.4]),
        }
        _, _, _, _, info = governed.step(actions)
        # Both agents should get DENY since they're too close
        self.assertEqual(info["governance"]["robot_0"]["decision"], "DENY")
        self.assertIn("min_separation", info["governance"]["robot_0"]["violations"])

    def test_separation_single_agent_no_check(self):
        """Single-agent env should skip separation check."""
        env = MockEnv()
        positions = {"0": {"x": 0.0, "y": 0.0, "z": 0.0}}
        policy = GovernancePolicy(min_separation=2.0)
        governed = GovernedEnvWrapper(
            env, governance_policy=policy,
            position_extractor=self._make_position_extractor(positions),
        )
        governed.reset()
        _, _, _, _, info = governed.step(np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(info["governance"]["decision"], "ALLOW")


class TestConfigurableLogDims(unittest.TestCase):
    """Tests for configurable action logging truncation."""

    def test_default_log_dims(self):
        """Default max_log_dims=64 should truncate long action vectors."""
        env = MockEnv(obs_shape=(4,))
        policy = GovernancePolicy()
        governed = GovernedEnvWrapper(env, governance_policy=policy)
        governed.reset()
        # 100-dim action
        governed.step(np.ones(100))
        record = governed.provenance.records[0]
        self.assertEqual(len(record["proposed_action"]), 64)

    def test_custom_log_dims(self):
        """Custom max_log_dims should control truncation."""
        env = MockEnv(obs_shape=(4,))
        policy = GovernancePolicy()
        governed = GovernedEnvWrapper(env, governance_policy=policy, max_log_dims=10)
        governed.reset()
        governed.step(np.ones(50))
        record = governed.provenance.records[0]
        self.assertEqual(len(record["proposed_action"]), 10)

    def test_zero_log_dims_logs_all(self):
        """max_log_dims=0 should log all dimensions."""
        env = MockEnv(obs_shape=(4,))
        policy = GovernancePolicy()
        governed = GovernedEnvWrapper(env, governance_policy=policy, max_log_dims=0)
        governed.reset()
        governed.step(np.ones(100))
        record = governed.provenance.records[0]
        self.assertEqual(len(record["proposed_action"]), 100)


class TestInfoDictPreservation(unittest.TestCase):
    """Tests that original info dict values are not silently dropped."""

    def test_non_dict_info_preserved(self):
        """Non-dict last element should be preserved under _original key."""
        env = MockEnv()
        env.step = MagicMock(return_value=(np.zeros(4), 1.0, False, False, "custom_info"))
        policy = GovernancePolicy()
        governed = GovernedEnvWrapper(env, governance_policy=policy)
        governed.reset()
        _, _, _, _, info = governed.step(np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(info["_original"], "custom_info")
        self.assertIn("governance", info)

    def test_existing_info_keys_preserved(self):
        """Existing keys in info dict should not be overwritten."""
        env = MockEnv()
        env.step = MagicMock(return_value=(
            np.zeros(4), 1.0, False, False, {"episode": {"reward": 42.0}}
        ))
        policy = GovernancePolicy()
        governed = GovernedEnvWrapper(env, governance_policy=policy)
        governed.reset()
        _, _, _, _, info = governed.step(np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(info["episode"]["reward"], 42.0)
        self.assertIn("governance", info)


if __name__ == "__main__":
    unittest.main()
