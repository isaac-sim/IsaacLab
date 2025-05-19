# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""
import unittest

import torch
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.envs.mdp.actions import ActionTermCfg
from isaaclab.managers import ActionTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from rai.eval_sim.utils.environment import (
    get_action_shape,
    random_actions,
    zero_actions,
)

NUM_ENVS = 1
ACTION_SIZE = 2
NUM_STEPS = 3
DEVICE = "cpu"
EPSILON = 1e-6


class DummyAction(ActionTerm):
    """Bare bones action that just has a dimension"""

    def __init__(self, cfg: DummyActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return self.cfg.action_size

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._actions = actions
        self._processed_actions = actions

    def apply_actions(self):
        pass


@configclass
class DummyActionCfg(ActionTermCfg):
    """Bare bones action configuration that only has a dimension"""

    class_type = DummyAction
    asset_name: str = "terrain"
    action_size: int = ACTION_SIZE


@configclass
class ActionCfg:
    dummy_action = DummyActionCfg()


class ObservationCfg:
    """Bare bones observation configuration"""


@configclass
class TestEnvCfg(ManagerBasedEnvCfg):
    """Bare bones environment with actions and no assets"""

    observations = ObservationCfg()
    scene = InteractiveSceneCfg()
    actions = ActionCfg()
    decimation = 1

    def __post_init__(self):
        """Post initialization."""
        # pass device down from test
        self.sim.device = DEVICE
        self.sim.dt = 0.005
        self.scene.num_envs = NUM_ENVS
        self.scene.env_spacing = 1.5


class TestEnvironmentUtils(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        """Create test environment"""
        self.env = ManagerBasedEnv(TestEnvCfg())

    @classmethod
    def tearDownClass(self) -> None:
        del self.env

    def test_get_action_shape(self):
        """Test the get_action_shape"""
        shape = get_action_shape(self.env)
        shape_N = get_action_shape(self.env, NUM_STEPS)

        # validate shape outputs
        self.assertEqual(shape, (NUM_ENVS, ACTION_SIZE))
        self.assertEqual(shape_N, (NUM_STEPS, NUM_ENVS, ACTION_SIZE))

    def test_zero_actions(self):
        """Test the zero_actions"""
        zeros = zero_actions(self.env)
        zeros_N = zero_actions(self.env, NUM_STEPS)

        # validate shape
        self.assertEqual(zeros.shape, (NUM_ENVS, ACTION_SIZE))
        self.assertEqual(zeros_N.shape, (NUM_STEPS, NUM_ENVS, ACTION_SIZE))
        # validate equal to zero within EPSILON
        for e in zeros:
            self.assertTrue(torch.max(torch.abs(0.0 - e)) <= EPSILON)
        for e in zeros_N:
            self.assertTrue(torch.max(torch.abs(0.0 - e)) <= EPSILON)

    def test_random_actions(self):
        """Test the random_actions"""
        randoms = random_actions(self.env)
        randoms_N = random_actions(self.env, NUM_STEPS)

        # validate shape
        self.assertEqual(randoms.shape, (NUM_ENVS, ACTION_SIZE))
        self.assertEqual(randoms_N.shape, (NUM_STEPS, NUM_ENVS, ACTION_SIZE))
        # check range of random values
        self.assertTrue(torch.max(randoms) <= 1.0)
        self.assertTrue(torch.min(randoms) >= -1.0)
        self.assertTrue(torch.max(randoms_N) <= 1.0)
        self.assertTrue(torch.min(randoms_N) >= -1.0)


if __name__ == "__main__":
    run_tests()
