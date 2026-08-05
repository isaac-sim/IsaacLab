# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for ManagerBasedRLEnv observation spaces."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import math

import gymnasium as gym
import numpy as np
import pytest
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

pytestmark = pytest.mark.integration


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class _ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)


@configclass
class _EventsCfg:
    reset_cart = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_pole = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


@configclass
class _RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class _TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class _CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """Minimal cartpole env shared across tests in this file."""

    @configclass
    class _ObsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
            joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    scene: _SceneCfg = _SceneCfg(num_envs=1, env_spacing=4.0)
    observations: _ObsCfg = _ObsCfg()
    actions: _ActionsCfg = _ActionsCfg()
    events: _EventsCfg = _EventsCfg()
    rewards: _RewardsCfg = _RewardsCfg()
    terminations: _TerminationsCfg = _TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_non_concatenated_obs_groups_contain_all_terms(device):
    """Test that non-concatenated observation groups contain all defined terms (issue #3133).

    Before the fix, only the last term in each non-concatenated group would be present
    in the observation space Dict. This test ensures all terms are correctly included.
    """
    sim_utils.create_new_stage()

    env_cfg = _CartpoleEnvCfg()
    env_cfg.scene.num_envs = 2
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.sim.device = device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    try:
        assert isinstance(env.observation_space, gym.spaces.Dict)
        policy_space = env.observation_space.spaces["policy"]
        assert isinstance(policy_space, gym.spaces.Dict)

        expected_policy_terms = ["joint_pos_rel", "joint_vel_rel"]

        assert list(policy_space.spaces) == expected_policy_terms
        for term_name in expected_policy_terms:
            assert isinstance(policy_space.spaces[term_name], gym.spaces.Box)

        # Test that observations match the space structure.
        env.reset()
        action = torch.tensor(env.action_space.sample(), device=env.device)
        obs, _, _, _, _ = env.step(action)
        assert list(obs["policy"]) == expected_policy_terms
    finally:
        env.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_obs_space_follows_clip_constraint(device):
    """Ensure observation space bounds reflect the clip constraint on each term.

    Uses a minimal Cartpole env where some obs terms have explicit clip bounds and
    others do not, verifying the obs-space Box bounds match in each case.
    """
    sim_utils.create_new_stage()

    @configclass
    class _MixedClipObsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-1.0, 1.0))
            joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)  # no clip → unbounded

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = False

        policy: PolicyCfg = PolicyCfg()

    env_cfg = _CartpoleEnvCfg()
    env_cfg.scene.num_envs = 2
    env_cfg.observations = _MixedClipObsCfg()
    env_cfg.sim.device = device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    try:
        for group_name, group_space in env.observation_space.spaces.items():
            assert isinstance(group_space, gym.spaces.Dict)
            for term_name, term_space in group_space.spaces.items():
                term_cfg = getattr(getattr(env_cfg.observations, group_name), term_name)
                low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
                high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
                assert isinstance(term_space, gym.spaces.Box), (
                    f"Expected Box space for {term_name} in {group_name}, got {type(term_space)}"
                )
                assert np.all(term_space.low == low)
                assert np.all(term_space.high == high)
    finally:
        env.close()
