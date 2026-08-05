# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for the action-state recorder manager using a minimal Cartpole environment."""

from isaaclab.app import AppLauncher

# launch the simulator
simulation_app = AppLauncher(headless=True).app


"""Rest everything follows."""

import math
import shutil
import tempfile
import uuid

import pytest
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
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


@pytest.fixture(scope="session", autouse=True)
def setup_carb_settings():
    """Set up settings to prevent simulation getting stuck."""
    get_settings_manager().set_bool("/physics/cooking/ujitsoCollisionCooking", False)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test datasets."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@configclass
class _SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


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


def compare_states(compared_state, ground_truth_state, ground_truth_env_id) -> tuple[bool, str]:
    """Compare a state with the given ground_truth.

    Args:
        compared_state: State to be compared.
        ground_truth_state: Ground truth state.
        ground_truth_env_id: Index of the environment in the ground_truth states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Error log if states don't match.
    """
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in ground_truth_state[asset_type].keys():
            for state_name in ground_truth_state[asset_type][asset_name].keys():
                runtime_asset_state = ground_truth_state[asset_type][asset_name][state_name][ground_truth_env_id]
                dataset_asset_state = compared_state[asset_type][asset_name][state_name][0]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    return False, f"State shape of {state_name} for asset {asset_name} don't match"
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        return (
                            False,
                            f'State ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n',
                        )
    return True, ""


def check_initial_state_recorder_term(env):
    """Check values recorded by the initial state recorder terms.

    Args:
        env: Environment instance.
    """
    current_state = env.unwrapped.scene.get_state(is_relative=True)
    for env_id in range(env.unwrapped.num_envs):
        recorded_initial_state = env.unwrapped.recorder_manager.get_episode(env_id).get_initial_state()
        are_states_equal, output_log = compare_states(recorded_initial_state, current_state, env_id)
        assert are_states_equal, output_log


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 2])
def test_action_state_recorder_terms(device, num_envs, temp_dir):
    """Check action state recorder terms record correct initial state in a minimal Cartpole environment."""
    sim_utils.create_new_stage()

    dummy_dataset_filename = f"{uuid.uuid4()}.hdf5"

    env_cfg = _CartpoleEnvCfg()
    env_cfg.sim.device = device
    env_cfg.scene.num_envs = num_envs
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = temp_dir
    env_cfg.recorders.dataset_filename = dummy_dataset_filename

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # reset all environment instances to trigger post-reset recorder callbacks
    env.reset()
    check_initial_state_recorder_term(env)

    # reset only one environment that is not the first one
    env.unwrapped.reset(env_ids=torch.tensor([num_envs - 1], device=env.unwrapped.device))
    check_initial_state_recorder_term(env)

    env.close()
