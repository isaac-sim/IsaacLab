# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
simulation_app = AppLauncher(headless=True).app


"""Rest everything follows."""

import shutil
import tempfile
import uuid

import gymnasium as gym
import pytest
import torch

import carb

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.fixture(scope="session", autouse=True)
def setup_carb_settings():
    """Set up carb settings to prevent simulation getting stuck."""
    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test datasets."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


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


@pytest.mark.parametrize("task_name", ["Isaac-Lift-Cube-Franka-v0"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 2])
def test_action_state_recorder_terms(task_name, device, num_envs, temp_dir):
    """Check action state recorder terms."""
    sim_utils.create_new_stage()

    dummy_dataset_filename = f"{uuid.uuid4()}.hdf5"

    # parse configuration
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
    # set recorder configurations for this test
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = temp_dir
    env_cfg.recorders.dataset_filename = dummy_dataset_filename

    # create environment
    env = gym.make(task_name, cfg=env_cfg)

    # reset all environment instances to trigger post-reset recorder callbacks
    env.reset()
    check_initial_state_recorder_term(env)

    # reset only one environment that is not the first one
    env.unwrapped.reset(env_ids=torch.tensor([num_envs - 1], device=env.unwrapped.device))
    check_initial_state_recorder_term(env)

    # close the environment
    env.close()
