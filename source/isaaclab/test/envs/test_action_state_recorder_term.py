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

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.test.env_cfgs import make_empty_manager_based_env_cfg
from isaaclab.test.integration_scene_cfgs import ArticulationRigidObjectSceneCfg
from isaaclab.test.utils import DeviceScope, test_devices

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


def check_initial_state_recorder_term(env: ManagerBasedEnv):
    """Check values recorded by the initial state recorder terms.

    Args:
        env: Environment instance.
    """
    current_state = env.scene.get_state(is_relative=True)
    for env_id in range(env.num_envs):
        recorded_initial_state = env.recorder_manager.get_episode(env_id).get_initial_state()
        are_states_equal, output_log = compare_states(recorded_initial_state, current_state, env_id)
        assert are_states_equal, output_log


# Two environments, so that resetting the last one is a partial reset; recorder bookkeeping is device independent.
@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_action_state_recorder_terms(device, temp_dir):
    """Check action state recorder terms."""
    num_envs = 2
    sim_utils.create_new_stage()

    dummy_dataset_filename = f"{uuid.uuid4()}.hdf5"

    # create a core-only environment with articulation and rigid-object state
    env_cfg = make_empty_manager_based_env_cfg(device=device, num_envs=num_envs)
    env_cfg.scene = ArticulationRigidObjectSceneCfg(num_envs=num_envs, env_spacing=2.5)
    # set recorder configurations for this test
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = temp_dir
    env_cfg.recorders.dataset_filename = dummy_dataset_filename

    # create environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # reset all environment instances to trigger post-reset recorder callbacks
    env.reset()
    check_initial_state_recorder_term(env)
    initial_state = env.scene.get_state(is_relative=True)

    # move the object in the first environment only, so its state differs from its recorded initial state
    # and from the state of the environment that is reset next
    rigid_object = env.scene["object"]
    root_pose = rigid_object.data.root_pose_w.torch.clone()
    root_pose[0, 2] += 0.5
    rigid_object.write_root_pose_to_sim_index(root_pose=root_pose)
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    moved_state = env.scene.get_state(is_relative=True)
    assert not compare_states(env.recorder_manager.get_episode(0).get_initial_state(), moved_state, 0)[0]

    # reset only one environment that is not the first one: only it records a new initial state
    env.reset(env_ids=torch.tensor([num_envs - 1], device=env.device))
    current_state = env.scene.get_state(is_relative=True)
    are_states_equal, output_log = compare_states(
        env.recorder_manager.get_episode(num_envs - 1).get_initial_state(), current_state, num_envs - 1
    )
    assert are_states_equal, output_log
    are_states_equal, output_log = compare_states(
        env.recorder_manager.get_episode(0).get_initial_state(), initial_state, 0
    )
    assert are_states_equal, output_log

    # close the environment
    env.close()
