# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import shutil
import tempfile
import torch
import unittest
import uuid

import carb
import omni.usd

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class TestActionStateRecorderManagerCfg(unittest.TestCase):
    """Test cases for ActionStateRecorderManagerCfg recorder terms."""

    @classmethod
    def setUpClass(cls):
        # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
        # test on many environments.
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    def setUp(self):
        # create a temporary directory to store the test datasets
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # delete the temporary directory after the test
        shutil.rmtree(self.temp_dir)

    def test_action_state_reocrder_terms(self):
        """Check action state recorder terms."""
        for task_name in ["Isaac-Lift-Cube-Franka-v0"]:
            for device in ["cuda:0", "cpu"]:
                for num_envs in [1, 2]:
                    with self.subTest(task_name=task_name, device=device):
                        omni.usd.get_context().new_stage()

                        dummy_dataset_filename = f"{uuid.uuid4()}.hdf5"

                        # parse configuration
                        env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
                        # set recorder configurations for this test
                        env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
                        env_cfg.recorders.dataset_export_dir_path = self.temp_dir
                        env_cfg.recorders.dataset_filename = dummy_dataset_filename

                        # create environment
                        env = gym.make(task_name, cfg=env_cfg)

                        # reset all environment instances to trigger post-reset recorder callbacks
                        env.reset()
                        self.check_initial_state_recorder_term(env)

                        # reset only one environment that is not the first one
                        env.unwrapped.reset(env_ids=torch.tensor([num_envs - 1], device=env.unwrapped.device))
                        self.check_initial_state_recorder_term(env)

                        # close the environment
                        env.close()

    def check_initial_state_recorder_term(self, env):
        """Check values recorded by the initial state recorder terms.

        Args:
            env: Environment instance.
        """
        current_state = env.unwrapped.scene.get_state(is_relative=True)
        for env_id in range(env.unwrapped.num_envs):
            recorded_initial_state = env.unwrapped.recorder_manager.get_episode(env_id).get_initial_state()
            are_states_equal, output_log = self.compare_states(recorded_initial_state, current_state, env_id)
            self.assertTrue(are_states_equal, msg=output_log)

    def compare_states(self, compared_state, ground_truth_state, ground_truth_env_id) -> (bool, str):
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


if __name__ == "__main__":
    run_tests()
