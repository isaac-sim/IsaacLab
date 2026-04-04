# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description=(
        "Test Isaac-Cartpole-RGB-Camera-Direct-v0 environment with different resolutions and number of environments."
    )
)
parser.add_argument("--save_images", action="store_true", default=False, help="Save out renders to file.")
parser.add_argument("unittest_args", nargs="*")

# parse the arguments
args_cli = parser.parse_args()
# set the sys.argv to the unittest_args
sys.argv[1:] = args_cli.unittest_args

# launch the simulator
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import sys

import gymnasium as gym
import pytest

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import save_images_to_file

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


@pytest.mark.skip(reason="Currently takes too long to run")
def test_tiled_resolutions_tiny():
    """Define settings for resolution and number of environments"""
    num_envs = 1024
    tile_widths = range(32, 48)
    tile_heights = range(32, 48)
    _launch_tests(tile_widths, tile_heights, num_envs)


@pytest.mark.skip(reason="Currently takes too long to run")
def test_tiled_resolutions_small():
    """Define settings for resolution and number of environments"""
    num_envs = 300
    tile_widths = range(128, 156)
    tile_heights = range(128, 156)
    _launch_tests(tile_widths, tile_heights, num_envs)


@pytest.mark.skip(reason="Currently takes too long to run")
def test_tiled_resolutions_medium():
    """Define settings for resolution and number of environments"""
    num_envs = 64
    tile_widths = range(320, 400, 20)
    tile_heights = range(320, 400, 20)
    _launch_tests(tile_widths, tile_heights, num_envs)


@pytest.mark.skip(reason="Currently takes too long to run")
def test_tiled_resolutions_large():
    """Define settings for resolution and number of environments"""
    num_envs = 4
    tile_widths = range(480, 640, 40)
    tile_heights = range(480, 640, 40)
    _launch_tests(tile_widths, tile_heights, num_envs)


@pytest.mark.skip(reason="Currently takes too long to run")
def test_tiled_resolutions_edge_cases():
    """Define settings for resolution and number of environments"""
    num_envs = 1000
    tile_widths = [12, 67, 93, 147]
    tile_heights = [12, 67, 93, 147]
    _launch_tests(tile_widths, tile_heights, num_envs)


@pytest.mark.skip(reason="Currently takes too long to run")
def test_tiled_num_envs_edge_cases():
    """Define settings for resolution and number of environments"""
    num_envs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 53, 359, 733, 927]
    tile_widths = [67, 93, 147]
    tile_heights = [67, 93, 147]
    for n_envs in num_envs:
        _launch_tests(tile_widths, tile_heights, n_envs)


# Helper functions


def _launch_tests(tile_widths: range, tile_heights: range, num_envs: int):
    """Run through different resolutions for tiled rendering"""
    device = "cuda:0"
    task_name = "Isaac-Cartpole-RGB-Camera-Direct-v0"
    # iterate over all registered environments
    for width in tile_widths:
        for height in tile_heights:
            # create a new stage
            sim_utils.create_new_stage()
            # parse configuration
            env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
            env_cfg.tiled_camera.width = width
            env_cfg.tiled_camera.height = height
            print(f">>> Running test for resolution: {width} x {height}")
            # check environment
            _run_environment(env_cfg)
            # close the environment
            print(f">>> Closing environment: {task_name}")
            print("-" * 80)


def _run_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg):
    """Run environment and capture a rendered image."""
    # create environment
    env: ManagerBasedRLEnv | DirectRLEnv = gym.make("Isaac-Cartpole-RGB-Camera-Direct-v0", cfg=env_cfg)
    # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
    # test on many environments.
    env.sim.set_setting("/physics/cooking/ujitsoCollisionCooking", False)

    # reset environment
    obs, _ = env.reset()
    # save image
    if args_cli.save_images:
        save_images_to_file(
            obs["policy"] + 0.93,
            f"output_{env.num_envs}_{env_cfg.tiled_camera.width}x{env_cfg.tiled_camera.height}.png",
        )

    # close the environment
    env.close()
