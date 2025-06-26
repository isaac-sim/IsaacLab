# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import sys

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import functools
from collections.abc import Callable

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from isaaclab.utils import replace_strings_with_slices

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import register_task_to_hydra


def hydra_task_config_test(task_name: str, agent_cfg_entry_point: str) -> Callable:
    """Copied from hydra.py hydra_task_config, since hydra.main requires a single point of entry,
    which will not work with multiple tests. Here, we replace hydra.main with hydra initialize
    and compose."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # register the task to Hydra
            env_cfg, agent_cfg = register_task_to_hydra(task_name, agent_cfg_entry_point)

            # replace hydra.main with initialize and compose
            with initialize(config_path=None, version_base="1.3"):
                hydra_env_cfg = compose(config_name=task_name, overrides=sys.argv[1:])
                # convert to a native dictionary
                hydra_env_cfg = OmegaConf.to_container(hydra_env_cfg, resolve=True)
                # replace string with slices because OmegaConf does not support slices
                hydra_env_cfg = replace_strings_with_slices(hydra_env_cfg)
                # update the configs with the Hydra command line arguments
                env_cfg.from_dict(hydra_env_cfg["env"])
                if isinstance(agent_cfg, dict):
                    agent_cfg = hydra_env_cfg["agent"]
                else:
                    agent_cfg.from_dict(hydra_env_cfg["agent"])
                # call the original function
                func(env_cfg, agent_cfg, *args, **kwargs)

        return wrapper

    return decorator


def test_hydra():
    """Test the hydra configuration system."""

    # set hardcoded command line arguments
    sys.argv = [
        sys.argv[0],
        "env.decimation=42",  # test simple env modification
        "env.events.physics_material.params.asset_cfg.joint_ids='slice(0 ,1, 2)'",  # test slice setting
        "env.scene.robot.init_state.joint_vel={.*: 4.0}",  # test regex setting
        "env.rewards.feet_air_time=null",  # test setting to none
        "agent.max_iterations=3",  # test simple agent modification
    ]

    @hydra_task_config_test("Isaac-Velocity-Flat-H1-v0", "rsl_rl_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        # env
        assert env_cfg.decimation == 42
        assert env_cfg.events.physics_material.params["asset_cfg"].joint_ids == slice(0, 1, 2)
        assert env_cfg.scene.robot.init_state.joint_vel == {".*": 4.0}
        assert env_cfg.rewards.feet_air_time is None
        # agent
        assert agent_cfg.max_iterations == 3

    main()
    # clean up
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_nested_iterable_dict():
    """Test the hydra configuration system when dict is nested in an Iterable."""

    @hydra_task_config_test("Isaac-Lift-Cube-Franka-v0", "rsl_rl_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        # env
        assert env_cfg.scene.ee_frame.target_frames[0].name == "end_effector"
        assert env_cfg.scene.ee_frame.target_frames[0].offset.pos[2] == 0.1034

    main()
    # clean up
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()
