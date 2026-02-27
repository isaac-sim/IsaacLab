# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
import pytest
from hydra import compose, initialize

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import process_hydra_config, register_task_to_hydra
from isaaclab_tasks.utils.render_config_store import NEWTON_WARP_AVAILABLE


def hydra_task_config_test(task_name: str, agent_cfg_entry_point: str) -> Callable:
    """Uses compose() instead of hydra.main (single entry point). Reuses process_hydra_config."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            env_cfg, agent_cfg = register_task_to_hydra(task_name, agent_cfg_entry_point)

            with initialize(config_path=None, version_base="1.3"):
                hydra_env_cfg = compose(config_name=task_name, overrides=sys.argv[1:])
                env_cfg, agent_cfg = process_hydra_config(hydra_env_cfg, env_cfg, agent_cfg)
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


def test_render_config_default():
    """Test that render config defaults to isaac_rtx when no override is passed."""

    sys.argv = [sys.argv[0]]

    @hydra_task_config_test("Isaac-Cartpole-RGB-Camera-Direct-v0", "rl_games_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        assert hasattr(env_cfg, "tiled_camera")
        assert env_cfg.tiled_camera.renderer_cfg.renderer_type == "isaac_rtx"

    main()

    # cleanup
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_render_config_override():
    """Test that render config group override is applied to cameras."""

    sys.argv = [sys.argv[0], "render=isaac_rtx"]

    @hydra_task_config_test("Isaac-Cartpole-RGB-Camera-Direct-v0", "rl_games_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        assert hasattr(env_cfg, "tiled_camera")
        assert env_cfg.tiled_camera.renderer_cfg.renderer_type == "isaac_rtx"

    main()

    # cleanup
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


@pytest.mark.skipif(not NEWTON_WARP_AVAILABLE, reason="isaaclab_newton not installed")
def test_render_config_override_newton_warp():
    """Test that render=newton_warp override is applied to cameras (requires isaaclab_newton)."""

    sys.argv = [sys.argv[0], "render=newton_warp"]

    @hydra_task_config_test("Isaac-Cartpole-RGB-Camera-Direct-v0", "rl_games_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        assert hasattr(env_cfg, "tiled_camera")
        assert env_cfg.tiled_camera.renderer_cfg.renderer_type == "newton_warp"

    main()

    # cleanup
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_render_config_invalid_raises():
    """Test that invalid render config raises an error."""

    sys.argv = [sys.argv[0], "render=invalid_renderer"]

    with pytest.raises(Exception, match="invalid_renderer|Could not find|No match"):
        env_cfg, agent_cfg = register_task_to_hydra("Isaac-Cartpole-RGB-Camera-Direct-v0", "rl_games_cfg_entry_point")
        with initialize(config_path=None, version_base="1.3"):
            compose(config_name="Isaac-Cartpole-RGB-Camera-Direct-v0", overrides=sys.argv[1:])

    # cleanup
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()
