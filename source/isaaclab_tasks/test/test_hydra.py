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
from hydra import compose, initialize
import pytest

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import process_hydra_config, register_task_to_hydra


def hydra_task_config_test(task_name: str, agent_cfg_entry_point: str) -> Callable:
    """Mirrors hydra_task_config: register task, compose, process_hydra_config, then run test."""

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
    """No override: default render config (isaac_rtx) is applied."""
    sys.argv = [sys.argv[0]]

    @hydra_task_config_test("Isaac-Cartpole-RGB-Camera-Direct-v0", None)
    def main(env_cfg, agent_cfg):
        assert env_cfg.tiled_camera.renderer_cfg is not None
        assert env_cfg.tiled_camera.renderer_cfg.renderer_type == "isaac_rtx"

    main()
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_render_config_override():
    """Override render=isaac_rtx is applied to cameras."""
    sys.argv = [sys.argv[0], "render=isaac_rtx"]

    @hydra_task_config_test("Isaac-Cartpole-RGB-Camera-Direct-v0", None)
    def main(env_cfg, agent_cfg):
        assert env_cfg.tiled_camera.renderer_cfg is not None
        assert env_cfg.tiled_camera.renderer_cfg.renderer_type == "isaac_rtx"

    main()
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_render_config_override_newton_warp():
    """Override render=newton_warp sets renderer_type on cameras. Skip if Newton not installed."""
    try:
        from isaaclab_newton.renderers import NewtonWarpRendererCfg  # noqa: F401
    except ImportError:
        pytest.skip("Newton Warp renderer not installed")
    sys.argv = [sys.argv[0], "render=newton_warp"]

    @hydra_task_config_test("Isaac-Cartpole-RGB-Camera-Direct-v0", None)
    def main(env_cfg, agent_cfg):
        assert env_cfg.tiled_camera.renderer_cfg is not None
        assert env_cfg.tiled_camera.renderer_cfg.renderer_type == "newton_warp"

    main()
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_render_config_invalid_raises():
    """Invalid render= choice raises."""
    sys.argv = [sys.argv[0], "render=invalid_renderer"]
    register_task_to_hydra("Isaac-Cartpole-RGB-Camera-Direct-v0", None)

    with pytest.raises(Exception):  # Hydra/OmegaConf error for unknown config group choice
        with initialize(config_path=None, version_base="1.3"):
            compose(config_name="Isaac-Cartpole-RGB-Camera-Direct-v0", overrides=sys.argv[1:])
    sys.argv = [sys.argv[0]]
    hydra.core.global_hydra.GlobalHydra.instance().clear()
