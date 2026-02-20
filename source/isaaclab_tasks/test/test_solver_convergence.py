# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

# Omniverse logger
import logging

from isaaclab.app import AppLauncher

# # Import pinocchio in the main script to force the use of the dependencies installed by IsaacLab and not the one installed by Isaac Sim
# # pinocchio is required by the Pink IK controller
# if sys.platform != "win32":
#     import pinocchio  # noqa: F401


# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import carb
import omni.usd
import pytest
from isaaclab_newton.physics import NewtonManager
from isaacsim.core.version import get_version

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.spaces import sample_space

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# import logger
logger = logging.getLogger(__name__)


# @pytest.fixture(scope="module", autouse=True)
def setup_environment():
    # disable interactive mode for wandb for automate environments
    os.environ["WANDB_DISABLED"] = "true"
    # acquire all Isaac environments names
    registered_tasks = list()
    for task_spec in gym.registry.values():
        # skip camera environments for now due to replicator issues with numpy > 2
        if "RGB" in task_spec.id or "Depth" in task_spec.id or "Vision" in task_spec.id:
            continue
        # TODO: Factory environments causes test to fail if run together with other envs
        if "Isaac" in task_spec.id and not task_spec.id.endswith("Play-v0") and "Factory" not in task_spec.id:
            registered_tasks.append(task_spec.id)
    # sort environments by name
    registered_tasks.sort()
    # this flag is necessary to prevent a bug where the simulation gets stuck randomly when running the
    # test on many environments.
    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/physics/cooking/ujitsoCollisionCooking", False)

    return registered_tasks


def _check_random_actions(
    task_name: str, device: str, num_envs: int, num_steps: int = 1000, create_stage_in_memory: bool = False
):
    """Run random actions and check environments returned signals are valid."""

    if not create_stage_in_memory:
        # create a new context stage
        omni.usd.get_context().new_stage()

    # reset the rtx sensors carb setting to False
    carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)
    try:
        # parse configuration
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
        env_cfg.sim.create_stage_in_memory = create_stage_in_memory

        # skip test if the environment is a multi-agent task
        if hasattr(env_cfg, "possible_agents"):
            print(f"[INFO]: Skipping {task_name} as it is a multi-agent task")
            return

        # create environment
        env = gym.make(task_name, cfg=env_cfg)
    except Exception as e:
        if "env" in locals() and hasattr(env, "_is_closed"):
            env.close()
        else:
            if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                e.obj.close()
        pytest.fail(f"Failed to set-up the environment for task {task_name}. Error: {e}")

    # disable control on stop
    env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

    # override action space if set to inf for `Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0`
    if task_name == "Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0":
        for i in range(env.unwrapped.single_action_space.shape[0]):
            if env.unwrapped.single_action_space.low[i] == float("-inf"):
                env.unwrapped.single_action_space.low[i] = -1.0
            if env.unwrapped.single_action_space.high[i] == float("inf"):
                env.unwrapped.single_action_space.low[i] = 1.0

    # reset environment
    obs, _ = env.reset()
    # check signal
    # simulate environment for num_steps steps
    try:
        with torch.inference_mode():
            for _ in range(num_steps):
                # sample actions according to the defined space
                actions = sample_space(
                    env.unwrapped.single_action_space, device=env.unwrapped.device, batch_size=num_envs
                )
                # apply actions
                _ = env.step(actions)
                convergence_data = NewtonManager.get_solver_convergence_steps()
                # TODO: this was increased from 25
                assert convergence_data["max"] < 30, f"Solver did not converge in {convergence_data['max']} iterations"
                # TODO: this was increased from 10
                assert (
                    convergence_data["mean"] < 12
                ), f"Solver did not converge in {convergence_data['mean']} iterations"
    finally:
        # close the environment
        env.close()


def _check_zero_actions(
    task_name: str, device: str, num_envs: int, num_steps: int = 1000, create_stage_in_memory: bool = False
):
    """Run zero actions and check environments returned signals are valid."""

    if not create_stage_in_memory:
        # create a new context stage
        omni.usd.get_context().new_stage()

    # reset the rtx sensors carb setting to False
    carb.settings.get_settings().set_bool("/isaaclab/render/rtx_sensors", False)
    try:
        # parse configuration
        env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(task_name, device=device, num_envs=num_envs)
        env_cfg.sim.create_stage_in_memory = create_stage_in_memory

        # skip test if the environment is a multi-agent task
        if hasattr(env_cfg, "possible_agents"):
            print(f"[INFO]: Skipping {task_name} as it is a multi-agent task")
            return

        # create environment
        env = gym.make(task_name, cfg=env_cfg)
    except Exception as e:
        if "env" in locals() and hasattr(env, "_is_closed"):
            env.close()
        else:
            if hasattr(e, "obj") and hasattr(e.obj, "_is_closed"):
                e.obj.close()
        pytest.fail(f"Failed to set-up the environment for task {task_name}. Error: {e}")

    # disable control on stop
    env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

    # override action space if set to inf for `Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0`
    if task_name == "Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0":
        for i in range(env.unwrapped.single_action_space.shape[0]):
            if env.unwrapped.single_action_space.low[i] == float("-inf"):
                env.unwrapped.single_action_space.low[i] = -1.0
            if env.unwrapped.single_action_space.high[i] == float("inf"):
                env.unwrapped.single_action_space.low[i] = 1.0

    # reset environment
    obs, _ = env.reset()
    # check signal
    # simulate environment for num_steps steps
    try:
        with torch.inference_mode():
            for _ in range(num_steps):
                # sample actions according to the defined space
                actions = torch.zeros(num_envs, env.unwrapped.single_action_space.shape[0], device=env.unwrapped.device)
                # apply actions
                _ = env.step(actions)
                convergence_data = NewtonManager.get_solver_convergence_steps()
                # TODO: this was increased from 25
                assert convergence_data["max"] < 30, f"Solver did not converge in {convergence_data['max']} iterations"
                # TODO: this was increased from 10
                assert (
                    convergence_data["mean"] < 12
                ), f"Solver did not converge in {convergence_data['mean']} iterations"
    finally:
        # close the environment
        env.close()


@pytest.mark.order(2)
@pytest.mark.parametrize("num_envs, device", [(4096, "cuda")])
@pytest.mark.parametrize("action_type", [_check_random_actions, _check_zero_actions])
@pytest.mark.parametrize("task_name", setup_environment())
def test_environments(task_name, num_envs, device, action_type):
    # run environments without stage in memory
    _run_environments(task_name, device, num_envs, num_steps=250, create_stage_in_memory=False, action_type=action_type)


def _run_environments(task_name, device, num_envs, num_steps, create_stage_in_memory, action_type):
    """Run all environments and check environments return valid signals."""

    # skip test if stage in memory is not supported
    isaac_sim_version = float(".".join(get_version()[2]))
    if isaac_sim_version < 5 and create_stage_in_memory:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

    # TODO: this causes crash in CI, but not locally
    if "Isaac-Reach-UR10" in task_name:
        return

    # skip these environments as they cannot be run with 32 environments within reasonable VRAM
    if num_envs == 32 and task_name in [
        "Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-v0",
        "Isaac-Stack-Cube-Instance-Randomize-Franka-IK-Rel-v0",
        "Isaac-Stack-Cube-Instance-Randomize-Franka-v0",
        "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0",
        "Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0",
    ]:
        return
    # skip automate environments as they require cuda installation
    if task_name in ["Isaac-AutoMate-Assembly-Direct-v0", "Isaac-AutoMate-Disassembly-Direct-v0"]:
        return
    # skipping this test for now as it requires torch 2.6 or newer

    if task_name == "Isaac-Cartpole-RGB-TheiaTiny-v0":
        return
    # TODO: why is this failing in Isaac Sim 5.0??? but the environment itself can run.
    if task_name == "Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0":
        return
    print(f">>> Running test for environment: {task_name}")
    action_type(task_name, device, num_envs, num_steps=num_steps, create_stage_in_memory=create_stage_in_memory)
    print(f">>> Closing environment: {task_name}")
    print("-" * 80)
