# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""


import argparse
import os

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--framework",
    type=str,
    default="torch",
    choices=["torch", "jax.jax", "jax.numpy"],
    help="Deep learning framework to use.",
)
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything follows."""


import gym

if args_cli.framework.startswith("torch"):
    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model, shared_model
elif args_cli.framework.startswith("jax"):
    import jax  # fmt:skip
    jax.Device = jax.xla.Device  # for Isaac Sim 2022.2.1 or earlier

    from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.utils.model_instantiators.jax import deterministic_model, gaussian_model

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import get_checkpoint_path, parse_env_cfg

if args_cli.framework == "torch":
    from omni.isaac.orbit_envs.utils.wrappers.skrl import SkrlTorchVecEnvWrapper as SkrlVecEnvWrapper
elif args_cli.framework.startswith("jax"):
    from omni.isaac.orbit_envs.utils.wrappers.skrl import SkrlJaxVecEnvWrapper as SkrlVecEnvWrapper

from config import convert_skrl_cfg, parse_skrl_cfg


def main():
    """Play with skrl agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    experiment_cfg = parse_skrl_cfg(args_cli.task)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)
    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env)  # same as: `wrap_env(env, wrapper="isaac-orbit")`

    # instantiate models using skrl model instantiator utility
    # https://skrl.readthedocs.io/en/latest/modules/skrl.utils.model_instantiators.html
    models = {}
    # force separated models for jax
    if args_cli.framework.startswith("jax"):
        experiment_cfg["models"]["separate"] = True
    # non-shared models
    if experiment_cfg["models"]["separate"]:
        models["policy"] = gaussian_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **convert_skrl_cfg(experiment_cfg["models"]["policy"], args_cli.framework),
        )
        models["value"] = deterministic_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **convert_skrl_cfg(experiment_cfg["models"]["value"], args_cli.framework),
        )
    # shared models
    else:
        models["policy"] = shared_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            structure=None,
            roles=["policy", "value"],
            parameters=[
                convert_skrl_cfg(experiment_cfg["models"]["policy"], args_cli.framework),
                convert_skrl_cfg(experiment_cfg["models"]["value"], args_cli.framework),
            ],
        )
        models["value"] = models["policy"]
    # initialize jax models' state dict
    if args_cli.framework.startswith("jax"):
        for role, model in models.items():
            model.init_state_dict(role)

    # configure and instantiate PPO agent
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    agent_cfg.update(convert_skrl_cfg(experiment_cfg["agent"], args_cli.framework))

    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})
    agent_cfg["experiment"]["write_interval"] = 0  # don't log to Tensorboard
    agent_cfg["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints

    agent = PPO(
        models=models,
        memory=None,  # memory is optional during evaluation
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, os.path.join("*", "checkpoints"), None)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # initialize agent
    agent.init()
    agent.load(resume_path)
    # set agent to evaluation mode
    agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    # simulate environment
    while simulation_app.is_running():
        # agent stepping
        actions = agent.act(obs, timestep=0, timesteps=0)[0]
        # env stepping
        obs, _, _, _, _ = env.step(actions)
        # check if simulator is stopped
        if env.sim.is_stopped():
            break

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
