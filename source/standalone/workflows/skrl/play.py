"""
Script to play a checkpoint if an RL agent from skrl.
Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in a more user-friendly way
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
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything follows."""


import gym
import torch
from datetime import datetime

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import ManualTrainer
from skrl.utils.model_instantiators import deterministic_model, gaussian_model, shared_model

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg
from omni.isaac.orbit_envs.utils.wrappers.skrl import SkrlVecEnvWrapper

from config import convert_skrl_cfg, parse_skrl_cfg


def main():
    """Play with skrl agent."""

    # parse env configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
    experiment_cfg = parse_skrl_cfg(args_cli.task)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = datetime.now().strftime("%b%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)
    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env)  # same as: `wrap_env(env, wrapper="isaac-orbit")`

    # instantiate models using skrl model instantiator utility
    # https://skrl.readthedocs.io/en/latest/modules/skrl.utils.model_instantiators.html
    models = {}
    # non-shared models
    if experiment_cfg["models"]["separate"]:
        models["policy"] = gaussian_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **convert_skrl_cfg(experiment_cfg["models"]["policy"]),
        )
        models["value"] = deterministic_model(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            **convert_skrl_cfg(experiment_cfg["models"]["value"]),
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
                convert_skrl_cfg(experiment_cfg["models"]["policy"]),
                convert_skrl_cfg(experiment_cfg["models"]["value"]),
            ],
        )
        models["value"] = models["policy"]

    # configure and instantiate PPO agent
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    agent_cfg.update(convert_skrl_cfg(experiment_cfg["agent"]))

    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})
    agent_cfg["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints

    agent = PPO(
        models=models,
        memory=None,  # memory is optional during evaluation
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # check checkpoint is valid
    if args_cli.checkpoint is None:
        raise ValueError("Checkpoint path is not valid.")
    # load agent from provided path
    print(f"Loading checkpoint from: {args_cli.checkpoint}")
    agent.load(args_cli.checkpoint)

    # test the agent according to the selected mode defined with "use_api":
    # - True: a skrl trainer will be used to evaluate the agent
    # - False: the interaction with the environment will be performed manually.
    #          This mode allows recording specific information about the environment

    # configure and instantiate the RL trainer
    # https://skrl.readthedocs.io/en/latest/modules/skrl.trainers.sequential.html
    if experiment_cfg["trainer"]["use_api"]:
        trainer_cfg = experiment_cfg["trainer"]
        trainer_cfg["disable_progressbar"] = True
        trainer = ManualTrainer(cfg=trainer_cfg, env=env, agents=agent)

        # simulate environment
        while simulation_app.is_running():
            # agent and environment stepping
            trainer.eval()
            # check if simulator is stopped
            if env.sim.is_stopped():
                break

    # execute the interaction with the environment without using a trainer
    # (note that skrl requires the execution of some additional agent methods for proper operation).
    # https://skrl.readthedocs.io/en/latest/modules/skrl.trainers.manual.html
    else:
        timestep = 0
        agent.init()
        agent.set_running_mode("eval")

        # reset environment
        obs, infos = env.reset()
        # simulate environment
        while simulation_app.is_running():
            timestep += 1
            # agent stepping
            actions = agent.act(obs, timestep=timestep, timesteps=None)[0]
            # env stepping
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            # track data
            agent.record_transition(
                states=obs,
                actions=actions,
                rewards=rewards,
                next_states=next_obs,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=None,
            )
            # log custom environment data
            if "episode" in infos:
                for k, v in infos["episode"].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        agent.track_data(f"Info / {k}", v.item())
            # write data to TensorBoard / Weights & Biases
            super(type(agent), agent).post_interaction(timestep=timestep, timesteps=None)
            # reset environments
            if terminated.any() or truncated.any():
                obs, infos = env.reset()
            else:
                obs.copy_(next_obs)
            # check if simulator is stopped
            if env.sim.is_stopped():
                break

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
