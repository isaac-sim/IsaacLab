"""
Script to train RL agent with skrl.
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
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
else:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"
# launch the simulator
simulation_app = SimulationApp(config, experience=app_experience)

"""Rest everything follows."""


import gym
import torch
from datetime import datetime

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators import deterministic_model, gaussian_model, shared_model

from omni.isaac.orbit.utils.io import dump_pickle, dump_yaml

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg
from omni.isaac.orbit_envs.utils.wrappers.skrl import SkrlVecEnvWrapper

from config import convert_skrl_cfg, parse_skrl_cfg


def main():
    """Train with skrl agent."""

    args_cli_seed = args_cli.seed

    # parse configuration
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
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)
    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env)  # same as: `wrap_env(env, wrapper="isaac-orbit")`

    # set seed for the experiment (override from command line)
    set_seed(args_cli_seed if args_cli_seed is not None else experiment_cfg["seed"])

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

    # instantiate a RandomMemory as rollout buffer (any memory can be used for this)
    # https://skrl.readthedocs.io/en/latest/modules/skrl.memories.random.html
    memory_size = experiment_cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # configure and instantiate PPO agent
    # https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    experiment_cfg["agent"]["rewards_shaper"] = None  # avoid 'dictionary changed size during iteration'
    agent_cfg.update(convert_skrl_cfg(experiment_cfg["agent"]))

    agent_cfg["state_preprocessor_kwargs"].update({"size": env.observation_space, "device": env.device})
    agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": env.device})

    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # train the agent according to the selected mode defined with "use_api":
    # - True: a skrl trainer will be used to train the agent
    # - False: the interaction with the environment will be performed manually.
    #          This mode allows recording specific information about the environment

    # configure and instantiate the RL trainer
    # https://skrl.readthedocs.io/en/latest/modules/skrl.trainers.sequential.html
    if experiment_cfg["trainer"]["use_api"]:
        trainer_cfg = experiment_cfg["trainer"]
        trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

        # train the agent
        trainer.train()

        # close the simulator (the environment is automatically closed by skrl)
        simulation_app.close()

    # execute the interaction with the environment without using a trainer
    # (note that skrl requires the execution of some additional agent methods for proper operation).
    # https://skrl.readthedocs.io/en/latest/modules/skrl.trainers.manual.html
    else:
        import tqdm

        timesteps = experiment_cfg["trainer"]["timesteps"]
        agent.init()
        agent.set_running_mode("train")

        # reset environment
        obs, infos = env.reset()
        # simulate environment
        for timestep in tqdm.tqdm(range(timesteps)):
            timestep += 1
            # pre-interaction stepping
            agent.pre_interaction(timestep=timestep, timesteps=timesteps)
            # agent stepping
            with torch.no_grad():
                actions = agent.act(obs, timestep=timestep, timesteps=timesteps)[0]
            # env stepping
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            # record environment transitions and track data
            with torch.no_grad():
                agent.record_transition(
                    states=obs,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_obs,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=timesteps,
                )
            # log custom environment data
            if "episode" in infos:
                for k, v in infos["episode"].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        agent.track_data(f"Info / {k}", v.item())
            # post-interaction stepping (write data to TensorBoard / Weights & Biases)
            agent.post_interaction(timestep=timestep, timesteps=timesteps)
            # reset environments
            with torch.no_grad():
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
