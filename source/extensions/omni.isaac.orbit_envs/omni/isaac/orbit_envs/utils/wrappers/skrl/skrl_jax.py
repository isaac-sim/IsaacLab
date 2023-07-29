"""Wrapper to configure an :class:`IsaacEnv` instance to skrl environment.

The following example shows how to wrap an environment for skrl:

.. code-block:: python

    from omni.isaac.orbit_envs.utils.wrappers.skrl import SkrlJaxVecEnvWrapper

    env = SkrlJaxVecEnvWrapper(env)

Or, equivalently, by directly calling the skrl library API as follows:

.. code-block:: python

    from skrl.envs.wrappers.jax import wrap_env

    env = wrap_env(env, wrapper="isaac-orbit")

"""

import torch
from typing import Optional

import tqdm

from omni.isaac.orbit_envs.isaac_env import IsaacEnv

__all__ = ["SkrlJaxVecEnvWrapper", "SkrlJaxVecTrainer"]


"""
Vectorized environment wrapper.
"""


def SkrlJaxVecEnvWrapper(env: IsaacEnv):
    """Wraps around Isaac Orbit environment for skrl using JAX.

    This function wraps around the Isaac Orbit environment. Since the :class:`IsaacEnv` environment
    wrapping functionality is defined within the skrl library itself, this implementation
    is maintained for compatibility with the structure of the extension that contains it.
    Internally it calls the :func:`wrap_env` from the skrl library API.

    Args:
        env: The environment to wrap around.

    Raises:
        ValueError: When the environment is not an instance of :class:`IsaacEnv`.

    Reference:
        https://skrl.readthedocs.io/en/latest/api/envs/wrapping.html
    """
    # check that input is valid
    if not isinstance(env.unwrapped, IsaacEnv):
        raise ValueError(f"The environment must be inherited from IsaacEnv. Environment type: {type(env)}")
    # wrap and return the environment
    from skrl.envs.wrappers.jax import wrap_env

    return wrap_env(env, wrapper="isaac-orbit")


"""
Custom trainer for skrl.
"""


class SkrlJaxVecTrainer:
    """Custom trainer with logging of episode information using JAX.

    This trainer implements the :class:`skrl.trainers.jax.Trainer` class.
    It is used to train and evaluate agents in vectorized environments.

    It differs from the :class:`skrl.trainers.jax.Trainer` in the following points:

    * Record log episode information to the agent's logger.
    * Don't close the environment at the end of the training.

    Reference:
        https://skrl.readthedocs.io/en/latest/api/trainers.html
    """

    def __init__(
        self,
        env: "Wrapper",  # noqa: F821
        agent: "Agent",  # noqa: F821
        cfg: Optional[dict] = None,
    ):
        """Initializes the trainer.

        Args:
            env (Wrapper): Environment to train on.
            agents (Union[Agent, List[Agent]]): Agents to train.
            agents_scope (Optional[List[int]], optional): Number of environments for each agent to
                train on. Defaults to None.
            cfg (Optional[dict], optional): Configuration dictionary. Defaults to None.
        """
        self.env = env
        self.agent = agent
        self.cfg = cfg if cfg is not None else {}

        # get configuration
        self.timesteps = self.cfg.get("timesteps", 0)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)

        # init agents
        self.agent.init()

    def train(self):
        """Train the agent in a vectorized environment.

        This method executes the training loop with the following steps:

        * Pre-interaction: Perform any pre-interaction operations.
        * Compute actions: Compute the actions for the agents.
        * Step the environments: Step the environments with the computed actions.
        * Record the environments' transitions: Record the transitions from the environments.
        * Log custom environment data: Log custom environment data.
        * Post-interaction: Perform any post-interaction operations.
        * Reset the environments: Reset the environments if they are terminated or truncated.

        """
        self.agent.set_running_mode("train")
        # reset env
        states, infos = self.env.reset()
        # training loop
        for timestep in tqdm.tqdm(range(self.timesteps), disable=self.disable_progressbar):
            # pre-interaction
            self.agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            # compute actions
            actions = self.agent.act(states, timestep=timestep, timesteps=self.timesteps)[0]
            # step env
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)
            # note: here we do not call render scene since it is done in the env.step() method
            # record transitions
            self.agent.record_transition(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=self.timesteps,
            )
            # log custom environment data
            if "episode" in infos:
                for k, v in infos["episode"].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        self.agent.track_data(f"EpisodeInfo / {k}", v.item())
            # post-interaction
            self.agent.post_interaction(timestep=timestep, timesteps=self.timesteps)
            # update states
            # note: here we do not call reset scene since it is done in the env.step() method
            states = next_states

    def eval(self) -> None:
        """Evaluate the agents in a vectorized environment.

        This method executes the evaluation loop with the following steps:

        * Compute actions: Compute the actions for the agents.
        * Step the environments: Step the environments with the computed actions.
        * Record the environments' transitions: Record the transitions from the environments.
        * Log custom environment data: Log custom environment data.
        """
        self.agent.set_running_mode("eval")
        # reset env
        states, infos = self.env.reset()
        # evaluation loop
        for timestep in tqdm.tqdm(range(self.timesteps), disable=self.disable_progressbar):
            # compute actions
            actions = self.agent.act(states, timestep=timestep, timesteps=self.timesteps)[0]
            # step env
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)
            # note: here we do not call render scene since it is done in the env.step() method
            # write data to TensorBoard
            self.agent.record_transition(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=self.timesteps,
            )
            # log custom environment data
            if "episode" in infos:
                for k, v in infos["episode"].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        self.agent.track_data(f"EpisodeInfo / {k}", v.item())
            # perform post-interaction
            super(type(self.agent), self.agent).post_interaction(timestep=timestep, timesteps=self.timesteps)
            # update states
            # note: here we do not call reset scene since it is done in the env.step() method
            states = next_states
