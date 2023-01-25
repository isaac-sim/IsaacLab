"""Wrapper to configure an :class:`IsaacEnv` instance to skrl environment

The following example shows how to wrap an environment for skrl:

.. code-block:: python

    from omni.isaac.orbit_envs.utils.wrappers.skrl import SkrlVecEnvWrapper

    env = SkrlVecEnvWrapper(env)

Or, equivalently, by directly calling the skrl library API as follows:

.. code-block:: python

    from skrl.envs.torch.wrappers import wrap_env

    env = wrap_env(env, wrapper="isaac-orbit")

"""


import torch
from typing import List, Optional, Union

import tqdm

# skrl
from skrl.agents.torch import Agent
from skrl.envs.torch.wrappers import Wrapper, wrap_env
from skrl.trainers.torch import Trainer

from omni.isaac.orbit_envs.isaac_env import IsaacEnv

__all__ = ["SkrlVecEnvWrapper"]


"""
Vectorized environment wrapper.
"""


def SkrlVecEnvWrapper(env: IsaacEnv):
    """Wraps around IsaacSim environment for skrl.

    This function wraps around the IsaacSim environment. Since the :class:`IsaacEnv` environment
    wrapping functionality is defined within the skrl library itself, this implementation
    is maintained for compatibility with the structure of the extension that contains it.
    Internally it calls the :func:`wrap_env` from the skrl library API.

    Reference:
        https://skrl.readthedocs.io/en/latest/modules/skrl.envs.wrapping.html
    """
    # check that input is valid
    if not isinstance(env.unwrapped, IsaacEnv):
        raise ValueError(f"The environment must be inherited from IsaacEnv. Environment type: {type(env)}")
    # wrap and return the environment
    return wrap_env(env, wrapper="isaac-orbit")


class SkrlLogTrainer(Trainer):
    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Customized trainer for tracking episode information

        Reference:
            https://skrl.readthedocs.io/en/latest/modules/skrl.trainers.base_class.html
        """
        default_cfg = {"timesteps": 1000, "disable_progressbar": False}
        default_cfg.update(cfg if cfg is not None else {})
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=default_cfg)

    def train(self):
        """Train the agent"""
        # init agent
        self.agents.init(trainer_cfg=self.cfg)
        self.agents.set_running_mode("train")
        # reset env
        states, infos = self.env.reset()
        # training loop
        for timestep in tqdm.tqdm(range(self.timesteps), disable=self.disable_progressbar):
            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]
            # step the environments
            next_states, rewards, terminated, truncated, infos = self.env.step(actions)
            # record the environments' transitions
            with torch.no_grad():
                self.agents.record_transition(
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
                        self.agents.track_data(f"Info / {k}", v.item())
            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)
            # update states
            states.copy_(next_states)
