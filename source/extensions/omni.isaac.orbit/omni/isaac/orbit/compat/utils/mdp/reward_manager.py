# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Reward manager for computing reward signals for a given world."""

import copy
import inspect
import torch
from prettytable import PrettyTable
from typing import Any, Dict, List, Optional


class RewardManager:
    """Manager for computing reward signals for a given world.

    The reward manager computes the total reward as a sum of the weighted reward terms. The reward
    terms are parsed from a nested dictionary containing the reward manger's settings and reward
    terms configuration. The manager looks for member functions with the name of the reward term's
    key in the settings.

    The following configuration settings are available:

    * ``only_positive_rewards``: A boolean indicating whether to return only positive rewards (if the
      accumulated reward is negative, then zero is returned). This is useful to facilitate learning
      towards solving the task first while pushing the negative penalties for later :cite:p:`rudin2022advanced`.

    Each reward term configuration must contain the key ``weight`` which decides the linear weighting
    of the reward term. Terms with weights as zero are ignored by the rewards manager. Additionally,
    the configuration dictionary can contain other parameters for the reward term's function, such
    as kernel parameters.

    .. note::

        The reward manager multiplies the reward terms by the ``weight`` key in the configuration
        dictionary with the time-step interval ``dt`` of the environment. This is done to ensure
        that the computed reward terms are balanced with respect to the chosen time-step interval.


    Usage:
        .. code-block:: python

            from collections import namedtuple

            from omni.isaac.orbit.utils.mdp.reward_manager import RewardManager


            class DummyRewardManager(RewardManager):
                def reward_term_1(self, env):
                    return 1

                def reward_term_2(self, env, bbq: bool):
                    return 1.0 * bbq

                def reward_term_3(self, env, hot: bool):
                    return 4.0 * hot

                def reward_term_4(self, env, hot: bool, bland: float):
                    return bland * hot * 0.5


            # dummy environment with 20 instances
            env = namedtuple("IsaacEnv", ["num_envs", "dt"])(20, 0.01)
            # dummy device
            device = "cpu"

            # create reward manager
            cfg = {
                "only_positive_rewards": False,
                "reward_term_1": {"weight": 10.0},
                "reward_term_2": {"weight": 5.0, "bbq": True},
                "reward_term_3": {"weight": 0.0, "hot": True},
                "reward_term_4": {"weight": 1.0, "hot": False, "bland": 2.0},
            }
            rew_man = DummyRewardManager(cfg, env, env.num_envs, env.dt, device)

            # print reward manager
            # shows active reward terms and their weights
            print(rew_man)

            # check number of active reward terms
            assert len(rew_man.active_terms) == 3

            # here we reset all environment ids, but it is possible to reset only a subset
            # of the environment ids based on the episode termination.
            rew_man.reset_idx(env_ids=list(range(env.num_envs)))

            # compute reward using manager
            rewards = rew_man.compute()
            # check the rewards shape
            assert rewards.shape == (env.num_envs,)

    """

    def __init__(self, cfg: Dict[str, Dict[str, Any]], env, num_envs: int, dt: float, device: str):
        """Construct a list of reward functions which are used to compute the total reward.

        Args:
            cfg (Dict[str, Dict[str, Any]]): Configuration for reward terms.
            env (IsaacEnv): A world instance used for accessing state.
            num_envs (int): Number of environment instances.
            dt (float): The time-stepping for the environment.
            device (int): The device on which create buffers.
        """
        # store input
        self._cfg = copy.deepcopy(cfg)
        self._env = env
        self._num_envs = num_envs  # We can get this from env?
        self._dt = dt  # We can get this from env?
        self._device = device
        # store reward manager settings
        self._enable_only_positive_rewards = self._cfg.pop("only_positive_rewards", False)
        # parse config to create reward terms information
        self._prepare_reward_terms()
        # prepare extra info to store individual reward term information
        self._episode_sums = dict()
        for term_name in self._reward_term_names:
            self._episode_sums[term_name] = torch.zeros(self._num_envs, dtype=torch.float, device=self._device)
        # create buffer for managing reward per environment
        self._reward_buf = torch.zeros(self._num_envs, dtype=torch.float, device=self._device)

    def __str__(self) -> str:
        """Returns: A string representation for reward manager."""
        msg = f"<RewardManager> contains {len(self._reward_term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Reward Terms"
        table.field_names = ["Index", "Name", "Weight", "Parameters"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Weight"] = "r"
        # add info on each term
        reward_terms = zip(self._reward_term_names, self._reward_term_weights, self._reward_term_params)
        for index, (name, weight, params) in enumerate(reward_terms):
            if any(params):
                table.add_row([index, name, weight, params])
            else:
                table.add_row([index, name, weight, None])
        # convert table to string
        msg += table.get_string()

        return msg

    """
    Properties.
    """

    @property
    def device(self) -> str:
        """Name of device for computation."""
        return self._device

    @property
    def active_terms(self) -> List[str]:
        """Name of active reward terms."""
        return self._reward_term_names

    @property
    def episode_sums(self) -> Dict[str, torch.Tensor]:
        """Contains the current episodic sum of individual reward terms."""
        return self._episode_sums

    """
    Operations.
    """

    def reset_idx(self, env_ids: torch.Tensor, episodic_extras: Optional[dict] = None):
        """Reset the reward terms computation for input environment indices.

        If `episodic_extras` is not None, then the collected sum of individual reward terms is stored
        in the dictionary. This is useful for logging episodic information.

        Args:
            env_ids (torch.Tensor): Indices of environment instances to reset.
            episodic_extras (Optional[dict], optional): Dictionary to store episodic information.
                Defaults to None.
        """
        if episodic_extras is None:
            # reset the collected sum
            for term_name in self._episode_sums:
                self._episode_sums[term_name][env_ids] = 0.0
        else:
            # episode length (in seconds from env)
            episode_length_s = self._env.cfg.env.episode_length_s
            # save collected sum and reset the rolling sums
            for term_name in self._episode_sums:
                # -- save the collected sum
                cumulated_score = self._episode_sums[term_name][env_ids]
                episodic_extras[f"rew_{term_name}"] = torch.mean(cumulated_score / episode_length_s)
                # -- reset the collected sum
                self._episode_sums[term_name][env_ids] = 0.0

    def compute(self) -> torch.Tensor:
        """Computes the reward signal as linearly weighted sum of individual terms.

        This function calls each reward term managed by the class and adds them to compute the net
        reward signal. It also updates the episodic sums corresponding to individual reward terms.

        Returns:
            torch.Tensor: The net reward signal of shape (num_envs,).
        """
        # reset computation
        self._reward_buf[:] = 0.0
        # iterate over all the reward terms
        for name, weight, params, func in zip(
            self._reward_term_names, self._reward_term_weights, self._reward_term_params, self._reward_term_functions
        ):
            # termination rewards: handled after clipping
            if name == "termination":
                continue
            # compute term's value
            value = func(self._env, **params) * weight
            # update total reward
            self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value
        # if enabled, consider rewards only when they yield a positive sum
        # TODO: (trick from Nikita) Add more documentation on why this might be helpful!
        if self._enable_only_positive_rewards:
            self._reward_buf = self._reward_buf.clip_(min=0.0)
        # add termination reward after clipping has been performed
        if "termination" in self._reward_term_names:
            weight = self._reward_term_weights["termination"]
            params = self._reward_term_params["termination"]
            func = self._reward_term_functions["termination"]
            # compute term's value
            value = func(self._env, **params) * weight
            # update total reward
            self._reward_buf += value
            # update episodic sum
            self._episode_sums["termination"] += value

        return self._reward_buf

    """
    Helper functions.
    """

    def _prepare_reward_terms(self):
        """Prepares a list of reward functions.

        Raises:
            KeyError: If reward term configuration does not have they key 'weight'.
            ValueError: If reward term configuration does not satisfy its function signature.
            AttributeError: If the reward term's function not found in the class.
        """
        # remove zero scales and multiply non-zero ones by dt
        # note: we multiply weights by dt to make them agnostic to control decimation
        for term_name in list(self._cfg):
            # extract term config
            term_cfg = self._cfg[term_name]
            # check for weight attribute
            if "weight" in term_cfg:
                if term_cfg["weight"] == 0:
                    self._cfg.pop(term_name)
                else:
                    term_cfg["weight"] *= self._dt
            else:
                raise KeyError(f"The key 'weight' not found for reward term: '{term_name}'.")

        # parse remaining reward terms and decimate their information
        # TODO: Make this more convenient by using data structures.
        self._reward_term_names = list()
        self._reward_term_weights = list()
        self._reward_term_params = list()
        self._reward_term_functions = list()

        for term_name, term_cfg in self._cfg.items():
            self._reward_term_names.append(term_name)
            self._reward_term_weights.append(term_cfg.pop("weight"))
            self._reward_term_params.append(term_cfg)
            # check if rewards manager has the term
            if hasattr(self, term_name):
                func = getattr(self, term_name)
                # check if reward term's arguments are matched by params
                term_params = list(term_cfg.keys())
                args = inspect.getfullargspec(func).args
                # ignore first two arguments for (self, env)
                # Think: Check for cases when kwargs are set inside the function?
                if len(args) > 2:
                    if set(args[2:]) != set(term_params):
                        msg = f"Reward term '{term_name}' expects parameters: {args[2:]}, but {term_params} provided."
                        raise ValueError(msg)
                # add function to list
                self._reward_term_functions.append(func)
            else:
                raise AttributeError(f"Reward term with the name '{term_name}' has no implementation.")
