# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import torch
from isaaclab.managers.action_manager import ActionManager
from isaaclab.managers.observation_manager import ObservationManager
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper


@configclass
class RslRlSymmetryCfg:
    """Configuration for the symmetry-augmentation in the training.

    When :meth:`use_data_augmentation` is True, the :meth:`data_augmentation_func` is used to generate
    augmented observations and actions. These are then used to train the model.

    When :meth:`use_mirror_loss` is True, the :meth:`mirror_loss_coeff` is used to weight the
    symmetry-mirror loss. This loss is directly added to the agent's loss function.

    If both :meth:`use_data_augmentation` and :meth:`use_mirror_loss` are False, then no symmetry-based
    training is enabled. However, the :meth:`data_augmentation_func` is called to compute and log
    symmetry metrics. This is useful for performing ablations.

    For more information, please check the work from :cite:`mittal2024symmetry`.
    """

    use_data_augmentation: bool = False
    """Whether to use symmetry-based data augmentation. Default is False."""

    use_mirror_loss: bool = False
    """Whether to use the symmetry-augmentation loss. Default is False."""

    data_augmentation_func: callable = MISSING
    """The symmetry data augmentation function.

    The function signature should be as follows:

    Args:

        env (VecEnv): The environment object. This is used to access the environment's properties.
        obs (torch.Tensor | None): The observation tensor. If None, the observation is not used.
        action (torch.Tensor | None): The action tensor. If None, the action is not used.
        obs_type (str): The name of the observation type. Defaults to "policy".
            This is useful when handling augmentation for different observation groups.

    Returns:
        A tuple containing the augmented observation and action tensors. The tensors can be None,
        if their respective inputs are None.
    """

    mirror_loss_coeff: float = 0.0
    """The weight for the symmetry-mirror loss. Default is 0.0."""


@configclass
class SymmetryTermCfg:
    """
    Configuration for a single symmetry term.

    For each possible symmetry configuration, specify the observation / action term names
    and the indices of elements inside the term to perform the operation on.

    - `swap_terms` swaps the elements at the specified indices, i.e. A_sym, B_sym = B, A
    - `swap_negate_terms` swaps and negates the elements at the specified indices, i.e. A_sym, B_sym = -B, -A
    - `negate_terms` negates the elements at the specified indices, i.e. A_sym = -A
    """

    # A_sym, B_sym = B, A
    swap_terms: dict[str, list[tuple[int, int]]] = MISSING

    # A_sym, B_sym = -B, -A
    swap_negate_terms: dict[str, list[tuple[int, int]]] = MISSING

    # A_sym = -A
    negate_terms: dict[str, list[int]] = MISSING


@configclass
class SymmetryCfg:
    """
    Configuration for the symmetry of the environment.

    For each possible symmetry configuration, specify the observation / action term names
    and the indices of elements inside the term to perform the operation on.
    """

    # Symmetry terms for actor observations
    actor_observations: SymmetryTermCfg = MISSING

    # Symmetry terms for critic observations
    critic_observations: SymmetryTermCfg = MISSING

    # Symmetry terms for policy actions
    actions: SymmetryTermCfg = MISSING


def __get_observation_term_index(observation_manager: ObservationManager, term_name: str) -> int:
    """
    Get the index of the first element of the specified observation term in the observation tensor.

    Args:
        observation_manager (ObservationManager): The observation manager.
        term_name (str): The name of the term to get the index of.

    Returns:
        int: The index of the first element of the specified observation term in the observation tensor.
    """
    term_index = 0

    # HACK: currently only works for policy group
    for name, dims in zip(
        observation_manager._group_obs_term_names["policy"],
        observation_manager._group_obs_term_dim["policy"],
    ):
        if name == term_name:
            break
        term_index += dims[-1]
    return term_index


def __get_action_term_index(action_manager: ActionManager, term_name: str) -> int:
    """
    Get the index of the first element of the specified action term in the action tensor.

    Args:
        action_manager (ActionManager): The action manager.
        term_name (str): The name of the term to get the index of.

    Returns:
        int: The index of the first element of the specified action term in the action tensor.
    """
    term_index = 0
    for (name, term) in action_manager._terms.items():
        dim = term.action_dim
        if name == term_name:
            break
        term_index += dim
    return term_index


def symmetry_data_augmentation_function(
    env: RslRlVecEnvWrapper,
    obs: torch.Tensor | None,
    actions: torch.Tensor | None,
    obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    The symmetry data augmentation function.

    This function implements symmetry-based data augmentation for the G1 robot walking task.
    It swaps and negates certain actions components to create symmetric training samples.

    Args:
        env (VecEnv): The environment object. This is used to access the environment's properties.
        obs (torch.Tensor | None): The observation tensor. If None, the observation is not used.
        actions (torch.Tensor | None): The actions tensor. If None, the actions is not used.
        obs_type (str): The name of the observation type. Defaults to "policy".
            This is useful when handling augmentation for different observation groups.

    Returns:
        tuple[torch.Tensor | None, torch.Tensor | None]: A tuple containing the augmented observation and actions tensors.
    """
    # TODO: change to search by class type, instead of attribute name
    symmetry_cfg: SymmetryCfg | None = getattr(getattr(env, "cfg", None), "symmetry", None)

    if symmetry_cfg is None:
        print("WARNING: No symmetry configuration found")
        return obs, actions

    # Augment the observation
    if obs is not None:
        symmetry_obs = obs.clone()
        manager = env.unwrapped.observation_manager

        # Swap pairs
        if obs_type == "policy":
            obs_cfg = symmetry_cfg.actor_observations
        elif obs_type == "critic":
            obs_cfg = symmetry_cfg.critic_observations

        for term_name in obs_cfg.swap_terms:
            # calculate the location of the term in the observation
            term_index = __get_observation_term_index(manager, term_name)

            for id_pair in obs_cfg.swap_terms[term_name]:
                symmetry_obs[..., term_index + id_pair[0]], symmetry_obs[..., term_index + id_pair[1]] = (
                    symmetry_obs[..., term_index + id_pair[1]],
                    symmetry_obs[..., term_index + id_pair[0]],
                )

        # Swap and negate pairs
        for term_name in obs_cfg.swap_negate_terms:
            # calculate the location of the term in the observation
            term_index = __get_observation_term_index(manager, term_name)

            for id_pair in obs_cfg.swap_negate_terms[term_name]:
                symmetry_obs[..., term_index + id_pair[0]], symmetry_obs[..., term_index + id_pair[1]] = (
                    -symmetry_obs[..., term_index + id_pair[1]],
                    -symmetry_obs[..., term_index + id_pair[0]],
                )

        # Negate
        for term_name in obs_cfg.negate_terms:
            # calculate the location of the term in the observation
            term_index = __get_observation_term_index(manager, term_name)

            for id in obs_cfg.negate_terms[term_name]:
                symmetry_obs[..., term_index + id] = -symmetry_obs[..., term_index + id]

        augmented_obs = torch.cat([obs, symmetry_obs], dim=0)
    else:
        augmented_obs = None

    # Augment the actions
    if actions is not None:
        symmetry_actions = actions.clone()
        manager = env.unwrapped.action_manager

        # Swap pairs
        for term_name in symmetry_cfg.actions.swap_terms:
            # calculate the location of the term in the action
            term_index = __get_action_term_index(manager, term_name)

            for id_pair in symmetry_cfg.actions.swap_terms[term_name]:
                symmetry_actions[..., term_index + id_pair[0]], symmetry_actions[..., term_index + id_pair[1]] = (
                    symmetry_actions[..., term_index + id_pair[1]],
                    symmetry_actions[..., term_index + id_pair[0]],
                )

        # Swap and negate pairs
        for term_name in symmetry_cfg.actions.swap_negate_terms:
            # calculate the location of the term in the action
            term_index = __get_action_term_index(manager, term_name)

            for id_pair in symmetry_cfg.actions.swap_negate_terms[term_name]:
                symmetry_actions[..., term_index + id_pair[0]], symmetry_actions[..., term_index + id_pair[1]] = (
                    -symmetry_actions[..., term_index + id_pair[1]],
                    -symmetry_actions[..., term_index + id_pair[0]],
                )

        # Negate
        for term_name in symmetry_cfg.actions.negate_terms:
            # calculate the location of the term in the action
            term_index = __get_action_term_index(manager, term_name)

            for id in symmetry_cfg.actions.negate_terms[term_name]:
                symmetry_actions[..., term_index + id] = -symmetry_actions[..., term_index + id]

        augmented_actions = torch.cat([actions, symmetry_actions], dim=0)
    else:
        augmented_actions = None

    return augmented_obs, augmented_actions
