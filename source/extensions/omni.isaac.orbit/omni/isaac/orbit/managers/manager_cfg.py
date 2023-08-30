# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration terms for different managers."""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, Callable

from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import NoiseCfg

if TYPE_CHECKING:
    from .action_manager import ActionTerm


@configclass
class SceneEntityCfg:
    """Configuration for a scene entity that is used by the manager's term.

    This class is used to specify the name of the scene entity that is queried from the
    :class:`InteractiveScene` and passed to the manager's term function.
    """

    name: str = MISSING
    """The name of the scene entity.

    This is the name defined in the scene configuration file. See the :class:`InteractiveSceneCfg`
    class for more details.
    """

    joint_names: str | list[str] | None = None
    """The names of the joints from the scene entity. Defaults to None.

    The names can be either joint names or a regular expression matching the joint names.

    These are converted to joint indices on initialization of the manager and passed to the term
    function as a list of joint indices under :attr:`dof_ids`.
    """

    joint_ids: list[int] | None = None
    """The indices of the joints from the asset required by the term. Defaults to None.

    If ``joint_names`` is specified, this is filled in automatically on initialization of the
    manager.
    """

    body_names: str | list[str] | None = None
    """The names of the bodies from the asset required by the term. Defaults to None.

    The names can be either body names or a regular expression matching the body names.

    These are converted to body indices on initialization of the manager and passed to the term
    function as a list of body indices under :attr:`body_ids`.
    """

    body_ids: list[int] | None = None
    """The indices of the bodies from the asset required by the term. Defaults to None.

    If ``body_names`` is specified, this is filled in automatically on initialization of the
    manager.
    """


@configclass
class ManagerBaseTermCfg:
    """Configuration for a manager term."""

    func: Callable = MISSING
    """The function to be called for the term.

    The function must take the environment object as the first argument.

    Note:
        It also supports `callable classes`_, i.e. classes that implement the :meth:`__call__`
        method.

    .. _`callable classes`: https://docs.python.org/3/reference/datamodel.html#object.__call__

    """

    params: dict[str, Any | SceneEntityCfg] = dict()
    """The parameters to be passed to the function as keyword arguments. Defaults to an empty dict.

    .. note::
        If the value is a :class:`SceneEntityCfg` object, the manager will query the scene entity
        from the :class:`InteractiveScene` and process the entity's joints and bodies as specified
        in the :class:`SceneEntityCfg` object.
    """


##
# Action manager.
##


@configclass
class ActionTermCfg:
    """Configuration for an action term."""

    cls: type[ActionTerm] = MISSING
    """Class of the action term."""

    asset_name: str = MISSING
    """Name of the asset (object or robot) on which action is applied."""


##
# Curriculum manager.
##


@configclass
class CurriculumTermCfg(ManagerBaseTermCfg):
    """Configuration for a curriculum term."""

    func: Callable[..., float | dict[str, float]] = MISSING
    """The name of the function to be called.

    This function should take the environment object, environment indices
    and any other parameters as input and return the curriculum state for
    logging purposes.
    """


##
# Observation manager.
##


@configclass
class ObservationTermCfg(ManagerBaseTermCfg):
    """Configuration for an observation term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the observation signal as torch float tensors of
    shape ``(num_envs, obs_term_dim)``.
    """

    noise: NoiseCfg | None = None
    """The noise to add to the observation. Defaults to None, in which case no noise is added."""

    clip: tuple[float, float] | None = None
    """The clipping range for the observation after adding noise. Defaults to None,
    in which case no clipping is applied."""

    scale: float | None = None
    """The scale to apply to the observation after clipping. Defaults to None,
    in which case no scaling is applied (same as setting scale to :obj:`1`)."""


@configclass
class ObservationGroupCfg:
    """Configuration for an observation group."""

    concatenate_terms: bool = True
    """Whether to concatenate the observation terms in the group. Defaults to True.

    If true, the observation terms in the group are concatenated along the last dimension.
    Otherwise, they are kept separate and returned as a dictionary.
    """

    enable_corruption: bool = False
    """Whether to enable corruption for the observation group. Defaults to False.

    If true, the observation terms in the group are corrupted by adding noise (if specified).
    Otherwise, no corruption is applied.
    """


##
# Randomization manager
##


@configclass
class RandomizationTermCfg(ManagerBaseTermCfg):
    """Configuration for a randomization term."""

    func: Callable[..., None] = MISSING
    """The name of the function to be called.

    This function should take the environment object, environment indices
    and any other parameters as input.
    """

    mode: str = MISSING
    """The mode in which the randomization term is applied.

    Note:
        The mode name ``"interval"`` is a special mode that is handled by the
        manager Hence, its name is reserved and cannot be used for other modes.
    """

    interval_range_s: tuple[float, float] | None = None
    """The range of time in seconds at which the term is applied.

    Based on this, the interval is sampled uniformly between the specified
    interval range for each environment instance and the term is applied for
    the environment instances if the current time hits the interval time.

    Note:
        This is only used if the mode is ``"interval"``.
    """


##
# Reward manager.
##


@configclass
class RewardTermCfg(ManagerBaseTermCfg):
    """Configuration for a reward term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the reward signals as torch float tensors of
    shape ``(num_envs,)``.
    """

    weight: float = MISSING
    """The weight of the reward term.

    This is multiplied with the reward term's value to compute the final
    reward.

    Note:
        If the weight is zero, the reward term is ignored.
    """


##
# Termination manager.
##


@configclass
class TerminationTermCfg(ManagerBaseTermCfg):
    """Configuration for a termination term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the termination signals as torch boolean tensors of
    shape ``(num_envs,)``.
    """

    time_out: bool = False
    """Whether the termination term contributes towards episodic timeouts. Defaults to False.

    Note:
        These usually correspond to tasks that have a fixed time limit.
    """
