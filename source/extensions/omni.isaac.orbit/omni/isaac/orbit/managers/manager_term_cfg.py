# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers."""

from __future__ import annotations

import torch
import warnings
from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any

from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import NoiseCfg

from .scene_entity_cfg import SceneEntityCfg

if TYPE_CHECKING:
    from .action_manager import ActionTerm
    from .command_manager import CommandTerm
    from .manager_base import ManagerTermBase


@configclass
class ManagerTermBaseCfg:
    """Configuration for a manager term."""

    func: Callable | ManagerTermBase = MISSING
    """The function or class to be called for the term.

    The function must take the environment object as the first argument.
    The remaining arguments are specified in the :attr:`params` attribute.

    It also supports `callable classes`_, i.e. classes that implement the :meth:`__call__`
    method. In this case, the class should inherit from the :class:`ManagerTermBase` class
    and implement the required methods.

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

    class_type: type[ActionTerm] = MISSING
    """The associated action term class.

    The class should inherit from :class:`omni.isaac.orbit.managers.action_manager.ActionTerm`.
    """

    asset_name: str = MISSING
    """The name of the scene entity.

    This is the name defined in the scene configuration file. See the :class:`InteractiveSceneCfg`
    class for more details.
    """


##
# Command manager.
##


@configclass
class CommandTermCfg:
    """Configuration for a command generator term."""

    class_type: type[CommandTerm] = MISSING
    """The associated command term class to use.

    The class should inherit from :class:`omni.isaac.orbit.managers.command_manager.CommandTerm`.
    """

    resampling_time_range: tuple[float, float] = MISSING
    """Time before commands are changed [s]."""
    debug_vis: bool = False
    """Whether to visualize debug information. Defaults to False."""


##
# Curriculum manager.
##


@configclass
class CurriculumTermCfg(ManagerTermBaseCfg):
    """Configuration for a curriculum term."""

    func: Callable[..., float | dict[str, float] | None] = MISSING
    """The name of the function to be called.

    This function should take the environment object, environment indices
    and any other parameters as input and return the curriculum state for
    logging purposes. If the function returns None, the curriculum state
    is not logged.
    """


##
# Observation manager.
##


@configclass
class ObservationTermCfg(ManagerTermBaseCfg):
    """Configuration for an observation term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the observation signal as torch float tensors of
    shape (num_envs, obs_term_dim).
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
# Event manager
##


@configclass
class EventTermCfg(ManagerTermBaseCfg):
    """Configuration for a event term."""

    func: Callable[..., None] = MISSING
    """The name of the function to be called.

    This function should take the environment object, environment indices
    and any other parameters as input.
    """

    mode: str = MISSING
    """The mode in which the event term is applied.

    Note:
        The mode name ``"interval"`` is a special mode that is handled by the
        manager Hence, its name is reserved and cannot be used for other modes.
    """

    interval_range_s: tuple[float, float] | None = None
    """The range of time in seconds at which the term is applied.

    Based on this, the interval is sampled uniformly between the specified
    range for each environment instance. The term is applied on the environment
    instances where the current time hits the interval time.

    Note:
        This is only used if the mode is ``"interval"``.
    """


@configclass
class RandomizationTermCfg(EventTermCfg):
    """Configuration for a randomization term.

    .. deprecated:: v0.3.0

        This class is deprecated and will be removed in v0.4.0. Please use :class:`EventTermCfg` instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Deprecation warning.
        warnings.warn(
            "The RandomizationTermCfg has been renamed to EventTermCfg and will be removed in v0.4.0. Please use"
            " EventTermCfg instead.",
            DeprecationWarning,
        )


##
# Reward manager.
##


@configclass
class RewardTermCfg(ManagerTermBaseCfg):
    """Configuration for a reward term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the reward signals as torch float tensors of
    shape (num_envs,).
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
class TerminationTermCfg(ManagerTermBaseCfg):
    """Configuration for a termination term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the termination signals as torch boolean tensors of
    shape (num_envs,).
    """

    time_out: bool = False
    """Whether the termination term contributes towards episodic timeouts. Defaults to False.

    Note:
        These usually correspond to tasks that have a fixed time limit.
    """
