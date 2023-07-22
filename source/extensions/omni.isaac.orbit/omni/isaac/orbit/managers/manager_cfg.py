# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration terms for different managers."""

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import Any, Callable, Sequence

from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.noise import NoiseCfg


@configclass
class ManagerBaseTermCfg:
    """Configuration for a curriculum term."""

    func: Callable = MISSING
    """The function to be called for the term.

    The function must take the environment object as the first argument.
    """
    sensor_name: str | None = None
    """The name of the sensor required by the term. Defaults to None.

    If the sensor is not already enabled, it will be enabled in the environment on initialization
    of the manager and passed to the term function as a string under :attr:`sensor_name`.
    """
    asset_name: str | None = None
    """The name of the asset used to resolve the joints and bodies required by the term. Defaults to None."""
    dof_names: Sequence[str] | None = None
    """The names of the joints from the asset required by the term. Defaults to None.

    The names can be either joint names or a regular expression matching the joint names.

    These are converted to joint indices on initialization of the manager and passed to the term
    function as a list of joint indices under :attr:`dof_ids`.
    """
    body_names: Sequence[str] | None = None
    """The names of the bodies from the asset required by the term. Defaults to None.

    The names can be either body names or a regular expression matching the body names.

    These are converted to body indices on initialization of the manager and passed to the term
    function as a list of body indices under :attr:`body_ids`.
    """
    params: dict[str, Any] = dict()
    """The parameters to be passed to the function as keyword arguments. Defaults to an empty dict."""


"""Observation manager."""


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


"""Reward manager."""


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
