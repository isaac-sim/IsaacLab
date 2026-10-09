# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Reflection rules for manager-based observation and action terms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

import torch

from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import ActionTermCfg, ObservationTermCfg
from isaaclab.utils.configclass import configclass

__all__ = ["MirrorObservationTermCfg", "MirrorActionTermCfg", "MirrorJointPositionActionCfg"]


@configclass
class MirrorObservationTermCfg(ObservationTermCfg):
    """An observation term with an explicit reflection rule."""

    mirror: Callable[..., torch.Tensor] = MISSING
    """Callable ``mirror(data, **mirror_params)``. Use :func:`mirror_identity` for invariant terms.

    The input contains the recorded, scaled observation, not live simulation data.
    Flattened histories are reshaped to ``(*batch, history, features)`` before calling
    the function. Other term shapes are preserved. The result must preserve shape,
    dtype, and device. Custom functions must handle arbitrary leading batch dimensions.
    """

    mirror_params: dict[str, Any] = {}
    """Keyword arguments passed to :attr:`mirror`, independent of observation parameters."""


@configclass
class MirrorActionTermCfg(ActionTermCfg):
    """Action configuration base that adds a reflection rule to concrete action configs."""

    mirror: Callable[..., torch.Tensor] = MISSING
    """Callable ``mirror(data, **mirror_params)`` acting on raw policy actions."""

    mirror_params: dict[str, Any] = {}
    """Keyword arguments passed to :attr:`mirror`."""


@configclass
class MirrorJointPositionActionCfg(JointPositionActionCfg, MirrorActionTermCfg):
    """Joint position actions with a configurable reflection of raw policy actions."""
