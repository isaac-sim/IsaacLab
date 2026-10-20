# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from dataclasses import MISSING, fields
from typing import Dict, Literal, TypeVar  # noqa: UP035

import gymnasium as gym
import torch
from isaaclab_visualizers.kit import KitBackendCfg

from ..utils import configclass
from ..visualizers import VisualizerCfg

##
# Deprecated: ViewerCfg
##


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera.

    .. deprecated::
        :class:`ViewerCfg` is deprecated and will be removed in a future release.
        Configure the viewport camera via :class:`~isaaclab.visualizers.VisualizerCfg`
        on :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` instead. Select its Kit backend:

    .. code-block:: python

        from isaaclab.sim import SimulationCfg
        from isaaclab.visualizers import VisualizerCfg
        from isaaclab_visualizers.kit import KitBackendCfg

        sim_cfg = SimulationCfg(
            visualizer_cfgs=VisualizerCfg(eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0), backend=KitBackendCfg())
        )
    """

    eye: tuple[float, float, float] = (7.5, 7.5, 7.5)
    """Initial camera position (in m). Default is (7.5, 7.5, 7.5)."""

    lookat: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Initial camera target position (in m). Default is (0.0, 0.0, 0.0)."""

    cam_prim_path: str = "/OmniverseKit_Persp"
    """The camera prim path to record images from. Default is "/OmniverseKit_Persp"."""

    resolution: tuple[int, int] = (1280, 720)
    """The resolution (width, height) of the camera. Default is (1280, 720)."""

    origin_type: Literal["world", "env", "asset_root", "asset_body"] = "world"
    """The frame in which the camera position (eye) and target (lookat) are defined. Default is "world"."""

    env_index: int = 0
    """The environment index for frame origin. Default is 0."""

    asset_name: str | None = None
    """The asset name in the interactive scene for the frame origin. Default is None."""

    body_name: str | None = None
    """The name of the body in :attr:`asset_name` for the frame origin. Default is None."""

    def __post_init__(self) -> None:
        # Bare ViewerCfg() declarations stay silent; warn once for any customized field.
        for field in fields(self):
            default = field.default if field.default_factory is MISSING else field.default_factory()
            if field.init and default is not MISSING and vars(self)[field.name] != default:
                warnings.warn(
                    "ViewerCfg is deprecated and will be removed in a future release. "
                    "Use VisualizerCfg on SimulationCfg.visualizer_cfgs instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                break


def _apply_deprecated_viewer_cfg(env_cfg: object) -> None:
    """Forward the deprecated viewport declaration to configured visualizers before scene creation."""
    viewer = env_cfg.viewer
    if viewer == ViewerCfg():
        return
    configs = env_cfg.sim.visualizer_cfgs
    configs = configs if isinstance(configs, list) else [configs]
    defaults = VisualizerCfg()
    for cfg in configs:
        # A deliberately configured common view takes precedence over the deprecated declaration.
        if (cfg.eye, cfg.lookat, cfg.window) != (defaults.eye, defaults.lookat, defaults.window):
            continue
        cfg.eye, cfg.lookat = tuple(viewer.eye), tuple(viewer.lookat)
        cfg.window.size = tuple(viewer.resolution)
        if isinstance(cfg.backend, KitBackendCfg):
            cfg.backend.origin_env_index = viewer.env_index
            cfg.backend.origin_type = viewer.origin_type
            if viewer.origin_type in ("asset_root", "asset_body"):
                cfg.backend.origin_type = "asset"
                cfg.backend.origin_track_path = viewer.asset_name
                if viewer.origin_type == "asset_body" and viewer.body_name is not None:
                    cfg.backend.origin_track_path = f"{viewer.asset_name}/{viewer.body_name}"


##
# Types.
##

SpaceType = TypeVar("SpaceType", gym.spaces.Space, int, set, tuple, list, dict)
"""A sentinel object to indicate a valid space type to specify states, observations and actions."""

VecEnvObs = Dict[str, torch.Tensor | Dict[str, torch.Tensor]]
"""Observation returned by the environment.

The observations are stored in a dictionary. The keys are the group to which the observations belong.
This is useful for various setups such as reinforcement learning with asymmetric actor-critic or
multi-agent learning. For non-learning paradigms, this may include observations for different components
of a system.

Within each group, the observations can be stored either as a dictionary with keys as the names of each
observation term in the group, or a single tensor obtained from concatenating all the observation terms.
For example, for asymmetric actor-critic, the observation for the actor and the critic can be accessed
using the keys ``"policy"`` and ``"critic"`` respectively.

Note:
    By default, most learning frameworks deal with default and privileged observations in different ways.
    This handling must be taken care of by the wrapper around the :class:`ManagerBasedRLEnv` instance.

    For included frameworks (RSL-RL, RL-Games, skrl), the observations must have the key "policy". In case,
    the key "critic" is also present, then the critic observations are taken from the "critic" group.
    Otherwise, they are the same as the "policy" group.

"""

VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]
"""The environment signals processed at the end of each step.

The tuple contains batched information for each sub-environment. The information is stored in the following order:

1. **Observations**: The observations from the environment.
2. **Rewards**: The rewards from the environment.
3. **Terminated Dones**: Whether the environment reached a terminal state, such as task success or robot falling etc.
4. **Timeout Dones**: Whether the environment reached a timeout state, such as end of max episode length.
5. **Extras**: A dictionary containing additional information from the environment.
"""

AgentID = TypeVar("AgentID")
"""Unique identifier for an agent within a multi-agent environment.

The identifier has to be an immutable object, typically a string (e.g.: ``"agent_0"``).
"""

ObsType = TypeVar("ObsType", torch.Tensor, Dict[str, torch.Tensor])
"""A sentinel object to indicate the data type of the observation.
"""

ActionType = TypeVar("ActionType", torch.Tensor, Dict[str, torch.Tensor])
"""A sentinel object to indicate the data type of the action.
"""

StateType = TypeVar("StateType", torch.Tensor, dict)
"""A sentinel object to indicate the data type of the state.
"""

EnvStepReturn = tuple[
    Dict[AgentID, ObsType],
    Dict[AgentID, torch.Tensor],
    Dict[AgentID, torch.Tensor],
    Dict[AgentID, torch.Tensor],
    Dict[AgentID, dict],
]
"""The environment signals processed at the end of each step.

The tuple contains batched information for each sub-environment (keyed by the agent ID).
The information is stored in the following order:

1. **Observations**: The observations from the environment.
2. **Rewards**: The rewards from the environment.
3. **Terminated Dones**: Whether the environment reached a terminal state, such as task success or robot falling etc.
4. **Timeout Dones**: Whether the environment reached a timeout state, such as end of max episode length.
5. **Extras**: A dictionary containing additional information from the environment.
"""
