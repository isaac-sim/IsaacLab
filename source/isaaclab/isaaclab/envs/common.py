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

from isaaclab.utils.configclass import configclass

##
# Deprecated: ViewerCfg
##


def _viewer_cfg_field_matches_default(value, default) -> bool:
    """Return True when *value* equals *default* (element-wise for tuples/lists)."""
    if isinstance(value, (tuple, list)):
        return type(value) is type(default) and tuple(value) == tuple(default)
    return value == default


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera.

    .. deprecated::
        :class:`ViewerCfg` is deprecated and will be removed in a future release.
        Configure the viewport camera via :class:`~isaaclab_visualizers.kit.KitVisualizerCfg`
        and add it to :attr:`~isaaclab.sim.SimulationCfg.visualizer_cfgs` instead::

            from isaaclab.sim import SimulationCfg
            from isaaclab_visualizers.kit import KitVisualizerCfg

            sim_cfg = SimulationCfg(visualizer_cfgs=[KitVisualizerCfg(eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))])
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
        # Warn only when the user configured a non-default field so that bare ``ViewerCfg()``
        # usage (e.g. in task configs that haven't been migrated yet) stays silent.
        #
        # @configclass stores mutable defaults (tuples, lists) via default_factory rather than
        # default, so we must check both to obtain the canonical default value.
        differing = []
        for f in fields(self):
            if not f.init:
                continue
            if f.default is not MISSING:
                default = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                default = f.default_factory()  # type: ignore[misc]
            else:
                continue
            if not _viewer_cfg_field_matches_default(getattr(self, f.name), default):
                differing.append(f.name)
        if differing:
            warnings.warn(
                "ViewerCfg is deprecated and will be removed in a future release. "
                "Use KitVisualizerCfg added to SimulationCfg.visualizer_cfgs instead.",
                DeprecationWarning,
                stacklevel=2,
            )


def _apply_deprecated_viewer_cfg(env_cfg: object) -> None:
    """Apply the deprecated ``viewer`` field to ``sim.default_visualizer_cfg`` if it was set.

    Detects any non-default value on ``env_cfg.viewer`` (eye, lookat, origin_type, asset
    tracking), emits a log warning, and writes the settings into
    ``env_cfg.sim.default_visualizer_cfg`` so the Kit visualizer still picks them up.

    Mapping from deprecated :class:`ViewerCfg` fields to :class:`~isaaclab_visualizers.kit.KitVisualizerCfg`:

    * ``eye`` / ``lookat``            → ``eye`` / ``lookat``
    * ``env_index``                   → ``origin_env_index``
    * ``origin_type="asset_root"``    → ``origin_type="asset"``, ``origin_track_path="<asset_name>"``
    * ``origin_type="asset_body"``    → ``origin_type="asset"``, ``origin_track_path="<asset_name>/<body_name>"``

    Must be called before :class:`~isaaclab.sim.SimulationContext` is constructed so
    that the translated cfg is visible to the context.
    """
    import logging as _logging

    viewer = getattr(env_cfg, "viewer", None)
    if viewer is None:
        return

    _defaults = ViewerCfg()
    eye_changed = not _viewer_cfg_field_matches_default(viewer.eye, _defaults.eye)
    lookat_changed = not _viewer_cfg_field_matches_default(viewer.lookat, _defaults.lookat)
    origin_changed = viewer.origin_type != _defaults.origin_type
    resolution_changed = not _viewer_cfg_field_matches_default(
        getattr(viewer, "resolution", _defaults.resolution), _defaults.resolution
    )
    cam_prim_path_changed = getattr(viewer, "cam_prim_path", _defaults.cam_prim_path) != _defaults.cam_prim_path

    if not (eye_changed or lookat_changed or origin_changed or resolution_changed):
        return

    if cam_prim_path_changed:
        _logging.getLogger(__name__).warning(
            "env_cfg.viewer.cam_prim_path=%r cannot be automatically forwarded to KitVisualizerCfg "
            "(no equivalent field). Set the camera prim path via the Kit viewport UI or configure "
            "a custom KitVisualizerCfg in env_cfg.sim.visualizer_cfgs.",
            viewer.cam_prim_path,
        )

    _logging.getLogger(__name__).warning(
        "env_cfg.viewer is deprecated. Set env_cfg.sim.default_visualizer_cfg = "
        "KitVisualizerCfg(eye=..., lookat=...) instead. The viewer values have been "
        "automatically forwarded for this run."
    )

    sim_cfg = getattr(env_cfg, "sim", None)
    if sim_cfg is None:
        return

    if getattr(sim_cfg, "default_visualizer_cfg", None) is not None:
        _logging.getLogger(__name__).warning(
            "env_cfg.viewer is deprecated, but its non-default values (eye, lookat, origin_type) "
            "could NOT be forwarded automatically because env_cfg.sim.default_visualizer_cfg is "
            "already set. To silence this warning and preserve your camera settings, migrate to "
            "KitVisualizerCfg: env_cfg.sim.default_visualizer_cfg = KitVisualizerCfg(eye=..., lookat=...)."
        )
        return

    # Map deprecated origin_type values to the new KitVisualizerCfg fields.
    new_origin_type = viewer.origin_type
    origin_track_path = None
    if viewer.origin_type in ("asset_root", "asset_body"):
        new_origin_type = "asset"
        if getattr(viewer, "asset_name", None) is not None:
            body_name = getattr(viewer, "body_name", None)
            if viewer.origin_type == "asset_body" and body_name is not None:
                origin_track_path = f"{viewer.asset_name}/{body_name}"
            else:
                origin_track_path = viewer.asset_name

    try:
        from isaaclab_visualizers.kit import KitVisualizerCfg

        resolution = getattr(viewer, "resolution", None)
        sim_cfg.default_visualizer_cfg = KitVisualizerCfg(
            eye=tuple(viewer.eye),
            lookat=tuple(viewer.lookat),
            origin_type=new_origin_type,
            origin_env_index=getattr(viewer, "env_index", 0),
            origin_track_path=origin_track_path,
            **({"window_width": resolution[0], "window_height": resolution[1]} if resolution is not None else {}),
        )
    except ImportError:
        from isaaclab.visualizers import VisualizerCfg

        sim_cfg.default_visualizer_cfg = VisualizerCfg(
            eye=tuple(viewer.eye),
            lookat=tuple(viewer.lookat),
        )


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
