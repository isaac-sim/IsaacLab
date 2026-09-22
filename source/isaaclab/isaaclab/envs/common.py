# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import MISSING, fields
from typing import TYPE_CHECKING, Any, Dict, Literal, TypeVar  # noqa: UP035

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.seed import configure_seed

if TYPE_CHECKING:
    from isaaclab.sim import SimulationContext

logger = logging.getLogger(__name__)

##
# Shared environment helpers.
##


def _seed_env(seed: int) -> int:
    """Seed the replicator (when available) and the common random number generators."""
    try:
        import omni.replicator.core as rep

        rep.set_global_seed(seed)
    except (ModuleNotFoundError, AttributeError):
        pass
    return configure_seed(seed)


def _log_env_info(env: Any) -> None:
    """Print the environment time-step summary and warn about render intervals below the decimation."""
    print("[INFO]: Base environment:")
    print(f"\tEnvironment device    : {env.device}")
    print(f"\tEnvironment seed      : {env.cfg.seed}")
    print(f"\tPhysics step-size     : {env.physics_dt}")
    print(f"\tRendering step-size   : {env.physics_dt * env.cfg.sim.render_interval}")
    print(f"\tEnvironment step-size : {env.step_dt}")
    if env.cfg.sim.render_interval < env.cfg.decimation:
        logger.warning(
            f"The render interval ({env.cfg.sim.render_interval}) is smaller than the decimation"
            f" ({env.cfg.decimation}). Multiple render calls will happen for each environment step."
            " If this is not intended, set the render interval to be equal to the decimation."
        )


def _warn_rerender_on_reset_deprecated(cfg: Any, cfg_name: str) -> None:
    """Map the deprecated ``rerender_on_reset`` flag onto ``num_rerenders_on_reset``."""
    if not cfg.rerender_on_reset:
        return
    warnings.warn(
        f"\033[93m\033[1m[DEPRECATION WARNING] {cfg_name}.rerender_on_reset is deprecated. Use"
        f" {cfg_name}.num_rerenders_on_reset instead.\033[0m",
        FutureWarning,
        stacklevel=3,
    )
    if cfg.num_rerenders_on_reset == 0:
        cfg.num_rerenders_on_reset = 1


def _step_physics(env: Any, apply_action: Callable[[], None], after_step: Callable[[], None] | None = None) -> None:
    """Advance the simulation by one environment step, honoring decimation and render cadence.

    Backends that fold the decimation into a single :meth:`~isaaclab.sim.SimulationContext.step` call are
    stepped once; otherwise the physics is stepped ``decimation`` times. Rendering happens whenever the
    simulation step counter hits a ``render_interval`` boundary and the simulation is rendering. When
    :attr:`render_enabled` is False the Kit app loop is skipped, but standalone visualizers still update.

    Args:
        env: The environment being stepped.
        apply_action: Callback that writes the processed actions into the asset buffers.
        after_step: Optional callback invoked after each physics step (e.g. recorder hooks).
    """
    is_rendering = env.sim.is_rendering
    render_interval = env.cfg.sim.render_interval
    if env._physics_handles_decimation:
        substeps, dt = 1, env.step_dt
        env._sim_step_counter += env.cfg.decimation
    else:
        substeps, dt = env.cfg.decimation, env.physics_dt
    for _ in range(substeps):
        if not env._physics_handles_decimation:
            env._sim_step_counter += 1
        apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if after_step is not None:
            after_step()
        if is_rendering and env._sim_step_counter % render_interval == 0:
            env.sim.render(skip_app_pumping=not env.render_enabled)
        env.scene.update(dt=dt)


def _render_env(env: Any, recompute: bool) -> np.ndarray | None:
    """Shared :meth:`render` implementation for the environment classes."""
    # with RTX sensors the step already rendered, so only render again when explicitly asked
    if not env.has_rtx_sensors and not recompute:
        env.sim.render()
    if env.render_mode == "rgb_array":
        warnings.warn(
            "render_mode='rgb_array' is deprecated and will be removed in a future release. "
            "Use VideoRecorderCfg on env_cfg.video_recorders to capture frames instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return None
    if env.render_mode == "human" or env.render_mode is None:
        return None
    raise NotImplementedError(
        f"Render mode '{env.render_mode}' is not supported. Please use: {env.metadata['render_modes']}."
    )


def _set_env_debug_vis(env: Any, debug_vis: bool) -> bool:
    """Shared :meth:`set_debug_vis` implementation for the direct environment classes."""
    if not env.has_debug_vis_implementation:
        return False
    env._set_debug_vis_impl(debug_vis)
    if debug_vis:
        if env._debug_vis_handle is None:
            env._debug_vis_handle = env.sim.vis_marker_registry.add_debug_vis_callback(env)
    else:
        env.sim.vis_marker_registry.clear_debug_vis_callback(env)
    return True


def _episode_scalar_sources(env: Any) -> dict[str, dict[str, Callable[[], float]]]:
    """Live-plot scalar sources for the mean reward and mean episode length."""

    def mean_reward() -> float:
        reward_buf = getattr(env, "reward_buf", None)
        return float(reward_buf.mean()) if reward_buf is not None else 0.0

    return {
        "episode": {
            "mean_reward": mean_reward,
            "episode_length": lambda: float(env.episode_length_buf.float().mean()),
        }
    }


def _kit_manager_visualizers(sim: SimulationContext) -> dict[str, Any]:
    """Collect the Kit ``ManagerLiveVisualizer`` widgets exposed by the active visualizers."""
    return {name: mlv for viz in sim.visualizers for name, mlv in getattr(viz, "kit_manager_visualizers", {}).items()}


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
        logger.warning(
            "env_cfg.viewer.cam_prim_path=%r cannot be automatically forwarded to KitVisualizerCfg "
            "(no equivalent field). Set the camera prim path via the Kit viewport UI or configure "
            "a custom KitVisualizerCfg in env_cfg.sim.visualizer_cfgs.",
            viewer.cam_prim_path,
        )

    logger.warning(
        "env_cfg.viewer is deprecated. Set env_cfg.sim.default_visualizer_cfg = "
        "KitVisualizerCfg(eye=..., lookat=...) instead. The viewer values have been "
        "automatically forwarded for this run."
    )

    sim_cfg = getattr(env_cfg, "sim", None)
    if sim_cfg is None:
        return

    if getattr(sim_cfg, "default_visualizer_cfg", None) is not None:
        logger.warning(
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
