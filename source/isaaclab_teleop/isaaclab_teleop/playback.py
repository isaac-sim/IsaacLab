# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configure and activate task-owned Capture resources for shared playback."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, ItemsView, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import gymnasium as gym

from isaaclab.cloner import expand_env_regex_ns, path
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg, ObservationPresentationCfg, SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import instantiate, validate
from isaaclab.utils.images import sensor_key_for_gt_type
from isaaclab.visualizers import ImageViewCfg, PerspectiveCameraCfg, SceneCameraCfg

from .input import TeleopInput
from .input_cfg import TeleopInputCfg
from .observation_cfg import TeleopObservationCfg
from .play_source_cfg import CaptureTeleopSourceCfg


def _items(config: Any) -> ItemsView[str, Any]:
    return config.items() if isinstance(config, Mapping) else vars(config).items()


def configured_teleop_inputs(env_cfg: Any) -> dict[str, TeleopInputCfg]:
    """Return optional Capture inputs without constructing input or controller resources."""
    actions = getattr(env_cfg, "actions", None)
    if actions is None:
        return {}
    return {
        name: cfg.input
        for name, cfg in _items(actions)
        if isinstance(cfg, ActionTermCfg) and isinstance(cfg.input, TeleopInputCfg)
    }


def _scene_entities(value: Any) -> Iterator[str]:
    """Visit explicit scene-entity selections, including nested observation parameters."""
    if isinstance(value, SceneEntityCfg):
        yield value.name
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _scene_entities(child)
    elif isinstance(value, tuple | list):
        for child in value:
            yield from _scene_entities(child)


def _viewer_selects_camera(selector: str, camera: CameraCfg, env_template: str) -> bool:
    """Match a configured camera selector against its single-environment scene path."""
    expression = expand_env_regex_ns(selector, env_template)
    camera_path = expand_env_regex_ns(camera.prim_path, env_template)
    if expression == camera_path:
        return True
    matched = path.match(camera_path, env_template)
    if matched is not None:
        camera_path = env_template.format(0) + matched.suffix
    expression = expression.replace("%d", "[^/]+").replace("{}", "[^/]+")
    expression = re.sub(r"(?<=/)\*(?=/|$)", "[^/]+", expression)
    return re.fullmatch(expression, camera_path) is not None


def _validate_presentation_viewers(env_cfg: Any, view: TeleopObservationCfg, cameras: dict[str, CameraCfg]) -> None:
    """Allow independent Lab views while excluding consumers of the leased XR cameras."""
    from isaaclab_visualizers.newton import NewtonRTXBackendCfg

    panels = {camera.name for camera in view.presentation.auxiliary_cameras}
    viewers = env_cfg.sim.visualizer_cfgs
    viewers = viewers if isinstance(viewers, list) else [viewers]
    for viewer in viewers:
        if not viewer.backend:
            continue
        selected = set()
        sources = [viewer.view.source] if viewer.view is not None else viewer.cameras
        if viewer.view is not None and isinstance(viewer.view.source, str):
            selected.add(viewer.view.source)
        elif sources is not None:
            for source in sources:
                if isinstance(source, SceneCameraCfg):
                    selected.update(
                        name
                        for name, camera in cameras.items()
                        if _viewer_selects_camera(source.prim_path, camera, env_cfg.scene.clone_cfg.clone_template)
                    )
        elif viewer.streaming_view:
            for name, camera in cameras.items():
                if viewer.streaming_sensor_prim_path is not None:
                    matches = _viewer_selects_camera(
                        viewer.streaming_sensor_prim_path, camera, env_cfg.scene.clone_cfg.clone_template
                    )
                else:
                    matches = all(
                        sensor_key_for_gt_type(channel, frozenset(camera.data_types), required=False) is not None
                        for channel in viewer.streaming_gt_types
                    )
                if matches:
                    selected.add(name)
        if selected & panels:
            raise ValueError(
                f"Lab visualizer reads XR-owned panel cameras {sorted(selected & panels)}."
                " Select an independently rendered scene camera or perspective view."
            )
        # Before launch expands perspective sources into scene cameras, reject
        # the same renderer conflict that the camera validation catches afterward.
        perspective = sources is None or any(isinstance(source, PerspectiveCameraCfg) for source in sources)
        backends = viewer.backend if isinstance(viewer.backend, list) else [viewer.backend]
        if perspective and any(
            isinstance(backend, NewtonRTXBackendCfg)
            and type(backend.renderer_cfg) is type(view.renderer)
            and backend.renderer_cfg == view.renderer
            for backend in backends
        ):
            raise ValueError("Lab RTX perspective view shares the renderer leased by XR. Select a separate renderer.")


def _validate_presentation_cameras(env_cfg: Any, view: TeleopObservationCfg) -> None:
    """Keep learning cameras intact and reject competing consumers of the XR renderer."""
    panels = {camera.name: camera for camera in view.presentation.auxiliary_cameras}
    cameras = {name: cfg for name, cfg in _items(env_cfg.scene) if isinstance(cfg, CameraCfg)}
    for name, panel in panels.items():
        camera = cameras.get(name)
        if camera is None or camera != panel.camera_cfg:
            raise ValueError(f"XR panel {name!r} must select the matching camera configuration in the scene.")
        if type(camera.renderer_cfg) is not type(view.renderer) or camera.renderer_cfg != view.renderer:
            raise ValueError(f"XR panel {name!r} must use the presentation renderer configuration.")
    if panels and not env_cfg.scene.lazy_sensor_update:
        raise ValueError("XR panel cameras require scene.lazy_sensor_update=True; Capture owns their rendering.")
    for name, camera in cameras.items():
        if camera.prim_path in view.presentation.camera_paths:
            raise ValueError(f"Scene camera {name!r} cannot use an operator-only XR eye camera path.")
        shared = type(camera.renderer_cfg) is type(view.renderer) and camera.renderer_cfg == view.renderer
        if shared and name not in panels:
            raise ValueError(
                f"Learning camera {name!r} shares the renderer leased by XR. Select a separate renderer;"
                " playback does not remove cameras or change policy observations."
            )
    for group_name, group in _items(env_cfg.observations):
        if group is None or isinstance(group, ObservationPresentationCfg):
            continue
        for term_name, term in _items(group):
            selected = set(_scene_entities(getattr(term, "params", {}))) & panels.keys()
            if selected:
                raise ValueError(
                    f"Observation {group_name}/{term_name} reads presentation-only XR cameras {sorted(selected)}."
                )
    for recorder in getattr(env_cfg, "video_recorders", ()):
        source = recorder.source
        if isinstance(source, ImageViewCfg):
            selected = source.source
        elif isinstance(source, str) and source.startswith("sensor:"):
            selected = source.split(":")[1]
        else:
            continue
        if isinstance(selected, str) and selected in panels:
            raise ValueError(f"Video recording cannot read presentation-only XR camera {selected!r}.")
    _validate_presentation_viewers(env_cfg, view, cameras)


def capture_input_config(env_cfg: ManagerBasedRLEnvCfg) -> tuple[str, TeleopInputCfg]:
    """Find the single Capture source shared by the task's participating inputs."""
    sources = {name: cfg for name, cfg in configured_teleop_inputs(env_cfg).items() if cfg.source is not None}
    if len(sources) != 1:
        raise ValueError("Capture playback currently requires exactly one TeleopInputCfg owning input.")
    return next(iter(sources.items()))


def presentation_config(env_cfg: ManagerBasedRLEnvCfg) -> TeleopObservationCfg | None:
    """Find the optional operator view without constructing a presentation runtime."""
    views = [cfg for _, cfg in _items(env_cfg.observations) if isinstance(cfg, TeleopObservationCfg)]
    if len(views) > 1:
        raise ValueError("Capture playback supports one operator presentation per process.")
    return views[0] if views else None


def prepare_playback_config(env_cfg: ManagerBasedRLEnvCfg) -> Any:
    """Apply the task's device preparation hook and expose the active XR renderer to launch."""
    _, input_cfg = capture_input_config(env_cfg)
    source_cfg = input_cfg.source
    if isinstance(source_cfg, CaptureTeleopSourceCfg) and source_cfg.device.prepare_environment is not None:
        source_cfg.device.prepare_environment(env_cfg)
    view = presentation_config(env_cfg)
    return (env_cfg, view.renderer) if view is not None else env_cfg


def validate_playback(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Reject unsupported input and rendering combinations after launch resolution."""
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError("Task-owned Capture playback requires a ManagerBasedRLEnvCfg.")
    owner_name, input_cfg = capture_input_config(env_cfg)
    for name, cfg in configured_teleop_inputs(env_cfg).items():
        validate(cfg)
        if cfg.env_ids != input_cfg.env_ids:
            raise ValueError(f"Teleop input {name!r} must control the same environment row as {owner_name!r}.")
    if not isinstance(input_cfg.source, CaptureTeleopSourceCfg):
        raise TypeError("Capture playback requires a CaptureTeleopSourceCfg input source.")
    if len(input_cfg.env_ids) != 1:
        raise ValueError("Capture playback controls exactly one environment row.")
    if input_cfg.env_ids[0] >= env_cfg.scene.num_envs:
        raise ValueError("Capture playback environment selection exceeds scene.num_envs.")
    view = presentation_config(env_cfg)
    if view is None:
        return
    validate(view)
    if env_cfg.scene.num_envs != 1 or list(input_cfg.env_ids) != [0]:
        raise ValueError("XR playback currently requires num_envs=1 and env_ids=[0].")
    if env_cfg.sim.physics is None:
        raise ValueError("XR playback requires a task with non-Kit physics.")
    if os.environ.get("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1").strip() != "1":
        raise ValueError("XR playback requires ISAAC_LAB_OVRTX_USE_OVSTAGE=1.")
    _validate_presentation_cameras(env_cfg, view)


@contextmanager
def activate_capture(env_cfg: ManagerBasedRLEnvCfg) -> Iterator[Callable[[gym.Env], None]]:
    """Prepare optional XR, then activate inputs on a constructed environment.

    The yielded function installs resources, without wrapping the environment.
    ActionManager subsequently owns input polling and teardown; the source owns
    its optional presentation. This construction scope covers failures before
    that ownership transfer, including a partially constructed environment.
    """
    from isaaclab.sim import SimulationContext

    owner_name, input_cfg = capture_input_config(env_cfg)
    view = presentation_config(env_cfg)
    previous_context = SimulationContext.instance()
    presentation = None
    env = None
    source = None
    source_owned = False

    def activate(created_env: gym.Env) -> None:
        nonlocal env, source, source_owned
        owner = None
        if env is not None:
            raise RuntimeError("Capture inputs can be activated on only one environment per launch.")
        env = created_env
        manager = env.unwrapped.action_manager
        followers = []
        for name in configured_teleop_inputs(env_cfg):
            action_input = manager.activate_input(name)
            if not isinstance(action_input, TeleopInput):
                raise TypeError(f"TeleopInputCfg for {name!r} must construct a TeleopInput.")
            if name == owner_name:
                owner = action_input
            else:
                followers.append(action_input)
        source = instantiate(
            input_cfg.source,
            env.unwrapped,
            env_ids=list(input_cfg.env_ids),
            presentation=presentation,
            startup_timeout_s=view.startup_timeout_s if view is not None else 30.0,
        )
        assert owner is not None
        owner.activate(source, followers=followers)
        source_owned = True
        source.start()

    def close() -> None:
        # Close inputs before controllers/managers; on failure retain every
        # dependency. The XR worker retires rendering, never the environment.
        if env is not None:
            manager = getattr(env.unwrapped, "action_manager", None)
            if manager is not None:
                manager.close_inputs()
        if source is not None and not source_owned:
            source.close()
        if presentation is not None:
            presentation.close()
        if env is not None:
            env.close()
        else:
            context = SimulationContext.instance()
            if context is not None and context is not previous_context:
                context.clear_instance()

    try:
        if view is not None:
            from .ov_libraries_xr import OvLibrariesXrSession
            from .ovxr_presentation import OvxrPresentation

            session = OvLibrariesXrSession.prepare(
                env_cfg.sim, view.presentation, renderer_cfg=view.renderer, environment_owner="host"
            )
            presentation = OvxrPresentation(
                session, excluded_visual_meshes=input_cfg.source.device.xr_cfg.excluded_visual_meshes
            )
        yield activate
    except BaseException as error:
        try:
            close()
        except BaseException as cleanup_error:
            error.add_note(f"Capture cleanup also failed: {cleanup_error!r}")
        raise
    else:
        close()
