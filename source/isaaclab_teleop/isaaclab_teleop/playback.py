# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture playback lifecycle for task-owned actions and operator presentation."""

from __future__ import annotations

import os
import re
from collections.abc import ItemsView, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import torch

from isaaclab.cloner import expand_env_regex_ns, path
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.managers import ObservationPresentationCfg, SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import instantiate, validate
from isaaclab.utils.images import sensor_key_for_gt_type
from isaaclab.visualizers import ImageViewCfg, PerspectiveCameraCfg, SceneCameraCfg

from .action_cfg import TeleopActionCfg
from .observation_cfg import TeleopObservationCfg
from .play_source_cfg import CaptureTeleopSourceCfg

if TYPE_CHECKING:
    from .action import TeleopAction
    from .ov_libraries_xr import OvLibrariesXrSession
    from .play_source import CaptureTeleopSource
    from .recording import DemoRecording, DemoRecordingCfg


class PlaybackStopped(Exception):
    """The operator closed presentation while playback was waiting for a frame."""


def _items(config: Any) -> ItemsView[str, Any]:
    return config.items() if isinstance(config, Mapping) else vars(config).items()


def configured_teleop_actions(env_cfg: Any) -> dict[str, TeleopActionCfg]:
    """Return task action terms that opt into Capture playback, without starting input."""
    actions = getattr(env_cfg, "actions", None)
    if actions is None:
        return {}
    return {name: cfg for name, cfg in _items(actions) if isinstance(cfg, TeleopActionCfg)}


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


class CapturePlaySession:
    """Own Capture input and optional XR presentation around one manager-based task.

    Construct and validate before launch, call :meth:`prepare` inside the launch
    context before constructing the environment, then :meth:`attach` before RL
    library wrappers. :meth:`close` also covers failures between these stages.
    The task's action term owns action selection; this class owns resource order,
    operator resets, and complete-scene publication to the independent XR worker.
    """

    def __init__(self, env_cfg: ManagerBasedRLEnvCfg) -> None:
        self._cfg = env_cfg
        self._actions = configured_teleop_actions(env_cfg)
        sources = {name: cfg for name, cfg in self._actions.items() if cfg.source is not None}
        if len(sources) != 1:
            raise ValueError("Capture playback currently requires exactly one TeleopActionCfg owning input.")
        self._term_name, self._term_cfg = next(iter(sources.items()))
        views = [cfg for _, cfg in _items(env_cfg.observations) if isinstance(cfg, TeleopObservationCfg)]
        if len(views) > 1:
            raise ValueError("Capture playback supports one operator presentation per process.")
        self._view = views[0] if views else None
        self._xr: OvLibrariesXrSession | None = None
        self._env: gym.Env | None = None
        self._term: TeleopAction | None = None
        self._followers: list[TeleopAction] = []
        self._source: CaptureTeleopSource | None = None
        self._recording_cfg: DemoRecordingCfg | None = None
        self._recording: DemoRecording | None = None
        self._prepared_context = None
        self._context_before_prepare = None
        self._prepared = False
        self._closed = False
        self._generation = 0
        self._pending: tuple[int, int] | None = None
        self._force_publication = False

    @property
    def uses_xr(self) -> bool:
        """Whether launch cleanup must defer to Capture's XR owner."""
        return self._view is not None

    @property
    def launch_config(self) -> Any:
        """Expose the active XR renderer to launch resolution only during playback."""
        return (self._cfg, self._view.renderer) if self._view is not None else self._cfg

    @property
    def shutdown_requested(self) -> bool:
        """Whether XR closed or the requested demonstration count has been reached."""
        return (self._xr is not None and self._xr.shutdown_requested) or (
            self._recording is not None and self._recording.complete
        )

    def configure_recording(self, cfg: DemoRecordingCfg, *, task_name: str) -> None:
        """Prepare episode collection before launching simulation or opening a dataset."""
        if self._prepared or self._closed or self._recording_cfg is not None:
            raise RuntimeError("Configure demonstration recording once, before preparing playback.")
        from .recording import prepare_demo_recording

        self._recording_cfg = prepare_demo_recording(self._cfg, cfg, task_name, self._term_cfg.env_ids)

    def prepare_environment(self) -> None:
        """Apply the task's optional configuration hook before creating resources.

        The hook must preserve runtime selections and learning observations.
        Call :meth:`validate` after launch resolution to check the final cameras.
        """
        source_cfg = self._term_cfg.source
        if isinstance(source_cfg, CaptureTeleopSourceCfg) and source_cfg.device.prepare_environment is not None:
            source_cfg.device.prepare_environment(self._cfg)

    def validate(self) -> None:
        """Reject unsupported task and rendering combinations before startup."""
        if not isinstance(self._cfg, ManagerBasedRLEnvCfg):
            raise ValueError("Task-owned Capture playback requires a ManagerBasedRLEnvCfg.")
        for name, term in self._actions.items():
            validate(term)
            if (
                term.env_ids != self._term_cfg.env_ids
                or term.controller.asset_name != self._term_cfg.controller.asset_name
            ):
                raise ValueError(
                    f"Teleop action {name!r} must control the same robot asset and environment row"
                    f" as {self._term_name!r}."
                )
        if not isinstance(self._term_cfg.source, CaptureTeleopSourceCfg):
            raise TypeError("Capture playback requires a CaptureTeleopSourceCfg action source.")
        if len(self._term_cfg.env_ids) != 1:
            raise ValueError("Capture playback controls exactly one environment row.")
        if self._term_cfg.env_ids[0] >= self._cfg.scene.num_envs:
            raise ValueError("Capture playback environment selection exceeds scene.num_envs.")
        if not self.uses_xr:
            return
        assert self._view is not None
        validate(self._view)
        if self._cfg.scene.num_envs != 1 or list(self._term_cfg.env_ids) != [0]:
            raise ValueError("XR playback currently requires num_envs=1 and env_ids=[0].")
        if self._cfg.sim.physics is None:
            raise ValueError("XR playback requires a task with non-Kit physics.")
        if os.environ.get("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1").strip() != "1":
            raise ValueError("XR playback requires ISAAC_LAB_OVRTX_USE_OVSTAGE=1.")
        _validate_presentation_cameras(self._cfg, self._view)

    def prepare(self) -> None:
        """Reserve the XR renderer before environment construction, if requested."""
        from isaaclab.sim import SimulationContext

        if self._closed or self._prepared:
            raise RuntimeError("Capture playback preparation requires a new session.")
        self.validate()
        self._context_before_prepare = SimulationContext.instance()
        if self._view is not None:
            from .ov_libraries_xr import OvLibrariesXrSession

            self._xr = OvLibrariesXrSession.prepare(
                self._cfg.sim, self._view.presentation, renderer_cfg=self._view.renderer
            )
            # The public bootstrap creates this context before an environment
            # can borrow it. Retain it for cleanup if gym.make subsequently fails.
            self._prepared_context = SimulationContext.instance()
        self._prepared = True

    def attach(self, env: gym.Env) -> gym.Wrapper:
        """Activate one input owner and its robot's action terms, then wrap playback lifecycle."""
        if not self._prepared or self._closed or self._env is not None:
            raise RuntimeError("Capture playback must be prepared and attached exactly once.")
        self._env = env
        term = env.unwrapped.action_manager.get_term(self._term_name)
        self._followers = [
            env.unwrapped.action_manager.get_term(name) for name in self._actions if name != self._term_name
        ]
        if self._recording_cfg is not None:
            from .recording import DemoRecording

            self._recording = DemoRecording(self._recording_cfg)
        self._source = instantiate(
            self._term_cfg.source, env.unwrapped, env_ids=list(self._term_cfg.env_ids), xr_session=self._xr
        )
        term.activate(self._source)
        self._term = term
        self._source.start()
        if self._xr is not None:
            assert self._view is not None
            device = self._source.device
            started = self._xr.start_render_worker(
                env,
                tracking_anchor_path=device.xr_render_anchor_path,
                required_openxr_extensions=device.required_openxr_extensions,
                excluded_visual_meshes=self._term_cfg.source.device.xr_cfg.excluded_visual_meshes,
                external_anchor_state=device.external_anchor_state,
                timeout_s=self._view.startup_timeout_s,
            )
            if not started:
                raise PlaybackStopped("XR presentation closed during startup.")
            self._publish(reset=True)
        return _CapturePlaybackEnv(env, self)

    def close(self) -> None:
        """Drain input, then XR, then release any environment still owned by the host.

        A failed drain retains the borrowed environment and session for a later
        cleanup attempt. Releasing them while Capture still uses them is unsafe.
        """
        if self._closed:
            return
        try:
            if self._term is not None:
                self._term.deactivate()
            elif self._source is not None:
                self._source.close()
            for term in self._followers:
                term.set_sample(None)
            if self._source is not None and self._source.session_cleanup_failed:
                raise RuntimeError("Capture input did not drain; retaining its XR session and environment.")
        except BaseException as input_error:
            if self._xr is not None:
                try:
                    self._xr.request_shutdown()
                except BaseException as shutdown_error:
                    input_error.add_note(f"Requesting XR shutdown also failed: {shutdown_error!r}")
            raise
        if self._xr is not None:
            if self._env is not None:
                # Native Lab windows belong to the host thread. Input has released
                # them before XR delegates environment teardown to its render worker.
                for visualizer in self._env.unwrapped.sim.visualizers:
                    visualizer.close()
            self._xr.close()
        if self._env is not None:
            if self._xr is None or not self._xr.async_worker_owns_environment:
                self._env.close()
        elif self._prepared_context is not None:
            self._prepared_context.clear_instance()
        elif self._prepared:
            from isaaclab.sim import SimulationContext

            # A desktop task may construct its context and fail before attach.
            # Retire only that new context, never one supplied by an embedding host.
            context = SimulationContext.instance()
            if context is not None and context is not self._context_before_prepare:
                context.clear_instance()
        self._closed = True

    def _publish(self, *, reset: bool = False, changed: bool = True) -> None:
        if self._xr is None:
            return
        assert self._env is not None
        sim = self._env.unwrapped.sim
        if reset:
            self._generation += 1
            self._force_publication = True
        if changed or reset:
            self._pending = self._generation, sim.get_physics_step_count()
        if self._pending is not None:
            admitted = self._xr.request_render(self._pending, force=self._force_publication)
            if admitted is not None:
                self._pending = None
                self._force_publication = False
        self._xr.report_host_timing(sim.get_physics_step_count(), sim.get_physics_dt())

    def _before_step(self) -> bool:
        """Commit pending reset anchors before reading Capture input."""
        if self._closed:
            raise RuntimeError("Capture playback is closed.")
        while self._force_publication:
            if self.shutdown_requested:
                raise PlaybackStopped("XR presentation closed while publishing a reset.")
            self._publish(changed=False)
            if self._force_publication:
                assert self._xr is not None and self._env is not None
                self._xr.wait_for_shutdown(min(self._env.unwrapped.step_dt, 0.05))
        if self.shutdown_requested:
            raise PlaybackStopped("XR presentation closed.")
        assert self._term is not None
        reset = self._term.poll_source()
        for term in self._followers:
            term.set_sample(self._term.sample)
        return reset


def _copy_observations(observations: dict | torch.Tensor) -> dict | torch.Tensor:
    if isinstance(observations, dict):
        return {name: _copy_observations(value) for name, value in observations.items()}
    return observations.clone()


class _CapturePlaybackEnv(gym.Wrapper):
    """Route lifecycle only; the action manager resolves every applied action."""

    def __init__(self, env: gym.Env, session: CapturePlaySession) -> None:
        super().__init__(env)
        self._session = session
        self._env_ids = torch.tensor(session._term_cfg.env_ids, device=env.unwrapped.device, dtype=torch.long)

    def reset(self, **kwargs: Any) -> tuple[dict, dict]:
        recording = self._session._recording
        if recording is not None:
            recording.before_explicit_reset(self.unwrapped)
        try:
            result = self.env.reset(**kwargs)
        finally:
            if recording is not None:
                recording.after_explicit_reset(self.unwrapped)
        self._session._publish(reset=True)
        return result

    def step(self, actions: torch.Tensor) -> VecEnvStepReturn:
        if self._session._before_step():
            # An operator interruption is a reset boundary, not a task action.
            # No physics/recorder step occurs and other rows pause for this call.
            raw = self.unwrapped
            final_obs = _copy_observations(raw.obs_buf)
            obs, extras = self.reset(env_ids=self._env_ids)
            terminated = torch.zeros(raw.num_envs, dtype=torch.bool, device=raw.device)
            truncated = torch.zeros_like(terminated)
            truncated[self._env_ids] = True
            extras = {**extras, "final_obs": final_obs, "teleop_reset": self._env_ids.clone()}
            return obs, torch.zeros(raw.num_envs, device=raw.device), terminated, truncated, extras
        result = self.env.step(actions)
        if self._session._recording is not None:
            self._session._recording.after_step(self.unwrapped, result[2] | result[3])
        # The action manager notifies the source as part of environment reset.
        # Publication follows all reset/physics work before the next input read.
        reset = bool(torch.any(result[2] | result[3]))
        self._session._publish(reset=reset)
        return result

    def close(self) -> None:
        self._session.close()
