# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture input lifecycle for task-owned action terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import replace
from isaaclab.utils.version import has_kit

from .control_events import poll_control_events
from .isaac_teleop_cfg import IsaacTeleopCfg
from .source import PlaybackStopped, TeleopSample, TeleopSource

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import OutputCombiner

    from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedRLEnv

    from .ovxr_presentation import OvxrPresentation
    from .play_source_cfg import CaptureTeleopSourceCfg


class CaptureTeleopSource(TeleopSource):
    """Read a Capture pipeline without stepping, rendering, or resetting simulation.

    One environment owns the pipeline's anchor and clutch state. Device pause
    disengages this source; the caller decides whether physics keeps advancing.
    Construct first, retain ownership, then call :meth:`start`. The source owns
    optional XR presentation and drains input before releasing its borrowed XR
    handles. If startup fails, the caller still owns and must close this source.
    Scene changes are published at the next input poll, before the next control
    step. A reset followed by no control step leaves XR's previous snapshot
    visible until polling resumes.
    """

    def __init__(
        self,
        cfg: CaptureTeleopSourceCfg,
        env: ManagerBasedRLEnv | DirectRLEnv | DirectMARLEnv,
        *,
        env_ids: list[int],
        presentation: OvxrPresentation | None = None,
        startup_timeout_s: float = 30.0,
    ) -> None:
        if len(env_ids) != 1:
            raise ValueError(
                "CaptureTeleopSource requires one selected environment for its anchor and retargeter state."
            )
        if type(env_ids[0]) is not int or not 0 <= env_ids[0] < env.num_envs:
            raise ValueError("CaptureTeleopSource requires a valid environment index.")
        if not isinstance(cfg.device, IsaacTeleopCfg):
            raise TypeError("CaptureTeleopSourceCfg.device must be an IsaacTeleopCfg.")
        if has_kit():
            raise ValueError("CaptureTeleopSource currently requires a non-Kit simulation runtime.")
        if presentation is None and (env.sim.get_setting("/isaaclab/xr/enabled") or cfg.device.xr_camera_feeds):
            raise ValueError("CaptureTeleopSource requires a task-configured XR presentation session for XR input.")

        from .device_selection import create_teleop_device

        xr_session = presentation.session if presentation is not None else None
        if xr_session is None:
            pipeline = _desktop_pipeline(cfg.device)
            device_cfg = replace(cfg.device, pipeline_builder=lambda: pipeline)
        else:
            device_cfg = cfg.device
        self._closed = False
        self._started = False
        self._start_active = cfg.start_active
        self._active = cfg.device.teleoperation_active_default
        self._env_id = env_ids[0]
        self._env = env
        self._presentation = presentation
        self._startup_timeout_s = startup_timeout_s
        self._device = create_teleop_device(
            device_cfg,
            env_prim_path=env.scene.env_prim_paths[env_ids[0]],
            sim_device=env.device,
            visualizers=env.sim.visualizers,
            cloudxr_env_file=cfg.cloudxr_env_file if xr_session is None else None,
            auto_launch_cloudxr=cfg.auto_launch_cloudxr and xr_session is None,
            use_kit_xr_bridge=False,
            external_oxr_handles_provider=xr_session.external_oxr_handles if xr_session is not None else None,
        )
        if presentation is not None:
            presentation.input_device = self._device

    def start(self) -> None:
        """Start input after the playback owner has retained this source.

        The caller must call :meth:`close` even if startup raises. Keeping startup
        outside construction lets the owner retain borrowed XR resources when
        an input failure is followed by an unsuccessful cleanup attempt.
        """
        if self._closed or self._started:
            raise RuntimeError("CaptureTeleopSource.start() requires a new source.")
        self._device.__enter__()
        if self._start_active:
            self._device.request_start()
            self._active = True
        if self._presentation is not None:
            if not self._presentation.start(
                self._env,
                timeout_s=self._startup_timeout_s,
                required_openxr_extensions=self._device.required_openxr_extensions,
            ):
                raise PlaybackStopped("XR presentation closed during startup.")
            self._presentation.publish(self._env, reset=True)
        self._started = True

    def advance(self) -> TeleopSample:
        """Return the latest device action and control events without advancing physics."""
        if self._closed:
            raise RuntimeError("CaptureTeleopSource is closed.")
        if not self._started:
            raise RuntimeError("CaptureTeleopSource.start() must be called before polling input.")
        target_T_world = None
        if self._presentation is not None:
            # The preceding control step is complete, including its episode
            # resets. Publish it once at this boundary; the XR worker renders
            # independently and never reads a partially updated simulation.
            self._presentation.publish(self._env, changed=True)
            if not self._presentation.wait_ready(self._env):
                raise PlaybackStopped("XR presentation closed.")
            if self._device.target_frame_prim_path is not None:
                target_T_world = self._presentation.session.get_target_frame_T_world(
                    self._device.target_frame_prim_path
                )
        actions = self._device.advance(target_T_world)
        events = poll_control_events(self._device)
        if events.is_active is not None:
            self._active = events.is_active
        # Device control events already deduplicate returned frames. Consecutive
        # distinct reset frames remain distinct operator commands.
        return TeleopSample(actions=actions, active=self._active and actions is not None, reset=events.should_reset)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset retargeter/anchor state after the environment has reset."""
        if self._closed:
            return
        if env_ids is not None and self._env_id not in env_ids:
            return
        self._device.notify_reset()
        if self._presentation is not None:
            # Manager reset runs before all task state/observations are ready.
            # The next control boundary publishes the completed reset state.
            self._presentation.mark_changed(self._env, reset=True)

    def close(self) -> None:
        """Drain input and XR before the borrowed environment and windows close."""
        if self._closed:
            return
        try:
            self._device.__exit__(None, None, None)
        except BaseException as cleanup_error:
            if self._presentation is not None:
                try:
                    self._presentation.session.request_shutdown()
                except BaseException as shutdown_error:
                    cleanup_error.add_note(f"Requesting XR shutdown also failed: {shutdown_error!r}")
            raise
        if self._presentation is not None:
            self._presentation.close()
        elif self._device.session_cleanup_failed:
            raise RuntimeError("Capture input did not drain; retaining its borrowed environment")
        self._closed = True


def _desktop_pipeline(cfg: IsaacTeleopCfg) -> OutputCombiner:
    """Build and validate input sources before starting the Capture runtime."""
    from isaacteleop.retargeting_engine.deviceio_source_nodes import (
        GamepadSource,
        KeyboardSource,
        SpaceMouseSource,
    )
    from isaacteleop.retargeting_engine.interface import ValueInput

    pipeline = cfg.pipeline_builder()
    leaves = pipeline.get_leaf_nodes()
    desktop_types = (KeyboardSource, GamepadSource, SpaceMouseSource)
    unsupported = [type(leaf).__name__ for leaf in leaves if not isinstance(leaf, (*desktop_types, ValueInput))]
    if unsupported or not any(isinstance(leaf, desktop_types) for leaf in leaves):
        raise ValueError(
            "CaptureTeleopSource currently supports keyboard, gamepad, and SpaceMouse input pipelines only."
            f" Unsupported input sources: {unsupported or ['no desktop source']}."
        )
    return pipeline
