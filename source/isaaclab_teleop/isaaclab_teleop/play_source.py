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
from .source import TeleopSample, TeleopSource

if TYPE_CHECKING:
    from isaacteleop.retargeting_engine.interface import OutputCombiner

    from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedRLEnv

    from .isaac_teleop_device import IsaacTeleopDevice
    from .ov_libraries_xr import OvLibrariesXrSession
    from .play_source_cfg import CaptureTeleopSourceCfg


class CaptureTeleopSource(TeleopSource):
    """Read a Capture pipeline without stepping, rendering, or resetting simulation.

    One environment owns the pipeline's anchor and clutch state. Device pause
    disengages this source; the caller decides whether physics keeps advancing.
    Construct first, retain ownership, then call :meth:`start`. If startup fails,
    the caller still owns this source and must drain it before releasing XR.
    """

    def __init__(
        self,
        cfg: CaptureTeleopSourceCfg,
        env: ManagerBasedRLEnv | DirectRLEnv | DirectMARLEnv,
        *,
        env_ids: list[int],
        xr_session: OvLibrariesXrSession | None = None,
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
        if xr_session is None and (env.sim.get_setting("/isaaclab/xr/enabled") or cfg.device.xr_camera_feeds):
            raise ValueError("CaptureTeleopSource requires a task-configured XR presentation session for XR input.")

        from .device_selection import create_teleop_device

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
        self._xr_session = xr_session
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
        self._started = True

    @property
    def device(self) -> IsaacTeleopDevice:
        """Input device whose anchors and required extensions configure XR startup."""
        return self._device

    @property
    def session_cleanup_failed(self) -> bool:
        """Whether input failed to drain, requiring its borrowed XR resources to remain alive."""
        return self._device.session_cleanup_failed

    def advance(self) -> TeleopSample:
        """Return the latest device action and control events without advancing physics."""
        if self._closed:
            raise RuntimeError("CaptureTeleopSource is closed.")
        if not self._started:
            raise RuntimeError("CaptureTeleopSource.start() must be called before polling input.")
        target_T_world = None
        if self._xr_session is not None and self._device.target_frame_prim_path is not None:
            target_T_world = self._xr_session.get_target_frame_T_world(self._device.target_frame_prim_path)
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

    def close(self) -> None:
        """Release Capture before the borrowed environment and windows close."""
        if not self._closed:
            self._device.__exit__(None, None, None)
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
