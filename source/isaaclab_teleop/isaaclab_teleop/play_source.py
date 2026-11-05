# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Capture device lifecycle adapted to environment-owned action selection."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.teleop import TeleopSample, TeleopSource
from isaaclab.utils import replace
from isaaclab.utils.version import has_kit

from .control_events import poll_control_events
from .isaac_teleop_cfg import IsaacTeleopCfg

if TYPE_CHECKING:
    from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedRLEnv

    from .play_source_cfg import CaptureTeleopSourceCfg

logger = logging.getLogger(__name__)


class CaptureTeleopSource(TeleopSource):
    """Read a Capture pipeline without stepping, rendering, or resetting simulation.

    One environment owns the pipeline's anchor and clutch state. Device pause
    disengages this source; the caller decides whether physics keeps advancing.
    """

    def __init__(
        self,
        cfg: CaptureTeleopSourceCfg,
        env: ManagerBasedRLEnv | DirectRLEnv | DirectMARLEnv,
        *,
        env_ids: list[int],
    ):
        if len(env_ids) != 1:
            raise ValueError(
                "CaptureTeleopSource requires one selected environment for its anchor and retargeter state."
            )
        if not isinstance(cfg.device, IsaacTeleopCfg):
            raise TypeError("CaptureTeleopSourceCfg.device must be an IsaacTeleopCfg.")
        if has_kit():
            raise ValueError("CaptureTeleopSource currently requires a non-Kit simulation runtime.")
        if env.sim.get_setting("/isaaclab/xr/enabled") or cfg.device.xr_camera_feeds:
            raise ValueError(
                "CaptureTeleopSource currently supports desktop input only; XR integration is not available in play."
            )

        from .device_selection import create_teleop_device

        # Keep the current desktop-only scope at the Capture boundary. Future
        # kitless XR integration can extend this without changing task routing.
        pipeline = _desktop_pipeline(cfg.device)
        device_cfg = replace(cfg.device, pipeline_builder=lambda: pipeline)
        self._closed = False
        self._active = cfg.device.teleoperation_active_default
        self._reset_seen = False
        self._device = create_teleop_device(
            device_cfg,
            env_prim_path=env.scene.env_prim_paths[env_ids[0]],
            sim_device=env.device,
            visualizers=env.sim.visualizers,
            cloudxr_env_file=cfg.cloudxr_env_file,
            auto_launch_cloudxr=cfg.auto_launch_cloudxr,
            use_kit_xr_bridge=False,
        )
        try:
            self._device.__enter__()
            if cfg.start_active:
                self._device.request_start()
                self._active = True
        except BaseException:
            try:
                self.close()
            except Exception:
                logger.exception("Failed to close Capture after source startup failed.")
            raise

    def advance(self) -> TeleopSample:
        """Return the latest device action and control events without advancing physics."""
        if self._closed:
            raise RuntimeError("CaptureTeleopSource is closed.")
        actions = self._device.advance()
        events = poll_control_events(self._device)
        if events.is_active is not None:
            self._active = events.is_active
        # Pipelined Capture polls may return the same completed frame more
        # than once. Treat its operator reset pulse as an edge.
        reset = events.should_reset and not self._reset_seen
        self._reset_seen = events.should_reset
        return TeleopSample(actions=actions, active=self._active and actions is not None, reset=reset)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset retargeter/anchor state after the environment has reset."""
        if self._closed:
            return
        if env_ids is not None and len(env_ids) == 0:
            return
        self._device.notify_reset()

    def close(self) -> None:
        """Release Capture before the borrowed environment and windows close."""
        if not self._closed:
            self._closed = True
            self._device.__exit__(None, None, None)


def _desktop_pipeline(cfg: IsaacTeleopCfg):
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
