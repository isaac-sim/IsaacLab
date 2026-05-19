# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Preset variant of the Cartpole camera env with optional frame stacking."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.configclass import resolve_cfg_presets

from .cartpole_camera_env import CartpoleCameraEnv

if TYPE_CHECKING:
    from .cartpole_camera_presets_env_cfg import CartpoleCameraPresetsEnvCfg


class CartpoleCameraPresetsEnv(CartpoleCameraEnv):
    """Cartpole camera env that wires up a :class:`~isaaclab.utils.buffers.CircularBuffer`
    when the active backend combo benefits from explicit temporal observations.

    Behavior is identical to :class:`CartpoleCameraEnv` when ``cfg.frame_stack == 1``;
    when it is ``> 1``, the policy observation becomes the channel-stacked output of
    the buffer (oldest → newest).
    """

    cfg: CartpoleCameraPresetsEnvCfg

    @staticmethod
    def _resolve_frame_stack_default(camera_cfg, physics_cfg) -> int:
        """Return ``2`` for the Newton + Warp combo (no implicit damping, no temporal AA),
        ``1`` otherwise."""
        from isaaclab_newton.physics import NewtonCfg
        from isaaclab_newton.renderers import NewtonWarpRendererCfg

        is_newton_warp = isinstance(physics_cfg, NewtonCfg) and isinstance(
            getattr(camera_cfg, "renderer_cfg", None), NewtonWarpRendererCfg
        )
        return 2 if is_newton_warp else 1

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        # Flatten preset wrappers so the isinstance check below sees concrete types.
        # Idempotent — base ``DirectRLEnv.__init__`` calls this again with no effect.
        resolve_cfg_presets(cfg)

        if cfg.frame_stack < 0:
            cfg.frame_stack = self._resolve_frame_stack_default(cfg.tiled_camera, cfg.sim.physics)
        elif cfg.frame_stack == 0:
            cfg.frame_stack = 1

        single_channels = int(cfg.observation_space[-1])
        if cfg.frame_stack > 1:
            cfg.observation_space = [*cfg.observation_space[:-1], single_channels * cfg.frame_stack]

        super().__init__(cfg, render_mode, **kwargs)

        self._stack: CircularBuffer | None = None
        if cfg.frame_stack > 1:
            self._stack = CircularBuffer(
                max_len=cfg.frame_stack,
                batch_size=self.num_envs,
                device=self.device,
            )

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        if self._stack is not None:
            self._stack.append(obs["policy"])
            # CircularBuffer.buffer is (B, K, H, W, C) oldest->newest along dim 1.
            # Channel-stack: move K next to C, then flatten so the last dim reads
            # oldest_C, ..., newest_C.
            stacked = self._stack.buffer
            b, k, h, w, c = stacked.shape
            obs["policy"] = stacked.permute(0, 2, 3, 1, 4).reshape(b, h, w, k * c).clone()
        return obs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if self._stack is not None:
            self._stack.reset(env_ids)
