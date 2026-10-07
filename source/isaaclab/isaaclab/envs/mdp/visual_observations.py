# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateful image processing configured through observation terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ...managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from ...sensors.post_processing import CameraPostProcessingChain, SensorPostProcessorCfg

if TYPE_CHECKING:
    from .. import ManagerBasedEnv


class processed_image(ManagerTermBase):
    """Apply an ordered processor chain to camera buffers once per rendered frame.

    Configure with :class:`~isaaclab.managers.ObservationTermCfg`. The observation
    manager prepares this term after scene spawning and before simulation reset,
    allowing processors to request renderer signals such as unexposed radiance. Each term
    owns its chain, intermediate buffers, and state, even when sharing a camera.
    Processed pixels are returned through this term; camera outputs retain their
    raw renderer buffers. Required renderer settings apply to the entire source camera.
    Camera binding, new-frame tracking, and resets are handled by
    :class:`~isaaclab.sensors.CameraPostProcessingChain`, which can also be used directly.

    Processor inputs are borrowed read-only. The returned tensor persists until the
    next rendered frame; the observation manager makes its usual snapshot copy
    before modifiers, noise, clipping, scaling, delay, and history. Direct callers should
    clone the result if they need to retain a frame or modify its contents.
    """

    @classmethod
    def prepare_scene(cls, cfg: ObservationTermCfg, env: ManagerBasedEnv) -> processed_image:
        """Resolve processing requirements before cameras create render products."""
        return cls(cfg, env)

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self._data_type = cfg.params.get("data_type", "rgb")
        self._normalize = cfg.params.get("normalize", False)
        self._permute = cfg.params.get("permute", False)
        if self._normalize and self._data_type not in {"rgb", "rgba"}:
            raise ValueError(
                "processed_image normalize=True supports 'rgb' and 'rgba'; use a processor for other outputs."
            )
        self._result: torch.Tensor | None = None
        self._image: torch.Tensor | None = None
        self._normalized: torch.Tensor | None = None
        self._mean: torch.Tensor | None = None
        self._chain: CameraPostProcessingChain | None = CameraPostProcessingChain(
            env.scene[sensor_cfg.name],
            cfg.params["processors"],
            [self._data_type],
            num_views=env.num_envs,
            device=env.device,
            stage=env.sim.stage,
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        processors: list[SensorPostProcessorCfg],
        data_type: str = "rgb",
        normalize: bool = False,
        permute: bool = False,
    ) -> torch.Tensor:
        """Read and process a new frame, or return the cached processed image.

        Args:
            env: Environment owning this observation term.
            sensor_cfg: Source camera. Resolved once during scene preparation.
            processors: Ordered processor configurations. Resolved once during preparation.
            data_type: Output name to return. Defaults to ``"rgb"``.
            normalize: For RGB/RGBA, convert to float32, divide by 255, and subtract
                each image's spatial mean, matching :func:`isaaclab.envs.mdp.image`.
            permute: Return BCHW instead of BHWC as a view of the persistent output.

        ``data_type``, ``normalize``, and ``permute`` are fixed when the term is prepared, because they
        select persistent buffers. Calls must pass the same values as the term's ``params``.

        Raises:
            ValueError: If ``data_type``, ``normalize``, or ``permute`` differs from the prepared value.
        """
        if self._chain is None:
            raise RuntimeError("Cannot read a closed processed_image observation term.")
        requested = (data_type, normalize, permute)
        prepared = (self._data_type, self._normalize, self._permute)
        if requested != prepared:
            raise ValueError(
                f"processed_image prepared (data_type, normalize, permute) = {prepared}, but was called with "
                f"{requested}. Set these values in the observation term's params."
            )
        processed = self._chain.update()
        if self._image is None:
            self._image = self._chain.outputs[self._data_type].torch
            if self._normalize:
                self._normalized = torch.empty(self._image.shape, dtype=torch.float32, device=self.device)
                self._mean = torch.empty(
                    (self.num_envs, 1, 1, self._image.shape[-1]), dtype=torch.float32, device=self.device
                )
            result = self._normalized if self._normalize else self._image
            self._result = result.permute(0, 3, 1, 2) if self._permute else result
        if processed and self._normalize:
            # The chain finished on the current Torch stream, so these Torch operations follow it.
            self._normalized.copy_(self._image).div_(255.0)
            torch.mean(self._normalized, dim=(1, 2), keepdim=True, out=self._mean)
            self._normalized.sub_(self._mean)
        return self._result

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset processor state for selected environments without reprocessing a cached frame."""
        if self._chain is not None:
            self._chain.reset(env_ids)

    def close(self) -> None:
        """Release owned processor state and buffers. Repeated calls are safe."""
        chain, self._chain = self._chain, None
        self._image = self._result = self._normalized = self._mean = None
        if chain is not None:
            chain.close()
