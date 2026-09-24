# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateful image processing configured through observation terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from ...managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from ...utils.visual_processing import VisualProcessingPipeline, VisualProcessorCfg, VisualProcessorContext

if TYPE_CHECKING:
    from ...sensors import Camera
    from .. import ManagerBasedEnv


@wp.kernel
def _find_new_camera_frames(
    current: wp.array(dtype=wp.int64), previous: wp.array(dtype=wp.int64), mask: wp.array(dtype=wp.bool)
):
    index = wp.tid()
    mask[index] = current[index] != previous[index]


class processed_image(ManagerTermBase):
    """Apply an ordered processor chain to camera buffers once per rendered frame.

    Configure with :class:`~isaaclab.managers.ObservationTermCfg`. The observation
    manager prepares this term after scene spawning and before simulation reset,
    allowing processors to request renderer inputs and neutral exposure. Each term
    owns its chain, intermediate buffers, and state, even when sharing a camera.
    The camera's public outputs remain unchanged.

    Processor inputs are borrowed read-only. The returned tensor persists until the
    next rendered frame; the observation manager makes its usual snapshot copy
    before modifiers, noise, clipping, scaling, and history. Direct callers should
    clone the result if they need to retain a frame or modify its contents.
    """

    @classmethod
    def prepare_scene(cls, cfg: ObservationTermCfg, env: ManagerBasedEnv) -> processed_image:
        """Resolve processing requirements before cameras create render products."""
        return cls(cfg, env)

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        self._camera: Camera = env.scene[sensor_cfg.name]
        if self._camera.cfg.isp_cfg is not None:
            raise ValueError(
                f"processed_image for {sensor_cfg.name!r} requires CameraCfg.isp_cfg=None. "
                "Move its value into PpispProcessorCfg(isp_cfg=...) in this observation term."
            )
        self._data_type = cfg.params.get("data_type", "rgb")
        self._normalize = cfg.params.get("normalize", False)
        self._permute = cfg.params.get("permute", False)
        if self._normalize and self._data_type not in {"rgb", "rgba"}:
            raise ValueError(
                "processed_image normalize=True supports 'rgb' and 'rgba'; use a processor for other outputs."
            )
        self._pipeline: VisualProcessingPipeline | None = None
        self._generation = -1
        self._result: torch.Tensor | None = None
        self._image: torch.Tensor | None = None
        self._normalized: torch.Tensor | None = None
        self._mean: torch.Tensor | None = None
        self._reset_mask: wp.array | None = None
        self._last_frames: wp.array | None = None
        self._process_mask: wp.array | None = None
        try:
            context = VisualProcessorContext(
                stage=env.sim.stage,
                camera_prim_paths=self._camera.camera_prim_paths,
                num_views=env.num_envs,
                height=self._camera.cfg.height,
                width=self._camera.cfg.width,
                device=env.device,
            )
            self._pipeline = VisualProcessingPipeline(
                cfg.params["processors"], context, self._camera.render_buffer_specs, [self._data_type]
            )
            self._camera.request_render_inputs(
                self._pipeline.render_data_types, neutral_exposure=self._pipeline.neutral_exposure
            )
        except Exception as exc:
            try:
                self.close()
            finally:
                raise exc

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        processors: list[VisualProcessorCfg],
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
        """
        if self._pipeline is None:
            raise RuntimeError("Cannot read a closed processed_image observation term.")
        # Render, Warp processing, and the manager's Torch snapshot share a stream.
        # Enter/exit waits also cover a camera frame rendered outside this term.
        stream = (
            wp.stream_from_torch(torch.cuda.current_stream(self.device)) if self.device.startswith("cuda") else None
        )
        with wp.ScopedStream(stream, sync_enter=True, sync_exit=True):
            raw = self._camera.render_outputs
            if self._image is not None and any(
                raw.get(name) is not buffer for name, buffer in self._pipeline.render_outputs.items()
            ):
                raise RuntimeError(
                    "Camera render buffers were recreated. Recreate the environment to rebind its image observations."
                )
            if self._image is None:
                self._image = self._pipeline.allocate(raw)[self._data_type].torch
                if self._reset_mask is None:
                    self._reset_mask = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
                self._last_frames = wp.full(self.num_envs, -1, dtype=wp.int64, device=self.device)
                self._process_mask = wp.empty(self.num_envs, dtype=wp.bool, device=self.device)
                if self._normalize:
                    self._normalized = torch.empty(self._image.shape, dtype=torch.float32, device=self.device)
                    self._mean = torch.empty(
                        (self.num_envs, 1, 1, self._image.shape[-1]), dtype=torch.float32, device=self.device
                    )
                result = self._normalized if self._normalize else self._image
                self._result = result.permute(0, 3, 1, 2) if self._permute else result
            generation = self._camera.render_generation
            if generation != self._generation:
                # A group may skip several renders with different partial-view masks.
                # Compare against this term's last consumed frames, not just the latest mask.
                frames = self._camera.frame.warp
                wp.launch(
                    _find_new_camera_frames,
                    dim=self.num_envs,
                    inputs=[frames, self._last_frames, self._process_mask],
                    device=self.device,
                )
                self._pipeline.process(self._process_mask)
                if self._normalize:
                    self._normalized.copy_(self._image).div_(255.0)
                    torch.mean(self._normalized, dim=(1, 2), keepdim=True, out=self._mean)
                    self._normalized.sub_(self._mean)
                wp.copy(self._last_frames, frames)
                self._generation = generation
            return self._result

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset processor state for selected environments without reprocessing a cached frame."""
        if self._pipeline is None:
            return
        if self._reset_mask is None:
            self._reset_mask = wp.zeros(self.num_envs, dtype=wp.bool, device=self.device)
        stream = (
            wp.stream_from_torch(torch.cuda.current_stream(self.device)) if self.device.startswith("cuda") else None
        )
        with wp.ScopedStream(stream, sync_enter=True, sync_exit=True):
            if env_ids is None:
                self._reset_mask.fill_(True)
            else:
                self._reset_mask.zero_()
                wp.to_torch(self._reset_mask)[env_ids] = True
            if self._last_frames is not None:
                wp.to_torch(self._last_frames)[slice(None) if env_ids is None else env_ids] = -1
            self._pipeline.reset(self._reset_mask)

    def close(self) -> None:
        """Release owned processor state and buffers. Repeated calls are safe."""
        pipeline, self._pipeline = self._pipeline, None
        self._image = self._result = self._normalized = self._mean = self._reset_mask = None
        self._last_frames = self._process_mask = None
        if pipeline is not None:
            pipeline.close()
