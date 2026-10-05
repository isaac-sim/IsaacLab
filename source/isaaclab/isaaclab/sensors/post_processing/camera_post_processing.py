# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera binding, new-frame tracking, and reset handling for sensor post-processing chains."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import warp as wp

from ...utils.warp import ProxyArray
from .post_processor import CameraPostProcessorContext, SensorPostProcessingPipeline

if TYPE_CHECKING:
    from ..camera import Camera
    from .post_processor_cfg import SensorPostProcessorCfg


@wp.kernel
def _find_new_camera_frames(
    current: wp.array(dtype=wp.int64),
    previous: wp.array(dtype=wp.int64),
    reset_generations: wp.array(dtype=wp.int64),
    generation: wp.int64,
    mask: wp.array(dtype=wp.bool),
):
    index = wp.tid()
    mask[index] = generation > reset_generations[index] and current[index] != previous[index]


@wp.kernel
def _record_processed_camera_frames(
    current: wp.array(dtype=wp.int64), previous: wp.array(dtype=wp.int64), mask: wp.array(dtype=wp.bool)
):
    index = wp.tid()
    if mask[index]:
        previous[index] = current[index]


class CameraPostProcessingChain:
    """Run an ordered post-processor chain on a camera's outputs once per new frame.

    Construct the chain after the camera is spawned and before simulation reset, so processors can
    request renderer signals such as unexposed radiance. Call :meth:`update` after rendering to
    process views that received a new capture, :meth:`reset` when environments reset, and
    :meth:`close` to release processor state. Each chain owns its processors, intermediate buffers,
    and state, even when several chains share a camera. Camera outputs keep their raw renderer
    buffers; processed buffers are available through :attr:`outputs`.

    :class:`isaaclab.envs.mdp.processed_image` wraps this chain as an observation term. Use the chain
    directly to post-process camera outputs outside observations.

    Args:
        camera: Source camera. It must not be initialized yet.
        processors: Ordered processor configurations.
        outputs: Output names to expose, including processor-produced names.
        num_views: Number of camera views (environments).
        device: Device of the camera buffers.
        stage: USD stage used for processor configuration discovery.
    """

    def __init__(
        self,
        camera: Camera,
        processors: list[SensorPostProcessorCfg],
        outputs: list[str],
        *,
        num_views: int,
        device: str,
        stage: Any,
    ):
        self._camera = camera
        self._num_views = num_views
        self._device = device
        self._pipeline: SensorPostProcessingPipeline | None = None
        self._outputs: dict[str, ProxyArray] | None = None
        self._generation = -1
        self._reset_mask: wp.array | None = None
        self._last_frames: wp.array | None = None
        self._reset_generations: wp.array | None = None
        self._process_mask: wp.array | None = None
        try:
            context = CameraPostProcessorContext(
                stage=stage,
                camera_prim_paths=camera.camera_prim_paths,
                num_views=num_views,
                height=camera.cfg.height,
                width=camera.cfg.width,
                device=device,
            )
            self._pipeline = SensorPostProcessingPipeline(processors, context, camera.render_buffer_specs, outputs)
            camera.request_render_inputs(self._pipeline.render_data_types)
        except Exception as exc:
            try:
                self.close()
            finally:
                raise exc

    @property
    def outputs(self) -> dict[str, ProxyArray] | None:
        """Persistent processed buffers, or ``None`` before the first :meth:`update`."""
        return self._outputs

    def update(self) -> bool:
        """Bind camera buffers on first use and process views with a new capture.

        Processing runs once per published camera capture, on the current Torch stream for CUDA
        devices. Views reset since their last capture keep their previous output until a capture
        published after the reset is available.

        Returns:
            Whether a new capture was processed.

        Raises:
            RuntimeError: If the chain is closed, or the camera recreated its render buffers.
        """
        if self._pipeline is None:
            raise RuntimeError("Cannot update a closed camera post-processing chain.")
        # Render, Warp processing, and Torch consumers share a stream. Enter/exit waits also
        # cover a camera frame rendered outside this chain.
        with wp.ScopedStream(self._stream(), sync_enter=True, sync_exit=True):
            raw = self._camera.render_outputs
            if self._outputs is not None and any(
                raw.get(name) is not buffer for name, buffer in self._pipeline.render_outputs.items()
            ):
                raise RuntimeError(
                    "Camera render buffers were recreated. Recreate the post-processing chain to rebind them."
                )
            if self._outputs is None:
                self._outputs = self._pipeline.allocate(raw)
                if self._reset_mask is None:
                    self._reset_mask = wp.zeros(self._num_views, dtype=wp.bool, device=self._device)
                self._last_frames = wp.full(self._num_views, -1, dtype=wp.int64, device=self._device)
                if self._reset_generations is None:
                    self._reset_generations = wp.full(self._num_views, -1, dtype=wp.int64, device=self._device)
                self._process_mask = wp.empty(self._num_views, dtype=wp.bool, device=self._device)
            generation = self._camera.render_generation
            if generation == self._generation:
                return False
            # A consumer may skip several renders with different partial-view masks.
            # Compare against this chain's last consumed frames, not just the latest mask.
            frames = self._camera.render_frame.warp
            wp.launch(
                _find_new_camera_frames,
                dim=self._num_views,
                inputs=[frames, self._last_frames, self._reset_generations, generation, self._process_mask],
                device=self._device,
            )
            self._pipeline.process(self._process_mask)
            # Retain reset sentinels until a post-reset capture is processed: episode frame
            # numbers can repeat, and an unconsumed pre-reset capture must not consume them.
            wp.launch(
                _record_processed_camera_frames,
                dim=self._num_views,
                inputs=[frames, self._last_frames, self._process_mask],
                device=self._device,
            )
            self._generation = generation
            return True

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Reset processor state for selected views without reprocessing a retained capture."""
        if self._pipeline is None:
            return
        if self._reset_mask is None:
            self._reset_mask = wp.zeros(self._num_views, dtype=wp.bool, device=self._device)
        with wp.ScopedStream(self._stream(), sync_enter=True, sync_exit=True):
            if self._reset_generations is None:
                self._reset_generations = wp.full(self._num_views, -1, dtype=wp.int64, device=self._device)
            indices = slice(None) if env_ids is None else env_ids
            wp.to_torch(self._reset_generations)[indices] = self._camera.render_generation
            if env_ids is None:
                self._reset_mask.fill_(True)
                if self._outputs is not None:
                    self._generation = self._camera.render_generation
            else:
                self._reset_mask.zero_()
                wp.to_torch(self._reset_mask)[env_ids] = True
            if self._last_frames is not None:
                wp.to_torch(self._last_frames)[indices] = -1
            self._pipeline.reset(self._reset_mask)

    def close(self) -> None:
        """Release processor state and buffers. Repeated calls are safe."""
        pipeline, self._pipeline = self._pipeline, None
        self._outputs = self._reset_mask = self._last_frames = None
        self._reset_generations = self._process_mask = None
        if pipeline is not None:
            pipeline.close()

    def _stream(self) -> wp.Stream | None:
        if self._device.startswith("cuda"):
            return wp.stream_from_torch(torch.cuda.current_stream(self._device))
        return None
