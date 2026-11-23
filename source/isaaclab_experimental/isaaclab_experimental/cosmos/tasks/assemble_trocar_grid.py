# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coordinate the trocar cameras around one native single-frame Cosmos stream per grid batch."""

from __future__ import annotations

import os
from contextlib import closing
from dataclasses import MISSING
from itertools import count

import torch
import warp as wp

from isaaclab.sim import BackendCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.warp import ProxyArray

from ..client.cosmos_model import (
    CosmosModel,
    _CameraGroup,
    _CameraStream,
    _CosmosStream,
    _validate_resets,
    service_endpoints,
    split_views,
)
from ..client.cosmos_model_cfg import CosmosModelCfg
from ..client.cosmos_processor import CosmosTransferProcessor
from ..client.cosmos_processor_cfg import CosmosTransferProcessorCfg

TROCAR_COSMOS_CAMERAS = ("front_camera", "left_wrist_camera", "right_wrist_camera")


@configclass
class TrocarCosmosModelCfg(CosmosModelCfg):
    """Cosmos settings for the trocar camera grid."""

    class_type: type | str = "{DIR}.assemble_trocar_grid:TrocarCosmosModel"
    max_batch_chunk: int = 16
    """Maximum number of single-frame environment grids generated together on each service."""


@configclass
class _GridGroupCfg(BackendCfg):
    """Simulation-owned camera coordination, separate from the resident Cosmos client."""

    class_type: type | str = "{DIR}.assemble_trocar_grid:_GridCameraGroup"
    model: CosmosModelCfg = MISSING


@configclass
class TrocarGridProcessorCfg(CosmosTransferProcessorCfg):
    """Publish a panel of the three-camera trocar grid with its rendered foreground."""

    func: type | str = "{DIR}.assemble_trocar_grid:TrocarGridProcessor"
    foreground_color: tuple[int, int, int] = (255, 255, 255)

    def validate_config(self):
        """Use the native processor for AR, and request the render buffers needed by the single-frame grid."""
        super().validate_config()
        if self.backend.generation_mode == "autoregressive":
            self.func = CosmosTransferProcessor
            self.inputs = [name for name in self.inputs if name != "semantic_segmentation"]
        else:
            self.func = TrocarGridProcessor
            for name in ("rgb", "semantic_segmentation"):
                if name not in self.inputs:
                    self.inputs.append(name)


class TrocarGridProcessor(CosmosTransferProcessor):
    """Collect one panel, then publish the corresponding crop when all cameras are ready."""

    def __init__(self, cfg: TrocarGridProcessorCfg, data: dict[str, ProxyArray]):
        super().__init__(cfg, data)
        self._grid_group.processors[cfg.camera] = self
        self._foreground_color = data["rgb"].torch.new_tensor(cfg.foreground_color)

    def _open_stream(self, model: CosmosModel, num_views: int, seeds: tuple[int, ...]) -> _CameraStream:
        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("The trocar camera grid requires a scene with all three camera sensors.")
        cfg = _GridGroupCfg(model=self._cfg.backend)
        group = sim.get_or_create_backend(cfg, model=model)
        if group._closed:
            sim.close_backend(group)
            group = sim.get_or_create_backend(cfg, model=model)
        stream = group.join(num_views, seeds)
        self._grid_group = group
        return stream

    def _generate_single_frame(self, data: dict[str, ProxyArray], env_mask: wp.array) -> dict[str, ProxyArray]:
        frames = self._controls(data, env_mask)
        self._source_rgb = data["rgb"].torch[..., :3]
        self._segmentation = data["semantic_segmentation"].torch
        self._captured = wp.to_torch(env_mask).to(self._output.device).view(-1, 1, 1, 1)
        resets, seeds, self._grid_episodes = self._take_resets()
        self._publish(self._stream.step(list(frames.split(1)), resets, seeds))
        return dict(self._outputs)

    def foreground_mask(self) -> torch.Tensor:
        """Return the preserved pixels of this capture in all environments."""
        return (self._segmentation[..., :3] == self._foreground_color).all(dim=-1)

    @torch.no_grad()
    def _publish(self, generated: list[torch.Tensor] | None, chunk: list[int] | None = None) -> None:
        if generated is None:
            return  # The last camera will publish this panel.
        pixels = self._source_rgb
        if generated:
            pixels = torch.where(self.foreground_mask().unsqueeze(-1), pixels, torch.cat(generated).to(pixels.device))
        torch.where(self._captured, pixels, self._output, out=self._output)
        self._resize()


class TrocarCosmosModel(CosmosModel):
    """Generate independent trocar grids in bounded batches using the native service."""

    def __init__(self, cfg: TrocarCosmosModelCfg):
        # RLinf exports the environment worker's rank to its simulator subprocess.
        super().__init__(cfg)
        self._rlinf_worker = "RLINF_CONFIG_FILE" in os.environ and "WORKER_NAME" in os.environ
        if cfg.generation_mode == "single_frame" and self._rlinf_worker and int(os.environ["WORLD_SIZE"]) > 1:
            endpoints = service_endpoints(cfg.endpoint)
            self._cfg = self._cfg.replace(endpoint=endpoints[int(os.environ["RANK"]) % len(endpoints)])

    def _open_session(self, num_views: int, seeds: tuple[int, ...]):
        if self._cfg.generation_mode == "autoregressive":
            return super()._open_session(num_views, seeds)
        if type(self._cfg.max_batch_chunk) is not int or self._cfg.max_batch_chunk < 1:
            raise ValueError("Trocar max_batch_chunk must be a positive integer.")
        if not self._rlinf_worker and num_views <= self._cfg.max_batch_chunk * len(
            service_endpoints(self._cfg.endpoint)
        ):
            return super()._open_session(num_views, seeds)
        with self._lock:
            if self._closed:
                raise RuntimeError("Cosmos client model is closed.")
            stream = _GridBatches(self, seeds, self._cfg.max_batch_chunk)
            self._streams.add(stream)
            return stream


class _GridBatches(_CosmosStream):
    """Keep native per-view seed and prompt state across short-lived single-frame service sessions."""

    def __init__(self, model: CosmosModel, seeds: tuple[int, ...], batch_size: int):
        self._endpoints = service_endpoints(model._cfg.endpoint)
        self._shares = split_views(len(seeds), len(self._endpoints))
        super().__init__(model, self._endpoints[0], seeds, range(len(seeds)), batched=len(seeds) > 1)
        self._batch_size = batch_size
        self._prompts = self._session_prompt()

    def step(
        self,
        controls: list[torch.Tensor],
        reset_rows: tuple[int, ...],
        seeds: tuple[int, ...],
        episodes: tuple[int, ...] | None = None,
    ) -> list[torch.Tensor]:
        """Generate one grid per environment and preserve their order across replicas."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Trocar grid stream is closed.")
            try:
                if len(controls) != len(self._seeds):
                    raise ValueError("Expected one trocar grid per environment.")
                _validate_resets(reset_rows, seeds, episodes, len(controls))
                return self._generate_batches(controls, reset_rows, seeds, episodes)
            except Exception:
                self._release(notify=False)
                raise

    def _generate_batches(
        self,
        controls: list[torch.Tensor],
        reset_rows: tuple[int, ...],
        seeds: tuple[int, ...],
        episodes: tuple[int, ...] | None,
    ) -> list[torch.Tensor]:
        # Advance the camera's native single-frame schedule once, independent of batch boundaries.
        frame = self._single_frame_step([1] * len(controls), reset_rows, seeds, episodes)
        self._prompts = frame.get("prompt", self._prompts)
        # Initialization and evaluation can capture outside the training rollout.
        self._model.activate()
        images = list(controls)
        for start in range(0, len(self._shares[0]), self._batch_size):
            slices = [share[start : start + self._batch_size] for share in self._shares]
            rows = [row for share in slices for row in share]
            endpoints = [endpoint for endpoint, share in zip(self._endpoints, slices, strict=True) if share]
            prompts = [self._prompts[i] for i in rows] if isinstance(self._prompts, list) else self._prompts
            settings = self._cfg.replace(endpoint=endpoints, prompt=prompts, single_frame_prompt_by_seed=False)
            batch_seeds = tuple(frame["seeds"][i] for i in rows)
            # Closing the session releases its buffers and lets the RLinf driver page the idle service.
            with closing(CosmosModel(settings)) as client:
                stream = client.open_stream(len(rows), batch_seeds)
                generated = stream.step([controls[i] for i in rows], (), ())
                for row, image in zip(rows, generated, strict=True):
                    images[row] = image
        return images


class _GridCameraGroup(_CameraGroup):
    """Generate one row-major 2x2 canvas per environment. An empty result publishes the rendered capture."""

    def __init__(self, cfg: _GridGroupCfg, model: CosmosModel):
        super().__init__(model, TROCAR_COSMOS_CAMERAS)
        self.processors: dict[str, TrocarGridProcessor] = {}
        self._settings = cfg.model
        self.policy_step = count().__next__  # Standalone camera processors count captures until a rollout begins.
        self.last_decision_step: int | None = None
        self._augment = False
        self._pending_resets: dict[int, tuple[int, int]] = {}

    def close(self) -> None:
        """Release the camera callbacks and their GPU buffers when the grid closes."""
        super().close()
        self.processors.clear()
        self._streams.clear()

    def _generate(self) -> dict[int, list[torch.Tensor]]:
        processors = [self.processors[name] for name in self._cameras]
        first = processors[0]
        _, rows, seeds = self._chunks[0]
        for index, processor in enumerate(processors[1:], start=1):
            if (
                processor._seeds != first._seeds
                or self._chunks[index][1:] != (rows, seeds)
                or processor._grid_episodes != first._grid_episodes
            ):
                raise ValueError("Grid cameras must share their environment seeds and episode resets.")
        for row, seed, episode in zip(rows, seeds, first._grid_episodes, strict=True):
            self._pending_resets[row] = (seed, episode)
        cfg = self._settings
        step = self.policy_step()
        if step < 0 or step % cfg.decision_interval:
            return {index: [] for index in self._chunks}
        if step != self.last_decision_step:
            self._augment = first._random.random() < cfg.augmentation_probability
            self.last_decision_step = step
        if not self._augment:
            return {index: [] for index in self._chunks}
        if any(not torch.equal(processor._captured, first._captured) for processor in processors[1:]):
            raise ValueError("Grid cameras must capture the same environments together.")
        if self._session is None:
            self._session = self._model.open_stream(self._num_views, first._seeds)
        grids = []
        for panels in zip(*(self._chunks[i][0] for i in range(len(self._cameras))), strict=True):
            panels = list(panels)
            panels.extend(torch.zeros_like(panels[0]) for _ in range(4 - len(panels)))
            grids.append(torch.cat((torch.cat(panels[:2], dim=2), torch.cat(panels[2:], dim=2)), dim=1))
        rows = tuple(sorted(self._pending_resets))
        images = self._session.step(
            grids,
            rows,
            tuple(self._pending_resets[r][0] for r in rows),
            episodes=tuple(self._pending_resets[r][1] for r in rows),
        )
        self._pending_resets.clear()
        if len(images) != len(grids) or any(
            im.shape != grid.shape or im.dtype != torch.uint8 for im, grid in zip(images, grids)
        ):
            raise ValueError("Cosmos must return one RGB grid matching each environment's control.")
        height, width = self._chunks[0][0][0].shape[1:3]
        return {
            k: [im[:, k // 2 * height : (k // 2 + 1) * height, k % 2 * width : (k % 2 + 1) * width] for im in images]
            for k in range(len(self._cameras))
        }
