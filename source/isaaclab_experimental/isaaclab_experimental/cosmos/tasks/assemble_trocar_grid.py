# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coordinate the trocar cameras around one native single-frame Cosmos stream per grid batch."""

from __future__ import annotations

from dataclasses import MISSING
from itertools import count

import torch
import warp as wp

from isaaclab.sim import BackendCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.warp import ProxyArray

from ..client.cosmos_model import CosmosModel, _CameraGroup, _CameraStream
from ..client.cosmos_model_cfg import CosmosModelCfg
from ..client.cosmos_processor import CosmosTransferProcessor
from ..client.cosmos_processor_cfg import CosmosTransferProcessorCfg

TROCAR_COSMOS_CAMERAS = ("front_camera", "left_wrist_camera", "right_wrist_camera")


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
