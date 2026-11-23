# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assemble-trocar tasks with Cosmos images on the front and both wrist cameras."""

import math
from contextlib import contextmanager

import torch
import warp as wp

from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import add_labels, clone
from isaaclab.utils import configclass
from isaaclab.utils.warp import ProxyArray

from isaaclab_tasks.contrib.assemble_trocar.g129_dex3_env_cfg import (
    AssembleTrocarSceneCfg,
    EventCfgFixTrayRotation,
    G1AssembleTrocarEnvCfg,
)

from isaaclab_experimental.image_transfer import DepthControlProcessor, DepthControlProcessorCfg, depth_to_control

from .. import CosmosModelCfg, CosmosTransferProcessorCfg, cosmos_camera, episode_frame_budget
from .assemble_trocar_grid import TROCAR_COSMOS_CAMERAS, TrocarCosmosModelCfg, TrocarGridProcessorCfg
from .prompts.assemble_trocar import TROCAR_GRID_PROMPTS, TROCAR_PROMPTS


def _trocar_depth_to_control(
    data: torch.Tensor,
    quantiles: tuple[float, float] | None = (0.02, 0.85),
    near: float = 0.1,
    far: float = 2.0,
) -> torch.Tensor:
    """Use rl2real's per-image robust depth mapping, or the native metric mapping with ``quantiles=None``.

    Camera controls are NHWC uint8; statistics use only finite positive depths in each image. No image is
    copied to the host. Quantile selection has a per-image synchronization cost, unlike fixed metric bounds.
    """
    if quantiles is None:
        return depth_to_control(data, near=near, far=far)
    if len(quantiles) != 2 or not all(math.isfinite(q) for q in quantiles) or not 0 <= quantiles[0] < quantiles[1] <= 1:
        raise ValueError("Depth quantiles must satisfy 0 <= lower < upper <= 1.")
    controls = torch.zeros((*data.shape[:-1], 3), dtype=torch.uint8, device=data.device)
    for index, image in enumerate(data[..., 0].float()):
        valid = torch.isfinite(image) & (image > 0)
        if not valid.any():
            continue
        values = image[valid]
        lower, upper = torch.quantile(values, torch.tensor(quantiles, device=image.device, dtype=image.dtype)).unbind()
        if upper <= lower + 1e-8:
            lower, upper = values.min(), values.max()
        pixels = torch.where(valid, 1 - (image - lower) / (upper - lower).clamp_min(1e-8), 0)
        controls[index] = pixels.clamp(0, 1).mul(255).round().to(torch.uint8).unsqueeze(-1)
    return controls


class TrocarDepthControlProcessor(DepthControlProcessor):
    """Write quantile-normalized depth into the native control buffer."""

    @torch.no_grad()
    def __call__(self, data: dict[str, ProxyArray], env_mask: wp.array | None = None) -> dict[str, ProxyArray]:
        cfg = self._cfg
        self._output.copy_(_trocar_depth_to_control(data[cfg.inputs[0]].torch, cfg.quantiles, cfg.near, cfg.far))
        return dict(self._outputs)


@configclass
class TrocarDepthControlCfg(DepthControlProcessorCfg):
    """Use finite per-image depth quantiles or fixed metric bounds."""

    func: type = TrocarDepthControlProcessor
    quantiles: tuple[float, float] | None = (0.02, 0.85)
    near: float = 0.1
    far: float = 2.0


_FOREGROUND_CLASSES = ("robot", "tray", "trocar", "trocar_device", "cart", "mayo_table")
_FOREGROUND_COLOR = (255, 255, 255)


@clone
def _spawn_labelled_room(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Label the two pieces of furniture on the source room before it is cloned."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd_file

    prim = spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    for name, label in (("Cart001", "cart"), ("InstrumentTrolley002", "mayo_table")):
        child = prim.GetChild(name)
        if child:
            add_labels(child, [label])
    return prim


def _cosmos_scene() -> AssembleTrocarSceneCfg:
    """Keep camera poses and output sizes, with synchronized captures from all three cameras."""
    scene = AssembleTrocarSceneCfg(num_envs=1, env_spacing=6.0, lazy_sensor_update=True)
    for name, label in (("robot", "robot"), ("tray", "tray"), ("trocar_1", "trocar"), ("trocar_2", "trocar_device")):
        getattr(scene, name).spawn.semantic_tags = [("class", label)]
    scene.scene.spawn.func = _spawn_labelled_room
    model = TrocarCosmosModelCfg(
        generation_mode="single_frame",
        decision_interval=16,
        augmentation_probability=0.5,
        single_frame_prompt_by_seed=True,
        prompt=list(TROCAR_GRID_PROMPTS),
        fps=30.0,
    )
    for name in TROCAR_COSMOS_CAMERAS:
        camera = getattr(scene, name).replace(update_period=1 / 30)
        wrapped = cosmos_camera(camera, model, control=[TrocarDepthControlCfg()], canvas=(640, 640), name=name)
        wrapped.width = wrapped.height = 320
        wrapped.tensor_processors[-1] = TrocarGridProcessorCfg(
            backend=model, camera=name, image_size=(camera.height, camera.width), foreground_color=_FOREGROUND_COLOR
        )
        setattr(scene, name, wrapped)
    return scene


@configclass
class G1AssembleTrocarCosmosEnvCfg(G1AssembleTrocarEnvCfg):
    """The trocar policy consumes generated RGB through its existing camera observations.

    Uses rl2real's 30 Hz control and 25-second horizon. AR shares one service across cameras.
    Single-frame mode tiles all three cameras on one canvas and restores their semantic foreground.
    """

    scene: AssembleTrocarSceneCfg = _cosmos_scene()
    cosmos_offload_during_update: bool = True
    """Page the single-frame services to CPU while the RLinf actor updates its weights."""

    @property
    def cosmos(self) -> CosmosModelCfg:
        """Shared settings, exposed for overrides such as ``env.cosmos.endpoint=...``."""
        return self._cosmos_transfers()[0].backend

    @cosmos.setter
    def cosmos(self, settings: CosmosModelCfg):
        for transfer in self._cosmos_transfers():
            transfer.backend = settings

    @property
    def cosmos_seed(self) -> int | list[int]:
        """One seed, or one seed per environment, applied to all three cameras."""
        return self._cosmos_transfers()[0].seed

    @cosmos_seed.setter
    def cosmos_seed(self, seed: int | list[int]):
        for transfer in self._cosmos_transfers():
            transfer.seed = seed

    @property
    def cosmos_depth_quantiles(self) -> tuple[float, float] | None:
        """Depth percentiles for every camera; ``None`` uses each control's metric ``near`` and ``far``."""
        return self.scene.front_camera.tensor_processors[0].quantiles

    @cosmos_depth_quantiles.setter
    def cosmos_depth_quantiles(self, quantiles: tuple[float, float] | None):
        for name in TROCAR_COSMOS_CAMERAS:
            getattr(self.scene, name).tensor_processors[0].quantiles = quantiles

    def validate_config(self):
        """Apply final overrides to all cameras and size their shared session for the episode."""
        cameras = [getattr(self.scene, name) for name in TROCAR_COSMOS_CAMERAS]
        if any(camera.update_period != cameras[0].update_period for camera in cameras[1:]):
            raise ValueError("Trocar Cosmos cameras sharing a session must have the same update_period.")
        model = self.cosmos
        transfers = self._cosmos_transfers()
        grid = model.generation_mode == "single_frame"
        # Single-frame camera synchronization belongs to the grid processor. The model sees one canvas per env.
        model.cameras = () if grid else TROCAR_COSMOS_CAMERAS
        if isinstance(model.prompt, (tuple, list)) and tuple(model.prompt) in (TROCAR_PROMPTS, TROCAR_GRID_PROMPTS):
            model.prompt = list(TROCAR_GRID_PROMPTS if grid else TROCAR_PROMPTS)
        if not grid and model.max_episode_frames is None:
            model.max_episode_frames = episode_frame_budget(self, cameras[0].update_period)[2]
        self.cosmos = model
        for camera in cameras:
            # The square optics are unchanged; a single-frame panel occupies one quarter of the service canvas.
            camera.width = camera.height = 320 if grid else 640
            if grid:
                renderer = camera.renderer_cfg
                # Directly constructed tasks can still hold backend presets before Hydra selects a renderer.
                renderers = (
                    [renderer]
                    if hasattr(renderer, "renderer_type")
                    else [
                        getattr(renderer, name) for name in ("default", "isaacsim_rtx", "newton_warp", "newton_optix")
                    ]
                )
                for backend in renderers:
                    if not hasattr(backend, "semantic_segmentation_mapping"):
                        raise ValueError(
                            f"Renderer {backend.renderer_type!r} needs semantic color mapping for trocar foreground "
                            "compositing. Choose isaacsim_rtx or newton_warp."
                        )
                    backend.colorize_semantic_segmentation = True
                    backend.semantic_segmentation_mapping = {
                        **{f"class:{label}": (*_FOREGROUND_COLOR, 255) for label in _FOREGROUND_CLASSES},
                        "class:BACKGROUND": (0, 0, 0, 255),
                        "class:UNLABELLED": (0, 0, 0, 255),
                    }
        # Child configs validate before the task. Recompute the RGB inputs after propagating overrides.
        for transfer in transfers:
            transfer.validate_config()
        self.cosmos_seed = self.cosmos_seed
        self.cosmos_depth_quantiles = self.cosmos_depth_quantiles

    def _cosmos_transfers(self) -> list[CosmosTransferProcessorCfg]:
        """Return the transfer step of each camera in session order."""
        return [
            next(
                processor
                for processor in getattr(self.scene, name).tensor_processors
                if isinstance(processor, CosmosTransferProcessorCfg)
            )
            for name in TROCAR_COSMOS_CAMERAS
        ]


@configclass
class G1AssembleTrocarCosmosEvalEnvCfg(G1AssembleTrocarCosmosEnvCfg):
    """Keep the original Eval task's deterministic per-environment tray yaw."""

    events: EventCfgFixTrayRotation = EventCfgFixTrayRotation()


@contextmanager
def offload_trocar_cosmos(env_cfg: G1AssembleTrocarCosmosEnvCfg):
    """Release GPU weights while the owning rollout process pauses captures for a learner update."""
    if env_cfg.cosmos.generation_mode != "single_frame":
        raise ValueError("Offloading an open trocar rollout requires single_frame generation.")
    model = SimulationContext.instance().get_or_create_backend(env_cfg.cosmos)
    model.offload()
    try:
        yield
    finally:
        model.activate()
