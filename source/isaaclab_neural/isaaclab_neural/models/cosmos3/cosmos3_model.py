# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos3 inference behind the :class:`~isaaclab_neural.world_model.WorldModel` interface.

Only visual dynamics live here. The robot's joint mapping is supplied by the task through
:class:`~.cosmos3_cfg.JointMappingCfg`.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import torch
from omegaconf import DictConfig, flag_override

from isaaclab.utils.configclass import instantiate

from ...world_model import WorldModel, WorldModelSpec

if TYPE_CHECKING:
    from pathlib import Path

    from cosmos_framework.inference.bimanual import BimanualActionContract
    from cosmos_framework.inference.bimanual_session import BimanualSession
    from cosmos_framework.inference.inference import OmniInference

    from .cosmos3_cfg import Cosmos3Cfg

NUM_CONTROLS = 59
"""Number of ``unified_v1`` controls; the model pads them to 64."""


def _allow_saved_config_rewrites() -> None:
    """Let the Framework update its saved config under OmegaConf 2.4.

    While building the network, the Framework normalizes and upgrades the saved ``multiview_attention`` config in
    place. OmegaConf 2.4 marks that config read-only, so the writes raise; with OmegaConf 2.3 they succeeded. The
    writes only fill defaults and normalize values, so each call runs with the read-only flag lifted.
    """
    module = importlib.import_module("cosmos_framework.configs.base.defaults.multiview_attention")
    for name in ("normalize_multiview_temporal_windows", "upgrade_saved_multiview_attention"):
        original = getattr(module, name)

        def writable(config, *args, _original=original, **kwargs):
            if isinstance(config, DictConfig):
                with flag_override(config, "readonly", False):
                    return _original(config, *args, **kwargs)
            return _original(config, *args, **kwargs)

        # Modules that imported the function by name keep their own reference, so replace those too.
        for other in list(sys.modules.values()):
            if getattr(other, name, None) is original:
                setattr(other, name, writable)


class Cosmos3Model(WorldModel):
    """Share one set of Cosmos3 weights across independent environments.

    ``window`` mode predicts every environment's chunk from its own last generated image. ``session`` mode drives
    one native causal session and supports a single environment.
    """

    cfg: Cosmos3Cfg

    def __init__(self, cfg: Cosmos3Cfg, num_envs: int):
        super().__init__(cfg, num_envs)
        self.pipe: OmniInference | None = None
        self.session: BimanualSession | None = None
        self.contract: BimanualActionContract | None = None
        self.mapping = None
        self.condition: list[torch.Tensor | None] = [None] * num_envs
        self.prompts = [""] * num_envs
        self.seeds = np.zeros(num_envs, np.int64)
        self.elapsed = np.zeros(num_envs, np.int64)

    def spec(self) -> WorldModelSpec:
        cfg = self.cfg
        return WorldModelSpec(
            chunk_length=cfg.chunk_size,
            fps=cfg.fps,
            image_size=cfg.image_size,
            max_envs=1 if cfg.mode == "session" else None,
        )

    # ---------------------------------------------------------------- loading

    def load(self) -> None:
        """Load a local HF export, or a DCP checkpoint with its saved YAML config."""
        if self.pipe is not None:
            return
        # The backend samples frozen dynamics; the Framework default enables its separate training stack on import.
        os.environ.setdefault("COSMOS_TRAINING", "0")
        from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
        from cosmos_framework.inference.args import OmniSetupOverrides
        from cosmos_framework.inference.bimanual import BimanualActionContract

        _allow_saved_config_rewrites()
        cfg = self.cfg
        kwargs = {"config_file": cfg.config_file} if cfg.config_file is not None else {}
        setup = OmniSetupOverrides(
            checkpoint_path=cfg.checkpoint_path,
            output_dir=cfg.output_dir,
            use_ema_weights=cfg.use_ema_weights,
            use_torch_compile=False,
            use_cuda_graphs=False,
            diffusion_cache=False,
            # The backend calls the model directly, which skips the guardrails of the Framework's own pipeline.
            guardrails=False,
            dp_replicate_size=1,
            dp_shard_size=1,
            cp_size=1,
            cfgp_size=1,
            **kwargs,
        ).build_setup()
        self.pipe = setup.get_inference_cls().create(setup)
        self.contract = getattr(self.pipe.model, "_bimanual_action_contract", None) or BimanualActionContract(
            NUM_CONTROLS, 64, {cfg.domain_name: get_domain_id(cfg.domain_name)}
        )

    # ----------------------------------------------------------------- frames

    def _pixels(self, rgb: torch.Tensor) -> np.ndarray:
        """Convert decoded RGB in [0, 1], shape ``(1, 3, T, H, W)``, to ``uint8`` frames ``(1, T, H, W, 3)``."""
        # Match the native public API before using pixels as the next condition.
        return rgb.detach().cpu().clamp(0, 1).mul(255).to(torch.uint8).permute(0, 2, 3, 4, 1).numpy()

    def _batch(self, condition: torch.Tensor, controls: np.ndarray, prompt: str) -> dict:
        from cosmos_framework.inference.action import build_action_batch
        from cosmos_framework.inference.args import ModelMode
        from cosmos_framework.inference.inference import _finalize_data_batch

        cfg = self.cfg
        payload = {"action_space": "normalized_unified_v1", "action": controls.tolist()}
        values, domains = self.contract.validate(payload, action_count=cfg.chunk_size, domain_name=cfg.domain_name)
        padded = torch.zeros(cfg.chunk_size, 64, dtype=torch.float32)
        padded[:, :NUM_CONTROLS] = torch.tensor(values, dtype=torch.float32)
        batch = build_action_batch(
            video=condition,
            action=padded,
            raw_action_dim=NUM_CONTROLS,
            prompt=prompt,
            view_point="ego_view",
            domain_name=cfg.domain_name,
            model_mode=ModelMode.FORWARD_DYNAMICS,
            action_chunk_size=cfg.chunk_size,
            fps=cfg.fps,
            resolution=cfg.resolution,
            input_video_key=self.pipe.model.input_video_key,
            device=self.pipe.model.tensor_kwargs["device"],
            action_domain_ids=torch.tensor(domains, dtype=torch.long),
            format_prompt_as_json=False,
        )
        return _finalize_data_batch(batch, 1, self.pipe.model)

    # ------------------------------------------------------------- interface

    @torch.inference_mode()
    def reset(self, env_ids: np.ndarray, images: list[Path], prompts: list[str], seeds: np.ndarray) -> np.ndarray:
        """Reset the selected rows from real images; other rows keep their generated conditions."""
        self.load()
        env_ids = np.asarray(env_ids)
        if self.cfg.mode == "session":
            if self.session is not None:
                self.session.close()
            self.session = self.pipe.open_bimanual_session(
                vision_path=images[0],
                prompt=prompts[0],
                domain_name=self.cfg.domain_name,
                fps=self.cfg.fps,
                seed=int(seeds[0]),
                resolution=self.cfg.resolution,
            )
            frames = self._pixels(self.session.observation)
        else:
            from cosmos_framework.inference.vision import read_media_frames

            prepared = []
            for image in images:
                condition, _ = read_media_frames(image, max_frames=1)
                prepared.append(self._pixels(condition[None].float() / 255))
            frames = np.concatenate(prepared)
        for env, frame, prompt, seed in zip(env_ids.tolist(), frames, prompts, np.asarray(seeds), strict=True):
            self.condition[env] = torch.from_numpy(frame[-1].copy()).permute(2, 0, 1)[:, None]
            self.prompts[env] = prompt
            self.seeds[env] = seed
            self.elapsed[env] = 0
        return frames[:, -1]

    def step_controls(self, controls: np.ndarray) -> np.ndarray:
        """Predict the next chunk from normalized ``unified_v1`` controls shaped ``(N, T, 59)``.

        Replay of recorded controls uses this directly; the manager goes through :meth:`step`.
        """
        cfg = self.cfg
        with torch.inference_mode():
            if cfg.mode == "session":
                rgb = self.session.step({"action_space": "normalized_unified_v1", "action": controls[0].tolist()})
                frames = self._pixels(rgb)
            else:
                chunks = []
                for env in range(self.num_envs):
                    batch = self._batch(self.condition[env], controls[env], self.prompts[env])
                    result = self.pipe.model.generate_samples_from_batch(
                        batch,
                        guidance=cfg.guidance,
                        shift=10.0 if cfg.shift is None else cfg.shift,
                        seed=[int((self.seeds[env] + self.elapsed[env]) % 2**32)],
                        num_steps=cfg.num_inference_steps or 30,
                        n_sample=1,
                    )
                    rgb = (self.pipe.model.decode(result["vision"][0]) + 1) / 2
                    chunks.append(self._pixels(rgb)[:, 1:])
                frames = np.concatenate(chunks)
        for env, frame in enumerate(frames):
            self.condition[env] = torch.from_numpy(frame[-1].copy()).permute(2, 0, 1)[:, None]
        self.elapsed += cfg.chunk_size
        return frames

    def step(self, joint_targets: np.ndarray, start_joint_state: np.ndarray) -> np.ndarray:
        """Convert joint targets to controls with the task's joint mapping, then predict the chunk."""
        if self.mapping is None:
            self.mapping = instantiate(self.cfg.joint_mapping)
        controls = np.stack(
            [self.mapping.encode(joint_targets[i], start_joint_state[i]) for i in range(self.num_envs)]
        ).astype(np.float32)
        return self.step_controls(controls)

    def close(self) -> None:
        if self.session is not None:
            self.session.close()
        self.session = self.pipe = self.contract = None
