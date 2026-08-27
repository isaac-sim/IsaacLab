# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GPU visual appearance randomization terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from isaaclab.assets import VisualMaterialCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.backend_utils import FactoryBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class randomize_visual_material(ManagerTermBase):
    """Sample numeric material channels on device and issue one batched runtime renderer write.

    This term requires an initialized renderer and therefore does not support ``prestartup`` mode.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if cfg.mode == "prestartup":
            raise ValueError("Visual-material writes do not support prestartup mode.")
        material_cfgs = cfg.params["materials"]
        material_cfgs = [material_cfgs] if isinstance(material_cfgs, SceneEntityCfg) else material_cfgs
        self._materials = [env.scene[material_cfg.name] for material_cfg in material_cfgs]
        if not self._materials:
            raise ValueError("Visual material randomization requires at least one material.")
        if not all(isinstance(material.cfg, VisualMaterialCfg) for material in self._materials):
            raise TypeError("Every material selector must resolve to a VisualMaterial asset.")
        scopes = {material.is_per_env for material in self._materials}
        if len(scopes) != 1:
            raise ValueError("Bucket and per-environment materials require separate event terms.")
        self._per_env = scopes.pop()
        self._samplers = {
            channel: _compile_distribution(spec, env.device) for channel, spec in cfg.params["channels"].items()
        }
        for channel in self._samplers:
            if not all(channel in material.channels for material in self._materials):
                raise ValueError(f"Channel {channel!r} must be declared by every selected material.")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        materials: list[SceneEntityCfg] | SceneEntityCfg,
        channels: dict[str, tuple | dict],
    ) -> None:
        del materials, channels
        if isinstance(env_ids, slice):
            env_ids = None
        count = env.scene.num_envs if env_ids is None else len(env_ids)
        shape = (len(self._materials), count) if self._per_env else (len(self._materials),)
        sampled = {channel: sampler(shape) for channel, sampler in self._samplers.items()}
        env.sim.render_context.write_visual_materials(self._materials, sampled, env_ids if self._per_env else None)


class randomize_visual_shape(FactoryBase, ManagerTermBase):
    """Randomize visual channels per selected shape on backends that expose shape storage."""

    @classmethod
    def _get_backend(cls, cfg: EventTermCfg, env: ManagerBasedEnv) -> str:
        consumers = (*env.sim.resolve_visualizer_types(), *env.sim.render_context.renderer_types)
        supported = ("newton_gl", "newton_rtx", "newton_warp")
        if consumers and all(name in supported for name in consumers):
            return "newton"
        raise NotImplementedError(
            "This renderer has no per-shape visual storage; use one VisualMaterialCfg per randomized part."
        )


def _compile_distribution(spec: Any, device: str):
    """Compile one public distribution spec into a device sampler."""
    if isinstance(spec, dict) and "choices" in spec:
        values = torch.as_tensor(spec["choices"], dtype=torch.float32, device=device)

        def sample_choices(shape):
            return values[torch.randint(len(values), shape, device=device)]

        return sample_choices
    if isinstance(spec, dict):
        low = tuple(spec[key][0] for key in ("r", "g", "b"))
        high = tuple(spec[key][1] for key in ("r", "g", "b"))
    else:
        low, high = spec
    low = torch.as_tensor(low, dtype=torch.float32, device=device)
    high = torch.as_tensor(high, dtype=torch.float32, device=device)
    span = high - low
    trailing = () if low.ndim == 0 else tuple(low.shape)

    def sample_uniform(shape):
        return torch.rand((*shape, *trailing), device=device).mul_(span).add_(low)

    return sample_uniform
