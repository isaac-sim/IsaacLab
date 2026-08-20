# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime-writable visual material asset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from pxr import Sdf, UsdShade

from isaaclab import cloner
from isaaclab.assets.asset_base import AssetBase
from isaaclab.sim import SimulationContext

from .visual_material_cfg import VisualMaterialCfg

_PREVIEW_CHANNELS = {
    "color": ("diffuseColor", (0.18, 0.18, 0.18)),
    "roughness": ("roughness", 0.5),
    "metallic": ("metallic", 0.0),
    "emissive_color": ("emissiveColor", (0.0, 0.0, 0.0)),
    "opacity": ("opacity", 1.0),
}
_PBR_CHANNELS = {
    "color": ("diffuse_color_constant", (0.18, 0.18, 0.18)),
    "roughness": ("reflection_roughness_constant", 0.5),
    "metallic": ("metallic_constant", 0.0),
    "specular": ("specular_level", 0.5),
    "emissive_color": ("emissive_color", (0.0, 0.0, 0.0)),
    "emissive_intensity": ("emissive_intensity", 0.0),
    "opacity": ("opacity_constant", 1.0),
    "uv_scale": ("texture_scale", (1.0, 1.0)),
    "uv_offset": ("texture_translate", (0.0, 0.0)),
    "uv_rotate": ("texture_rotate", 0.0),
}
_GLASS_CHANNELS = {
    "color": ("glass_color", (1.0, 1.0, 1.0)),
    "roughness": ("frosting_roughness", 0.0),
    "ior": ("glass_ior", 1.491),
}


class VisualMaterial(AssetBase):
    """A visual material cloned and finalized through the normal asset lifecycle."""

    cfg: VisualMaterialCfg

    def __init__(self, cfg: VisualMaterialCfg):
        self._clone_cfg_id = id(cfg)
        if "texture" in cfg.channels:
            raise NotImplementedError("Runtime texture swaps are not part of the numeric visual-material pipeline.")
        if len(cfg.texture_pool) > 1:
            raise NotImplementedError("Runtime texture selection is unsupported; provide at most one static texture.")
        super().__init__(cfg)
        self._render_context = SimulationContext.instance().render_context
        self._source_material_path = self.cfg.spawn.spawn_path if self.cfg.spawn is not None else None
        self._source_material_path = self._source_material_path or self.cfg.prim_path
        self._is_per_env = self._source_material_path != self.cfg.prim_path
        material = UsdShade.Material(self.stage.GetPrimAtPath(self._source_material_path))
        if not material:
            raise ValueError(f"Visual material {self._source_material_path!r} is not a UsdShade.Material.")
        outputs = (material.GetSurfaceOutput("mdl"), material.GetSurfaceOutput())
        connected = next(
            (output.GetConnectedSource() for output in outputs if output and output.HasConnectedSource()), None
        )
        if connected is None:
            raise ValueError(f"Visual material {self._source_material_path!r} has no connected surface shader.")
        shader = UsdShade.Shader(connected[0].GetPrim())
        self._source_shader_path = str(shader.GetPrim().GetPath())
        channel_specs = _channel_specs(shader)
        if self.cfg.texture_pool:
            if "uv_scale" not in channel_specs:
                raise ValueError("texture_pool is only valid for an OmniPBR material.")
            shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(self.cfg.texture_pool[0]))
            if "color" in channel_specs:
                channel_specs["color"] = ("diffuse_tint", (1.0, 1.0, 1.0))

        self._input_names: dict[str, str] = {}
        self._initial_values: dict[str, torch.Tensor] = {}
        for channel in self.cfg.channels:
            if channel not in channel_specs:
                raise ValueError(
                    f"Material {self._source_material_path!r} does not support channel {channel!r}; "
                    f"available channels are {tuple(channel_specs)}."
                )
            input_name, default = channel_specs[channel]
            value_type = (
                Sdf.ValueTypeNames.Float
                if isinstance(default, float)
                else {
                    2: Sdf.ValueTypeNames.Float2,
                    3: Sdf.ValueTypeNames.Color3f,
                }[len(default)]
            )
            shader_input = shader.CreateInput(input_name, value_type)
            if shader_input.Get() is None:
                shader_input.Set(default)
            self._input_names[channel] = input_name
            self._initial_values[channel] = torch.as_tensor(shader_input.Get(), dtype=torch.float32)
        self._material_paths: tuple[str, ...] = ()
        self._shader_paths: tuple[str, ...] = ()
        self._values: dict[str, torch.Tensor] = {}
        self._offsets: dict[str, int] = {}

    @property
    def material_prim_path(self) -> str:
        """Source material prim path."""
        return self._source_material_path

    @property
    def channels(self) -> tuple[str, ...]:
        """Runtime-writable channel names."""
        return tuple(self._input_names)

    @property
    def is_per_env(self) -> bool:
        """Whether this material owns one clone per environment."""
        return self._is_per_env

    @property
    def env_material_paths(self) -> tuple[str, ...]:
        """Final material clone paths in environment order."""
        return self._material_paths if self._is_per_env else ()

    @property
    def num_instances(self) -> int:
        return len(self._material_paths)

    @property
    def data(self) -> dict[str, torch.Tensor]:
        return self._values

    @staticmethod
    def write_channels(
        materials: Sequence[VisualMaterial],
        channels: dict[str, torch.Tensor],
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Write aligned numeric channels to bucket or selected environment rows."""
        if materials:
            materials[0]._render_context.write_visual_materials(materials, channels, env_ids)

    def read_channel(self, channel: str, env_id: int | None = None) -> Any:
        """Read one buffered channel value; this diagnostic operation synchronizes to the host."""
        if self._is_per_env:
            row = 0 if env_id is None else env_id
        elif env_id is not None:
            raise ValueError("env_id is only valid for a per-environment material.")
        else:
            row = 0
        value = self._values[channel][row].detach().cpu()
        return float(value) if value.ndim == 0 else tuple(float(component) for component in value)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        pass

    def write_data_to_sim(self) -> None:
        pass

    def update(self, dt: float) -> None:
        pass

    def _initialize_impl(self) -> None:
        plan = SimulationContext.instance().get_clone_plan()
        if self._is_per_env:
            assert plan is not None and plan.env_ids is not None
            row = plan.cfg_rows[self._clone_cfg_id][0]
            num_envs = plan.clone_mask.shape[1]
            if not bool(plan.clone_mask[row].all()):
                raise ValueError(
                    f"Per-environment material {self._source_material_path!r} must populate every environment."
                )
            self._material_paths = tuple(
                cloner.path.rebase(self._source_material_path, plan.sources[row], plan.destinations[row].format(env_id))
                for env_id in range(num_envs)
            )
            shader_suffix = self._source_shader_path.removeprefix(self._source_material_path)
            self._shader_paths = tuple(path + shader_suffix for path in self._material_paths)
        else:
            self._material_paths = (self._source_material_path,)
            self._shader_paths = (self._source_shader_path,)
        if not self._values:
            for channel, value in self._initial_values.items():
                self._values[channel] = value.to(self.device).expand(len(self._material_paths), *value.shape).clone()
        self._render_context.register_visual_material(self)


def _channel_specs(shader: UsdShade.Shader) -> dict[str, tuple[str, float | tuple[float, ...]]]:
    if shader.GetImplementationSource() == UsdShade.Tokens.id and shader.GetShaderId() == "UsdPreviewSurface":
        return dict(_PREVIEW_CHANNELS)
    identifier = shader.GetSourceAssetSubIdentifier("mdl")
    if identifier and identifier.startswith("OmniGlass"):
        return dict(_GLASS_CHANNELS)
    if identifier and identifier.startswith("OmniPBR"):
        channels = dict(_PBR_CHANNELS)
        texture = shader.GetInput("diffuse_texture")
        if texture and texture.Get():
            channels["color"] = ("diffuse_tint", (1.0, 1.0, 1.0))
        return channels
    raise TypeError("VisualMaterial requires PreviewSurfaceCfg, PbrMdlCfg, or GlassMdlCfg.")
