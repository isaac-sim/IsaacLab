# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.sim.spawners.materials import VisualMaterialCfg as VisualMaterialSpawnerCfg
from isaaclab.utils import REQUIRED


@dataclass
class VisualMaterialCfg(AssetBaseCfg):
    """A runtime-writable material declared like any other scene asset.

    An absolute :attr:`prim_path` declares one shared material. For independent per-environment
    values, declare the material below its owning cloned asset and bind it with an asset-relative
    path through :attr:`isaaclab.sim.spawners.from_files.from_files_cfg.FileCfg.visual_material_bindings`.
    """

    class_type: type | str = "isaaclab.assets.visual_material.visual_material:VisualMaterial"
    cloning_contexts: tuple[str | type, ...] | None = ()
    spawn: VisualMaterialSpawnerCfg | None = REQUIRED
    """Material spawner, or ``None`` to wrap an existing material prim."""
    channels: tuple[str, ...] = ("color",)
    """Numeric shader channels writable at runtime.

    Preview Surface supports ``color``, ``roughness``, ``metallic``, ``emissive_color``, and
    ``opacity``. OmniPBR additionally supports ``specular``, ``emissive_intensity``, ``uv_scale``,
    ``uv_offset``, and ``uv_rotate``. OmniGlass supports ``color``, ``roughness``, and ``ior``.
    """
