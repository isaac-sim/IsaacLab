# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render-material events for the AGX Orin packing task."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from pxr import Gf, Sdf, Usd, UsdShade

from isaaclab.managers import SceneEntityCfg

from .rewards import get_pack_agx_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger(__name__)


def reset_task_stage(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    agx_orin_cfg: SceneEntityCfg = SceneEntityCfg("agx_orin"),
    print_log: bool = False,
) -> None:
    """Reset Pack-AGX stage trackers and capture the randomized start height."""
    if len(env_ids) == 0:
        return

    state = get_pack_agx_state(env)
    previous_stage = state.task_stage[env_ids].clone()
    if print_log:
        reached = [int((previous_stage >= stage_id).sum().item()) for stage_id in (1, 2, 3)]
        logger.info(
            "[PACK_AGX_STAGE_SUMMARY] total=%d reached_stage_1/2/3=%s",
            len(env_ids),
            reached,
        )

    state.task_stage[env_ids] = 0
    state.prev_stage_lift[env_ids] = 0
    state.prev_stage_align[env_ids] = 0
    state.prev_stage_seat[env_ids] = 0
    state.stage_hold_counter[env_ids] = 0

    agx_orin = env.scene[agx_orin_cfg.name]
    state.initial_agx_z[env_ids] = agx_orin.data.root_pos_w.torch[env_ids, 2]


def _preview_material(
    stage,
    path: str,
    diffuse: float,
    metallic: float,
    roughness: float,
    emissive: float = 0.0,
):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(diffuse))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(emissive))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def align_table_material(
    env,
    env_ids,
    diffuse: float = 0.75,
    emissive: float = 0.8,
) -> None:
    """Keep the tabletop brighter than the backdrop under the real room lighting."""
    del env_ids
    stage = env.scene.stage
    material = _preview_material(stage, "/World/Looks/AlignedTable", diffuse, 0.0, 0.8, emissive)

    for index in range(env.scene.num_envs):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{index}/Table")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)


def align_prop_material(
    env,
    env_ids,
    asset: str,
    metallic: float,
    roughness: float,
    albedo_brightness: float | None = None,
    albedo_add: float | None = None,
) -> None:
    """Tame an authored OmniPBR prop so its baked textures stay visible.

    The Lightwheel props apply their ORM texture's metallic channel at full
    influence.  In this room that turns the AGX into a mirror of a dark
    environment and it renders at 2/255 instead of the real 120, so the metallic
    and roughness channels are driven by constants while the diffuse, normal and
    occlusion textures keep providing the surface detail.
    """
    del env_ids
    stage = env.scene.stage
    inputs: dict[str, tuple[Sdf.ValueTypeName, object]] = {
        "metallic_texture_influence": (Sdf.ValueTypeNames.Float, 0.0),
        "reflection_roughness_texture_influence": (Sdf.ValueTypeNames.Float, 0.0),
        "metallic_constant": (Sdf.ValueTypeNames.Float, metallic),
        "reflection_roughness_constant": (Sdf.ValueTypeNames.Float, roughness),
    }
    if albedo_brightness is not None:
        inputs["albedo_brightness"] = (Sdf.ValueTypeNames.Float, albedo_brightness)
    if albedo_add is not None:
        inputs["albedo_add"] = (Sdf.ValueTypeNames.Float, albedo_add)

    for index in range(env.scene.num_envs):
        root = stage.GetPrimAtPath(f"/World/envs/env_{index}/{asset}")
        for prim in Usd.PrimRange(root):
            shader = UsdShade.Shader(prim)
            if not shader:
                continue
            # These are MDL inputs, so the ones the asset never authored have to
            # be created before they can be driven.
            for name, (value_type, value) in inputs.items():
                shader.CreateInput(name, value_type).Set(value)
