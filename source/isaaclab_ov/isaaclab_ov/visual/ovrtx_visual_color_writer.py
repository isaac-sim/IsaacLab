# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless OVRTX backend of the visual ``write_colors`` contract.

Reset-time per-env diffuse-color randomization on the OVRTX renderer without Kit, via the renderer's
``write_attribute`` channel on the OmniPBR MDL input ``inputs:diffuse_color_constant``.

Lifecycle:
  * ``pre_physics_ready_setup`` (before bake): strip any asset-bundled ``material:binding`` so the
    post-bake rebind is not shadowed.
  * first ``write_colors``: inject one OmniPBR material per target and rebind each mesh to it
    (deferred to here because the bake resets the runtime stage).
  * each ``write_colors``: write the per-env diffuse colors, then ``Renderer.reset()`` to flush the
    path-traced accumulator.

Requires ``replicate_physics=False`` (un-instanced per-env prims, which also disables OVRTX cloning);
articulation-root scoping is handled upstream in ``randomize_visual_color``.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.visual_color import env_index_from_prim_path as _env_index_from_prim_path
from isaaclab.utils.visual_color import select_visual_color_targets

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)

_DIFFUSE_ATTR = "inputs:diffuse_color_constant"


def _unique_materials_root(mesh_prim_path: str) -> str:
    """A stage-unique top-level scope for the DR materials.

    Avoids the "A prim already exists" rejection from ovrtx: ``/Looks`` already exists, and a shared
    root would collide across per-term writers. Deriving from ``mesh_prim_path`` keeps it unique.
    """
    return "/DRColorMaterials_" + re.sub(r"[^A-Za-z0-9]+", "_", mesh_prim_path).strip("_")


def _build_materials_sublayer_usda(
    num_materials: int,
    # Dedicated scope, never ``/Looks`` (pre-exists and would collide); see _unique_materials_root.
    materials_root: str = "/DRColorMaterials",
    material_prefix: str = "mat",
    initial_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    initial_roughness: float = 1.0,
) -> tuple[str, list[str], list[str]]:
    """Build a USDA sublayer with ``num_materials`` OmniPBR Materials. Returns (usda, material_paths, shader_paths)."""
    r, g, b = initial_color
    material_paths: list[str] = []
    shader_paths: list[str] = []
    blocks: list[str] = []
    root_segments = materials_root.strip("/").split("/")
    default_prim_name = root_segments[0]

    for i in range(num_materials):
        mat_name = f"{material_prefix}_{i}"
        mat_path = f"{materials_root}/{mat_name}"
        shader_path = f"{mat_path}/Shader"
        material_paths.append(mat_path)
        shader_paths.append(shader_path)
        blocks.append(
            f'        def Material "{mat_name}"\n'
            f"        {{\n"
            f"            token outputs:mdl:surface.connect = <{shader_path}.outputs:out>\n"
            f'            def Shader "Shader"\n'
            f"            {{\n"
            f'                uniform token info:implementationSource = "sourceAsset"\n'
            f"                uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@\n"
            f'                uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"\n'
            f"                color3f inputs:diffuse_color_constant = ({r}, {g}, {b})\n"
            f"                float inputs:reflection_roughness_constant = {initial_roughness}\n"
            f'                token outputs:out (renderType = "material")\n'
            f"            }}\n"
            f"        }}"
        )

    nested_open = ""
    nested_close = ""
    for seg in root_segments[1:]:
        nested_open += f'    def Scope "{seg}"\n    {{\n'
        nested_close = "}\n" + nested_close
    body = "\n\n".join(blocks)
    usda = (
        f'#usda 1.0\n(\n    defaultPrim = "{default_prim_name}"\n)\n'
        f'def Scope "{default_prim_name}"\n{{\n{nested_open}{body}\n{nested_close}}}\n'
    )
    return usda, material_paths, shader_paths


class OVRTXVisualColorWriter:
    """Kitless OVRTX color writer constructed from the IsaacLab env (the dispatch entry point).

    Args:
        env: The manager-based environment.
        mesh_prim_path: The (regex) prim-path pattern of the target visual meshes, resolved upstream
            in :class:`~isaaclab.envs.mdp.events.randomize_visual_color` (the ``/visuals`` scoping).
    """

    @classmethod
    def _resolve_target_prims(cls, mesh_prim_path: str) -> list:
        """Return the target ``UsdGeom.Gprim`` leaves under ``mesh_prim_path``, un-instancing the chain
        (ancestors up to ``/World`` plus all descendants) so USD-stage authoring on them takes effect.
        """
        from pxr import Usd, UsdGeom  # noqa: PLC0415

        roots = list(sim_utils.find_matching_prims(mesh_prim_path))
        if not roots:
            raise ValueError(f"OVRTXVisualColorWriter found no prims under pattern '{mesh_prim_path}'.")
        target_prims: list = []
        for root in roots:
            ancestor = root
            while ancestor and ancestor.IsValid() and str(ancestor.GetPath()) not in ("/World", "/"):
                ancestor.SetInstanceable(False)
                ancestor = ancestor.GetParent()
            for descendant in Usd.PrimRange(root):
                descendant.SetInstanceable(False)
            if UsdGeom.Gprim(root):
                target_prims.append(root)
                continue
            for descendant in Usd.PrimRange(root):
                if descendant != root and UsdGeom.Gprim(descendant):
                    target_prims.append(descendant)
        if not target_prims:
            raise ValueError(f"OVRTXVisualColorWriter found no Gprim leaves under pattern '{mesh_prim_path}'.")
        return target_prims

    @classmethod
    def pre_physics_ready_setup(cls, env: ManagerBasedEnv, mesh_prim_path: str) -> None:
        """Strip asset-bundled ``material:binding`` from the target meshes (the pre-PHYSICS_READY hook).

        Must run before the OVRTX bake: OVRTX caches a mesh's binding at bake and silently drops a
        post-bake rebind, so the meshes must reach the bake unbound for the later rebind to land.
        """
        from pxr import UsdShade  # noqa: PLC0415

        target_prims = cls._resolve_target_prims(mesh_prim_path)
        unbound = 0
        for prim in target_prims:
            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            if binding_api.GetDirectBinding().GetMaterial():
                binding_api.UnbindAllBindings()
                unbound += 1
        logger.info(
            "OVRTXVisualColorWriter.pre_physics_ready_setup: stripped material:binding from %d/%d meshes",
            unbound,
            len(target_prims),
        )

    def __init__(self, env: ManagerBasedEnv, mesh_prim_path: str):
        self._renderer = self._resolve_ovrtx_renderer(env)

        # Collect the target mesh paths (already unbound by pre_physics_ready_setup before the bake).
        target_prims = self._resolve_target_prims(mesh_prim_path)
        target_mesh_paths = [str(prim.GetPath()) for prim in target_prims]
        self._env_of_target = [_env_index_from_prim_path(p) for p in target_mesh_paths]
        self.num_targets = len(target_mesh_paths)

        # Defer runtime authoring (material inject + rebind) to first write_colors: the bake resets the
        # runtime stage, so the authoring must land post-bake to take effect.
        sublayer_usda, material_paths, shader_paths = _build_materials_sublayer_usda(
            self.num_targets, materials_root=_unique_materials_root(mesh_prim_path)
        )
        self._target_mesh_paths = target_mesh_paths
        self._material_paths = material_paths
        self._sublayer_usda = sublayer_usda
        self._shader_paths = shader_paths
        self._post_bake_initialized = False

    @staticmethod
    def _resolve_ovrtx_renderer(env: ManagerBasedEnv):
        """Return the live ``ovrtx.Renderer`` handle from the env.

        Checks the render context's renderer entries first, then the scene's camera sensors, returning
        the first handle exposing ``write_attribute``.
        """
        render_context = getattr(env.sim, "render_context", None)
        # _renderer_entries: list[tuple[cfg, backend]] (or a dict / list of bare backends on older builds).
        entries = getattr(render_context, "_renderer_entries", None) or []
        for entry in entries.values() if hasattr(entries, "values") else entries:
            backend = entry[1] if isinstance(entry, tuple) and len(entry) >= 2 else entry
            # backend may wrap the raw ovrtx.Renderer (._renderer) or be it directly.
            raw = getattr(backend, "_renderer", None) or getattr(backend, "renderer", None) or backend
            if raw is not None and hasattr(raw, "write_attribute"):
                return raw
        for sensor in env.scene.sensors.values():
            renderer = getattr(getattr(sensor, "_renderer", None), "_renderer", None)
            if renderer is not None and hasattr(renderer, "write_attribute"):
                return renderer
        raise RuntimeError(
            "OVRTXVisualColorWriter could not resolve the ovrtx.Renderer handle from env."
            " Expected a render context entry or camera sensor exposing `write_attribute`."
        )

    def write_colors(self, env_ids: torch.Tensor, colors: torch.Tensor) -> None:
        """Apply one diffuse color per target Shader prim, for targets whose env is being reset.

        The first call performs the deferred post-bake authoring (material inject + rebind). After every
        write, ``Renderer.reset()`` flushes the path-traced accumulator: without it a subset write does
        not appear for many frames and auto-exposure drifts the colors toward grey.
        """
        if not self._post_bake_initialized:
            self._renderer.add_usd_reference_from_string(
                self._sublayer_usda, "/" + self._material_paths[0].strip("/").split("/")[0]
            )
            self._renderer.write_array_attribute(
                prim_paths=self._target_mesh_paths,
                attribute_name="material:binding",
                tensors=[[self._material_paths[g]] for g in range(self.num_targets)],
            )
            self._post_bake_initialized = True
        targets = select_visual_color_targets(env_ids, colors, self._env_of_target, self.num_targets)
        if not targets:
            return
        rgb = colors.detach().to("cpu", torch.float32).contiguous().numpy()
        sel_paths = [self._shader_paths[g] for g in targets]
        sel_rgb = np.asarray(rgb[targets], dtype=np.float32)
        self._renderer.write_attribute(prim_paths=sel_paths, attribute_name=_DIFFUSE_ATTR, tensor=sel_rgb)
        self._renderer.reset()


__all__ = ["OVRTXVisualColorWriter"]
