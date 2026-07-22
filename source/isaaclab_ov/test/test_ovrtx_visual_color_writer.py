# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :class:`OVRTXVisualColorWriter` using a recording (mock) renderer.

Covers the OVRTX-specific logic that the shared/Newton tests cannot: the OmniPBR material-sublayer
builder, the deferred (first-write) material binding, that ``write_colors`` dispatches the right
shader prims for an ``env_ids`` subset, and that ``pre_physics_ready_setup`` strips pre-existing
``material:binding`` direct opinions on the target meshes. The real ``ovrtx.Renderer`` API (the
actual pixel write) is out of scope here -- that is proven by the render-readback contract test
on GPU.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

_REQUIRED_MODULES = ("isaaclab_ov", "pxr")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.visual.ovrtx_visual_color_writer import (  # noqa: E402
        OVRTXVisualColorWriter,
        _build_materials_sublayer_usda,
    )

    from pxr import Sdf, Usd, UsdShade  # noqa: E402


def _scene_layer(num_envs: int, with_prior_binding: bool) -> str:
    """Return USDA for a 2-cube-per-env cartpole-shaped scene.

    When ``with_prior_binding`` is True, each cube mesh ships with its own
    ``material:binding`` opinion (the scenario that previously broke OVRTX rebinds).
    """
    parts: list[str] = ['#usda 1.0\n(\n    defaultPrim = "World"\n)\n\ndef Xform "World"\n{\n']
    parts.append('    def Xform "envs"\n    {\n')
    for env_id in range(num_envs):
        parts.append(f'        def Xform "env_{env_id}"\n        {{\n')
        parts.append('            def Xform "Robot"\n            {\n')
        for body in ("cart", "pole"):
            parts.append(f'                def Xform "{body}"\n                {{\n')
            parts.append('                    def Xform "visuals"\n                    {\n')
            # Authored as siblings of mesh_0 (not children) so the rel target path resolves.
            if with_prior_binding:
                parts.append('                        def Scope "Looks"\n                        {\n')
                parts.append('                            def Material "material"\n                            {\n')
                parts.append(
                    "                                token outputs:mdl:surface.connect = "
                    "<Looks/material/Shader.outputs:out>\n"
                )
                parts.append('                                def Shader "Shader"\n                                {\n')
                parts.append("                                    uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@\n")
                parts.append(
                    "                                    color3f inputs:diffuse_color_constant = (0.5, 0.5, 0.5)\n"
                )
                parts.append(
                    "                                }\n                            }\n                        }\n"
                )
            parts.append('                        def Cube "mesh_0"\n                        {\n')
            if with_prior_binding:
                parts.append(
                    f"                            rel material:binding = "
                    f"</World/envs/env_{env_id}/Robot/{body}/visuals/Looks/material>\n"
                )
            parts.append("                        }\n                    }\n                }\n")
        parts.append("            }\n        }\n")
    parts.append("    }\n}\n")
    return "".join(parts)


def _make_stage(num_envs: int = 2, with_prior_binding: bool = False) -> Usd.Stage:
    """Build an in-memory USD stage shaped like the cartpole multi-env layout."""
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(_scene_layer(num_envs, with_prior_binding))
    return Usd.Stage.Open(layer)


class _RecordingRenderer:
    """Stand-in for ``ovrtx.Renderer`` that records the calls the writer makes."""

    def __init__(self):
        self.usd_refs: list[tuple[str, str]] = []
        self.binding_calls: list[tuple[list[str], str, list]] = []
        self.write_calls: list[tuple[list[str], str, np.ndarray]] = []
        self.reset_calls: int = 0

    def add_usd_reference_from_string(self, usda, root):
        self.usd_refs.append((usda, root))

    def write_array_attribute(self, prim_paths, attribute_name, tensors):
        self.binding_calls.append((list(prim_paths), attribute_name, [list(t) for t in tensors]))

    def write_attribute(self, prim_paths, attribute_name, tensor):
        self.write_calls.append((list(prim_paths), attribute_name, np.asarray(tensor)))

    def reset(self):
        # Writer calls this after every write_colors to flush the accumulator; record the count.
        self.reset_calls += 1


def _make_writer(monkeypatch, renderer, stage):
    """Construct an :class:`OVRTXVisualColorWriter` against a mock renderer + a real USD stage."""
    import isaaclab_ov.visual.ovrtx_visual_color_writer as mod

    monkeypatch.setattr(mod.OVRTXVisualColorWriter, "_resolve_ovrtx_renderer", staticmethod(lambda env: renderer))
    monkeypatch.setattr(
        mod.sim_utils,
        "find_matching_prims",
        lambda pattern, *args, **kwargs: list(stage.Traverse())
        and [prim for prim in stage.Traverse() if prim.GetPath().pathString.endswith("/visuals")],
    )
    return OVRTXVisualColorWriter(env=None, mesh_prim_path="/World/envs/env_.*/Robot/.*/visuals")


def _saturated_colors_for(num_targets: int) -> torch.Tensor:
    """One distinct fully-saturated colour per target, just enough for subset-write assertions."""
    return torch.tensor([[0.1 * (i + 1), 0.0, 0.0] for i in range(num_targets)], dtype=torch.float32)


def test_target_resolution_maps_env_per_mesh(monkeypatch):
    stage = _make_stage(num_envs=2)
    writer = _make_writer(monkeypatch, _RecordingRenderer(), stage)
    # 2 envs * 2 bodies (cart, pole) = 4 Gprim leaves (the ``mesh_0`` Cubes).
    assert writer.num_targets == 4
    assert writer._env_of_target == [0, 0, 1, 1]


def test_pre_physics_ready_setup_unbinds_existing_bindings():
    """The pre-PHYSICS_READY classmethod strips asset-bundled ``material:binding`` direct
    opinions so the post-bake rebind isn't shadowed by them."""
    stage = _make_stage(num_envs=2, with_prior_binding=True)
    # Sanity: each mesh ships bound to its asset material. ``GetMaterial`` returns an invalid
    # UsdShade.Material when no binding is authored, so check via the underlying prim's validity.
    sample = stage.GetPrimAtPath("/World/envs/env_0/Robot/cart/visuals/mesh_0")
    bound_before = UsdShade.MaterialBindingAPI(sample).GetDirectBinding().GetMaterial()
    assert bound_before.GetPrim().IsValid()
    # Run pre-PHYSICS_READY using the stage's prim lookup as the resolver.
    import unittest.mock as _mock

    import isaaclab_ov.visual.ovrtx_visual_color_writer as mod

    with _mock.patch.object(
        mod.sim_utils,
        "find_matching_prims",
        lambda *_a, **_k: [p for p in stage.Traverse() if p.GetPath().pathString.endswith("/visuals")],
    ):
        OVRTXVisualColorWriter.pre_physics_ready_setup(env=None, mesh_prim_path="/World/envs/env_.*/Robot/.*/visuals")
    # After the hook, the direct binding is gone (returned Material's prim is invalid).
    bound_after = UsdShade.MaterialBindingAPI(sample).GetDirectBinding().GetMaterial()
    assert not bound_after.GetPrim().IsValid()


def test_init_defers_runtime_authoring_until_first_write(monkeypatch):
    """The writer's ``__init__`` no longer calls the runtime renderer -- those go through
    ``write_colors`` on first invocation (so they land post-bake, in OVRTX's live layer)."""
    renderer = _RecordingRenderer()
    _make_writer(monkeypatch, renderer, _make_stage(num_envs=2))
    assert renderer.usd_refs == []
    assert renderer.binding_calls == []
    assert renderer.write_calls == []
    assert renderer.reset_calls == 0


def test_write_colors_dispatches_only_selected_envs(monkeypatch):
    renderer = _RecordingRenderer()
    writer = _make_writer(monkeypatch, renderer, _make_stage(num_envs=2))
    colors = _saturated_colors_for(writer.num_targets)
    writer.write_colors(torch.tensor([1]), colors)  # only env 1
    # First write_colors performs the deferred USD-reference + material:binding wiring.
    assert len(renderer.binding_calls) == 1
    prim_paths, attribute_name, tensors = renderer.binding_calls[0]
    assert attribute_name == "material:binding"
    assert len(tensors) == 4
    assert len({t[0] for t in tensors}) == 4  # each mesh bound to its own material
    # The diffuse-color write hits only env 1's two prims and uses their colors.
    assert len(renderer.write_calls) == 1
    write_paths, write_attr, write_tensor = renderer.write_calls[0]
    assert write_paths == [writer._shader_paths[2], writer._shader_paths[3]]  # env 1's two prims
    assert write_attr == "inputs:diffuse_color_constant"
    assert np.allclose(write_tensor, colors[[2, 3]].numpy())


def test_write_colors_flushes_rt_accumulator_via_reset(monkeypatch):
    # Every write_colors must flush the renderer's path-traced accumulator; without it a subset
    # write does not appear for several frames.
    renderer = _RecordingRenderer()
    writer = _make_writer(monkeypatch, renderer, _make_stage(num_envs=2))
    colors = _saturated_colors_for(writer.num_targets)
    # full-set
    writer.write_colors(torch.tensor([0, 1]), colors)
    assert renderer.reset_calls == 1
    # subset
    writer.write_colors(torch.tensor([1]), colors)
    assert renderer.reset_calls == 2
    # empty env_ids is a no-op (also no reset; cheap optimization).
    writer.write_colors(torch.empty(0, dtype=torch.long), colors)
    assert renderer.reset_calls == 2


def test_build_materials_sublayer_usda_aligned():
    usda, material_paths, shader_paths = _build_materials_sublayer_usda(3)
    assert len(material_paths) == len(shader_paths) == 3
    assert len(set(material_paths)) == 3  # distinct materials
    for material_path, shader_path in zip(material_paths, shader_paths):
        assert shader_path.startswith(material_path + "/")  # shader nested under its material
    assert usda.lstrip().startswith("#usda 1.0")
    assert usda.count("def Material") == 3
    assert "diffuse_color_constant" in usda
