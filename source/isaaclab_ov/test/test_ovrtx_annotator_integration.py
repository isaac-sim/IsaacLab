# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for the OVRTX renderer's idToLabels/idToSemantics resolution.

Builds semantically-labeled USD scenes programmatically with ``pxr.Usd``/
``pxr.UsdSemantics``, renders them through a real ``ovrtx.Renderer`` using the
production render-var wiring (:func:`isaaclab_ov.renderers.ovrtx_usd.build_render_product_as_string`),
then drives the actual :class:`~isaaclab_ov.renderers.ovrtx_renderer.OVRTXRenderer`
methods (``_process_id_segmentation_render_var`` -> ``_update_segmentation_info``
-> ``read_output``) against the resulting frame and asserts the expected
prim paths/labels land in ``CameraData.info``.

Driving the real ``OVRTXRenderer`` methods (rather than calling
:mod:`isaaclab_ov.renderers.annotator_utils` directly, which is already
covered by ``test_annotator_utils.py``'s synthetic-buffer unit tests) is the
point of this file: it is the only place that exercises the actual wiring in
``ovrtx_renderer.py`` -- render var selection, CPU mapping, and
``CameraData.info`` propagation -- against a real render.

Requires an NVIDIA RTX-capable GPU: this constructs and steps a real
``ovrtx.Renderer``.

.. note::
    The renderer occasionally drops exactly one labeled instance's
    ``StableIdMap`` entry per render (its ``idToSemantics`` label still
    resolves correctly; only the ``idToLabels`` prim path falls back to
    ``"UNLABELLED"``). This looks like an OVRTX-internal timing/ordering
    quirk unrelated to the ``annotator_utils`` decode logic exercised here --
    it reproduces with a hand-authored ``.usda`` file too, not just
    programmatically-built stages. The tests below tolerate up to one such
    fallback per scene rather than asserting exact 1:1 resolution for every
    instance; see the assertions in
    ``test_instance_segmentation_fast_id_to_labels_resolve_to_real_prim_paths``.
"""

from __future__ import annotations

import importlib.util
import math
import os

import pytest

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx", "pxr")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")

    import ovrtx
    import warp as wp
    from pxr import Usd, UsdGeom, UsdLux, UsdSemantics

    from isaaclab.sensors.camera.camera_data import CameraData, RenderBufferKind

    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderData, OVRTXRenderer
    from isaaclab_ov.renderers.ovrtx_renderer_cfg import OVRTXRendererCfg
    from isaaclab_ov.renderers.ovrtx_usd import build_render_product_as_string
else:
    ovrtx = None
    wp = None
    Usd = None
    UsdGeom = None
    UsdLux = None
    UsdSemantics = None
    CameraData = None
    RenderBufferKind = None
    OVRTXRenderData = None
    OVRTXRenderer = None
    OVRTXRendererCfg = None
    build_render_product_as_string = None


CAMERA_REL_PATH = "Camera"
_WIDTH, _HEIGHT = 320, 240
_WARMUP_STEPS = 3


def _require_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("OVRTX integration tests require a CUDA-capable GPU")


def _apply_labels(prim: Usd.Prim, taxonomy: str, labels: list[str]) -> None:
    UsdSemantics.LabelsAPI.Apply(prim, taxonomy).CreateLabelsAttr(labels)


def _xform_at(stage: Usd.Stage, name: str, translate: tuple[float, float, float]) -> Usd.Prim:
    xform = UsdGeom.Xform.Define(stage, f"/World/envs/env_0/{name}")
    xform.AddTranslateOp().Set(translate)
    return xform.GetPrim()


def build_labeled_scene_stage() -> Usd.Stage:
    """Build a single-env USD stage with several distinctly-labeled prims.

    Mirrors ``ovrtx_aov_test/semantic_maps_scene.usda``, constructed
    programmatically via ``pxr.Usd``/``pxr.UsdSemantics`` instead of hand-typed
    USDA text:

    * ``Cone`` -- two semantic taxonomies (``class:cone``, ``instance:cone_01``).
    * ``Cube`` -- a single semantic taxonomy (``class:cube``).
    * ``Sphere`` -- a single semantic taxonomy (``class:sphere``).
    * ``Vehicle_Xform`` -- carries ``class:vehicle``; its child ``Wheel`` has no
      label of its own, so it inherits the label from its ancestor.
    * ``GroundPlane`` -- unlabeled, so the reserved UNLABELLED id also appears
      on screen.

    Returns:
        An in-memory USD stage with prims under ``/World/envs/env_0``, a
        camera at ``/World/envs/env_0/Camera`` framing all objects, and a
        distant light.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")

    cone_xform = _xform_at(stage, "Cone_Xform", (-3.0, 0.5, 0.0))
    cone = UsdGeom.Cone.Define(stage, cone_xform.GetPath().AppendChild("Cone"))
    cone.CreateHeightAttr(1.0)
    cone.CreateRadiusAttr(0.5)
    cone.CreateDisplayColorAttr([(0.8, 0.2, 0.2)])
    _apply_labels(cone.GetPrim(), "class", ["cone"])
    _apply_labels(cone.GetPrim(), "instance", ["cone_01"])

    cube_xform = _xform_at(stage, "Cube_Xform", (-1.0, 0.5, 0.0))
    cube = UsdGeom.Cube.Define(stage, cube_xform.GetPath().AppendChild("Cube"))
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([(0.2, 0.6, 0.8)])
    _apply_labels(cube.GetPrim(), "class", ["cube"])

    sphere_xform = _xform_at(stage, "Sphere_Xform", (3.0, 0.5, 0.0))
    sphere = UsdGeom.Sphere.Define(stage, sphere_xform.GetPath().AppendChild("Sphere"))
    sphere.CreateRadiusAttr(0.5)
    sphere.CreateDisplayColorAttr([(0.3, 0.8, 0.3)])
    _apply_labels(sphere.GetPrim(), "class", ["sphere"])

    vehicle_xform = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Vehicle_Xform")
    vehicle_xform.AddTranslateOp().Set((0.0, 0.5, 3.0))
    _apply_labels(vehicle_xform.GetPrim(), "class", ["vehicle"])
    wheel = UsdGeom.Cylinder.Define(stage, "/World/envs/env_0/Vehicle_Xform/Wheel")
    wheel.CreateHeightAttr(0.3)
    wheel.CreateRadiusAttr(0.6)
    wheel.CreateAxisAttr("X")
    wheel.CreateDisplayColorAttr([(0.1, 0.1, 0.1)])

    ground = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/GroundPlane")
    ground.CreateExtentAttr([(-10, 0, -10), (10, 0, 10)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreatePointsAttr([(-10, 0, -10), (10, 0, -10), (10, 0, 10), (-10, 0, 10)])
    ground.CreateNormalsAttr([(0, 1, 0)] * 4)

    light = UsdLux.DistantLight.Define(stage, "/World/envs/env_0/DistantLight")
    light.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(light).AddRotateXYZOp().Set((300.0, 45.0, 0.0))

    camera = UsdGeom.Camera.Define(stage, f"/World/envs/env_0/{CAMERA_REL_PATH}")
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateClippingRangeAttr((0.1, 1000.0))
    camera_xformable = UsdGeom.Xformable(camera)
    camera_xformable.AddTranslateOp().Set((0.0, 4.0, 10.0))
    camera_xformable.AddRotateXYZOp().Set((-20.0, 0.0, 0.0))

    return stage


def _render_frame(stage: Usd.Stage, data_type: str):
    """Render ``stage`` for ``data_type`` and return the resulting ovrtx renderer, step result, and frame.

    The caller must keep both the returned renderer AND the step result
    (``products``) alive for as long as ``frame`` is used: the render var
    output handles ``frame`` wraps are only valid while ``products`` -- not
    just the renderer -- has not been garbage collected.
    """
    render_product_string, render_product_path = build_render_product_as_string(
        width=_WIDTH,
        height=_HEIGHT,
        num_envs=1,
        data_types=[data_type],
        camera_rel_path=CAMERA_REL_PATH,
    )
    full_usd_string = stage.ExportToString() + "\n\n" + render_product_string

    renderer = ovrtx.Renderer()
    renderer.open_usd_from_string(full_usd_string)
    for _ in range(_WARMUP_STEPS):
        renderer.step(render_products={render_product_path}, delta_time=0.0)
    products = renderer.step(render_products={render_product_path}, delta_time=0.0)
    frame = products[render_product_path].frames[0]
    return renderer, products, frame


_DEVICE = "cuda:0"


def _make_ovrtx_renderer_stub(**cfg_overrides) -> OVRTXRenderer:
    """Build an OVRTXRenderer with just enough state for _process_render_frame."""
    renderer = OVRTXRenderer.__new__(OVRTXRenderer)
    renderer.cfg = OVRTXRendererCfg(**cfg_overrides)
    renderer._device = _DEVICE
    renderer._output_id_color_buffers = {}
    return renderer


def _make_ovrtx_render_data_stub(num_envs: int, height: int, width: int) -> OVRTXRenderData:
    render_data = OVRTXRenderData.__new__(OVRTXRenderData)
    render_data.num_envs = num_envs
    render_data.height = height
    render_data.width = width
    render_data.num_cols = math.ceil(math.sqrt(num_envs))
    render_data.num_rows = math.ceil(num_envs / render_data.num_cols)
    render_data.warp_buffers = {}
    render_data.renderer_info = {}
    render_data.ppisp_pipeline = None
    return render_data


def _run_segmentation_info(stage: Usd.Stage, data_type: str, **cfg_overrides) -> dict:
    """Render ``stage`` for ``data_type`` and return the ``idToLabels``/``idToSemantics`` info dict.

    Drives the real ``OVRTXRenderer._process_render_frame`` -- the same entry
    point ``render()`` calls in production -- against a real ovrtx frame, then
    ``read_output`` to confirm the result lands in ``CameraData.info``. Only
    the public ``data_type`` name (and, for the two instance types,
    ``cfg_overrides`` to select colorize True/False) is known here; which AOV
    backs it, its buffer_key, and dtype are all resolved the same way
    production code resolves them (``supported_output_types()``), not
    hardcoded in the test.
    """
    renderer_, products_, frame = _render_frame(stage, data_type)
    try:
        renderer = _make_ovrtx_renderer_stub(**cfg_overrides)
        render_data = _make_ovrtx_render_data_stub(num_envs=1, height=_HEIGHT, width=_WIDTH)

        spec = renderer.supported_output_types()[RenderBufferKind(data_type)]
        output_buffers = {
            data_type: wp.zeros((1, _HEIGHT, _WIDTH, spec.channels), dtype=spec.dtype, device=_DEVICE),
        }

        renderer._process_render_frame(render_data, frame, output_buffers)
    finally:
        del products_, renderer_

    assert data_type in render_data.renderer_info, (
        f"OVRTXRenderer never populated renderer_info[{data_type!r}] -- check that"
        f" ovrtx_usd.get_render_var_configs authors the render vars _update_segmentation_info needs"
    )

    camera_data = CameraData()
    camera_data.info = {}
    camera_data._output = {}
    renderer.read_output(render_data, camera_data)
    assert camera_data.info == render_data.renderer_info

    return camera_data.info[data_type]


def test_semantic_segmentation_id_to_labels_matches_authored_labels():
    """OVRTXRenderer resolves semantic_segmentation's idToLabels via a real render."""
    _require_gpu()
    stage = build_labeled_scene_stage()
    info = _run_segmentation_info(stage, "semantic_segmentation")

    id_to_labels = info["idToLabels"]
    assert id_to_labels[0] == "BACKGROUND"
    assert id_to_labels[1] == "UNLABELLED"
    resolved_labels = set(id_to_labels.values())
    assert {"class:cube", "class:sphere", "class:vehicle"} <= resolved_labels
    assert any("instance:cone_01" in label and "class:cone" in label for label in resolved_labels)


@pytest.mark.parametrize("colorize", [True, False], ids=["colorize", "raw_ids"])
def test_instance_segmentation_fast_id_to_semantics_matches_authored_labels(colorize: bool):
    """OVRTXRenderer resolves instance_segmentation_fast's idToSemantics via a real render.

    Unlike idToLabels (prim paths, see the flakiness noted in the module
    docstring), idToSemantics only depends on the positional
    StableIdSemanticIdMap -> SemanticIdMap join and has been reliable in
    practice across every instance. Parametrized over colorize since the info
    dict is populated identically either way but the pixel-extraction branch
    ``_process_id_segmentation_render_var`` takes differs (see the
    torch.uint32/wp.uint32 conversion fixed in ovrtx_renderer.py, which only
    reproduced on the ``colorize=False`` branch).
    """
    _require_gpu()
    stage = build_labeled_scene_stage()
    info = _run_segmentation_info(stage, "instance_segmentation_fast", colorize_instance_segmentation=colorize)

    resolved_labels = set(info["idToSemantics"].values())
    assert {"class:cube", "class:sphere", "class:vehicle"} <= resolved_labels
    assert any("instance:cone_01" in label and "class:cone" in label for label in resolved_labels)


@pytest.mark.parametrize("colorize", [True, False], ids=["colorize", "raw_ids"])
def test_instance_segmentation_fast_id_to_labels_resolve_to_real_prim_paths(colorize: bool):
    """OVRTXRenderer resolves instance_segmentation_fast's idToLabels to real (or fallback) prim paths."""
    _require_gpu()
    stage = build_labeled_scene_stage()
    info = _run_segmentation_info(stage, "instance_segmentation_fast", colorize_instance_segmentation=colorize)
    id_to_labels = info["idToLabels"]

    fallback_values = {"BACKGROUND", "UNLABELLED", "<unknown prim>"}
    real_prim_paths = {v for v in id_to_labels.values() if v not in fallback_values}

    # Every resolved value is either a real prim that exists on the stage, or
    # one of the documented fallbacks -- never garbage.
    for prim_path in real_prim_paths:
        assert stage.GetPrimAtPath(prim_path).IsValid(), f"resolved a prim path not on the stage: {prim_path}"

    # See the module docstring: the renderer occasionally drops one instance's
    # StableIdMap entry, so tolerate at most one fallback among the 4 labeled
    # instances (Cone, Cube, Sphere, Vehicle_Xform) rather than requiring all 4.
    assert len(real_prim_paths) >= 3, f"expected at least 3 of 4 instances to resolve, got: {id_to_labels}"


@pytest.mark.parametrize("colorize", [True, False], ids=["colorize", "raw_ids"])
def test_instance_id_segmentation_fast_id_to_labels_resolve_to_real_prim_paths(colorize: bool):
    """OVRTXRenderer resolves instance_id_segmentation_fast's idToLabels to real (or fallback) prim paths."""
    _require_gpu()
    stage = build_labeled_scene_stage()
    info = _run_segmentation_info(stage, "instance_id_segmentation_fast", colorize_instance_id_segmentation=colorize)
    id_to_labels = info["idToLabels"]

    fallback_values = {"BACKGROUND", "UNLABELLED", "<unknown prim>"}
    real_prim_paths = {v for v in id_to_labels.values() if v not in fallback_values}

    for prim_path in real_prim_paths:
        assert stage.GetPrimAtPath(prim_path).IsValid(), f"resolved a prim path not on the stage: {prim_path}"

    # See the module docstring for the known single-instance fallback quirk.
    assert len(real_prim_paths) >= 3, f"expected at least 3 of 4 instances to resolve, got: {id_to_labels}"


def test_render_var_wiring_authors_expected_render_vars_per_data_type():
    """get_render_var_configs()-driven render products expose the render vars annotator_utils needs."""
    _require_gpu()
    stage = build_labeled_scene_stage()

    expectations = {
        "semantic_segmentation": {"SemanticSegmentation", "SemanticIdMap"},
        "instance_segmentation_fast": {
            "NonStableInstanceSegmentation",
            "SemanticIdMap",
            "StableIdMap",
            "StableIdSemanticIdMap",
        },
        "instance_id_segmentation_fast": {
            "InstanceSegmentationSD",
            "StableIdMap",
            "StableIdSemanticIdMap",
            "InstanceMap",
        },
    }
    for data_type, expected_render_vars in expectations.items():
        renderer, products, frame = _render_frame(stage, data_type)
        try:
            assert expected_render_vars <= set(frame.render_vars.keys())
        finally:
            del products, renderer
