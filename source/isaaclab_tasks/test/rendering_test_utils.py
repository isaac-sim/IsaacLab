# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers for rendering correctness tests."""

import logging
import os
import re
import tempfile
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from PIL import Image, ImageChops

from isaaclab.utils.images import make_camera_output_grid, normalize_camera_output_for_display
from isaaclab.utils.warp import ProxyArray

if TYPE_CHECKING:
    from isaaclab.sensors.camera import CameraData

logger = logging.getLogger(__name__)

# Directory containing golden images.
_GOLDEN_IMAGES_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_images")

# Directory containing golden USD stage files.
_GOLDEN_STAGES_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_stages")

# Pixel L2 norm difference threshold. L2 norm difference is the Euclidean distance between two pixels:
#
#   d = sqrt((R1 - R2)^2 + (G1 - G2)^2 + (B1 - B2)^2)
#
# If the difference between two pixels is less than this threshold, consider them "equal" (i.e. within the tolerance).
#
_PIXEL_L2_NORM_DIFFERENCE_THRESHOLD = 10.0

# The max percentage of pixels allowed to differ. If the percentage exceeds this value, the test will fail.
# The value is set case by case based on the screen space taken up by the env in camera output images. It
# needs to be large enough to tolerate minor rendering noise while small enough to catch unexpected changes.
MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME = {
    # RTX anti-aliasing along the ground-plane edges varies slightly across GPU and driver environments.
    "cartpole": 1.5,
    # Aliasing artifacts of shadow on the table.
    "franka_cloth": 8.0,
    "franka_soft": 8.0,
    # Shadow-hand renderings (incl. ``Isaac-Reorient-Cube-Shadow-Camera-Direct``) show up to
    # ~3.28 % per-pixel diff from anti-aliasing noise along the many finger/cube edges. 5.0 gives
    # headroom above that without masking real regressions, which the SSIM gate still catches.
    "shadow_hand": 5.0,
    # Texture aliasing artifacts on the ground (NVBUG#6116767)
    "dexsuite_kuka_homo": 8.0,
    "dexsuite_kuka_hetero": 8.0,
}

# Minimum SSIM score below which two images are considered structurally different. SSIM is a perceptual metric
# robust to uniform per-pixel noise that penalises structural changes (geometry shifts, swapped colours, missing
# materials, etc.), so it complements the per-pixel L2 gate by catching regressions that survive a loosened pixel
# threshold.
_SSIM_THRESHOLD = 0.985

# Per-env SSIM overrides. Envs not listed fall back to ``_SSIM_THRESHOLD``. Loosened individually
# (not globally) to keep the strict gate active everywhere it already passes.
_SSIM_THRESHOLD_BY_ENV_NAME = {
    # Texture aliasing artifacts on the ground (NVBUG#6116767)
    "dexsuite_kuka_homo": 0.95,
    "dexsuite_kuka_hetero": 0.95,
}

# Data types for which the SSIM gate is not enforced. SSIM assumes natural-image statistics and is unreliable on
# outputs where the per-pixel value distribution is highly non-uniform after normalisation (e.g. depth, where we
# divide by the max value so tiny absolute differences near the far plane dominate windowed variance). For these
# data types we still compute SSIM for reporting, but only the per-pixel L2 gate is used to decide pass/fail.
# ``motion_vectors`` is included for the same reason: per-pixel (u, v) offsets are normalized by their max
# magnitude, so small absolute differences in near-static regions can dominate windowed variance.
_SSIM_DISABLED_DATA_TYPES: set[str] = {
    "depth",
    "distance_to_camera",
    "distance_to_image_plane",
    "instance_segmentation_fast",
    "instance_id_segmentation_fast",
    "motion_vectors",
}

# Directory for comparison images saved during the test session.
# Located under the pytest output root so it gets copied alongside test reports.
_COMPARISON_IMAGES_DIR = os.path.join(os.getcwd(), "tests", "comparison-images")
_COMPARISON_IMAGE_SUBDIR = "images"

_COPY_ICON_SVG = (
    '<svg class="copy-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"'
    ' stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>'
    '<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>'
    "</svg>"
)
_CHECK_ICON_SVG = (
    '<svg class="check-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"'
    ' stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M20 6 9 17l-5-5"/>'
    "</svg>"
)

# ---------------------------------------------------------------------------
# Parametrization: (physics_backend, renderer, data_type)
# ---------------------------------------------------------------------------

# Low-resolution camera outputs from RTX renderers are not deterministic enough to pass golden image testing
# on every CI run. (NVBUG#6152566)
_FLAKY_MARK = pytest.mark.flaky(max_runs=3, min_passes=1)

# Expand this tuple to test additional camera sensor data types
_DEFAULT_SENSOR_DATA_TYPES = (
    "rgb",
    "albedo",
    "simple_shading_constant_diffuse",
    "simple_shading_diffuse_mdl",
    "simple_shading_full_mdl",
    "semantic_segmentation",
    "depth",
    "distance_to_camera",
    "distance_to_image_plane",
    "normals",
    "instance_segmentation_fast",
    "instance_id_segmentation_fast",
    "motion_vectors",
)


# Data types the Newton Warp renderer (``newton_renderer``) supports. Expand this tuple as the
# renderer gains support for additional output types.
_NEWTON_WARP_DATA_TYPES = (
    "rgb",
    "depth",
    "distance_to_camera",
    "distance_to_image_plane",
    "normals",
    "semantic_segmentation",
    "instance_segmentation_fast",
)

# Data types the OVRTX renderer supports. ``instance_id_segmentation_fast`` is intentionally
# excluded: it has no real-world sensor equivalent, so the OVRTX integration does not support it.
# Users should use ``instance_segmentation_fast`` or ``semantic_segmentation`` instead.
_OVRTX_DATA_TYPES = tuple(dt for dt in _DEFAULT_SENSOR_DATA_TYPES if dt != "instance_id_segmentation_fast")

_OVRTX_TEXTURE_READINESS_DATA_TYPES = (
    "albedo",
    "simple_shading_diffuse_mdl",
    "simple_shading_full_mdl",
)
_OVRTX_TEXTURE_READINESS_XFAIL_REASON = "OVRTX 0.4 may return before textured materials are ready (NVBUG#6505191)."


def make_xfail_rendering_params(
    params: list[pytest.param],
    expected_failures: dict[tuple[str, str, str], str],
) -> list[pytest.param]:
    """Mark selected rendering parameter combinations as expected failures.

    Args:
        params: Rendering parameters containing physics backend, renderer, and data type values.
        expected_failures: Mapping from parameter value tuples to expected-failure reasons.

    Returns:
        Rendering parameters with non-strict ``xfail`` marks applied to matching combinations.
    """
    marked_params = []
    for param in params:
        reason = expected_failures.get(tuple(param.values))
        if reason is None:
            marked_params.append(param)
            continue
        # Expected failures should run once instead of consuming the RTX flaky-retry budget.
        marks = [mark for mark in param.marks if mark.name != "flaky"]
        marked_params.append(
            pytest.param(
                *param.values,
                id=param.id,
                marks=[*marks, pytest.mark.xfail(reason=reason, strict=False)],
            )
        )
    return marked_params


def _make_sensor_data_type_params(
    physics_backend: str,
    renderer: str,
    sensor_data_types: list[str] | None = None,
    *,
    flaky: bool = True,
    renderer_label: str | None = None,
) -> list[pytest.param]:
    """Create golden-image parameter entries for the given data types.

    Args:
        physics_backend: Physics backend label (e.g. ``"physx"``, ``"newton"``, ``"ovphysx"``).
        renderer: Renderer nickname; the camera renderer argument becomes ``f"{renderer}_renderer"``.
        sensor_data_types: Data types to parametrize. Defaults to :data:`_DEFAULT_SENSOR_DATA_TYPES`.
        flaky: Whether to apply the retry mark. RTX renderers are non-deterministic and need it; the
            deterministic Warp rasterizer does not.
        renderer_label: Overrides the renderer segment of the test id. Defaults to ``renderer``; used
            to keep the ``newton_warp`` id distinct from its ``newton_renderer`` argument.
    """
    sensor_data_types = list(sensor_data_types or _DEFAULT_SENSOR_DATA_TYPES)
    label = renderer_label or renderer
    marks = _FLAKY_MARK if flaky else ()
    return [
        pytest.param(
            physics_backend,
            f"{renderer}_renderer",
            data_type,
            id=f"{physics_backend}-{label}-{data_type}",
            marks=marks,
        )
        for data_type in sensor_data_types
    ]


PHYSICS_RENDERER_AOV_COMBINATIONS = [
    *_make_sensor_data_type_params("physx", "isaacsim_rtx"),
    *_make_sensor_data_type_params("newton", "isaacsim_rtx"),
    *_make_sensor_data_type_params(
        "physx", "newton", _NEWTON_WARP_DATA_TYPES, flaky=False, renderer_label="newton_warp"
    ),
]

KITLESS_PHYSICS_RENDERER_AOV_COMBINATIONS = make_xfail_rendering_params(
    [
        *_make_sensor_data_type_params("ovphysx", "ovrtx", _OVRTX_DATA_TYPES),
        *_make_sensor_data_type_params("newton", "ovrtx", _OVRTX_DATA_TYPES),
        *_make_sensor_data_type_params(
            "ovphysx", "newton", _NEWTON_WARP_DATA_TYPES, flaky=False, renderer_label="newton_warp"
        ),
        *_make_sensor_data_type_params(
            "newton", "newton", _NEWTON_WARP_DATA_TYPES, flaky=False, renderer_label="newton_warp"
        ),
    ],
    {
        ("newton", "ovrtx_renderer", data_type): _OVRTX_TEXTURE_READINESS_XFAIL_REASON
        for data_type in _OVRTX_TEXTURE_READINESS_DATA_TYPES
    },
)


# Tolerances for the numeric transform comparison. Transform entries mix unit-scale rotation
# components with translations in metres, so a small absolute floor plus a relative term absorbs
# per-platform / per-Kit-build float noise while still catching real pose changes.
_STAGE_TRANSFORM_RTOL = 1e-4
_STAGE_TRANSFORM_ATOL = 1e-5

# Cap on the number of reported stage differences so a large regression stays readable.
_MAX_STAGE_DIFF_LINES = 80


def extract_stage_structure_and_transforms(usd_path: str) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    """Open a USD stage and return its prim structure and per-prim world transforms.

    Args:
        usd_path: Path to a ``.usda``/``.usd`` file to open and compose.

    Returns:
        A ``(structure, transforms)`` pair. ``structure`` maps every prim path to its type name.
        ``transforms`` maps each :class:`~pxr.UsdGeom.Xformable` prim path to its 4x4 local-to-world
        transform as a ``float64`` array.
    """
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage at {usd_path}.")

    structure: dict[str, str] = {}
    transforms: dict[str, np.ndarray] = {}
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        structure[path] = str(prim.GetTypeName())
        if UsdGeom.Xformable(prim):
            matrix = xform_cache.GetLocalToWorldTransform(prim)
            transforms[path] = np.array([[matrix[row][col] for col in range(4)] for row in range(4)], dtype=np.float64)
    return structure, transforms


def compare_golden_stage(golden_path: str, result_path: str) -> list[str]:
    """Compare two USD stages by prim structure (exact) and world transforms (``allclose``).

    Prim structure — the set of prim paths and their type names — must match exactly. Transforms of
    prims present in both stages are compared with :func:`numpy.allclose` so per-platform float noise
    does not trip the comparison. Returns a list of human-readable difference descriptions (empty
    when the stages match).
    """
    golden_structure, golden_transforms = extract_stage_structure_and_transforms(golden_path)
    result_structure, result_transforms = extract_stage_structure_and_transforms(result_path)

    problems: list[str] = []
    golden_paths, result_paths = set(golden_structure), set(result_structure)
    for path in sorted(result_paths - golden_paths):
        problems.append(f"+ added prim {path} ({result_structure[path]})")
    for path in sorted(golden_paths - result_paths):
        problems.append(f"- removed prim {path} ({golden_structure[path]})")
    for path in sorted(golden_paths & result_paths):
        if golden_structure[path] != result_structure[path]:
            problems.append(f"~ type changed {path}: {golden_structure[path]} -> {result_structure[path]}")

    for path in sorted(set(golden_transforms) & set(result_transforms)):
        golden_matrix, result_matrix = golden_transforms[path], result_transforms[path]
        if not np.allclose(golden_matrix, result_matrix, rtol=_STAGE_TRANSFORM_RTOL, atol=_STAGE_TRANSFORM_ATOL):
            max_diff = float(np.max(np.abs(golden_matrix - result_matrix)))
            problems.append(f"~ transform {path}: max abs diff {max_diff:.3e} exceeds tolerance")

    return problems


def _sanitize_golden_stage_text(text: str) -> str:
    """Strip machine-specific provenance from a flattened golden so it commits reproducibly.

    Blanks the volatile ``doc`` provenance block (which embeds the generating host's temp path) and
    masks absolute filesystem asset paths left in material/texture attributes. Neither is consulted
    by the structure/transform comparison, so masking keeps the committed baseline free of local,
    host-specific paths without affecting correctness. Portable asset tokens (e.g. ``@OmniPBR.mdl@``)
    have no leading drive/slash and are left untouched.
    """
    text = re.sub(r'doc = """.*?"""', 'doc = """"""', text, flags=re.DOTALL)
    text = re.sub(r"@(?:[A-Za-z]:)?[\\/][^@\n]*@", "@<MASKED_ASSET_PATH>@", text)
    # Ensure a single trailing newline so regeneration stays clean under the end-of-file hook.
    return text.rstrip("\n") + "\n"


def maybe_save_stage(
    test_name: str,
    physics_backend: str,
    renderer: str,
    data_type: str,
    *,
    compare_golden: bool = False,
) -> None:
    """Dump the current USD stage and optionally compare it against a golden USDA file.

    When ``ISAAC_LAB_SAVE_STAGES`` is set, the stage is written to that directory. When
    ``compare_golden`` is True, the exported stage is validated against
    ``golden_stages/<test_name>/<physics_backend>-<renderer>-<data_type>.usda`` by opening both
    stages and comparing prim structure exactly and world transforms with :func:`numpy.allclose`
    (see :func:`compare_golden_stage`). A missing baseline is bootstrapped and the test fails.
    """
    out_dir = os.environ.get("ISAAC_LAB_SAVE_STAGES")
    if not out_dir and not compare_golden:
        return

    import isaaclab.sim as sim_utils

    safe_test_name = test_name.replace("/", "_")
    stage_basename = f"{safe_test_name}-{physics_backend}-{renderer}-{data_type}.usda"

    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as tmp_file:
        stage_path = tmp_file.name

    try:
        if not sim_utils.save_stage(stage_path, save_and_reload_in_place=False):
            pytest.fail(f"save_stage reported failure while writing the USD stage to {stage_path}.")

        from pxr import Usd  # noqa: PLC0415

        # Flatten the saved stage to inline sublayer references and resolve asset paths.
        opened_stage = Usd.Stage.Open(stage_path)
        if opened_stage is None:
            pytest.fail(f"Could not open the saved stage at {stage_path} to flatten.")
        flat_layer = opened_stage.Flatten()
        if flat_layer is None:
            pytest.fail(f"Could not flatten the saved stage at {stage_path}.")

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, stage_basename)
            if not flat_layer.Export(out_path):
                pytest.fail(f"Failed to export the flattened stage to {out_path}.")
            logger.info("[ISAAC_LAB_SAVE_STAGES] wrote %s", out_path)

        if compare_golden:
            golden_dir = os.path.join(_GOLDEN_STAGES_DIRECTORY, safe_test_name)
            os.makedirs(golden_dir, exist_ok=True)
            golden_path = os.path.join(golden_dir, f"{physics_backend}-{renderer}-{data_type}.usda")

            if not os.path.exists(golden_path):
                if not flat_layer.Export(golden_path):
                    pytest.fail(f"Failed to export the flattened golden baseline to {golden_path}.")
                # Strip host-specific provenance/paths so the committed baseline is reproducible.
                with open(golden_path, encoding="utf-8") as file:
                    golden_text = file.read()
                with open(golden_path, "w", encoding="utf-8", newline="\n") as file:
                    file.write(_sanitize_golden_stage_text(golden_text))
                pytest.fail(f"Golden stage not found at {golden_path}. A new baseline was written.")

            # Write the flattened result to a temp file so both sides of the comparison use
            # the same representation as the bootstrap wrote for the golden.
            with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as flat_tmp:
                flat_stage_path = flat_tmp.name
            try:
                if not flat_layer.Export(flat_stage_path):
                    pytest.fail("Failed to write flattened result stage for comparison.")
                problems = compare_golden_stage(golden_path, flat_stage_path)
            finally:
                if os.path.exists(flat_stage_path):
                    os.unlink(flat_stage_path)

            if problems:
                diff_summary = "\n".join(problems[:_MAX_STAGE_DIFF_LINES])
                if len(problems) > _MAX_STAGE_DIFF_LINES:
                    diff_summary += f"\n... ({len(problems) - _MAX_STAGE_DIFF_LINES} more differences)"
                pytest.fail(
                    f"{test_name} (physics={physics_backend}, renderer={renderer}, data_type={data_type}) "
                    f"USD stage mismatch:\n{diff_summary}"
                )
    finally:
        if os.path.exists(stage_path):
            os.unlink(stage_path)


def _apply_overrides_to_env_cfg(env_cfg: Any, override_args: list[str]) -> Any:
    """Apply override args to env_cfg using parse_overrides and apply_overrides."""
    from isaaclab_tasks.utils.hydra import apply_overrides, collect_presets, parse_overrides

    presets = {"env": collect_presets(env_cfg)}
    global_presets, preset_sel, preset_scalar, _ = parse_overrides(override_args, presets)
    hydra_cfg = {"env": env_cfg.to_dict()}
    env_cfg, _ = apply_overrides(env_cfg, None, hydra_cfg, global_presets, preset_sel, preset_scalar, presets)
    return env_cfg


def _redirect_ovrtx_renderer_log_to_stdout(env_cfg: Any) -> None:
    """Point OVRTX renderer logs at process stdout for kitless rendering tests.

    Walks camera cfgs (``env_cfg.tiled_camera`` for direct envs, ``env_cfg.scene.base_camera`` / ``wrist_camera`` for
    manager-based envs) and sets :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.log_file_path` on each camera whose
    resolved ``renderer_cfg.renderer_type`` is ``"ovrtx"``. Uses ``/dev/stdout`` on Linux and ``CON`` on Windows so
    pytest captures OVRTX renderer log.
    """
    camera_cfgs: list[Any] = []

    # direct envs
    tiled_camera = getattr(env_cfg, "tiled_camera", None)
    if tiled_camera is not None:
        camera_cfgs.append(tiled_camera)

    # manager-based envs
    scene = getattr(env_cfg, "scene", None)
    if scene is not None:
        for camera_name in ("base_camera", "wrist_camera", "tiled_camera"):
            camera_cfg = getattr(scene, camera_name, None)
            if camera_cfg is not None:
                camera_cfgs.append(camera_cfg)

    # redirect OVRTX renderer log to stdout
    for camera_cfg in camera_cfgs:
        renderer_cfg = getattr(camera_cfg, "renderer_cfg", None)
        if renderer_cfg is not None and getattr(renderer_cfg, "renderer_type", None) == "ovrtx":
            renderer_cfg.log_file_path = "CON" if os.name == "nt" else "/dev/stdout"


def _maybe_enable_physx_determinism_for_motion(env_cfg: Any, physics_backend: str, data_type: str) -> None:
    """Trade PhysX solver performance for determinism/accuracy when testing ``motion_vectors``.

    PhysX's default TGS solver settings produce noisy per-step velocities (see the "TGS solver ... may
    cause noisy velocities" warning logged by ``physx_manager``), and ``motion_vectors`` encodes velocity
    directly. That noise differs run-to-run, making golden-image comparisons for this AOV flaky on CI.
    Applies to both PhysX backends (``"physx"`` and ``"ovphysx"``, which share the same underlying PhysX
    solver). No-op for any other ``(physics_backend, data_type)`` combination.

    Args:
        env_cfg: The resolved environment config, exposing ``sim.physics`` as a
            :class:`~isaaclab_physx.physics.PhysxCfg` when ``physics_backend == "physx"``, or an
            :class:`~isaaclab_ovphysx.physics.OvPhysxCfg` when ``physics_backend == "ovphysx"``.
        physics_backend: The physics backend under test (``"physx"``, ``"newton"``, or ``"ovphysx"``).
        data_type: The camera data type under test.
    """
    if physics_backend not in ("physx", "ovphysx") or data_type != "motion_vectors":
        return

    env_cfg.sim.physics.enable_enhanced_determinism = True
    env_cfg.sim.physics.enable_external_forces_every_iteration = True


def _maybe_disable_instancing_for_current_stage(physics_backend: str, renderer: str, data_type: str) -> None:
    """Disable USD instancing for the current stage to work around NVBUG#6418121.

    HDC_TODO: Remove this temporary workaround once NVBUG#6418121 is fixed.
    """
    disable_instancing = False

    if renderer == "isaacsim_rtx_renderer":
        disable_instancing = True

    if disable_instancing:
        from isaaclab.sim.utils.prims import get_current_stage, make_uninstanceable

        stage = get_current_stage()
        make_uninstanceable("/World/envs", stage)

        print("[rendering_test_utils] Disabled USD instancing for the current stage to work around NVBUG#6418121.")


def _skip_if_newton_motion_vectors(physics_backend: str, data_type: str) -> None:
    """Skip ``motion_vectors`` golden-image tests running on the Newton physics backend.

    Newton is not yet deterministic enough for the ``motion_vectors`` AOV: per-pixel (u, v) offsets
    encode sub-step body motion, which varies run-to-run under the Newton solver, so golden-image
    comparison is unreliable. No-op for any other ``(physics_backend, data_type)`` combination.

    Args:
        physics_backend: The physics backend under test (``"physx"``, ``"newton"``, or ``"ovphysx"``).
        data_type: The camera data type under test.
    """
    if physics_backend == "newton" and data_type == "motion_vectors":
        pytest.skip("Newton physics is not deterministic enough for motion_vectors golden-image testing.")


def _physics_preset_name(physics_backend: str) -> str:
    """Translate the historical ``"newton"`` backend label (still used by golden-image
    filenames and ``pytest.param`` IDs) to the renamed Hydra preset
    ``"newton_mjwarp"``. Other labels (``"physx"`` etc.) pass through unchanged.
    """
    return "newton_mjwarp" if physics_backend == "newton" else physics_backend


def _physics_preset_name_deformable(physics_backend: str) -> str:
    """Map deformable-test physics labels to Hydra preset names."""
    return "newton_mjwarp_vbd" if physics_backend == "newton" else physics_backend


def _skip_if_physics_preset_unsupported(env_cfg: Any, physics_preset_name: str) -> None:
    """Skip the test when the env does not support the given physics preset.

    An env cfg may intentionally support only a subset of the physics presets - newton, physx, ovphysx.
    Rather than hard-coding the unsupported-backend list per test, this inspects the resolved
    ``env_cfg.sim.physics`` :class:`~isaaclab_tasks.utils.PresetCfg` and skips any backend whose
    Hydra preset name is not declared as a field.

    Args:
        env_cfg: The environment config, exposing its physics presets at ``sim.physics``.
        physics_preset_name: The physics preset name (e.g. ``"newton_mjwarp"``).
    """
    from isaaclab_tasks.utils import PresetCfg

    physics_cfg = getattr(getattr(env_cfg, "sim", None), "physics", None)
    if not isinstance(physics_cfg, PresetCfg):
        return

    # Preset variants are declared as annotated fields; aliases such as ``default`` are not.
    supported_physics_preset_names = set(getattr(type(physics_cfg), "__dataclass_fields__", {}))
    if physics_preset_name not in supported_physics_preset_names:
        pytest.skip(f"{type(env_cfg).__name__} does not support '{physics_preset_name}'.")


def _save_comparison_image(img: Image.Image, filename: str) -> str:
    """Save a PIL image under the comparison images directory."""
    path = os.path.join(_COMPARISON_IMAGES_DIR, _COMPARISON_IMAGE_SUBDIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, format="PNG")
    return path


def _format_bcompare_command(actual_path: str, golden_path: str) -> str:
    """Build a shell command that opens actual and golden images in Beyond Compare."""
    return f"bcompare \\\n  {actual_path} \\\n  {golden_path}"


def generate_html_report(comparison_scores: list[dict], report_filename: str) -> None:
    """Generate and save an HTML report of comparison scores."""
    if not comparison_scores:
        return

    os.makedirs(_COMPARISON_IMAGES_DIR, exist_ok=True)
    report_path = os.path.join(_COMPARISON_IMAGES_DIR, report_filename)
    sorted_scores = sorted(comparison_scores, key=lambda e: -e["diff_pct"])

    rows = []
    for entry in sorted_scores:
        status_class = "pass" if entry["passed"] else "fail"
        status_text = status_class.upper()

        actual_img_html = ""
        golden_img_html = ""
        compare_html = ""
        if entry.get("img_result_path"):
            actual_fname = os.path.relpath(entry["img_result_path"], _COMPARISON_IMAGES_DIR)
            golden_fname = os.path.relpath(entry["img_golden_path"], _COMPARISON_IMAGES_DIR)
            actual_img_html = f'<a href="{actual_fname}"><img src="{actual_fname}" width="120" loading="lazy"></a>'
            golden_img_html = f'<a href="{golden_fname}"><img src="{golden_fname}" width="120" loading="lazy"></a>'
            compare_cmd = _format_bcompare_command(actual_fname, golden_fname)
            compare_html = (
                '<div class="compare-cmd-wrap">'
                f'<code class="compare-cmd">{compare_cmd}</code>'
                '<button type="button" class="copy-btn" title="Copy command" aria-label="Copy command"'
                f' onclick="copyCompareCmd(this)">{_COPY_ICON_SVG}{_CHECK_ICON_SVG}</button>'
                "</div>"
            )

        ssim_checked = entry.get("ssim_checked", True)
        ssim_cell_class = "" if ssim_checked else ' class="ssim-disabled"'
        entry_ssim_threshold = entry.get("ssim_threshold", _SSIM_THRESHOLD)
        ssim_threshold_cell = f"{entry_ssim_threshold:.4f}" if ssim_checked else "N/A"
        ssim_title = "" if ssim_checked else ' title="SSIM gate disabled for this data type; score is informational."'

        rows.append(
            f'<tr class="{status_class}">'
            f"<td>{entry['test']}</td>"
            f"<td>{entry['backend']}</td>"
            f"<td>{entry['renderer']}</td>"
            f"<td>{entry['aov']}</td>"
            f"<td>{entry['diff_pct']:.2f}</td>"
            f"<td>{entry['threshold']:.1f}</td>"
            f"<td{ssim_cell_class}{ssim_title}>{entry['ssim']:.4f}</td>"
            f"<td{ssim_cell_class}{ssim_title}>{ssim_threshold_cell}</td>"
            f'<td class="status-{status_class}">{status_text}</td>'
            f"<td>{actual_img_html}</td>"
            f"<td>{golden_img_html}</td>"
            f'<td class="compare-cell">{compare_html}</td>'
            "</tr>"
        )

    report_html = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>Rendering Correctness - Image Comparison Report</title>\n"
        "<style>\n"
        "  body { font-family: sans-serif; font-size: 13px; margin: 16px; }\n"
        "  h1 { font-size: 1.3em; margin-bottom: 4px; }\n"
        "  p { margin-top: 4px; color: #555; }\n"
        "  table { border-collapse: collapse; width: 100%; }\n"
        "  th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; vertical-align: middle; }\n"
        "  th { background: #f0f0f0; white-space: nowrap; }\n"
        "  tr.fail { background: #fff0f0; }\n"
        "  tr.pass:hover, tr.fail:hover { filter: brightness(0.96); }\n"
        "  .status-pass { color: #2a7a2a; font-weight: bold; }\n"
        "  .status-fail { color: #cc0000; font-weight: bold; }\n"
        "  .ssim-disabled { color: #999; font-style: italic; }\n"
        "  img { display: block; max-width: 120px; height: auto; }\n"
        "  .compare-cell { max-width: 420px; }\n"
        "  .compare-cmd-wrap { position: relative; }\n"
        "  .compare-cmd { display: block; font-size: 11px; white-space: pre-wrap; word-break: break-all;"
        " background: #f8f8f8; padding: 4px 28px 4px 6px; border: 1px solid #ddd; border-radius: 3px;"
        " user-select: all; }\n"
        "  .copy-btn { position: absolute; top: 4px; right: 4px; width: 22px; height: 22px; padding: 0;"
        " border: none; border-radius: 4px; background: transparent; color: #666; cursor: pointer; }\n"
        "  .copy-btn:hover { background: #e8e8e8; color: #333; }\n"
        "  .copy-btn svg { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);"
        " width: 14px; height: 14px; }\n"
        "  .copy-btn .check-icon { display: none; }\n"
        "  .copy-btn.copied .copy-icon { display: none; }\n"
        "  .copy-btn.copied .check-icon { display: block; }\n"
        "</style>\n"
        "<script>\n"
        "function copyCompareCmd(btn) {\n"
        "  const text = btn.previousElementSibling.textContent;\n"
        "  navigator.clipboard.writeText(text).then(() => {\n"
        "    btn.classList.add('copied');\n"
        "    btn.title = 'Copied!';\n"
        "    setTimeout(() => {\n"
        "      btn.classList.remove('copied');\n"
        "      btn.title = 'Copy command';\n"
        "    }, 1500);\n"
        "  });\n"
        "}\n"
        "</script>\n"
        "</head>\n"
        "<body>\n"
        "<h1>Rendering Correctness - Image Comparison Report</h1>\n"
        f"<p>Sorted by PixelDiff&nbsp;% (desc) - {len(sorted_scores)}&nbsp; total.</p>\n"
        "<table>\n"
        "<thead><tr>"
        "<th>Test</th>"
        "<th>Backend</th>"
        "<th>Renderer</th>"
        "<th>AOV</th>"
        "<th>PixelDiff&nbsp;%</th>"
        "<th>PixelDiff Threshold&nbsp;%</th>"
        "<th>SSIM</th>"
        "<th>SSIM Threshold</th>"
        "<th>Status</th>"
        "<th>ACTUAL</th>"
        "<th>GOLDEN</th>"
        "<th>Beyond Compare Command</th>"
        "</tr></thead>\n"
        "<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>\n"
        f"<p>Generated:&nbsp;{datetime.now().astimezone().isoformat(timespec='seconds')}.</p>\n"
        "</body>\n"
        "</html>\n"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report_html)


def attach_comparison_properties(
    request: pytest.FixtureRequest, comparison_scores: list[dict], initial_count: int
) -> None:
    """Attach pixel-diff, SSIM scores, and failure images as JUnit XML properties."""
    for entry in comparison_scores[initial_count:]:
        label = f"{entry['backend']}-{entry['renderer']}-{entry['aov']}"
        request.node.user_properties.append((f"diff_pct:{label}", f"{entry['diff_pct']:.2f}"))
        ssim_value = f"{entry['ssim']:.4f}" if entry.get("ssim_checked", True) else f"{entry['ssim']:.4f} (N/A)"
        request.node.user_properties.append((f"ssim:{label}", ssim_value))
        request.node.user_properties.append((f"threshold:{label}", f"{entry['threshold']:.1f}"))
        if entry.get("img_result_path"):
            request.node.user_properties.append((f"img_result:{label}", entry["img_result_path"]))
            request.node.user_properties.append((f"img_golden:{label}", entry["img_golden_path"]))


def make_determinism_fixture():
    """Create an autouse fixture that enables determinism for each test."""

    @pytest.fixture(autouse=True)
    def _determinism_fixture():
        """Enable determinism for each test."""
        from isaaclab.utils.seed import configure_seed

        configure_seed(42, torch_deterministic=True)

        yield

        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()

    return _determinism_fixture


def make_generate_html_report_fixture(comparison_scores: list[dict], report_filename: str):
    """Create a session fixture that writes the HTML report for one module.

    Args:
        comparison_scores: Module-local comparison score storage.
        report_filename: Output report filename.
    """

    @pytest.fixture(scope="session", autouse=True)
    def _generate_html_report():
        """Generate an HTML comparison report after all tests in the session complete."""
        yield
        generate_html_report(comparison_scores, report_filename)

    return _generate_html_report


def make_attach_comparison_properties_fixture(comparison_scores: list[dict]):
    """Create an autouse fixture that attaches JUnit properties for one module.

    Args:
        comparison_scores: Module-local comparison score storage.
    """

    @pytest.fixture(autouse=True)
    def _attach_comparison_properties(request):
        """Attach pixel-diff, SSIM scores, and failure images as JUnit XML properties."""
        initial_count = len(comparison_scores)
        yield
        attach_comparison_properties(request, comparison_scores, initial_count)

    return _attach_comparison_properties


def make_require_ovlibs_install_fixture():
    """Create an autouse fixture that fails fast when OV libraries are required but not installed.

    Only parametrized cases with ``renderer == "ovrtx_renderer"`` or ``physics_backend == "ovphysx"`` are checked.
    Install with ``./isaaclab.sh -i 'ov[all]'`` (or the equivalent in your environment).
    """

    @pytest.fixture(autouse=True)
    def _require_ovlibs_install(request, monkeypatch: pytest.MonkeyPatch):
        # TODO: Remove once usd-core>=26.5 is the minimum - that release fixes the race condition.
        # Limit OpenUSD's work-thread pool to one thread to avoid race condition in usd-core<26.5
        monkeypatch.setenv("PXR_WORK_THREAD_LIMIT", "1")

        callspec = getattr(request.node, "callspec", None)
        if callspec is None:
            return

        if callspec.params.get("renderer") == "ovrtx_renderer":
            monkeypatch.setenv("ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS", "0")
            try:
                import ovrtx

                print(f"ovrtx version: {ovrtx.__version__}")
            except ImportError as exc:
                pytest.fail(
                    "Kitless OVRTX rendering tests require the optional dependency ov[ovrtx]. "
                    "Install with: ./isaaclab.sh -i 'ov[ovrtx]'\n"
                    f"ImportError: {exc}"
                )

        if callspec.params.get("physics_backend") == "ovphysx":
            try:
                import ovphysx

                print(f"ovphysx version: {ovphysx.__version__}")
            except ImportError as exc:
                pytest.fail(
                    "Kitless OVPhysX rendering tests require the optional dependency ov[ovphysx]. "
                    "Install with: ./isaaclab.sh -i 'ov[ovphysx]'\n"
                    f"ImportError: {exc}"
                )

    return _require_ovlibs_install


def _ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """Compute mean SSIM between two (1, C, H, W) float tensors in [0, 1]."""
    c1 = 0.01**2
    c2 = 0.03**2
    channels = img1.shape[1]
    pad = window_size // 2

    kernel = torch.ones(channels, 1, window_size, window_size, device=img1.device, dtype=img1.dtype) / (
        window_size * window_size
    )

    mu1 = torch.nn.functional.conv2d(img1, kernel, padding=pad, groups=channels)
    mu2 = torch.nn.functional.conv2d(img2, kernel, padding=pad, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, kernel, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


def _pixel_diff_percentage(
    result_image: Image.Image,
    golden_image: Image.Image,
    pixel_diff_threshold: float = _PIXEL_L2_NORM_DIFFERENCE_THRESHOLD,
) -> float:
    """Compute the percentage of pixels whose L2 norm difference exceeds a threshold."""
    diff_array = np.array(ImageChops.difference(result_image, golden_image))
    l2_norm_array = np.linalg.norm(diff_array, axis=2)
    num_different_pixels = np.sum(l2_norm_array > pixel_diff_threshold)
    return 100.0 * num_different_pixels / l2_norm_array.size


def _compare_images(
    result_image: Image.Image,
    golden_image: Image.Image,
    max_different_pixels_percentage: float,
    check_ssim: bool = True,
    ssim_threshold: float = _SSIM_THRESHOLD,
) -> tuple[bool, str | None, float, float]:
    """Compare result and golden images against pixel and SSIM thresholds."""
    if result_image.size != golden_image.size:
        return False, f"Size mismatch: expected {golden_image.size}, got {result_image.size}.", 0.0, 0.0

    if result_image.mode != golden_image.mode:
        return False, f"Mode mismatch: expected {golden_image.mode}, got {result_image.mode}.", 0.0, 0.0

    diff_pct = _pixel_diff_percentage(result_image, golden_image)

    result_tensor = torch.from_numpy(np.array(result_image, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    golden_tensor = torch.from_numpy(np.array(golden_image, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    ssim_score = _ssim(result_tensor, golden_tensor)

    if diff_pct > max_different_pixels_percentage:
        return (
            False,
            f"The percentage of different pixels ({diff_pct:.2f}%) exceeds the threshold of"
            f" {max_different_pixels_percentage:.2f}%. SSIM={ssim_score:.4f}.",
            diff_pct,
            ssim_score,
        )

    if check_ssim and ssim_score < ssim_threshold:
        return (
            False,
            f"SSIM ({ssim_score:.4f}) is below the threshold of {ssim_threshold:.4f}."
            f" Different pixels: {diff_pct:.2f}%.",
            diff_pct,
            ssim_score,
        )

    return True, None, diff_pct, ssim_score


def validate_camera_outputs(
    test_name: str,
    physics_backend: str,
    renderer: str,
    camera_outputs: dict[str, ProxyArray],
    max_different_pixels_percentage: float,
    comparison_scores: list[dict],
) -> None:
    """Validate correctness and consistency of camera outputs."""
    assert len(camera_outputs) > 0, f"[{test_name}] No camera outputs produced by {physics_backend} + {renderer}."

    golden_image_dir = os.path.join(_GOLDEN_IMAGES_DIRECTORY, test_name)
    os.makedirs(golden_image_dir, exist_ok=True)

    ssim_threshold = _SSIM_THRESHOLD_BY_ENV_NAME.get(test_name, _SSIM_THRESHOLD)
    failed_data_types = {}

    for data_type, output in camera_outputs.items():
        tensor = output if isinstance(output, torch.Tensor) else output.torch
        condition = torch.logical_or(torch.isinf(tensor), torch.isnan(tensor))
        corrected = torch.where(condition, torch.zeros_like(tensor), tensor)
        max_val = corrected.abs().max()
        if max_val <= 0:
            failed_data_types[data_type] = f"Camera output '{data_type}' has no non-zero pixels."
            continue

        normalized = normalize_camera_output_for_display(corrected, data_type)
        grid = make_camera_output_grid(normalized)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        result_image = Image.fromarray(ndarr)

        golden_path = os.path.join(golden_image_dir, f"{physics_backend}-{renderer}-{data_type}.png")
        if not os.path.exists(golden_path):
            failed_data_types[data_type] = f"Golden image not found at {golden_path}."
            result_image.save(golden_path)
            continue

        try:
            golden_image = Image.open(golden_path)
        except Exception as error:  # noqa: BLE001
            failed_data_types[data_type] = f"Error opening golden image: {error}"
            continue

        # Instance seg colors are non-stable across runs (NonStableInstanceSegmentation /
        # InstanceSegmentationSD IDs vary). Compare only the alpha channel (instance mask shape)
        # by zeroing RGB on both images.
        result_image_for_comparison = result_image
        golden_image_for_comparison = golden_image
        if data_type in {"instance_segmentation_fast", "instance_id_segmentation_fast"}:

            def _alpha_only(img: Image.Image) -> Image.Image:
                r, g, b, a = img.split()
                zero = r.point(lambda _: 0)
                return Image.merge("RGBA", (zero, zero, zero, a))

            result_image_for_comparison = _alpha_only(result_image)
            golden_image_for_comparison = _alpha_only(golden_image)

        check_ssim = data_type not in _SSIM_DISABLED_DATA_TYPES
        succeeded, error_message, diff_pct, ssim_score = _compare_images(
            result_image_for_comparison,
            golden_image_for_comparison,
            max_different_pixels_percentage,
            check_ssim=check_ssim,
            ssim_threshold=ssim_threshold,
        )

        entry = {
            "test": test_name,
            "backend": physics_backend,
            "renderer": renderer,
            "aov": data_type,
            "diff_pct": diff_pct,
            "ssim": ssim_score,
            "ssim_checked": check_ssim,
            "threshold": max_different_pixels_percentage,
            "ssim_threshold": ssim_threshold,
            "passed": succeeded,
            "img_result_path": None,
            "img_golden_path": None,
        }

        # We want to keep the record in the test artifact so that we can have a chance to see whether
        # test cases that appear successful is a false positive or not, e.g. image diff threshold is set
        # 8%, if the result image is 7% different from the golden, it is considered a pass
        if diff_pct > 0:
            prefix = f"{test_name}-{physics_backend}-{renderer}-{data_type}"
            entry["img_result_path"] = _save_comparison_image(result_image, f"{prefix}-actual.png")
            entry["img_golden_path"] = _save_comparison_image(golden_image, f"{prefix}-golden.png")

        comparison_scores.append(entry)

        if not succeeded:
            failed_data_types[data_type] = error_message

    if failed_data_types:
        reason = f"{test_name} (physics={physics_backend}, renderer={renderer}) failed for the following data types:\n"
        for data_type, error_message in failed_data_types.items():
            reason += f"- {data_type}: {error_message}\n"
        reason += f"Comparison images were written to {_COMPARISON_IMAGES_DIR}."
        pytest.fail(reason)


def maybe_validate_semantic_segmentation(
    data_type: str,
    info: dict[str, Any] | None,
    expected_id_to_labels: dict[tuple[int, int, int, int], dict[str, str]],
    stringify_keys: bool = False,
) -> None:
    """For the ``semantic_segmentation`` data type, assert ``camera.data.info`` matches ground truth.

    No-op for any other data type. The ``idToLabels`` mapping (keyed by the colorized RGBA value for the
    default colorized output) is a Replicator contract that both the Isaac RTX and OVRTX renderers must
    satisfy, so it is compared for exact equality against the expected mapping.

    ``expected_id_to_labels`` is keyed by raw ``(r, g, b, a)`` **tuples** (the form the OVRTX renderer and
    Replicator's fast nodes produce). isaacsim's Isaac RTX ``semantic_segmentation`` annotator, however,
    returns the stringified ``"(r, g, b, a)"`` keys, so pass ``stringify_keys=True`` for that renderer to cast
    the expected keys to ``str`` before comparing.

    Args:
        data_type: The camera data type under test.
        info: The ``camera.data.info`` dict (or ``None``) produced by the renderer.
        expected_id_to_labels: Expected ``idToLabels`` mapping (RGBA-tuple key -> ``{semantic_type: label}``).
        stringify_keys: Cast the expected keys to ``str`` before comparing, to match a renderer that returns
            stringified color keys (isaacsim Isaac RTX ``semantic_segmentation``).
    """
    if data_type != "semantic_segmentation":
        return

    expected = (
        {str(key): value for key, value in expected_id_to_labels.items()} if stringify_keys else expected_id_to_labels
    )

    assert info is not None, "camera.data.info is None; renderer did not populate segmentation metadata."
    assert info["semantic_segmentation"]["idToLabels"] == expected, (
        f"semantic_segmentation idToLabels mismatch.\n"
        f"  expected: {expected}\n"
        f"  actual:   {info['semantic_segmentation']}"
    )


def maybe_validate_instance_segmentation_fast(
    data_type: str,
    camera_data: "CameraData",
    expected_prim_paths: set[str],
    expected_semantics: list[dict[str, str]],
) -> None:
    """For the ``instance_segmentation_fast`` data type, assert ``camera.data.info`` matches ground truth.

    No-op for any other data type. Instance segmentation exposes two mappings that the Isaac RTX and OVRTX
    renderers must both satisfy: ``idToLabels`` (instance -> USD prim path) and ``idToSemantics`` (instance ->
    ``{semantic_type: label}``), both keyed by the colorized ``(r, g, b, a)`` instance value.

    The colorized keys come from ``NonStableInstanceSegmentation`` ids, whose id -> prim assignment is not
    stable across runs, so the keys cannot be hard-coded. Instead they are validated for self-consistency
    against the rendered image: each map's key set must equal the unique colors in ``instance_image`` plus the
    reserved BACKGROUND / UNLABELLED keys (which the maps always include even when those pixels are not
    rendered). The *values* are then compared against the expected ground truth.

    Concretely it checks:

    - each map's key set equals the unique image colors plus the reserved BACKGROUND / UNLABELLED keys,
    - the reserved BACKGROUND / UNLABELLED entries (fixed colorized keys) are present and correct,
    - the set of non-reserved prim paths (``idToLabels`` values) equals ``expected_prim_paths``, and
    - the multiset of non-reserved semantic labels (``idToSemantics`` values) equals ``expected_semantics``.

    Args:
        data_type: The camera data type under test.
        camera_data: The camera's :class:`~isaaclab.sensors.camera.CameraData`; its ``info`` and colorized
            ``instance_segmentation_fast`` output image are read internally.
        expected_prim_paths: Expected set of non-reserved ``idToLabels`` values (one USD prim path per instance).
        expected_semantics: Expected multiset of non-reserved ``idToSemantics`` values (``{semantic_type: label}``
            per instance).
    """
    if data_type != "instance_segmentation_fast":
        return

    info = camera_data.info
    assert info is not None, "camera.data.info is None; renderer did not populate segmentation metadata."
    segmentation = info["instance_segmentation_fast"]
    id_to_labels = segmentation["idToLabels"]
    id_to_semantics = segmentation["idToSemantics"]

    # Reserved BACKGROUND / UNLABELLED entries have fixed colorized (RGBA tuple) keys and are always present in
    # the maps even when the corresponding pixels are not rendered (e.g. no unlabelled prim in view).
    reserved_keys = {(0, 0, 0, 0), (0, 0, 0, 255)}

    # The colorized keys are non-stable across runs, so validate them against the rendered image instead of
    # hard-coding: the map keys must be exactly the unique colors present in the image plus the reserved keys.
    instance_image = camera_data.output["instance_segmentation_fast"]
    color_image = instance_image if isinstance(instance_image, torch.Tensor) else instance_image.torch
    color_image = color_image.reshape(-1, color_image.shape[-1]).cpu().numpy()
    unique_colors = {tuple(int(channel) for channel in row) for row in np.unique(color_image, axis=0)}
    assert set(id_to_labels) == unique_colors | reserved_keys, (
        f"instance_segmentation_fast idToLabels keys != rendered colors + reserved.\n"
        f"  image colors:    {sorted(unique_colors)}\n"
        f"  idToLabels keys: {sorted(id_to_labels)}"
    )
    assert set(id_to_semantics) == unique_colors | reserved_keys, (
        f"instance_segmentation_fast idToSemantics keys != rendered colors + reserved.\n"
        f"  image colors:       {sorted(unique_colors)}\n"
        f"  idToSemantics keys: {sorted(id_to_semantics)}"
    )
    assert id_to_labels.get((0, 0, 0, 0)) == "BACKGROUND"
    assert id_to_labels.get((0, 0, 0, 255)) == "UNLABELLED"
    assert id_to_semantics.get((0, 0, 0, 0)) == {"class": "BACKGROUND"}
    assert id_to_semantics.get((0, 0, 0, 255)) == {"class": "UNLABELLED"}

    actual_prim_paths = {value for key, value in id_to_labels.items() if key not in reserved_keys}
    assert actual_prim_paths == expected_prim_paths, (
        f"instance_segmentation_fast idToLabels prim paths mismatch.\n"
        f"  expected: {sorted(expected_prim_paths)}\n"
        f"  actual:   {sorted(actual_prim_paths)}"
    )

    def _as_multiset(semantics: list[dict[str, str]]) -> list[tuple[tuple[str, str], ...]]:
        return sorted(tuple(sorted(entry.items())) for entry in semantics)

    actual_semantics = [value for key, value in id_to_semantics.items() if key not in reserved_keys]
    assert _as_multiset(actual_semantics) == _as_multiset(expected_semantics), (
        f"instance_segmentation_fast idToSemantics labels mismatch.\n"
        f"  expected: {expected_semantics}\n"
        f"  actual:   {actual_semantics}"
    )


def maybe_step_env_for_motion(env: Any, data_type: str, num_steps: int = 2, action_value: float = 0.0) -> None:
    """Step ``env`` so motion-vector AOVs have real inter-frame motion to encode.

    Motion vectors compare the current frame's transforms against the previous frame's; the first frame
    rendered after a reset has no render history, so its motion vectors would otherwise be all zero. No-op
    for any ``data_type`` other than ``"motion_vectors"``.

    Args:
        env: The environment to step. Must expose ``action_space`` and ``step`` (``DirectRLEnv`` /
            ``ManagerBasedRLEnv``).
        data_type: The camera data type under test.
        num_steps: Number of steps to take before capturing camera output.
        action_value: Constant value applied to every action component on every step. Defaults to
            ``0.0`` (motion then comes only from physics already in flight, e.g. gravity/initial velocity).
            Pass a non-zero value to additionally induce actuated motion (e.g. cartpole cart translation).
    """
    if data_type != "motion_vectors":
        return

    action = torch.full(env.action_space.shape, action_value, device=env.device)
    for _ in range(num_steps):
        env.step(action)


def rendering_test_shadow_hand(
    physics_backend: str,
    renderer: str,
    data_type: str,
    comparison_scores: list[dict],
) -> None:
    if physics_backend == "ovphysx":
        pytest.skip("ovphysx is not supported yet.")
    _skip_if_newton_motion_vectors(physics_backend, data_type)

    from isaaclab.utils.configclass import configclass

    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_camera_env import ShadowHandCameraEnv
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_camera_env_cfg import (
        ShadowHandCameraEnvCfg,
        ShadowHandTiledCameraCfg,
        _ShadowHandBaseTiledCameraCfg,
    )

    @configclass
    class _ShadowHandTiledCameraTestCfg(ShadowHandTiledCameraCfg):
        distance_to_camera = _ShadowHandBaseTiledCameraCfg(data_types=["distance_to_camera"])
        distance_to_image_plane = _ShadowHandBaseTiledCameraCfg(data_types=["distance_to_image_plane"])
        normals = _ShadowHandBaseTiledCameraCfg(data_types=["normals"])
        instance_segmentation_fast = _ShadowHandBaseTiledCameraCfg(data_types=["instance_segmentation_fast"])
        instance_id_segmentation_fast = _ShadowHandBaseTiledCameraCfg(data_types=["instance_id_segmentation_fast"])
        motion_vectors = _ShadowHandBaseTiledCameraCfg(data_types=["motion_vectors"])

    @configclass
    class _ShadowHandCameraTestEnvCfg(ShadowHandCameraEnvCfg):
        tiled_camera = _ShadowHandTiledCameraTestCfg()

    override_args = [f"presets={_physics_preset_name(physics_backend)},{renderer},{data_type}"]

    env_cfg = _ShadowHandCameraTestEnvCfg()
    env_cfg = _apply_overrides_to_env_cfg(env_cfg, override_args)

    env_cfg.scene.num_envs = 4

    if renderer == "ovrtx_renderer":
        _redirect_ovrtx_renderer_log_to_stdout(env_cfg)

    _maybe_enable_physx_determinism_for_motion(env_cfg, physics_backend, data_type)

    if data_type in {"depth", "distance_to_camera", "distance_to_image_plane", "motion_vectors"}:
        # Disable CNN forward pass as it cannot be meaningfully trained from depth/motion vectors alone
        # and will raise a ValueError.
        env_cfg.feature_extractor.enabled = False

    env = None

    try:
        env = ShadowHandCameraEnv(env_cfg)
        maybe_step_env_for_motion(env, data_type)
        maybe_save_stage("shadow_hand", physics_backend, renderer, data_type)

        validate_camera_outputs(
            "shadow_hand",
            physics_backend,
            renderer,
            env._tiled_camera.data.output,
            max_different_pixels_percentage=MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME["shadow_hand"],
            comparison_scores=comparison_scores,
        )

        # The ShadowHand task has a ``class:cube`` on the manipulated object; the hand is UNLABELLED and empty
        # space is BACKGROUND. Both the Isaac RTX (ground truth) and OVRTX renderers must expose this same
        # idToLabels mapping in camera.data.info.
        maybe_validate_semantic_segmentation(
            data_type,
            env._tiled_camera.data.info,
            expected_id_to_labels={
                (0, 0, 0, 0): {"class": "BACKGROUND"},
                (0, 0, 0, 255): {"class": "UNLABELLED"},
                (33, 243, 3, 255): {"class": "cube"},
            },
            # isaacsim Isaac RTX semantic_segmentation returns stringified color keys; OVRTX returns raw tuples.
            stringify_keys=(renderer == "isaacsim_rtx_renderer"),
        )

        # Instance segmentation yields one instance per env's ``class:cube`` object (num_envs=4). The colorized
        # keys are non-stable, so they are validated against the rendered image; only the values are hard-coded.
        maybe_validate_instance_segmentation_fast(
            data_type,
            env._tiled_camera.data,
            expected_prim_paths={f"/World/envs/env_{i}/object" for i in range(4)},
            expected_semantics=[{"class": "cube"} for _ in range(4)],
        )
    finally:
        if env is not None:
            env.close()

            # This invokes camera sensor and renderer cleanup explicitly before pytest teardown, otherwise OV
            # native code could probably complain about leaks and trigger segmentation fault.
            env = None


def rendering_test_cartpole(
    physics_backend: str,
    renderer: str,
    data_type: str,
    comparison_scores: list[dict],
    *,
    compare_golden: bool = False,
) -> None:
    _skip_if_newton_motion_vectors(physics_backend, data_type)

    from isaaclab.utils.configclass import configclass

    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env import CartpoleCameraEnv
    from isaaclab_tasks.core.cartpole.cartpole_direct_camera_env_cfg import CartpoleCameraEnvCfg, CartpoleTiledCameraCfg

    from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

    @configclass
    class _CartpoleTiledCameraTestCfg(CartpoleTiledCameraCfg):
        distance_to_camera = CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(data_types=["distance_to_camera"])
        distance_to_image_plane = CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(
            data_types=["distance_to_image_plane"]
        )
        normals = CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(data_types=["normals"])
        instance_segmentation_fast = CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(
            data_types=["instance_segmentation_fast"]
        )
        instance_id_segmentation_fast = CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(
            data_types=["instance_id_segmentation_fast"]
        )
        motion_vectors = CartpoleTiledCameraCfg.BaseCartpoleTiledCameraCfg(data_types=["motion_vectors"])

    @configclass
    class _BaseCartpoleCameraEnvTestCfg(CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg):
        robot_cfg = CARTPOLE_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            spawn=CARTPOLE_CFG.spawn.replace(semantic_tags=[("class", "cartpole")]),
        )

    @configclass
    class _CartpoleCameraTestEnvCfg(CartpoleCameraEnvCfg):
        # Use the semantically-tagged robot (class:cartpole) so semantic_segmentation produces a non-trivial
        # idToLabels mapping; the base env's semantic_segmentation variant leaves the robot untagged.
        semantic_segmentation = _BaseCartpoleCameraEnvTestCfg(
            observation_space=[4, 96, 96], tiled_camera=_CartpoleTiledCameraTestCfg()
        )
        distance_to_camera = _BaseCartpoleCameraEnvTestCfg(
            observation_space=[1, 96, 96], tiled_camera=_CartpoleTiledCameraTestCfg()
        )
        distance_to_image_plane = _BaseCartpoleCameraEnvTestCfg(
            observation_space=[1, 96, 96], tiled_camera=_CartpoleTiledCameraTestCfg()
        )
        normals = _BaseCartpoleCameraEnvTestCfg(
            observation_space=[3, 96, 96], tiled_camera=_CartpoleTiledCameraTestCfg()
        )
        instance_segmentation_fast = _BaseCartpoleCameraEnvTestCfg(
            observation_space=[4, 96, 96], tiled_camera=_CartpoleTiledCameraTestCfg()
        )
        instance_id_segmentation_fast = _BaseCartpoleCameraEnvTestCfg(
            observation_space=[4, 96, 96], tiled_camera=_CartpoleTiledCameraTestCfg()
        )
        motion_vectors = CartpoleCameraEnvCfg.BaseCartpoleCameraEnvCfg(
            observation_space=[2, 96, 96], tiled_camera=_CartpoleTiledCameraTestCfg()
        )

    env_cfg = _CartpoleCameraTestEnvCfg()
    env_cfg = _apply_overrides_to_env_cfg(
        env_cfg, [f"presets={_physics_preset_name(physics_backend)},{renderer},{data_type}"]
    )

    env_cfg.scene.num_envs = 4
    if getattr(env_cfg.tiled_camera.renderer_cfg, "renderer_type", None) == "newton_warp":
        env_cfg.tiled_camera.renderer_cfg.render_order = "pixel_priority"

    if renderer == "ovrtx_renderer":
        _redirect_ovrtx_renderer_log_to_stdout(env_cfg)

    _maybe_enable_physx_determinism_for_motion(env_cfg, physics_backend, data_type)

    env = None

    try:
        env = CartpoleCameraEnv(env_cfg)
        # Nudge the cart with a small constant force so motion vectors also capture cart translation,
        # not just pole dynamics already in flight from the randomized reset.
        maybe_step_env_for_motion(env, data_type, action_value=0.5)
        camera_outputs = env._tiled_camera.data.output
        if renderer == "ovrtx_renderer":
            # The first output access creates the selected OVRTX render-variable mapping. Give
            # it a few frames to compile and populate instead of validating its zeroed buffer.
            for _ in range(10):
                has_valid_outputs = all(
                    torch.count_nonzero(output if isinstance(output, torch.Tensor) else output.torch).item() > 0
                    for output in camera_outputs.values()
                )
                if has_valid_outputs:
                    break
                env.sim.render()
                env.scene.update(dt=env.physics_dt)
                camera_outputs = env._tiled_camera.data.output
        maybe_save_stage(
            "cartpole",
            physics_backend,
            renderer,
            data_type,
            compare_golden=compare_golden and data_type == "rgb",
        )
        validate_camera_outputs(
            "cartpole",
            physics_backend,
            renderer,
            camera_outputs,
            max_different_pixels_percentage=MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME["cartpole"],
            comparison_scores=comparison_scores,
        )

        # We augment the cartpole task to author ``class:cartpole`` on the robot; everything else is UNLABELLED
        # and empty space is BACKGROUND. Both the Isaac RTX (ground truth) and OVRTX renderers must expose this
        # same idToLabels mapping in camera.data.info.
        maybe_validate_semantic_segmentation(
            data_type,
            env._tiled_camera.data.info,
            expected_id_to_labels={
                (0, 0, 0, 0): {"class": "BACKGROUND"},
                (0, 0, 0, 255): {"class": "UNLABELLED"},
                (33, 243, 3, 255): {"class": "cartpole"},
            },
            # isaacsim Isaac RTX semantic_segmentation returns stringified color keys; OVRTX returns raw tuples.
            stringify_keys=(renderer == "isaacsim_rtx_renderer"),
        )

        # Instance segmentation yields one instance per env's ``class:cartpole`` robot (num_envs=4). The colorized
        # keys are non-stable, so they are validated against the rendered image; only the values are hard-coded.
        maybe_validate_instance_segmentation_fast(
            data_type,
            env._tiled_camera.data,
            expected_prim_paths={f"/World/envs/env_{i}/Robot" for i in range(4)},
            expected_semantics=[{"class": "cartpole"} for _ in range(4)],
        )
    finally:
        if env is not None:
            env.close()

            # This invokes camera sensor and renderer cleanup explicitly before pytest teardown, otherwise OV
            # native code could probably complain about leaks and trigger segmentation fault.
            env = None


def rendering_test_dexsuite_kuka(
    physics_backend: str,
    renderer: str,
    data_type: str,
    setup_homogeneous_envs: bool,
    comparison_scores: list[dict],
) -> None:
    if physics_backend == "ovphysx":
        pytest.skip("ovphysx is not supported yet.")
    _skip_if_newton_motion_vectors(physics_backend, data_type)

    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils.configclass import configclass

    from isaaclab_tasks.core.dexsuite.config.kuka_allegro.camera_cfg import (
        BASE_CAMERA_CFG,
        BaseTiledCameraCfg,
        SingleCameraObservationsCfg,
    )
    from isaaclab_tasks.core.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_camera_env_cfg import (
        _SCENE_KWARGS,
        DexsuiteKukaAllegroLiftCameraEnvCfg,
        SingleCameraSceneCfg,
    )
    from isaaclab_tasks.core.dexsuite.config.kuka_allegro.dexsuite_kuka_allegro_env_cfg import (
        DexsuiteKukaAllegroLiftEnvCfg,
    )

    @configclass
    class _DexsuiteBaseTiledCameraTestCfg(BaseTiledCameraCfg):
        distance_to_camera64 = BASE_CAMERA_CFG.replace(data_types=["distance_to_camera"], width=64, height=64)
        distance_to_camera128 = BASE_CAMERA_CFG.replace(data_types=["distance_to_camera"], width=128, height=128)
        distance_to_camera256 = BASE_CAMERA_CFG.replace(data_types=["distance_to_camera"], width=256, height=256)
        distance_to_image_plane64 = BASE_CAMERA_CFG.replace(data_types=["distance_to_image_plane"], width=64, height=64)
        distance_to_image_plane128 = BASE_CAMERA_CFG.replace(
            data_types=["distance_to_image_plane"], width=128, height=128
        )
        distance_to_image_plane256 = BASE_CAMERA_CFG.replace(
            data_types=["distance_to_image_plane"], width=256, height=256
        )
        normals64 = BASE_CAMERA_CFG.replace(data_types=["normals"], width=64, height=64)
        normals128 = BASE_CAMERA_CFG.replace(data_types=["normals"], width=128, height=128)
        normals256 = BASE_CAMERA_CFG.replace(data_types=["normals"], width=256, height=256)
        instance_segmentation_fast64 = BASE_CAMERA_CFG.replace(
            data_types=["instance_segmentation_fast"], width=64, height=64
        )
        instance_segmentation_fast128 = BASE_CAMERA_CFG.replace(
            data_types=["instance_segmentation_fast"], width=128, height=128
        )
        instance_segmentation_fast256 = BASE_CAMERA_CFG.replace(
            data_types=["instance_segmentation_fast"], width=256, height=256
        )
        instance_id_segmentation_fast64 = BASE_CAMERA_CFG.replace(
            data_types=["instance_id_segmentation_fast"], width=64, height=64
        )
        instance_id_segmentation_fast128 = BASE_CAMERA_CFG.replace(
            data_types=["instance_id_segmentation_fast"], width=128, height=128
        )
        instance_id_segmentation_fast256 = BASE_CAMERA_CFG.replace(
            data_types=["instance_id_segmentation_fast"], width=256, height=256
        )
        motion_vectors64 = BASE_CAMERA_CFG.replace(data_types=["motion_vectors"], width=64, height=64)
        motion_vectors128 = BASE_CAMERA_CFG.replace(data_types=["motion_vectors"], width=128, height=128)
        motion_vectors256 = BASE_CAMERA_CFG.replace(data_types=["motion_vectors"], width=256, height=256)

    @configclass
    class _DexsuiteSingleCameraTestSceneCfg(SingleCameraSceneCfg):
        base_camera: CameraCfg = _DexsuiteBaseTiledCameraTestCfg()

    @configclass
    class _DexsuiteKukaAllegroLiftCameraTestEnvCfg(DexsuiteKukaAllegroLiftCameraEnvCfg):
        single_camera = DexsuiteKukaAllegroLiftEnvCfg(
            scene=_DexsuiteSingleCameraTestSceneCfg(**_SCENE_KWARGS),
            observations=SingleCameraObservationsCfg(),
        )
        default = single_camera

    override_arg = f"presets={_physics_preset_name(physics_backend)},{renderer},{data_type}64,single_camera"

    # The default setup uses heterogeneous environments with multiple asset spawner to place random objects.
    # For homogeneous environments, we use a fixed object config - cube.
    if setup_homogeneous_envs:
        override_arg += ",cube"

    env_cfg = _DexsuiteKukaAllegroLiftCameraTestEnvCfg()
    env_cfg = _apply_overrides_to_env_cfg(env_cfg, [override_arg])

    env_cfg.scene.num_envs = 4

    if renderer == "ovrtx_renderer":
        _redirect_ovrtx_renderer_log_to_stdout(env_cfg)

    _maybe_enable_physx_determinism_for_motion(env_cfg, physics_backend, data_type)

    # Disable the observation point-cloud visualisation markers (/Visuals/ObservationPointCloud).
    # The underlying point sampling uses the global numpy/torch RNG, so marker positions shift
    # across processes and show up as random red dots in the rendered camera output. Since this
    # test only cares about rendering correctness of the scene itself, we hide the markers here
    # rather than making the sampler deterministic globally.
    point_cloud_term = getattr(env_cfg.observations.perception, "object_point_cloud", None)
    if point_cloud_term is not None:
        point_cloud_term.params["visualize"] = False

    # The success and failure markers are placed exactly at the same location. If both markers are
    # visible, the rendering order will determine which one is visible in the camera output. Hide
    # both markers to avoid this nondeterministic behavior.
    for marker_cfg in env_cfg.commands.object_pose.success_visualizer_cfg.markers.values():
        marker_cfg.visible = False

    test_name = f"dexsuite_kuka_{'homo' if setup_homogeneous_envs else 'hetero'}"

    env = None

    try:
        env = ManagerBasedRLEnv(env_cfg)
        maybe_step_env_for_motion(env, data_type)
        maybe_save_stage(test_name, physics_backend, renderer, data_type)
        validate_camera_outputs(
            test_name,
            physics_backend,
            renderer,
            env.scene.sensors["base_camera"].data.output,
            max_different_pixels_percentage=MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME[test_name],
            comparison_scores=comparison_scores,
        )
    finally:
        if env is not None:
            env.close()

            # This invokes camera sensor and renderer cleanup explicitly before pytest teardown, otherwise OV
            # native code could probably complain about leaks and trigger segmentation fault.
            env = None


def _make_franka_cloth_camera_env_cfg(data_type: str):
    """Create a test-local Franka cloth camera env cfg without exposing a production task."""
    import isaaclab.sim as sim_utils
    from isaaclab.envs import mdp as env_mdp
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils.configclass import configclass

    from isaaclab_tasks.core.lift.config.franka_soft.franka_cloth_env_cfg import FrankaClothEnvCfg, FrankaClothSceneCfg
    from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

    @configclass
    class TestFrankaClothCameraSceneCfg(FrankaClothSceneCfg):
        """Franka cloth scene with a test-only camera sensor."""

        tiled_camera: CameraCfg = CameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.85, -0.55, 0.42),
                rot=(0.5080, 0.2114, 0.318, 0.7720),
                convention="opengl",
            ),
            data_types=[data_type],
            spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 3.0)),
            width=128,
            height=128,
            renderer_cfg=MultiBackendRendererCfg(),
        )

    @configclass
    class TestFrankaClothCameraObservationsCfg:
        """Image-only observations for the local rendering test env."""

        @configclass
        class PolicyCfg(ObsGroup):
            image = ObsTerm(
                func=env_mdp.image,
                params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": data_type, "permute": True},
            )

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: ObsGroup = PolicyCfg()

    @configclass
    class TestFrankaClothCameraEnvCfg(FrankaClothEnvCfg):
        """Test-only camera variant of ``Isaac-Lift-Cloth-Franka``."""

        scene: TestFrankaClothCameraSceneCfg = TestFrankaClothCameraSceneCfg(
            num_envs=4, env_spacing=3.0, replicate_physics=True
        )
        observations: TestFrankaClothCameraObservationsCfg = TestFrankaClothCameraObservationsCfg()

        def __post_init__(self) -> None:
            super().__post_init__()
            self.commands.deformable_pose.debug_vis = False
            self.events.reset_deformable.params["position_range"] = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            }

    return TestFrankaClothCameraEnvCfg()


def rendering_test_franka_cloth(
    physics_backend: str,
    renderer: str,
    data_type: str,
    comparison_scores: list[dict],
) -> None:
    if renderer == "ovrtx_renderer" and data_type == "instance_segmentation_fast":
        pytest.skip("instance_segmentation_fast crashes with the OVRTX renderer on franka_cloth (NVBUG#6463802).")

    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = _make_franka_cloth_camera_env_cfg(data_type)

    physics_preset_name = _physics_preset_name_deformable(physics_backend)
    _skip_if_physics_preset_unsupported(env_cfg, physics_preset_name)

    env_cfg = _apply_overrides_to_env_cfg(env_cfg, [f"presets={physics_preset_name},{renderer}"])

    if renderer == "ovrtx_renderer":
        _redirect_ovrtx_renderer_log_to_stdout(env_cfg)

    _maybe_enable_physx_determinism_for_motion(env_cfg, physics_backend, data_type)

    test_name = "franka_cloth"
    env = None

    try:
        env = ManagerBasedRLEnv(env_cfg)

        _maybe_disable_instancing_for_current_stage(physics_backend, renderer, data_type)

        maybe_save_stage(test_name, physics_backend, renderer, data_type)

        # We step only once to let the cloth fall uniformly on the gravity but not collide with the cube on the table.
        # This is to limit the inconsistent nodal poses and pixels from run to run due to solver scheduling and
        # numerical precision.
        zero_actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        env.step(zero_actions)

        validate_camera_outputs(
            test_name,
            physics_backend,
            renderer,
            env.scene.sensors["tiled_camera"].data.output,
            max_different_pixels_percentage=MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME[test_name],
            comparison_scores=comparison_scores,
        )
    finally:
        if env is not None:
            env.close()

            # This invokes camera sensor and renderer cleanup explicitly before pytest teardown, otherwise OV
            # native code could probably complain about leaks and trigger segmentation fault.
            env = None


def _make_franka_soft_camera_env_cfg(data_type: str):
    """Create a test-local Franka soft camera env cfg without exposing a production task."""
    import isaaclab.sim as sim_utils
    from isaaclab.envs import mdp as env_mdp
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils.configclass import configclass

    from isaaclab_tasks.core.lift.config.franka_soft.franka_soft_env_cfg import FrankaSoftEnvCfg, _FrankaSoftSceneCfg
    from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

    @configclass
    class TestFrankaSoftCameraSceneCfg(_FrankaSoftSceneCfg):
        """Franka soft scene with a test-only camera sensor."""

        tiled_camera: CameraCfg = CameraCfg(
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.85, -0.55, 0.42),
                rot=(0.5080, 0.2114, 0.318, 0.7720),
                convention="opengl",
            ),
            data_types=[data_type],
            spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.01, 3.0)),
            width=128,
            height=128,
            renderer_cfg=MultiBackendRendererCfg(),
        )

    @configclass
    class TestFrankaSoftCameraObservationsCfg:
        """Image-only observations for the local rendering test env."""

        @configclass
        class PolicyCfg(ObsGroup):
            image = ObsTerm(
                func=env_mdp.image,
                params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": data_type, "permute": True},
            )

            def __post_init__(self) -> None:
                self.enable_corruption = False
                self.concatenate_terms = True

        policy: ObsGroup = PolicyCfg()

    @configclass
    class TestFrankaSoftCameraEnvCfg(FrankaSoftEnvCfg):
        """Test-only camera variant of ``Isaac-Lift-Soft-Franka``."""

        scene: TestFrankaSoftCameraSceneCfg = TestFrankaSoftCameraSceneCfg(
            num_envs=4, env_spacing=3.0, replicate_physics=True
        )
        observations: TestFrankaSoftCameraObservationsCfg = TestFrankaSoftCameraObservationsCfg()

        def __post_init__(self) -> None:
            super().__post_init__()
            self.commands.deformable_pose.debug_vis = False
            self.events.reset_deformable.params["position_range"] = {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            }

    return TestFrankaSoftCameraEnvCfg()


def rendering_test_franka_soft(
    physics_backend: str,
    renderer: str,
    data_type: str,
    comparison_scores: list[dict],
) -> None:
    if physics_backend == "physx" or renderer == "isaacsim_rtx_renderer":
        pytest.skip("Random teardown hangs in the kit-based combinations (OMPE-101977).")

    if renderer == "ovrtx_renderer" and data_type == "instance_segmentation_fast":
        pytest.skip("instance_segmentation_fast crashes with the OVRTX renderer on franka_soft (NVBUG#6463802).")

    if renderer == "isaacsim_rtx_renderer" and data_type == "motion_vectors":
        pytest.skip("The test cases will be enabled after NVBUG#6418121 is fixed.")

    _skip_if_newton_motion_vectors(physics_backend, data_type)

    from isaaclab.envs import ManagerBasedRLEnv

    env_cfg = _make_franka_soft_camera_env_cfg(data_type)

    physics_preset_name = _physics_preset_name_deformable(physics_backend)
    _skip_if_physics_preset_unsupported(env_cfg, physics_preset_name)

    env_cfg = _apply_overrides_to_env_cfg(env_cfg, [f"presets={physics_preset_name},{renderer}"])

    if renderer == "ovrtx_renderer":
        _redirect_ovrtx_renderer_log_to_stdout(env_cfg)

    _maybe_enable_physx_determinism_for_motion(env_cfg, physics_backend, data_type)

    test_name = "franka_soft"
    env = None

    try:
        env = ManagerBasedRLEnv(env_cfg)

        if data_type == "motion_vectors":
            # Command a valid absolute IK pose with a small displacement (0.05m) so the renderer sees arm motion.
            arm_action = env.action_manager.get_term("arm_action")
            ee_pos_curr, ee_quat_curr = arm_action._compute_frame_pose()
            ee_pos_curr[0] += 0.05

            actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            actions[:, 0:3] = ee_pos_curr
            actions[:, 3:7] = ee_quat_curr

            env.step(actions)

        # This workaround invalidates the physx data views and would lead to issues if it was done before stepping.
        _maybe_disable_instancing_for_current_stage(physics_backend, renderer, data_type)

        maybe_save_stage(test_name, physics_backend, renderer, data_type)

        validate_camera_outputs(
            test_name,
            physics_backend,
            renderer,
            env.scene.sensors["tiled_camera"].data.output,
            max_different_pixels_percentage=MAX_DIFFERENT_PIXELS_PERCENTAGE_BY_ENV_NAME[test_name],
            comparison_scores=comparison_scores,
        )
    finally:
        if env is not None:
            env.close()

            # This invokes camera sensor and renderer cleanup explicitly before pytest teardown, otherwise OV
            # native code could probably complain about leaks and trigger segmentation fault.
            env = None
