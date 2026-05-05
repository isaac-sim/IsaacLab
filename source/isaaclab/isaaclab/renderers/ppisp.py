# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared PPISP configuration and USD/SPG authoring helpers."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isaaclab.utils import configclass

logger = logging.getLogger(__name__)

PPISP_INPUT_RENDER_VAR = "HdrColor"
PPISP_OUTPUT_RENDER_VAR = "PPISPColor"
PPISP_LDR_RENDER_VAR = "LdrColor"
PPISP_SHADER_NAME = "PPISP"
PPISP_SPG_EXTENSION = "omni.rtx.spg"
PPISP_SPG_ENABLED_SETTING = "/rtx/spg/enabled"
PPISP_SPG_USDA_FILE = "ppisp_usd_spg.slang.usda"
PPISP_SPG_SLANG_FILE = "ppisp_usd_spg.slang"
PPISP_SPG_LUA_FILE = "ppisp_usd_spg.slang.lua"
PPISP_SPG_SUB_IDENTIFIER = "ppispProcess"
PPISP_SPG_FILES = (PPISP_SPG_USDA_FILE, PPISP_SPG_SLANG_FILE, PPISP_SPG_LUA_FILE)

PPISP_NO_ISP_EXPOSURE = 0.0
PPISP_NO_ISP_EXPOSURE_FSTOP = 1.0
PPISP_NO_ISP_EXPOSURE_ISO = 100.0
PPISP_NO_ISP_EXPOSURE_RESPONSIVITY = 1.0
PPISP_NO_ISP_EXPOSURE_TIME = 1.0

PPISP_FLOAT2_INPUTS = {
    "vignettingCenterR",
    "vignettingCenterG",
    "vignettingCenterB",
    "colorLatentBlue",
    "colorLatentRed",
    "colorLatentGreen",
    "colorLatentNeutral",
}

PPISP_DEFAULT_INPUTS: dict[str, float | tuple[float, float]] = {
    "exposureOffset": 0.0,
    "vignettingCenterR": (0.0, 0.0),
    "vignettingAlpha1R": 0.0,
    "vignettingAlpha2R": 0.0,
    "vignettingAlpha3R": 0.0,
    "vignettingCenterG": (0.0, 0.0),
    "vignettingAlpha1G": 0.0,
    "vignettingAlpha2G": 0.0,
    "vignettingAlpha3G": 0.0,
    "vignettingCenterB": (0.0, 0.0),
    "vignettingAlpha1B": 0.0,
    "vignettingAlpha2B": 0.0,
    "vignettingAlpha3B": 0.0,
    "colorLatentBlue": (0.0, 0.0),
    "colorLatentRed": (0.0, 0.0),
    "colorLatentGreen": (0.0, 0.0),
    "colorLatentNeutral": (0.0, 0.0),
    "crfToeR": 0.013659,
    "crfShoulderR": 0.013659,
    "crfGammaR": 0.378165,
    "crfCenterR": 0.0,
    "crfToeG": 0.013659,
    "crfShoulderG": 0.013659,
    "crfGammaG": 0.378165,
    "crfCenterG": 0.0,
    "crfToeB": 0.013659,
    "crfShoulderB": 0.013659,
    "crfGammaB": 0.378165,
    "crfCenterB": 0.0,
}


def default_ppisp_inputs() -> dict[str, float | tuple[float, float]]:
    """Return a copy of the PPISP identity/default input dictionary."""
    return dict(PPISP_DEFAULT_INPUTS)


def get_ppisp_spg_dir() -> Path:
    """Return the directory containing bundled PPISP SPG sidecar files."""
    return Path(__file__).with_name("ppisp_spg")


def get_ppisp_spg_file_paths() -> dict[str, Path]:
    """Return bundled PPISP SPG sidecar paths keyed by filename."""
    spg_dir = get_ppisp_spg_dir()
    return {filename: spg_dir / filename for filename in PPISP_SPG_FILES}


def copy_ppisp_spg_files(target_dir: str | Path) -> dict[str, Path]:
    """Copy bundled PPISP SPG sidecar files into ``target_dir``."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    copied = {}
    for filename, source_path in get_ppisp_spg_file_paths().items():
        destination_path = target_path / filename
        shutil.copy2(source_path, destination_path)
        copied[filename] = destination_path
    return copied


def localize_ppisp_spg_assets(stage: Any, target_dir: str | Path | None = None) -> None:
    """Copy bundled PPISP sidecars beside a stage and rewrite PPISP shader assets.

    This mirrors 3DGRUT exports: the PPISP shader references
    ``ppisp_usd_spg.slang.usda`` and that shader definition references
    ``ppisp_usd_spg.slang`` from the same directory. Keeping these paths relative
    lets Kit discover the adjacent ``.slang.lua`` launcher.
    """
    from pxr import Sdf

    stage_dir = _stage_asset_dir(stage, target_dir)
    if stage_dir is None:
        return
    copy_ppisp_spg_files(stage_dir)

    for prim in stage.Traverse():
        source_asset_attr = prim.GetAttribute("info:spg:sourceAsset")
        source_asset = source_asset_attr.Get() if source_asset_attr else None
        source_asset_path = getattr(source_asset, "path", source_asset)
        is_ppisp_shader = prim.GetName() == PPISP_SHADER_NAME or str(source_asset_path).endswith(PPISP_SPG_SLANG_FILE)
        if not is_ppisp_shader:
            continue

        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(PPISP_SPG_USDA_FILE)
        prim.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token, custom=False).Set("sourceAsset")
        prim.CreateAttribute("info:spg:sourceAsset", Sdf.ValueTypeNames.Asset, custom=False).Set(
            Sdf.AssetPath(PPISP_SPG_SLANG_FILE)
        )
        prim.CreateAttribute("info:spg:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token, custom=False).Set(
            PPISP_SPG_SUB_IDENTIFIER
        )


@configclass
class PPISPCfg:
    """Configuration for PPISP post-processing.

    PPISP inputs are static in IsaacLab. If imported from animated USD shader inputs,
    the first authored time sample is used and later samples are ignored.
    """

    shader_prim_path: str | None = None
    """Optional source USD shader prim path used to populate :attr:`inputs`."""

    spg_usda_file: str = PPISP_SPG_USDA_FILE
    """SPG shader definition asset referenced by authored PPISP shader prims."""

    spg_slang_file: str = PPISP_SPG_SLANG_FILE
    """Slang source asset used by the SPG shader definition."""

    inputs: dict[str, float | tuple[float, float]] = field(default_factory=default_ppisp_inputs)
    """Flat PPISP shader input values keyed by USD input name."""


@dataclass
class RenderProductInfo:
    """Parsed USD RenderProduct information used for PPISP validation."""

    render_product_path: str
    camera_paths: list[str]
    resolution: tuple[int, int] | None
    ordered_vars: list[str]
    ppisp: PPISPCfg | None
    camera_xform_time_samples: list[float]


def is_ppisp_enabled(ppisp_cfg: PPISPCfg | dict[str, Any] | None) -> bool:
    """Return whether a PPISP configuration is present."""
    return ppisp_cfg is not None


def enable_ppisp_spg() -> None:
    """Enable Kit SPG support required by PPISP render-product graphs.

    This is a no-op outside Kit. Inside Kit, missing ``omni.rtx.spg`` is logged
    by the extension manager; the RTX setting is still authored so environments
    with the extension available do not require extra user CLI flags.
    """
    try:
        from isaacsim.core.experimental.utils.app import enable_extension

        enable_extension(PPISP_SPG_EXTENSION)
    except Exception as exc:
        logger.debug("Could not enable PPISP SPG extension '%s': %s", PPISP_SPG_EXTENSION, exc)

    try:
        from isaaclab.app.settings_manager import get_settings_manager

        get_settings_manager().set_bool(PPISP_SPG_ENABLED_SETTING, True)
    except Exception as exc:
        logger.debug("Could not set PPISP SPG setting '%s': %s", PPISP_SPG_ENABLED_SETTING, exc)


def normalize_ppisp_cfg(ppisp_cfg: PPISPCfg | dict[str, Any] | None, stage: Any | None = None) -> PPISPCfg | None:
    """Convert supported user PPISP representations to :class:`PPISPCfg`."""
    if ppisp_cfg is None:
        return None
    if isinstance(ppisp_cfg, PPISPCfg):
        input_overrides = dict(ppisp_cfg.inputs)
        if ppisp_cfg.shader_prim_path and stage is not None:
            ppisp_cfg = _merge_shader_inputs_with_cfg(ppisp_cfg, stage, input_overrides)
        else:
            ppisp_cfg.inputs = _normalized_inputs(input_overrides)
        return ppisp_cfg
    if isinstance(ppisp_cfg, dict):
        input_overrides = ppisp_cfg.get(
            "inputs", {key: value for key, value in ppisp_cfg.items() if key in PPISP_DEFAULT_INPUTS}
        )
        cfg = PPISPCfg()
        cfg.inputs = _normalized_inputs(input_overrides)
        shader_prim_path = ppisp_cfg.get("shader_prim_path")
        if shader_prim_path is not None:
            cfg.shader_prim_path = str(shader_prim_path)
            if stage is not None:
                cfg = _merge_shader_inputs_with_cfg(cfg, stage, input_overrides)
        return cfg
    raise TypeError(f"Unsupported PPISP configuration type: {type(ppisp_cfg)!r}")


def ppisp_cfg_from_usd_shader(shader: Any) -> PPISPCfg:
    """Create :class:`PPISPCfg` from a ``UsdShade.Shader`` prim.

    Animated inputs are collapsed to their first authored time sample.
    """
    cfg = PPISPCfg(shader_prim_path=str(shader.GetPath()))
    values = default_ppisp_inputs()
    for input_name in values:
        shader_input = shader.GetInput(input_name)
        if not shader_input:
            continue
        attr = shader_input.GetAttr()
        value = _read_first_authored_value(attr)
        if value is not None:
            values[input_name] = _normalize_input_value(input_name, value)
    cfg.inputs = values
    return cfg


def ppisp_cfg_from_usd_stage(stage: Any, shader_prim_path: str) -> PPISPCfg:
    """Create :class:`PPISPCfg` from a shader prim path in a USD stage."""
    from pxr import UsdShade

    shader = UsdShade.Shader(stage.GetPrimAtPath(shader_prim_path))
    if not shader:
        raise ValueError(f"PPISP shader prim not found at path: {shader_prim_path}")
    return ppisp_cfg_from_usd_shader(shader)


def parse_render_product(stage: Any, render_product_path: str) -> RenderProductInfo:
    """Parse a USD RenderProduct and optional PPISP shader configuration."""
    render_product = stage.GetPrimAtPath(render_product_path)
    if not render_product.IsValid() or render_product.GetTypeName() != "RenderProduct":
        raise ValueError(f"RenderProduct not found at path: {render_product_path}")

    camera_rel = render_product.GetRelationship("camera")
    camera_paths = [str(path) for path in camera_rel.GetTargets()] if camera_rel else []
    if not camera_paths:
        raise ValueError(f"RenderProduct at path '{render_product_path}' has no camera relationship targets.")

    resolution = None
    resolution_attr = render_product.GetAttribute("resolution")
    if resolution_attr:
        resolution_value = resolution_attr.Get()
        if resolution_value is not None:
            resolution = (int(resolution_value[0]), int(resolution_value[1]))

    ordered_vars_rel = render_product.GetRelationship("orderedVars")
    ordered_vars = [str(path) for path in ordered_vars_rel.GetTargets()] if ordered_vars_rel else []

    ppisp = None
    ppisp_prim = stage.GetPrimAtPath(f"{render_product_path}/{PPISP_SHADER_NAME}")
    if ppisp_prim.IsValid():
        from pxr import UsdShade

        ppisp = ppisp_cfg_from_usd_shader(UsdShade.Shader(ppisp_prim))

    return RenderProductInfo(
        render_product_path=render_product_path,
        camera_paths=camera_paths,
        resolution=resolution,
        ordered_vars=ordered_vars,
        ppisp=ppisp,
        camera_xform_time_samples=collect_camera_xform_time_samples(stage, camera_paths),
    )


def parse_render_product_file(usd_path: str, render_product_path: str) -> RenderProductInfo:
    """Open a USD file and parse a RenderProduct."""
    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage at path: {usd_path}")
    return parse_render_product(stage, render_product_path)


def collect_camera_xform_time_samples(stage: Any, camera_paths: list[str]) -> list[float]:
    """Collect authored xform time samples from cameras and inherited source cameras."""
    time_samples = set()
    for camera_path in camera_paths:
        prim = stage.GetPrimAtPath(camera_path)
        if not prim.IsValid():
            continue
        _collect_xform_attr_time_samples(prim, time_samples)
        for inherited_path in prim.GetInherits().GetAllDirectInherits():
            inherited_prim = stage.GetPrimAtPath(inherited_path)
            if inherited_prim.IsValid():
                _collect_xform_attr_time_samples(inherited_prim, time_samples)
    if not time_samples:
        start_time = stage.GetStartTimeCode()
        end_time = stage.GetEndTimeCode()
        if start_time != end_time:
            time_samples.update([start_time, end_time])
        else:
            time_samples.add(start_time)
    return sorted(time_samples)


def render_render_product_animation(
    usd_path: str,
    render_product_path: str,
    annotator_name: str = "rgb",
    device: str = "cuda",
) -> list[Any]:
    """Render an existing USD RenderProduct at each authored camera xform sample.

    This function requires Isaac Sim/Kit and is intended for PPISP validation of
    authored USD render products. It attaches a Replicator annotator directly to
    the existing RenderProduct path, so the USD-authored SPG graph is preserved.
    """
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd

    enable_ppisp_spg()
    usd_context = omni.usd.get_context()
    if not usd_context.open_stage(usd_path):
        raise RuntimeError(f"Failed to open USD stage at path: {usd_path}")
    stage = usd_context.get_stage()
    render_product_info = parse_render_product(stage, render_product_path)

    annotator = rep.AnnotatorRegistry.get_annotator(annotator_name, device=device, do_array_copy=False)
    annotator.attach([render_product_path])
    timeline = omni.timeline.get_timeline_interface()
    time_codes_per_second = stage.GetTimeCodesPerSecond() or 1.0
    frames = []
    try:
        for time_code in render_product_info.camera_xform_time_samples:
            timeline.set_current_time(float(time_code) / time_codes_per_second)
            ensure_isaac_rtx_render_update()
            frames.append(annotator.get_data())
    finally:
        annotator.detach([render_product_path])
    return frames


def ensure_no_isp_camera(stage: Any, render_product_path: str) -> list[Any]:
    """Redirect a RenderProduct to hidden cameras with neutral exposure settings.

    The hidden cameras inherit from the original camera targets so existing camera
    transform updates keep driving the render product.
    """
    from pxr import Sdf, UsdGeom

    render_product = stage.GetPrimAtPath(render_product_path)
    if not render_product.IsValid():
        raise ValueError(f"RenderProduct not found at path: {render_product_path}")

    camera_rel = render_product.GetRelationship("camera")
    camera_targets = camera_rel.GetTargets() if camera_rel else []
    if not camera_targets:
        raise ValueError(f"RenderProduct at path '{render_product_path}' has no camera relationship targets.")

    no_isp_targets = []
    for index, source_camera_path in enumerate(camera_targets):
        source_camera_prim = stage.GetPrimAtPath(source_camera_path)
        if not source_camera_prim.IsValid():
            raise ValueError(f"RenderProduct camera target does not exist: {source_camera_path}")

        no_isp_name = _make_no_isp_camera_name(source_camera_path, index)
        no_isp_camera_path = render_product.GetPath().AppendChild(no_isp_name)
        no_isp_camera_prim = stage.DefinePrim(no_isp_camera_path, "Camera")
        no_isp_camera_prim.SetHidden(True)
        UsdGeom.Imageable(no_isp_camera_prim).CreateVisibilityAttr().Set("invisible")
        no_isp_camera_prim.GetInherits().AddInherit(source_camera_path)
        no_isp_camera_prim.CreateAttribute("exposure", Sdf.ValueTypeNames.Float).Set(PPISP_NO_ISP_EXPOSURE)
        no_isp_camera_prim.CreateAttribute("exposure:fStop", Sdf.ValueTypeNames.Float).Set(PPISP_NO_ISP_EXPOSURE_FSTOP)
        no_isp_camera_prim.CreateAttribute("exposure:iso", Sdf.ValueTypeNames.Float).Set(PPISP_NO_ISP_EXPOSURE_ISO)
        no_isp_camera_prim.CreateAttribute("exposure:responsivity", Sdf.ValueTypeNames.Float).Set(
            PPISP_NO_ISP_EXPOSURE_RESPONSIVITY
        )
        no_isp_camera_prim.CreateAttribute("exposure:time", Sdf.ValueTypeNames.Float).Set(PPISP_NO_ISP_EXPOSURE_TIME)
        no_isp_targets.append(no_isp_camera_path)

    camera_rel.SetTargets(no_isp_targets)
    return no_isp_targets


def author_ppisp_render_product(stage: Any, render_product_path: str, ppisp_cfg: PPISPCfg | dict[str, Any]) -> Any:
    """Author ``HdrColor -> PPISP -> LdrColor`` under a USD RenderProduct."""
    from pxr import Sdf, UsdShade

    cfg = normalize_ppisp_cfg(ppisp_cfg)
    if cfg is None:
        raise ValueError("Cannot author PPISP render product without a PPISP configuration.")

    render_product = stage.GetPrimAtPath(render_product_path)
    if not render_product.IsValid():
        raise ValueError(f"RenderProduct not found at path: {render_product_path}")

    ensure_no_isp_camera(stage, render_product_path)

    input_var_path = f"{render_product_path}/{PPISP_INPUT_RENDER_VAR}"
    input_var = stage.DefinePrim(input_var_path, "RenderVar")
    input_var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(PPISP_INPUT_RENDER_VAR)
    input_var.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False)

    shader_path = f"{render_product_path}/{PPISP_SHADER_NAME}"
    shader = UsdShade.Shader.Define(stage, shader_path)
    spg_usda_file, spg_slang_file = _resolve_stage_spg_assets(stage, cfg)

    shader.GetPrim().GetReferences().AddReference(spg_usda_file)
    shader.GetPrim().CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token, custom=False).Set(
        "sourceAsset"
    )
    shader.GetPrim().CreateAttribute("info:spg:sourceAsset", Sdf.ValueTypeNames.Asset, custom=False).Set(
        Sdf.AssetPath(spg_slang_file)
    )
    shader.GetPrim().CreateAttribute("info:spg:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token, custom=False).Set(
        PPISP_SPG_SUB_IDENTIFIER
    )

    hdr_input = shader.CreateInput(PPISP_INPUT_RENDER_VAR, Sdf.ValueTypeNames.Opaque)
    hdr_input.GetAttr().SetConnections([Sdf.Path(f"../{PPISP_INPUT_RENDER_VAR}.omni:rtx:aov")])
    shader.CreateOutput(PPISP_OUTPUT_RENDER_VAR, Sdf.ValueTypeNames.Opaque)
    _set_shader_inputs(shader, cfg.inputs)

    ppisp_output_path = shader.GetPath().AppendProperty(f"outputs:{PPISP_OUTPUT_RENDER_VAR}")
    ldr_var_path = f"{render_product_path}/{PPISP_LDR_RENDER_VAR}"
    ldr_var = stage.DefinePrim(ldr_var_path, "RenderVar")
    ldr_var.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set(PPISP_LDR_RENDER_VAR)
    ldr_var.CreateAttribute("omni:rtx:aov", Sdf.ValueTypeNames.Opaque, custom=False).SetConnections([ppisp_output_path])

    ordered_vars = render_product.GetRelationship("orderedVars")
    if not ordered_vars:
        ordered_vars = render_product.CreateRelationship("orderedVars")
    targets = [
        target
        for target in ordered_vars.GetTargets()
        if target.name not in (PPISP_INPUT_RENDER_VAR, PPISP_LDR_RENDER_VAR)
    ]
    targets.extend(
        [
            Sdf.Path(PPISP_INPUT_RENDER_VAR),
            Sdf.Path(PPISP_LDR_RENDER_VAR),
        ]
    )
    ordered_vars.SetTargets(targets)

    return shader.GetPrim()


def _set_shader_inputs(shader: Any, inputs: dict[str, float | tuple[float, float]]) -> None:
    from pxr import Gf, Sdf

    normalized_inputs = _normalized_inputs(inputs)
    for input_name, value in normalized_inputs.items():
        if input_name in PPISP_FLOAT2_INPUTS:
            shader.CreateInput(input_name, Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(value[0], value[1]))
        else:
            shader.CreateInput(input_name, Sdf.ValueTypeNames.Float).Set(float(value))


def _normalized_inputs(inputs: dict[str, Any]) -> dict[str, float | tuple[float, float]]:
    values = default_ppisp_inputs()
    for input_name, value in inputs.items():
        if input_name not in values:
            raise ValueError(f"Unknown PPISP input: {input_name}")
        values[input_name] = _normalize_input_value(input_name, value)
    return values


def _merge_shader_inputs_with_cfg(ppisp_cfg: PPISPCfg, stage: Any, input_overrides: dict[str, Any]) -> PPISPCfg:
    parsed_cfg = ppisp_cfg_from_usd_stage(stage, ppisp_cfg.shader_prim_path)
    parsed_cfg.spg_usda_file = ppisp_cfg.spg_usda_file
    parsed_cfg.spg_slang_file = ppisp_cfg.spg_slang_file
    if input_overrides != PPISP_DEFAULT_INPUTS:
        parsed_cfg.inputs.update(_normalized_input_overrides(input_overrides))
    return parsed_cfg


def _normalized_input_overrides(inputs: dict[str, Any]) -> dict[str, float | tuple[float, float]]:
    values = {}
    for input_name, value in inputs.items():
        if input_name not in PPISP_DEFAULT_INPUTS:
            raise ValueError(f"Unknown PPISP input: {input_name}")
        values[input_name] = _normalize_input_value(input_name, value)
    return values


def _normalize_input_value(input_name: str, value: Any) -> float | tuple[float, float]:
    if input_name in PPISP_FLOAT2_INPUTS:
        if len(value) != 2:
            raise ValueError(f"PPISP input '{input_name}' expects two values.")
        return (float(value[0]), float(value[1]))
    return float(value)


def _read_first_authored_value(attr: Any) -> Any:
    time_samples = attr.GetTimeSamples()
    if time_samples:
        return attr.Get(time_samples[0])
    return attr.Get()


def _make_no_isp_camera_name(source_camera_path: Any, index: int) -> str:
    base_name = re.sub(r"[^A-Za-z0-9_]", "_", str(source_camera_path).strip("/"))
    return f"{base_name}_no_isp_{index}"


def _resolve_spg_asset_path(asset_path: str, default_filename: str) -> str:
    if asset_path == default_filename:
        return str(get_ppisp_spg_dir() / default_filename)
    return asset_path


def _resolve_stage_spg_assets(stage: Any, cfg: PPISPCfg) -> tuple[str, str]:
    """Resolve SPG assets, copying bundled sidecars beside file-backed stages.

    SPG discovers the optional ``.slang.lua`` launcher beside the Slang asset. For
    file-backed stages we mirror 3DGRUT exports and keep the assets relative to
    the USD layer so the launcher is found by Kit and by exported stages.
    """
    if cfg.spg_usda_file != PPISP_SPG_USDA_FILE or cfg.spg_slang_file != PPISP_SPG_SLANG_FILE:
        return (
            _resolve_spg_asset_path(cfg.spg_usda_file, PPISP_SPG_USDA_FILE),
            _resolve_spg_asset_path(cfg.spg_slang_file, PPISP_SPG_SLANG_FILE),
        )

    stage_dir = _stage_asset_dir(stage)
    if stage_dir is None:
        # Anonymous Kit stages do not provide a layer directory for relative SPG
        # assets. Use a unique colocated sidecar directory so the Slang source and
        # optional .slang.lua launcher resolve like exported 3DGRUT USDs.
        stage_dir = Path(tempfile.mkdtemp(prefix="isaaclab_ppisp_spg_"))
        copy_ppisp_spg_files(stage_dir)
        return str(stage_dir / PPISP_SPG_USDA_FILE), str(stage_dir / PPISP_SPG_SLANG_FILE)

    copy_ppisp_spg_files(stage_dir)
    return PPISP_SPG_USDA_FILE, PPISP_SPG_SLANG_FILE


def _stage_asset_dir(stage: Any, target_dir: str | Path | None = None) -> Path | None:
    if target_dir is not None:
        return Path(target_dir).expanduser().resolve()

    root_layer = stage.GetRootLayer()
    root_path = getattr(root_layer, "realPath", "") or getattr(root_layer, "identifier", "")
    if not root_path or root_path.startswith("anon:"):
        return None
    return Path(root_path).expanduser().resolve().parent


def _collect_xform_attr_time_samples(prim: Any, time_samples: set[float]) -> None:
    for attr in prim.GetAttributes():
        if attr.GetName().startswith("xformOp:"):
            time_samples.update(attr.GetTimeSamples())
