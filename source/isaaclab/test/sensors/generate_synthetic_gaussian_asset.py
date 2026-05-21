# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a tiny synthetic Gaussian-Splat USD asset for camera PPISP tests.

Avoids dependencies on heavyweight Nucleus assets by authoring a few large
opaque gaussians of known colors, bound to ``ParticleFieldEmissive.mdl`` with
``apply_inverse_tonemap=0`` and ``apply_srgb_linear=0`` so the wrapper PPISP is
the sole ISP authority. Tests assert *semantic invariants* of the PPISP
behavior (vignetting darkens corners, exposure offset increases mean, the CRF
compresses highlights, etc.) instead of doing a fidelity-against-baked
comparison — which sidesteps cross-renderer HDR-magnitude calibration drift
entirely.
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from isaaclab_ppisp import PpispCfg, normalize_ppisp_cfg

from pxr import Gf, Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from collections.abc import Iterator

    from isaaclab.renderers.renderer_cfg import RendererCfg
    from isaaclab.sim import SimulationCfg, SimulationContext


# SH degree-0 evaluation constant ``Y_0 = 1 / (2 * sqrt(pi))``. The standard
# 3DGS convention encodes a particle's base color as
# ``color = 0.5 + Y_0 * dc`` so inverting gives ``dc = (color - 0.5) / Y_0``.
_SH_Y0 = 1.0 / (2.0 * math.sqrt(math.pi))


@dataclass
class SyntheticGaussian:
    """One opaque gaussian in the synthetic scene."""

    position: tuple[float, float, float]
    """World-space position (x, y, z) in metres."""

    color: tuple[float, float, float]
    """Target final color in [0, 1] linear scene-referred space. Encoded into SH so
    the rendered (pre-PPISP) HDR pixels at the gaussian center approximate this color."""

    scale: float = 0.3
    """Isotropic scale (radius) of the gaussian ellipsoid in metres."""

    opacity: float = 1.0
    """Opacity in [0, 1]. Use 1.0 for fully opaque coverage."""


@dataclass
class SyntheticGaussianScene:
    """Scene description consumed by :func:`make_synthetic_gaussian_usd`.

    Defaults arrange four large fully-opaque gaussians (R, G, B, W) in a 2x2
    grid in the X-Y plane at Z=0, with a camera placed on +Z looking at the
    grid origin.
    """

    gaussians: list[SyntheticGaussian] = field(
        default_factory=lambda: [
            SyntheticGaussian(position=(-0.6, +0.6, 0.0), color=(0.9, 0.1, 0.1)),  # red
            SyntheticGaussian(position=(+0.6, +0.6, 0.0), color=(0.1, 0.9, 0.1)),  # green
            SyntheticGaussian(position=(-0.6, -0.6, 0.0), color=(0.1, 0.1, 0.9)),  # blue
            SyntheticGaussian(position=(+0.6, -0.6, 0.0), color=(0.9, 0.9, 0.9)),  # white
        ]
    )
    """Gaussians in the scene. Default forms a 2x2 RGBW grid."""

    camera_position: tuple[float, float, float] = (0.0, 0.0, 3.0)
    """Camera position. Default looks at the grid origin from +Z."""

    focal_length: float = 24.0
    """Pinhole camera focal length in mm."""

    horizontal_aperture: float = 20.955
    """Camera horizontal aperture in mm."""


def make_synthetic_gaussian_usd(path: str, scene: SyntheticGaussianScene | None = None) -> str:
    """Author a tiny gaussian-splat USD at ``path`` and return that path.

    The asset references ``ParticleFieldEmissive.mdl`` with ``apply_inverse_tonemap=0``
    and ``apply_srgb_linear=0`` so the wrapper PPISP is the sole ISP authority.
    The default prim is ``World``; cameras live at ``/World/Cameras/test_cam``
    and the gaussians at ``/World/Scene/gaussians/Gaussians/gaussians``.
    """
    if scene is None:
        scene = SyntheticGaussianScene()

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    stage = Usd.Stage.CreateNew(path)
    stage.SetMetadata("metersPerUnit", 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # Default prim ``/World`` so this asset can be referenced under any parent.
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # Camera under ``/World/Cameras/test_cam``. Authored without time samples so
    # ``UsdGeom.XformCache`` resolves the world transform at the default time.
    UsdGeom.Xform.Define(stage, "/World/Cameras")
    cam = UsdGeom.Camera.Define(stage, "/World/Cameras/test_cam")
    cam.GetFocalLengthAttr().Set(scene.focal_length)
    cam.GetHorizontalApertureAttr().Set(scene.horizontal_aperture)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
    cam.GetFStopAttr().Set(1.0)
    # Look from camera_position toward origin (-Z view direction).
    cx, cy, cz = scene.camera_position
    cam.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))

    # Gaussian particle field. Use a deeply nested path matching the typical
    # 3DGS export layout so user-side scene wiring matches real assets.
    UsdGeom.Xform.Define(stage, "/World/Scene")
    UsdGeom.Xform.Define(stage, "/World/Scene/gaussians")
    UsdGeom.Xform.Define(stage, "/World/Scene/gaussians/Gaussians")
    gauss_prim_path = "/World/Scene/gaussians/Gaussians/gaussians"
    gauss_prim = stage.DefinePrim(gauss_prim_path, "ParticleField3DGaussianSplat")

    def _attr(name: str, type_name: Sdf.ValueTypeName, value):
        attr = gauss_prim.CreateAttribute(name, type_name)
        attr.Set(value)
        return attr

    _attr("positions", Sdf.ValueTypeNames.Point3fArray, [Gf.Vec3f(*g.position) for g in scene.gaussians])
    _attr(
        "orientations",
        Sdf.ValueTypeNames.QuatfArray,
        # Identity quaternion (w, x, y, z) - Gf.Quatf takes (real, imaginary).
        [Gf.Quatf(1.0, 0.0, 0.0, 0.0) for _ in scene.gaussians],
    )
    _attr(
        "scales",
        Sdf.ValueTypeNames.Float3Array,
        [Gf.Vec3f(g.scale, g.scale, g.scale) for g in scene.gaussians],
    )
    _attr("opacities", Sdf.ValueTypeNames.FloatArray, [float(g.opacity) for g in scene.gaussians])

    # Encode the desired final color in SH degree-0 coefficients. With
    # apply_inverse_tonemap=0 and apply_srgb_linear=0, the MDL evaluates the
    # gaussian color as ``0.5 + Y_0 * dc * emission_intensity``. Solving for dc
    # gives the inverse encoding used here.
    sh_coeffs = [
        Gf.Vec3f(
            (g.color[0] - 0.5) / _SH_Y0,
            (g.color[1] - 0.5) / _SH_Y0,
            (g.color[2] - 0.5) / _SH_Y0,
        )
        for g in scene.gaussians
    ]
    _attr("radiance:sphericalHarmonicsCoefficients", Sdf.ValueTypeNames.Float3Array, sh_coeffs)
    sh_degree_attr = gauss_prim.CreateAttribute(
        "radiance:sphericalHarmonicsDegree", Sdf.ValueTypeNames.Int, custom=False
    )
    sh_degree_attr.Set(0)

    # Conservative extent — bounding box of all gaussian centers expanded by
    # their largest scale.
    if scene.gaussians:
        max_scale = max(g.scale for g in scene.gaussians)
        positions = [g.position for g in scene.gaussians]
        lo = (
            min(p[0] for p in positions) - max_scale,
            min(p[1] for p in positions) - max_scale,
            min(p[2] for p in positions) - max_scale,
        )
        hi = (
            max(p[0] for p in positions) + max_scale,
            max(p[1] for p in positions) + max_scale,
            max(p[2] for p in positions) + max_scale,
        )
        gauss_prim.CreateAttribute("extent", Sdf.ValueTypeNames.Float3Array).Set([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])

    # Material binding: ``ParticleFieldEmissive.mdl`` with the two boolean
    # ``apply_*`` inputs set to false so the wrapper PPISP is the sole ISP
    # authority and the gaussian color comes out of the renderer as linear
    # scene-referred radiance.
    UsdGeom.Xform.Define(stage, "/World/Scene/gaussians/Looks")
    material = stage.DefinePrim("/World/Scene/gaussians/Looks/ParticleFieldEmissive", "Material")
    shader = stage.DefinePrim("/World/Scene/gaussians/Looks/ParticleFieldEmissive/Shader", "Shader")
    shader.CreateAttribute("info:implementationSource", Sdf.ValueTypeNames.Token).Set("sourceAsset")
    shader.CreateAttribute("info:mdl:sourceAsset", Sdf.ValueTypeNames.Asset).Set("ParticleFieldEmissive.mdl")
    shader.CreateAttribute("info:mdl:sourceAsset:subIdentifier", Sdf.ValueTypeNames.Token).Set("ParticleFieldEmissive")
    shader.CreateAttribute("inputs:apply_inverse_tonemap", Sdf.ValueTypeNames.Bool, custom=True).Set(False)
    shader.CreateAttribute("inputs:apply_srgb_linear", Sdf.ValueTypeNames.Bool, custom=True).Set(False)
    shader.CreateAttribute("outputs:out", Sdf.ValueTypeNames.Token, custom=True)
    for output in ("mdl:displacement", "mdl:surface", "mdl:volume"):
        material.CreateAttribute(f"outputs:{output}", Sdf.ValueTypeNames.Token).AddConnection(
            shader.GetPath().AppendProperty("outputs:out")
        )
    gauss_prim.CreateRelationship("material:binding").SetTargets([material.GetPath()])

    stage.GetRootLayer().Save()
    return path


# PPISP cfg helpers ----------------------------------------------------------


# Strong negative radial coefficient — the warp kernel uses
# ``factor = clamp(1 + alpha1 * r^2 + alpha2 * r^4 + alpha3 * r^6, 0, 1)`` where
# ``r`` is normalised by ``max(W, H)``. With this value the corner of a square
# frame (``r^2 = 0.5``) attenuates by ``factor = 1 + (-1.5)(0.5) = 0.25``, i.e.
# corners drop to ~25% of center intensity. Visible but not fully black.
_AGGRESSIVE_VIGNETTING_ALPHA1 = -1.8

# Negative exposure offset (input × 2^-5 = ÷32) tuned so the aggressive cfg
# safely brings the RTX-bearing backends' gaussian HDR magnitudes (~10
# single-tile, ~17 multi-tile observed on OVRTX) below the CRF's [0,1] clamp
# before tonemapping. Newton's much lower native HDR scale is normalised
# separately via :func:`make_aggressive_ppisp_cfg`'s ``responsivity`` kwarg.
_AGGRESSIVE_EXPOSURE_OFFSET = -5.0


def make_aggressive_ppisp_cfg(*, responsivity: float = 1.0) -> PpispCfg:
    """Return a :class:`~isaaclab_ppisp.PpispCfg` with every PPISP feature engaged enough
    to be assertable in a downstream test.

    Each input is dialed past the "subtle correction" defaults so an integration
    test can check semantic invariants of the wrapper PPISP pipeline:

    * **Exposure**: ``exposureOffset = -5`` (input × 2^-5 = ÷32) — tuned so that
      a near-typical RTX-style gaussian HDR magnitude (≈10–17) lands below the
      CRF's [0,1] clamp before tonemapping, then CRF compresses to upper LDR.
    * **Vignetting**: per-channel ``alpha1 = -1.8`` — corners drop to ~0% of
      center intensity for a square frame. Slight per-channel imbalance
      (R < G < B in alpha2) produces a non-uniform corner colour cast.
    * **Color homography**: ``red_latent`` pulls the red anchor outward and
      ``green_latent`` pulls the green anchor down — input white pixels acquire
      a visible warm hue shift.
    * **CRF**: per-channel toe/shoulder/gamma/center values that meaningfully
      compress highlights (no overflow above 1.0 ⇒ max LDR uint8 stays at 255
      only when the wrapper actually clamps; under-engaged CRF would let the
      explicit ``clamp(.., 0, 1)`` in the kernel do all the work).

    Args:
        responsivity: PPISP achromatic ``responsivity`` factor applied **before**
            exposure. Defaults to ``1.0`` (calibrated for RTX-bearing backends'
            HDR magnitude). The Newton backend produces a much lower-magnitude
            HDR for the same scene and tests pass a value > 1 to bring its
            effective signal in line with the RTX backends.
    """
    inputs: dict[str, float | tuple[float, float]] = {
        "responsivity": responsivity,
        "exposureOffset": _AGGRESSIVE_EXPOSURE_OFFSET,
        # Vignetting: identical optical center for all channels (image center),
        # with a slight per-channel falloff offset so a vignetted region has
        # a faint chromatic gradient — verifies the per-channel paths are wired.
        "vignettingCenterR": (0.0, 0.0),
        "vignettingAlpha1R": _AGGRESSIVE_VIGNETTING_ALPHA1,
        "vignettingAlpha2R": -0.4,
        "vignettingAlpha3R": 0.0,
        "vignettingCenterG": (0.0, 0.0),
        "vignettingAlpha1G": _AGGRESSIVE_VIGNETTING_ALPHA1,
        "vignettingAlpha2G": -0.2,
        "vignettingAlpha3G": 0.0,
        "vignettingCenterB": (0.0, 0.0),
        "vignettingAlpha1B": _AGGRESSIVE_VIGNETTING_ALPHA1,
        "vignettingAlpha2B": 0.0,
        "vignettingAlpha3B": 0.0,
        # Color homography: shift the red and green anchors so the output picks
        # up a clear hue rotation. Blue and neutral remain near identity.
        "colorLatentRed": (0.4, 0.0),
        "colorLatentGreen": (0.0, -0.4),
        "colorLatentBlue": (0.0, 0.0),
        "colorLatentNeutral": (0.0, 0.0),
        # CRF: stronger shoulder than the default highlight knee so a saturated
        # input is compressed rather than clipped. Per-channel gammas are split
        # to produce a subtle warm cast.
        "crfToeR": 0.05,
        "crfShoulderR": 0.20,
        "crfGammaR": 0.50,
        "crfCenterR": 0.0,
        "crfToeG": 0.05,
        "crfShoulderG": 0.20,
        "crfGammaG": 0.45,
        "crfCenterG": 0.0,
        "crfToeB": 0.05,
        "crfShoulderB": 0.20,
        "crfGammaB": 0.40,
        "crfCenterB": 0.0,
    }
    return normalize_ppisp_cfg(PpispCfg(inputs=inputs))


def assert_ppisp_invariants(
    rgb_tile: torch.Tensor,
    *,
    patch: int = 16,
    vignetting_corner_ratio_max: float = 0.5,
    label: str = "",
) -> None:
    """Assert the four PPISP signatures expected from :func:`make_aggressive_ppisp_cfg`
    on a single ``[H, W, C>=3]`` rgb tile (uint8-range floats).

    1. Non-degenerate frame: ``5 < mean < 250``.
    2. Vignetting: each of the 4 corner patches is below
       ``vignetting_corner_ratio_max`` times the center patch (``alpha1=-1.8``
       drives the pre-CRF corner factor to ~0; after CRF compression the
       per-renderer corner/center ratio sits well below 0.5).
    3. Exposure: center patch mean > 50 (the aggressive cfg's
       ``responsivity * 2^exposureOffset`` is tuned so the per-renderer HDR
       magnitude lands solidly into mid-to-upper LDR after CRF).
    4. CRF clamping: output stays in ``[0, 255]`` (also catches NaNs implicitly).

    ``label`` is included in every assertion message so the caller can identify
    which renderer / tile failed.
    """
    prefix = f"[{label}] " if label else ""
    h, w = rgb_tile.shape[:2]

    mean = rgb_tile.mean().item()
    assert 5.0 < mean < 250.0, f"{prefix}render is degenerate (mean={mean:.1f})"

    cy, cx = h // 2 - patch // 2, w // 2 - patch // 2
    center_mean = rgb_tile[cy : cy + patch, cx : cx + patch, :3].mean().item()
    assert center_mean > 1.0, f"{prefix}center patch is degenerate (mean={center_mean:.1f})"

    for corner_name, y0, x0 in (
        ("top-left", 0, 0),
        ("top-right", 0, w - patch),
        ("bottom-left", h - patch, 0),
        ("bottom-right", h - patch, w - patch),
    ):
        corner_mean = rgb_tile[y0 : y0 + patch, x0 : x0 + patch, :3].mean().item()
        ratio = corner_mean / center_mean
        assert ratio < vignetting_corner_ratio_max, (
            f"{prefix}vignetting too weak at {corner_name}: corner/center = {ratio:.3f} "
            f"(expected < {vignetting_corner_ratio_max}). "
            f"corner_mean={corner_mean:.1f}, center_mean={center_mean:.1f}"
        )

    assert center_mean > 50.0, (
        f"{prefix}aggressive PPISP cfg should land the center patch above 50 (mid-LDR); "
        f"got {center_mean:.1f}. Check responsivity/exposureOffset and that the renderer is producing HDR > 0."
    )

    assert rgb_tile.max().item() <= 255.0, f"{prefix}output overflow: max={rgb_tile.max().item():.1f}"
    assert rgb_tile.min().item() >= 0.0, f"{prefix}output underflow: min={rgb_tile.min().item():.1f}"


def assert_ppisp_lifts_exposure(
    hdr_tile: torch.Tensor,
    rgb_tile: torch.Tensor,
    *,
    patch: int = 16,
    hdr_center_min: float = 1.0e-2,
    ldr_center_norm_range: tuple[float, float] = (0.1, 0.95),
    label: str = "",
) -> None:
    """Assert PPISP normalises the renderer's HDR into a useful LDR range.

    Different renderer backends produce wildly different HDR magnitudes for the
    same synthetic gaussian scene (Newton's emissive scale is ~10× lower than
    the RTX backends'). The aggressive cfg's ``responsivity`` knob is tuned per
    backend to bring the effective signal in line; this assertion then only
    enforces that:

    * the renderer is producing an HDR AOV (lower bound on the HDR center)
    * PPISP delivers a non-degenerate LDR center (not black, not fully
      saturated)

    Localises failures:

    * **Renderer not producing HDR** — caught by the HDR lower bound.
    * **PPISP saturated or black** — caught by the LDR range bound; suggests
      either ``responsivity``/``exposureOffset`` mis-tuned or the pipeline
      not running.

    Args:
        hdr_tile: ``[H, W, 3]`` float HDR tile (from ``output["rgb_hdr"]``).
        rgb_tile: ``[H, W, C>=3]`` uint8-range float LDR tile (from ``output["rgb"]``).
        patch: Center-patch window size in pixels.
        hdr_center_min: Lower bound on the raw HDR center mean.
        ldr_center_norm_range: ``(min, max)`` for the LDR center mean / 255.
        label: Included in every assertion message so the caller can identify
            which renderer / tile failed.
    """
    prefix = f"[{label}] " if label else ""
    h, w = rgb_tile.shape[:2]
    cy, cx = h // 2 - patch // 2, w // 2 - patch // 2

    hdr_center = hdr_tile[cy : cy + patch, cx : cx + patch, :3].float().mean().item()
    ldr_center_norm = rgb_tile[cy : cy + patch, cx : cx + patch, :3].float().mean().item() / 255.0

    assert hdr_center > hdr_center_min, (
        f"{prefix}HDR center too dark (mean={hdr_center:.4f}) — renderer not producing the HDR AOV?"
    )
    ldr_lo, ldr_hi = ldr_center_norm_range
    assert ldr_lo < ldr_center_norm < ldr_hi, (
        f"{prefix}PPISP LDR center out of range: ldr_norm={ldr_center_norm:.4f} "
        f"(expected {ldr_lo} < x < {ldr_hi}; hdr_center={hdr_center:.4f}). "
        f"Likely saturated or black — check responsivity/exposureOffset tuning."
    )


# ──────────────────────────────────────────────────────────────────────────────
# InteractiveScene helpers shared by the Isaac RTX-, Newton-, and OVRTX-backed
# gaussian tests.
# ──────────────────────────────────────────────────────────────────────────────

SYNTHETIC_GAUSSIAN_SCENE_REL_PATH = "Scene"
"""Asset prim path under each environment in :class:`SyntheticGaussianSceneCfg`."""

SYNTHETIC_GAUSSIAN_CAMERA_NAME = "test_cam"
"""Camera prim name authored inside the synthesised asset USD."""

SYNTHETIC_GAUSSIAN_CAMERA_REGEX = (
    f"/World/envs/env_.*/{SYNTHETIC_GAUSSIAN_SCENE_REL_PATH}/Cameras/{SYNTHETIC_GAUSSIAN_CAMERA_NAME}"
)
"""Regex camera prim path that resolves to one camera per env (single or tiled)."""


@configclass
class SyntheticGaussianSceneCfg(InteractiveSceneCfg):
    """Minimal :class:`~isaaclab.scene.InteractiveScene` cfg wrapping the synthesised gaussian asset.

    The :attr:`anchor` rigid body exists solely to give Newton-backed physics
    a non-empty body table — it is invisible at the camera viewpoint and far
    enough below the scene to never appear in the render.

    The ``gaussian`` asset URL is filled in at runtime by
    :func:`fresh_synthetic_gaussian_interactive_scene`.
    """

    env_spacing: float = 2.0

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    gaussian = AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{SYNTHETIC_GAUSSIAN_SCENE_REL_PATH}",
        spawn=sim_utils.UsdFileCfg(usd_path=""),  # filled in at runtime
    )

    anchor = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Anchor",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
    )


@contextlib.contextmanager
def fresh_synthetic_gaussian_interactive_scene(
    usd_path: str,
    sim_cfg: SimulationCfg,
    *,
    num_envs: int = 1,
) -> Iterator[SimulationContext]:
    """Yield a fresh :class:`~isaaclab.sim.SimulationContext` with the synthesised
    gaussian asset referenced under each env via :class:`SyntheticGaussianSceneCfg`.

    The InteractiveScene is held alive for the lifetime of the context — its
    callbacks register *weak* refs to the parent's bound methods; if the scene
    is dropped, the next ``dispatch_event`` raises ``ReferenceError`` from a
    dead weakref.

    Args:
        usd_path: Path to the synthesised gaussian USD asset (typically produced
            by :func:`make_synthetic_gaussian_usd`).
        sim_cfg: The simulation cfg (caller-provided, since the physics backend
            and timestep are renderer-specific).
        num_envs: Number of tiled envs to spawn.

    Yields:
        The constructed :class:`SimulationContext`.
    """
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = SyntheticGaussianSceneCfg(num_envs=num_envs)
    scene_cfg.gaussian.spawn = sim_utils.UsdFileCfg(usd_path=usd_path)
    scene = InteractiveScene(scene_cfg)  # noqa: F841 — kept alive intentionally
    try:
        yield sim
    finally:
        with contextlib.suppress(Exception):
            sim.stop()
        with contextlib.suppress(Exception):
            sim.clear_instance()


def render_synthetic_gaussian_scene(
    usd_path: str,
    *,
    sim_cfg: SimulationCfg,
    renderer_cfg: RendererCfg,
    data_types: list[str],
    num_envs: int = 1,
    height: int = 128,
    width: int = 128,
    sim_dt: float = 1.0 / 60.0,
    stabilisation_steps: int = 5,
    responsivity: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Render the synthesised gaussian asset with the aggressive wrapper ISP.

    Builds an :class:`~isaaclab.scene.InteractiveScene` via
    :func:`fresh_synthetic_gaussian_interactive_scene`, instantiates a
    :class:`~isaaclab.sensors.camera.Camera` whose prim path is
    :data:`SYNTHETIC_GAUSSIAN_CAMERA_REGEX` (one camera per env), drives the
    sim for ``stabilisation_steps`` ticks, and returns every requested output.

    Args:
        usd_path: Path to the synthesised gaussian USD asset.
        sim_cfg: Caller-provided simulation cfg (carries the physics backend).
        renderer_cfg: Renderer cfg (typically ``IsaacRtxRendererCfg``,
            ``NewtonWarpRendererCfg``, or ``OVRTXRendererCfg``).
        data_types: List passed through to :attr:`~isaaclab.sensors.camera.CameraCfg.data_types`.
            Include ``"rgb_hdr"`` here when callers need access to the renderer's HDR AOV
            (e.g. for :func:`assert_ppisp_lifts_exposure`).
        num_envs: Number of tiled envs.
        height: Render height [pixels].
        width: Render width [pixels].
        sim_dt: Simulation timestep [s] used for ``camera.update``.
        stabilisation_steps: Sim steps to run before reading the final frame.

    Returns:
        A dict mapping every key present in ``camera.data.output`` to a
        ``[num_envs, height, width, channels]`` float32 CPU tensor (uint8 LDR
        buffers are cast to float for downstream arithmetic).
    """
    isp_cfg = make_aggressive_ppisp_cfg(responsivity=responsivity)
    with fresh_synthetic_gaussian_interactive_scene(usd_path, sim_cfg, num_envs=num_envs) as sim:
        cfg = CameraCfg(
            prim_path=SYNTHETIC_GAUSSIAN_CAMERA_REGEX,
            update_period=0.0,
            height=height,
            width=width,
            data_types=data_types,
            spawn=None,
            isp_cfg=isp_cfg,
            renderer_cfg=renderer_cfg,
        )
        camera = Camera(cfg)
        sim.reset()
        for _ in range(stabilisation_steps):
            sim.step()
        camera.update(sim_dt)
        outputs = {name: tensor.clone().detach().cpu().to(torch.float32) for name, tensor in camera.data.output.items()}
        del camera
        return outputs
