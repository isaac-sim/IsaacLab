# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a tiny synthetic Gaussian-Splat USD asset for camera PPISP tests.

Avoids dependencies on heavyweight Nucleus assets by authoring a few large
opaque gaussians of known colors, bound to ``ParticleFieldEmissive.mdl`` with
``apply_inverse_tonemap=0`` and ``apply_srgb_linear=0`` so the wrapper PPISP is
the sole ISP authority. Tests assert *semantic invariants* of the PPISP
behavior (non-degenerate LDR output from renderer HDR, vignetting darkens
corners, the CRF keeps values bounded, etc.) instead of doing a
fidelity-against-baked comparison — which sidesteps cross-renderer
HDR-magnitude calibration drift entirely.
"""

from __future__ import annotations

import contextlib
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from isaaclab_ppisp import PpispCfg, normalize_ppisp_cfg

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.camera_isp import CameraISPMode
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
class SyntheticGaussianAnimation:
    """Time-sampled gaussian animation authored alongside the static grid.

    Mirrors how NuRec-exported reconstructions author motion: each animated
    *track* is an ``Xform`` with a single ``ParticleField3DGaussianSplat``
    child, where

    * **rigid** tracks time-sample the track ``Xform``'s ``xformOp:transform``
      and keep their per-particle arrays static, and
    * **deformable** tracks time-sample the per-particle ``positionsh`` and
      ``orientationsh`` arrays and keep their track ``Xform`` static.

    The two mechanisms are kept on separate tracks (rather than combined on one,
    as the reference captures do) so a renderer that implements only one of them
    is unambiguous to diagnose visually.

    Time-sampling the per-particle arrays is also what makes USD population
    classify the prim as time varying, which is the precondition for the RTX
    backends to update animated gaussians in place instead of destroying and
    rebuilding the geometry every frame.
    """

    num_frames: int = 24
    """Number of authored time samples per animated attribute, starting at time code 0."""

    time_codes_per_second: float = 24.0
    """Stage time codes per second [1/s]. Also the intended playback frame rate."""

    track_height: float = 0.9
    """Height of both animated tracks above the static grid plane [m]."""

    rigid_track_center: tuple[float, float] = (-0.45, 0.0)
    """Center of the rigid track's circular orbit in the X-Y plane [m]."""

    rigid_orbit_radius: float = 0.3
    """Radius of the rigid track's orbit [m]."""

    deformable_track_center: tuple[float, float] = (0.45, 0.0)
    """Center of the deformable track in the X-Y plane [m]."""

    ring_radius: float = 0.22
    """Rest radius of each track's particle ring [m]."""

    deformable_pulse: float = 0.5
    """Peak radial pulse of the deformable ring, as a fraction of its rest radius."""

    num_particles_per_track: int = 6
    """Number of gaussians in each animated track."""

    particle_scale: tuple[float, float, float] = (0.24, 0.06, 0.06)
    """Anisotropic scale of every animated gaussian [m]. Deliberately elongated so
    the time-sampled ``orientationsh`` produces visible rotation."""

    rigid_color: tuple[float, float, float] = (0.9, 0.9, 0.1)
    """Linear color of the rigid track's gaussians in [0, 1]."""

    deformable_color: tuple[float, float, float] = (0.1, 0.9, 0.9)
    """Linear color of the deformable track's gaussians in [0, 1]."""


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

    animation: SyntheticGaussianAnimation | None = None
    """When set, two extra animated gaussian tracks are authored next to the static grid.
    See :class:`SyntheticGaussianAnimation`."""


def make_offscreen_gaussian_scene() -> SyntheticGaussianScene:
    """Return a control scene whose only gaussian sits far outside every camera frustum.

    The asset still carries a populated gaussian field — so the renderer follows the same
    ingest path — but contributes nothing to the image. Rendering it with otherwise
    identical settings yields the gaussian-free baseline that
    :func:`assert_gaussian_contribution` compares against.
    """
    return SyntheticGaussianScene(gaussians=[SyntheticGaussian(position=(5000.0, 5000.0, 0.0), color=(0.9, 0.1, 0.1))])


def make_synthetic_gaussian_usd(path: str, scene: SyntheticGaussianScene | None = None) -> str:
    """Author a tiny gaussian-splat USD at ``path`` and return that path.

    The asset references ``ParticleFieldEmissive.mdl`` with ``apply_inverse_tonemap=0``
    and ``apply_srgb_linear=0`` so the wrapper PPISP is the sole ISP authority.
    The default prim is ``World``; cameras live at ``/World/Cameras/test_cam``
    and the gaussians at ``/World/Scene/gaussians/Gaussians/gaussians``.

    When :attr:`SyntheticGaussianScene.animation` is set, two animated tracks are
    additionally authored at :data:`SYNTHETIC_GAUSSIAN_RIGID_TRACK_PATH` and
    :data:`SYNTHETIC_GAUSSIAN_DEFORMABLE_TRACK_PATH`.
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
    _author_particle_field_hints(gauss_prim)

    # Conservative extent — bounding box of all gaussian centers expanded by
    # their largest scale.
    if scene.gaussians:
        _author_extent(
            gauss_prim,
            [g.position for g in scene.gaussians],
            padding=max(g.scale for g in scene.gaussians),
        )

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

    if scene.animation is not None:
        _author_animated_tracks(stage, scene.animation, material_path=material.GetPath())

    stage.GetRootLayer().Save()
    return path


# Animated gaussian authoring ------------------------------------------------

SYNTHETIC_GAUSSIAN_RIGID_TRACK_PATH = "/World/Scene/gaussians/NuRec/track_rigid"
"""Path of the rigid animated track ``Xform`` inside the synthesised asset.

Its time-sampled ``xformOp:transform`` drives the whole splat; its single
``ParticleField3DGaussianSplat`` child is named ``track_rigid_gaussians``.
"""

SYNTHETIC_GAUSSIAN_DEFORMABLE_TRACK_PATH = "/World/Scene/gaussians/NuRec/track_deformable"
"""Path of the deformable animated track ``Xform`` inside the synthesised asset.

Its ``Xform`` is static; its ``track_deformable_gaussians`` child time-samples the
per-particle ``positionsh`` and ``orientationsh`` arrays.
"""


_PARTICLE_FIELD_HINTS = {
    "colorSpace:name": "lin_rec709_scene",
    "projectionModeHint": "perspective",
    "sortingModeHint": "zDepth",
}
"""Render hints NuRec exports author on the gaussian prim, with their exported values.

Copied from the reference capture used by the PPISP demos
(``Samples/Scene_ParticleField/valiant_auto.usdz``) so the synthesised asset exercises the
same attribute set real captures deliver. ``sortingModeHint`` selects the depth metric the
renderer sorts splats by; ``zDepth`` is both what NuRec exports and the renderer's default
when the token is absent, so authoring it changes no rendering behaviour on its own.
"""


def _author_particle_field_hints(prim: Usd.Prim) -> None:
    """Author the color space, projection and sorting hint tokens onto a gaussian prim.

    Args:
        prim: The ``ParticleField3DGaussianSplat`` prim to author the hints on.
    """
    for name, value in _PARTICLE_FIELD_HINTS.items():
        prim.CreateAttribute(name, Sdf.ValueTypeNames.Token).Set(value)


def _author_extent(prim: Usd.Prim, positions: list[tuple[float, float, float]], *, padding: float) -> None:
    """Author a conservative ``extent`` covering ``positions`` expanded by ``padding`` [m].

    Args:
        prim: The ``ParticleField3DGaussianSplat`` prim to author ``extent`` on.
        positions: Particle centers [m] the extent must cover. For an animated prim
            this must be the union over every authored time sample, since ``extent``
            itself is static.
        padding: Isotropic padding [m] added on every side.
    """
    lo = tuple(min(p[axis] for p in positions) - padding for axis in range(3))
    hi = tuple(max(p[axis] for p in positions) + padding for axis in range(3))
    prim.CreateAttribute("extent", Sdf.ValueTypeNames.Float3Array).Set([Gf.Vec3f(*lo), Gf.Vec3f(*hi)])


def _sh_dc_from_color(color: tuple[float, float, float]) -> Gf.Vec3h:
    """Return the half-precision SH degree-0 DC coefficient encoding ``color``.

    Inverse of the 3DGS convention ``color = 0.5 + Y_0 * dc`` also used for the
    static grid's full-float coefficients.
    """
    return Gf.Vec3h(*((channel - 0.5) / _SH_Y0 for channel in color))


def _spin_quath(angle: float) -> Gf.Quath:
    """Return a half-precision quaternion rotating ``angle`` [rad] about +Z."""
    return Gf.Quath(math.cos(0.5 * angle), 0.0, 0.0, math.sin(0.5 * angle))


def _author_animated_tracks(
    stage: Usd.Stage, animation: SyntheticGaussianAnimation, *, material_path: Sdf.Path
) -> None:
    """Author the rigid and deformable animated gaussian tracks onto ``stage``.

    See :class:`SyntheticGaussianAnimation` for the authoring convention and the
    division of labour between the two tracks.

    Args:
        stage: Stage being authored. Its time-code range is set from ``animation``.
        animation: Animation description.
        material_path: ``ParticleFieldEmissive`` material bound to both tracks.
    """
    stage.SetTimeCodesPerSecond(animation.time_codes_per_second)
    stage.SetStartTimeCode(0.0)
    stage.SetEndTimeCode(float(animation.num_frames - 1))

    UsdGeom.Xform.Define(stage, "/World/Scene/gaussians/NuRec")

    # Both tracks share the same local ring layout; only what is time-sampled differs.
    num_particles = animation.num_particles_per_track
    ring = [2.0 * math.pi * index / num_particles for index in range(num_particles)]
    frames = [(frame, 2.0 * math.pi * frame / animation.num_frames) for frame in range(animation.num_frames)]

    # Rigid track: static local particles, time-sampled track transform.
    rigid_positions = [
        (
            animation.ring_radius * math.cos(theta),
            animation.ring_radius * math.sin(theta),
            0.0,
        )
        for theta in ring
    ]
    rigid_prim = _author_half_gaussian_prim(
        stage,
        f"{SYNTHETIC_GAUSSIAN_RIGID_TRACK_PATH}/track_rigid_gaussians",
        positions=rigid_positions,
        orientations=[_spin_quath(theta) for theta in ring],
        animation=animation,
        color=animation.rigid_color,
        material_path=material_path,
    )
    # The track orbits, so the *local* extent only needs to cover the local particles.
    _author_extent(rigid_prim, rigid_positions, padding=max(animation.particle_scale))

    center_x, center_y = animation.rigid_track_center
    transform_op = UsdGeom.Xform(stage.GetPrimAtPath(SYNTHETIC_GAUSSIAN_RIGID_TRACK_PATH)).AddTransformOp()
    for frame, phase in frames:
        transform = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), math.degrees(phase)))
        transform.SetTranslateOnly(
            Gf.Vec3d(
                center_x + animation.rigid_orbit_radius * math.cos(phase),
                center_y + animation.rigid_orbit_radius * math.sin(phase),
                animation.track_height,
            )
        )
        transform_op.Set(transform, Usd.TimeCode(frame))

    # Deformable track: static track transform, time-sampled per-particle arrays.
    # The ring pulses radially and every particle spins about +Z.
    deformable_positions_by_frame = [
        [
            (
                animation.ring_radius * (1.0 + animation.deformable_pulse * math.sin(phase)) * math.cos(theta),
                animation.ring_radius * (1.0 + animation.deformable_pulse * math.sin(phase)) * math.sin(theta),
                0.0,
            )
            for theta in ring
        ]
        for _, phase in frames
    ]
    deformable_prim = _author_half_gaussian_prim(
        stage,
        f"{SYNTHETIC_GAUSSIAN_DEFORMABLE_TRACK_PATH}/track_deformable_gaussians",
        positions=deformable_positions_by_frame[0],
        orientations=[_spin_quath(theta) for theta in ring],
        animation=animation,
        color=animation.deformable_color,
        material_path=material_path,
    )
    positions_attr = deformable_prim.GetAttribute("positionsh")
    orientations_attr = deformable_prim.GetAttribute("orientationsh")
    for (frame, phase), positions in zip(frames, deformable_positions_by_frame, strict=True):
        positions_attr.Set(Vt.Vec3hArray([Gf.Vec3h(*position) for position in positions]), Usd.TimeCode(frame))
        orientations_attr.Set(
            Vt.QuathArray([_spin_quath(theta + phase) for theta in ring]),
            Usd.TimeCode(frame),
        )

    # ``extent`` is static, so it must cover the union of every authored frame.
    _author_extent(
        deformable_prim,
        [position for positions in deformable_positions_by_frame for position in positions],
        padding=max(animation.particle_scale),
    )

    center_x, center_y = animation.deformable_track_center
    UsdGeom.Xform(stage.GetPrimAtPath(SYNTHETIC_GAUSSIAN_DEFORMABLE_TRACK_PATH)).AddTranslateOp().Set(
        Gf.Vec3d(center_x, center_y, animation.track_height)
    )


def _author_half_gaussian_prim(
    stage: Usd.Stage,
    prim_path: str,
    *,
    positions: list[tuple[float, float, float]],
    orientations: list[Gf.Quath],
    animation: SyntheticGaussianAnimation,
    color: tuple[float, float, float],
    material_path: Sdf.Path,
) -> Usd.Prim:
    """Author a half-precision gaussian prim under an ``Xform`` track parent.

    Uses the ``positionsh``/``orientationsh``/``scalesh``/``opacitiesh`` half-precision
    attribute set that NuRec exports, rather than the static grid's full-float set, so
    both spellings are exercised. Callers author ``extent`` and any time samples.

    Args:
        stage: Stage being authored.
        prim_path: Path of the gaussian prim. Its parent is defined as an ``Xform`` track.
        positions: Particle centers [m] in the track's local frame, at the default time.
        orientations: Per-particle orientations at the default time.
        animation: Animation description supplying scale and particle count.
        color: Linear color in [0, 1] shared by every particle of this track.
        material_path: ``ParticleFieldEmissive`` material to bind.

    Returns:
        The authored ``ParticleField3DGaussianSplat`` prim.
    """
    UsdGeom.Xform.Define(stage, Sdf.Path(prim_path).GetParentPath().pathString)
    prim = stage.DefinePrim(prim_path, "ParticleField3DGaussianSplat")

    prim.CreateAttribute("positionsh", Sdf.ValueTypeNames.Point3hArray).Set(
        Vt.Vec3hArray([Gf.Vec3h(*position) for position in positions])
    )
    prim.CreateAttribute("orientationsh", Sdf.ValueTypeNames.QuathArray).Set(Vt.QuathArray(list(orientations)))
    prim.CreateAttribute("scalesh", Sdf.ValueTypeNames.Half3Array).Set(
        Vt.Vec3hArray([Gf.Vec3h(*animation.particle_scale)] * len(positions))
    )
    prim.CreateAttribute("opacitiesh", Sdf.ValueTypeNames.HalfArray).Set(Vt.HalfArray([1.0] * len(positions)))
    prim.CreateAttribute("radiance:sphericalHarmonicsCoefficientsh", Sdf.ValueTypeNames.Half3Array).Set(
        Vt.Vec3hArray([_sh_dc_from_color(color)] * len(positions))
    )
    prim.CreateAttribute("radiance:sphericalHarmonicsDegree", Sdf.ValueTypeNames.Int, custom=False).Set(0)
    _author_particle_field_hints(prim)
    prim.CreateRelationship("material:binding").SetTargets([material_path])
    return prim


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


_PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN = 241_961
_PPISP_CONTROLLER_OFF_CONV1_W = 0
_PPISP_CONTROLLER_OFF_CONV1_B = _PPISP_CONTROLLER_OFF_CONV1_W + 16 * 3
_PPISP_CONTROLLER_OFF_CONV2_W = _PPISP_CONTROLLER_OFF_CONV1_B + 16
_PPISP_CONTROLLER_OFF_CONV2_B = _PPISP_CONTROLLER_OFF_CONV2_W + 32 * 16
_PPISP_CONTROLLER_OFF_CONV3_W = _PPISP_CONTROLLER_OFF_CONV2_B + 32
_PPISP_CONTROLLER_OFF_CONV3_B = _PPISP_CONTROLLER_OFF_CONV3_W + 64 * 32
_PPISP_CONTROLLER_OFF_TRUNK0_W = _PPISP_CONTROLLER_OFF_CONV3_B + 64
_PPISP_CONTROLLER_OFF_TRUNK0_B = _PPISP_CONTROLLER_OFF_TRUNK0_W + 128 * 1601
_PPISP_CONTROLLER_OFF_TRUNK1_W = _PPISP_CONTROLLER_OFF_TRUNK0_B + 128
_PPISP_CONTROLLER_OFF_TRUNK1_B = _PPISP_CONTROLLER_OFF_TRUNK1_W + 128 * 128
_PPISP_CONTROLLER_OFF_TRUNK2_W = _PPISP_CONTROLLER_OFF_TRUNK1_B + 128
_PPISP_CONTROLLER_OFF_TRUNK2_B = _PPISP_CONTROLLER_OFF_TRUNK2_W + 128 * 128
_PPISP_CONTROLLER_OFF_EXP_W = _PPISP_CONTROLLER_OFF_TRUNK2_B + 128
_PPISP_CONTROLLER_OFF_EXP_B = _PPISP_CONTROLLER_OFF_EXP_W + 128
_PPISP_CONTROLLER_OFF_COL_W = _PPISP_CONTROLLER_OFF_EXP_B + 1
_PPISP_CONTROLLER_OFF_COL_B = _PPISP_CONTROLLER_OFF_COL_W + 8 * 128

_PPISP_CONTROLLER_TOTAL_WEIGHTS = _PPISP_CONTROLLER_OFF_COL_B + 8
if _PPISP_CONTROLLER_TOTAL_WEIGHTS != _PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN:
    raise RuntimeError(
        "Synthetic PPISP controller fixture offsets are inconsistent: "
        f"{_PPISP_CONTROLLER_TOTAL_WEIGHTS} != {_PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN}."
    )


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


def make_neutral_ppisp_cfg(*, responsivity: float = 1.0) -> PpispCfg:
    """Return a mild static PPISP cfg used as the camera-attribute negative control."""
    return normalize_ppisp_cfg(PpispCfg(inputs={"responsivity": responsivity, "exposureOffset": 0.0}))


def assert_images_meaningfully_different(
    reference_rgb: torch.Tensor,
    candidate_rgb: torch.Tensor,
    *,
    min_mean_abs_diff: float = 3.0,
    label: str = "",
) -> None:
    """Assert two LDR RGB tiles differ enough to prove PPISP attributes changed output."""
    prefix = f"[{label}] " if label else ""
    diff = (reference_rgb[..., :3].float() - candidate_rgb[..., :3].float()).abs()
    mean_abs_diff = diff.mean().item()
    assert mean_abs_diff > min_mean_abs_diff, (
        f"{prefix}image difference too small: mean_abs_diff={mean_abs_diff:.3f}, "
        f"expected > {min_mean_abs_diff}. The authored PPISP camera attributes may not be applied."
    )


def assert_gaussian_contribution(
    rgb_tile: torch.Tensor,
    control_rgb_tile: torch.Tensor,
    *,
    min_mean_abs_diff: float = 5.0,
    label: str = "",
) -> None:
    """Assert a rendered tile actually contains gaussian content.

    The PPISP signature assertions hold equally well on a render whose gaussians are
    entirely absent — the background alone satisfies them — so a renderer that silently
    drops splats otherwise passes. Comparing against a control render of
    :func:`make_offscreen_gaussian_scene` attributes any difference to the splats.

    Args:
        rgb_tile: Tile rendered from the default scene, shape ``[H, W, C>=3]``.
        control_rgb_tile: Matching tile rendered from :func:`make_offscreen_gaussian_scene`.
        min_mean_abs_diff: Minimum per-pixel mean absolute difference [LDR units].
        label: Included in the assertion message to identify the renderer / tile.
    """
    prefix = f"[{label}] " if label else ""
    mean_abs_diff = (rgb_tile[..., :3].float() - control_rgb_tile[..., :3].float()).abs().mean().item()
    assert mean_abs_diff > min_mean_abs_diff, (
        f"{prefix}tile is indistinguishable from a gaussian-free render: "
        f"mean_abs_diff={mean_abs_diff:.3f}, expected > {min_mean_abs_diff}. The renderer produced no "
        "gaussian contribution for this tile — check that the gaussian prim is ingested and rendered "
        "for this env's camera."
    )


def assert_ppisp_controller_matches_static(
    static_rgb: torch.Tensor,
    controller_rgb: torch.Tensor,
    *,
    max_mean_abs_diff: float = 8.0,
    label: str = "",
) -> None:
    """Assert deterministic controller output matches the equivalent static PPISP cfg."""
    prefix = f"[{label}] " if label else ""
    diff = (static_rgb[..., :3].float() - controller_rgb[..., :3].float()).abs()
    mean_abs_diff = diff.mean().item()
    assert mean_abs_diff < max_mean_abs_diff, (
        f"{prefix}controller PPISP differs from static reference: mean_abs_diff={mean_abs_diff:.3f}, "
        f"expected < {max_mean_abs_diff}."
    )


def _deterministic_controller_weights(ppisp_cfg: PpispCfg) -> tuple[float, ...]:
    inputs = ppisp_cfg.inputs
    weights = [0.0] * _PPISP_CONTROLLER_EXPECTED_WEIGHTS_LEN
    weights[_PPISP_CONTROLLER_OFF_EXP_B] = float(inputs["exposureOffset"])
    color_values = (
        *_float2(inputs["colorLatentBlue"]),
        *_float2(inputs["colorLatentRed"]),
        *_float2(inputs["colorLatentGreen"]),
        *_float2(inputs["colorLatentNeutral"]),
    )
    for i, value in enumerate(color_values):
        weights[_PPISP_CONTROLLER_OFF_COL_B + i] = value
    return tuple(weights)


def _float2(value: float | tuple[float, float]) -> tuple[float, float]:
    assert not isinstance(value, float)
    return (float(value[0]), float(value[1]))


def _camera_path_for_env(env_id: int = 0) -> str:
    return f"/World/envs/env_{env_id}/{SYNTHETIC_GAUSSIAN_SCENE_REL_PATH}/Cameras/{SYNTHETIC_GAUSSIAN_CAMERA_NAME}"


def _set_ppisp_camera_attrs(
    stage: Usd.Stage,
    inputs: dict[str, float | tuple[float, float]],
    *,
    controller_weights: tuple[float, ...] | None = None,
) -> None:
    camera_prim = stage.GetPrimAtPath(_camera_path_for_env(0))
    if not camera_prim or not camera_prim.IsValid():
        raise RuntimeError(f"Synthetic PPISP camera prim not found: {_camera_path_for_env(0)}")
    for name, value in inputs.items():
        if isinstance(value, tuple):
            camera_prim.CreateAttribute(f"ppisp:{name}", Sdf.ValueTypeNames.Float2).Set(
                Gf.Vec2f(float(value[0]), float(value[1]))
            )
        else:
            camera_prim.CreateAttribute(f"ppisp:{name}", Sdf.ValueTypeNames.Float).Set(float(value))
    if controller_weights is not None:
        camera_prim.CreateAttribute("ppisp:controllerWeights", Sdf.ValueTypeNames.FloatArray).Set(
            Vt.FloatArray(controller_weights)
        )


def author_static_ppisp_camera_attrs(stage: Usd.Stage, *, ppisp_cfg: PpispCfg) -> None:
    """Author static PPISP camera attributes on the synthetic camera."""
    _set_ppisp_camera_attrs(stage, ppisp_cfg.inputs)


def author_controller_ppisp_camera_attrs(stage: Usd.Stage, *, ppisp_cfg: PpispCfg) -> None:
    """Author PPISP camera attributes plus deterministic controller weights."""
    _set_ppisp_camera_attrs(
        stage,
        ppisp_cfg.inputs,
        controller_weights=_deterministic_controller_weights(ppisp_cfg),
    )


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


def assert_tiled_views_match(
    tiled_images: torch.Tensor,
    *,
    max_mean_abs_diff: float = 12.75,
    max_relative_mean_abs_diff: float | None = None,
    label: str = "",
) -> None:
    """Assert that all views in a tiled image batch contain the same content.

    Args:
        tiled_images: Image batch with shape ``[num_tiles, H, W, C]``.
        max_mean_abs_diff: Maximum allowed mean absolute difference from tile zero.
            For LDR images scaled to ``[0, 255]``, the default of ``12.75``
            (``0.05 * 255``) matches the existing tiled-camera consistency
            tolerance. This absolute tolerance should not be used for HDR
            images, whose raw physical float scale is renderer-dependent.
        max_relative_mean_abs_diff: Maximum allowed mean absolute difference
            from tile zero relative to tile zero's mean absolute value. If set,
            this relative threshold is used instead of :paramref:`max_mean_abs_diff`.
            Use this for ``rgb_hdr`` tiles.
        label: Included in the assertion message.
    """
    prefix = f"[{label}] " if label else ""
    assert tiled_images.ndim == 4 and tiled_images.shape[0] > 1, (
        f"{prefix}expected a multi-tile image batch, got shape={tuple(tiled_images.shape)}"
    )

    reference = tiled_images[0][..., :3].float()
    reference_mean_abs = reference.abs().mean().item()
    for tile_index in range(1, tiled_images.shape[0]):
        mean_abs_diff = (reference - tiled_images[tile_index][..., :3].float()).abs().mean().item()
        if max_relative_mean_abs_diff is not None:
            relative_mean_abs_diff = mean_abs_diff / max(reference_mean_abs, 1.0e-6)
            assert relative_mean_abs_diff < max_relative_mean_abs_diff, (
                f"{prefix}tile {tile_index} differs from tile 0: "
                f"relative_mean_abs_diff={relative_mean_abs_diff:.4f}, "
                f"mean_abs_diff={mean_abs_diff:.4f}, reference_mean_abs={reference_mean_abs:.4f}, "
                f"expected relative_mean_abs_diff < {max_relative_mean_abs_diff}"
            )
            continue
        assert mean_abs_diff < max_mean_abs_diff, (
            f"{prefix}tile {tile_index} differs from tile 0: mean_abs_diff={mean_abs_diff:.3f}, "
            f"expected < {max_mean_abs_diff}"
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
    f"/World/envs/env_[^/]+/{SYNTHETIC_GAUSSIAN_SCENE_REL_PATH}/Cameras/{SYNTHETIC_GAUSSIAN_CAMERA_NAME}"
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

    # Comfortably wider than the ~2.6 m the default camera frames at its 3 m standoff, so
    # each env's camera sees only its own gaussians. At the scene's own ~1.8 m width the
    # neighbouring copies fall inside the frame and tiles stop being comparable.
    env_spacing: float = 10.0

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        # Keep the background in the calibrated HDR range independently of the default plane's appearance.
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
    )

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
    """Render the synthesised gaussian asset with the aggressive wrapper PPISP.

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
        # Camera is constructed after the scene's ReplicateSession has exited, so its
        # queued USD replication needs an explicit drain (Path B). Reuse the scene's
        # env positions so env_origins stays consistent.
        published = sim.get_clone_plan()
        positions = published.positions if published is not None else None
        src, dst = "/World/envs/env_0", "/World/envs/env_{}"
        camera_plan = cloner.clone_plan_from_env_0(src, dst, num_envs, str(sim.device), positions)
        cloner.replicate(camera_plan, stage=sim.stage)
        sim.reset()
        for _ in range(stabilisation_steps):
            sim.step()
        camera.update(sim_dt)
        outputs = {
            name: tensor.torch.clone().detach().cpu().to(torch.float32) for name, tensor in camera.data.output.items()
        }
        del camera
        return outputs


def render_synthetic_gaussian_scene_with_static_ppisp_attrs(
    usd_path: str,
    *,
    sim_cfg: SimulationCfg,
    renderer_cfg: RendererCfg,
    ppisp_cfg: PpispCfg,
    data_types: list[str],
    num_envs: int = 1,
    height: int = 128,
    width: int = 128,
    sim_dt: float = 1.0 / 60.0,
    stabilisation_steps: int = 5,
) -> dict[str, torch.Tensor]:
    """Render the synthesised gaussian asset through authored static PPISP camera attributes.

    The camera uses :class:`CameraISPMode.AUTO_CAMERA`; renderer backends must
    discover the camera-authored PPISP attributes and route them through their
    PPISP workflow.
    """
    with fresh_synthetic_gaussian_interactive_scene(usd_path, sim_cfg, num_envs=num_envs) as sim:
        author_static_ppisp_camera_attrs(sim.stage, ppisp_cfg=ppisp_cfg)
        return _render_synthetic_gaussian_camera(
            renderer_cfg=renderer_cfg,
            data_types=data_types,
            height=height,
            width=width,
            sim_dt=sim_dt,
            stabilisation_steps=stabilisation_steps,
            isp_cfg=CameraISPMode.AUTO_CAMERA,
            sim=sim,
        )


def render_synthetic_gaussian_scene_with_controller_ppisp_attrs(
    usd_path: str,
    *,
    sim_cfg: SimulationCfg,
    renderer_cfg: RendererCfg,
    ppisp_cfg: PpispCfg,
    data_types: list[str],
    num_envs: int = 1,
    height: int = 128,
    width: int = 128,
    sim_dt: float = 1.0 / 60.0,
    stabilisation_steps: int = 5,
) -> dict[str, torch.Tensor]:
    """Render the synthesised gaussian asset through camera-authored controller weights."""
    with fresh_synthetic_gaussian_interactive_scene(usd_path, sim_cfg, num_envs=num_envs) as sim:
        author_controller_ppisp_camera_attrs(sim.stage, ppisp_cfg=ppisp_cfg)
        return _render_synthetic_gaussian_camera(
            renderer_cfg=renderer_cfg,
            data_types=data_types,
            height=height,
            width=width,
            sim_dt=sim_dt,
            stabilisation_steps=stabilisation_steps,
            isp_cfg=CameraISPMode.AUTO_CAMERA,
            sim=sim,
        )


def _render_synthetic_gaussian_camera(
    *,
    renderer_cfg: RendererCfg,
    data_types: list[str],
    height: int,
    width: int,
    sim_dt: float,
    stabilisation_steps: int,
    isp_cfg: PpispCfg | CameraISPMode | None,
    sim: SimulationContext,
) -> dict[str, torch.Tensor]:
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
    outputs = {
        name: tensor.torch.clone().detach().cpu().to(torch.float32) for name, tensor in camera.data.output.items()
    }
    del camera
    return outputs
