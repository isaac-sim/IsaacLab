# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Connector asset provenance, validation, and CAD geometry."""

from __future__ import annotations

import hashlib
import math
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from isaaclab_assets.robots import rizon_sharpa

# Preserve the original task imports while callers move to the shared asset definition.
RIZON_SHARPA_ARM_JOINT_NAMES = rizon_sharpa.ARM_JOINT_NAMES
RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES = rizon_sharpa.HAND_JOINT_NAMES
RIZON_SHARPA_END_EFFECTOR_BODY_NAME = rizon_sharpa.PALM_BODY_NAME
RIZON_SHARPA_ASSET_ROOT_ENV = rizon_sharpa.ROBOT_ASSET_ROOT_ENV
RIZON_SHARPA_REVISION = rizon_sharpa.ROBOT_REVISION
RIZON_SHARPA_USD_NAME = rizon_sharpa.ROBOT_USD_NAME
RIZON_SHARPA_USD_PATH = rizon_sharpa.ROBOT_USD_PATH
RIZON_SHARPA_BUNDLE_SHA256 = rizon_sharpa.ROBOT_BUNDLE_SHA256
RIZON_SHARPA_BUNDLE_SIZE_BYTES = rizon_sharpa.ROBOT_BUNDLE_SIZE_BYTES
RIZON_SHARPA_REPOSITORY = "https://gitlab-master.nvidia.com/dex/fabrics-sim.git"
RIZON_SHARPA_RELATIVE_PATH = (
    f"src/fabrics_sim/models/robots/urdf/rizon4s_sharpa/rizon4s_sharpa_no_spheres/{RIZON_SHARPA_USD_NAME}"
)


# The measured socket geometry uses this CAD revision as its SDF cache key.
GB300_SIMREADY_EXTERNAL_USD_SHA256 = "5e0b7b3b58d005b24909b8d2e735c49997f8dbea72352b51911326343ef1e7bb"

RIZON_SHARPA_RIGHT_HAND_LIMITS_RAD = (
    (-0.1745, 1.9199),
    (-0.3491, 0.3491),
    (-0.5236, 1.3963),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (0.0, 0.2618),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
)

RJ45_PLUG_SHA256 = "50c95bcfb63544777f9148d548aac6f16b62f65cacbaaa9316453d579de4b4fa"
RJ45_PLUG_ASSET_ENV = "ISAACLAB_RJ45_PLUG_USD"
RJ45_PLUG_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/newton-physics/newton/"
    "c2ca70bec5998062b0b2d35869865cbe3a49feee/newton/examples/assets/rj45_plug.usd"
)
# CAD-calibrated compression stop [rad], retaining approximately 65 micrometers of housing clearance.
LATCH_LIMIT_LOWER = -0.225
LATCH_LIMIT_UPPER = 0.3
LATCH_SPRING_KE = 0.15


def configured_rizon_sharpa_usd() -> str:
    """Return the hosted robot USD or verify the configured local asset."""
    warnings.warn(
        "configured_rizon_sharpa_usd is deprecated. Use isaaclab_assets.robots.rizon_sharpa.resolve_robot_usd.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rizon_sharpa.resolve_robot_usd()


def verify_rizon_sharpa_bundle(root: str | Path) -> Path:
    """Verify the pinned robot bundle and return its USD path."""
    warnings.warn(
        "verify_rizon_sharpa_bundle is deprecated. Use isaaclab_assets.robots.rizon_sharpa.verify_robot_bundle.",
        DeprecationWarning,
        stacklevel=2,
    )
    return rizon_sharpa.verify_robot_bundle(root)


def configured_rj45_plug_usd() -> Path:
    """Resolve and verify the canonical connector CAD without downloading it."""
    from newton.examples import get_asset

    configured = os.environ.get(RJ45_PLUG_ASSET_ENV)
    candidate = Path(configured).expanduser() if configured else Path(get_asset("rj45_plug.usd"))
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Canonical RJ45 USD is missing: {candidate}. Download {RJ45_PLUG_DOWNLOAD_URL} "
            f"and set {RJ45_PLUG_ASSET_ENV} to the downloaded file."
        )
    with candidate.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != RJ45_PLUG_SHA256:
        raise RuntimeError(f"RJ45 USD SHA-256 mismatch: expected {RJ45_PLUG_SHA256}, got {digest}.")
    return candidate.resolve()


@dataclass(frozen=True)
class ConnectorRenderPart:
    """Immutable mesh in the connector-head frame, with points [m] and authored normals."""

    name: str
    points: tuple[tuple[float, float, float], ...]
    face_vertex_counts: tuple[int, ...]
    face_vertex_indices: tuple[int, ...]
    normals: tuple[tuple[float, float, float], ...] = ()
    normals_interpolation: str = "faceVarying"


@lru_cache(maxsize=4)
def connector_render_parts(connector_rigid_span_m: float) -> tuple[ConnectorRenderPart, ConnectorRenderPart]:
    """Read canonical plug/clip CAD relative to the cable's embedded strain relief [m]."""
    from pxr import Gf, Usd, UsdGeom

    if not math.isfinite(connector_rigid_span_m) or connector_rigid_span_m <= 0:
        raise ValueError("connector_rigid_span_m must be finite and positive.")
    path = configured_rj45_plug_usd()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open the canonical RJ45 USD: {path}")
    curve_prim = stage.GetPrimAtPath("/World/CableCurve")
    if not curve_prim.IsA(UsdGeom.BasisCurves):
        raise ValueError("Canonical RJ45 USD has no /World/CableCurve.")
    curve_transform = UsdGeom.Xformable(curve_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    attachment = curve_transform.Transform(UsdGeom.BasisCurves(curve_prim).GetPointsAttr().Get()[0])
    parts = []
    for name in ("Plug", "Latch"):
        prim = stage.GetPrimAtPath(f"/World/{name}")
        if not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Canonical RJ45 USD has no mesh /World/{name}.")
        mesh = UsdGeom.Mesh(prim)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        points = []
        for source_point in mesh.GetPointsAttr().Get():
            x, y, z = transform.Transform(source_point) - attachment
            # Source cable -Y becomes head +Z. Segment centers are half a span
            # beyond the authored attachment, which lies inside the rear housing.
            points.append((float(x), float(z), float(-y - connector_rigid_span_m / 2)))
        normal_transform = transform.GetInverse().GetTranspose()
        normals = []
        for normal in mesh.GetNormalsAttr().Get() or ():
            x, y, z = normal_transform.TransformDir(Gf.Vec3d(*normal)).GetNormalized()
            normals.append((float(x), float(z), float(-y)))
        parts.append(
            ConnectorRenderPart(
                name.lower(),
                tuple(points),
                tuple(int(value) for value in mesh.GetFaceVertexCountsAttr().Get()),
                tuple(int(value) for value in mesh.GetFaceVertexIndicesAttr().Get()),
                tuple(normals),
                mesh.GetNormalsInterpolation(),
            )
        )
    return parts[0], parts[1]
