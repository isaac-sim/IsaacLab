# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""External asset provenance, validation, and immutable connector CAD geometry.

Licensed USD payloads are not redistributed with the task. The robot needs its
complete texture bundle.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

RIZON_SHARPA_REPOSITORY = "https://gitlab-master.nvidia.com/dex/fabrics-sim.git"
RIZON_SHARPA_REVISION = "d0dbd1ddaefc4996db546949a7dfb37e39afcbeb"
RIZON_SHARPA_RELATIVE_PATH = (
    "src/fabrics_sim/models/robots/urdf/rizon4s_sharpa/"
    "rizon4s_sharpa_no_spheres/rizon4s_sharpa_no_spheres_generated.usd"
)
RIZON_SHARPA_USD_NAME = "rizon4s_sharpa_no_spheres_generated.usd"
RIZON_SHARPA_USD_PATH = (
    f"{ISAACLAB_NUCLEUS_DIR}/Environments/RizonSharpaRJ45/Robots/Rizon4sSharpa/{RIZON_SHARPA_USD_NAME}"
)
RIZON_SHARPA_BUNDLE_SHA256 = "ae5d22792b44fb6d29a7691d4276bc061a5529132f01e7a0eb5795a482595d63"
RIZON_SHARPA_BUNDLE_SIZE_BYTES = 379_258_308
RIZON_SHARPA_ASSET_ROOT_ENV = "ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT"

# The measured socket geometry uses this CAD revision as its SDF cache key.
GB300_SIMREADY_EXTERNAL_USD_SHA256 = "5e0b7b3b58d005b24909b8d2e735c49997f8dbea72352b51911326343ef1e7bb"

# Fabrics-Sim authors this fixed child from measured knuckle geometry. Its
# canonical axes are X toward the knuckles and Z out of the palm. The native
# Sharpa palm swaps X/Z and must never be used directly as a pose objective.
RIZON_SHARPA_END_EFFECTOR_BODY_NAME = "r_palm_ctrl"
RIZON_SHARPA_ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
RIZON_SHARPA_RIGHT_HAND_JOINT_NAMES = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)

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

_REQUIRED_PATHS = (
    RIZON_SHARPA_USD_NAME,
    "textures/t_Rizon4s_EmissiveMask.1001.png",
    "textures/t_Rizon4s_EmissiveMask.1002.png",
    "textures/t_Rizon4s_alb.1001.png",
    "textures/t_Rizon4s_alb.1002.png",
    "textures/t_Rizon4s_alb.1003.png",
    "textures/t_Rizon4s_nor.1001.png",
    "textures/t_Rizon4s_nor.1002.png",
    "textures/t_Rizon4s_nor.1003.png",
    "textures/t_Rizon4s_orm.1001.png",
    "textures/t_Rizon4s_orm.1002.png",
    "textures/t_Rizon4s_orm.1003.png",
    "textures/t_Sharpa_alb.png",
    "textures/t_Sharpa_nor.png",
    "textures/t_Sharpa_orm.png",
)


def _bundle_digest(root: Path) -> tuple[str, int]:
    manifest = hashlib.sha256()
    total_size = 0
    for relative_path in _REQUIRED_PATHS:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Pinned Rizon4s Sharpa dependency is missing: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest.update(f"{digest}  {relative_path}\n".encode())
        total_size += path.stat().st_size
    return manifest.hexdigest(), total_size


@lru_cache(maxsize=4)
def verify_rizon_sharpa_bundle(root: str | Path) -> Path:
    """Verify the exact upstream USD bundle and return its USD path."""
    resolved = Path(root).expanduser().resolve()
    if resolved.is_file():
        resolved = resolved.parent
    digest, total_size = _bundle_digest(resolved)
    if digest != RIZON_SHARPA_BUNDLE_SHA256:
        raise RuntimeError(
            f"Rizon4s Sharpa bundle SHA-256 mismatch: expected {RIZON_SHARPA_BUNDLE_SHA256}, got {digest}."
        )
    if total_size != RIZON_SHARPA_BUNDLE_SIZE_BYTES:
        raise RuntimeError(
            f"Rizon4s Sharpa bundle size mismatch: expected {RIZON_SHARPA_BUNDLE_SIZE_BYTES}, got {total_size}."
        )
    return resolved / RIZON_SHARPA_USD_NAME


def configured_rizon_sharpa_usd() -> str:
    """Return the public USD URL or verify an explicitly configured local bundle.

    Returns:
        Public asset URL or verified local USD path.
    """
    configured_root = os.environ.get(RIZON_SHARPA_ASSET_ROOT_ENV)
    if not configured_root:
        return RIZON_SHARPA_USD_PATH
    root = Path(configured_root).expanduser()
    candidate = root.parent if root.suffix.lower() in (".usd", ".usda", ".usdc") else root
    if not (candidate / RIZON_SHARPA_USD_NAME).is_file():
        raise FileNotFoundError(
            "The Rizon Sharpa task needs the pinned Fabrics-Sim bundle. "
            f"Check out {RIZON_SHARPA_REPOSITORY} at {RIZON_SHARPA_REVISION}, copy "
            f"{RIZON_SHARPA_RELATIVE_PATH} and its sibling textures to {candidate}, "
            f"or set {RIZON_SHARPA_ASSET_ROOT_ENV}."
        )
    return str(verify_rizon_sharpa_bundle(candidate))


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
