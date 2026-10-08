# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read a berry's Gaussians and MPM tissue from its USD/USDZ asset, and turn its Gaussians' shading."""

from pathlib import Path

import numpy as np

from pxr import Usd

from isaaclab.utils.assets import retrieve_file_path


def _read_data(stage: Usd.Stage, path: str):
    """Read nested metadata written as typed attributes and ordered child prims."""
    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise ValueError(f"Missing task metadata: {path}")
    kind = prim.GetAttribute("data:kind").Get()
    if kind in {"dict", "list"}:
        values = [_read_data(stage, f"{path}/item_{i}") for i in range(prim.GetAttribute("data:count").Get())]
        return dict(zip(prim.GetAttribute("data:keys").Get(), values, strict=True)) if kind == "dict" else values
    if kind == "null":
        return None
    if kind != "scalar":
        raise ValueError(f"Unknown task metadata kind at {path}: {kind}")
    return prim.GetAttribute("data:value").Get()


def _read_array(prim: Usd.Prim, name: str) -> np.ndarray:
    """Read a numeric array written with its shape and dtype as sibling attributes."""
    return np.asarray(prim.GetAttribute(name).Get(), dtype=prim.GetAttribute(name + ":dtype").Get()).reshape(
        tuple(prim.GetAttribute(name + ":shape").Get())
    )


def load_berry_asset(path: str | Path) -> tuple[Usd.Stage, dict, dict, dict]:
    """Load a berry's Gaussians, tissue particles and simulation metadata.

    Args:
        path: Nucleus URL or local path; it is resolved and cached locally by
            :func:`isaaclab.utils.assets.retrieve_file_path`.

    Returns:
        The opened stage; the Gaussians at rest (``xyz``, ``scales``, ``rotations``, ``alpha``, ``sh`` and extra arrays
        such as ``regions``); the tissue particles at rest (``xyz`` [m], ``regions``, ``interface``, ``spacing`` and
        ``particle_volume``); and the simulation metadata.
    """
    stage = Usd.Stage.Open(retrieve_file_path(str(path)))
    if not stage or stage.GetDefaultPrim().GetAttribute("berry:schemaVersion").Get() != 2:
        raise ValueError(f"Expected a schema-v2 self-contained berry USD asset: {path}")
    field = stage.GetPrimAtPath("/Berry/Gaussians")
    gaussians = {}
    for key, name in {
        "xyz": "positions",
        "scales": "scales",
        "rotations": "orientations",
        "alpha": "opacities",
        "sh": "radiance:sphericalHarmonicsCoefficients",
    }.items():
        gaussians[key] = np.asarray(field.GetAttribute(name).Get(), dtype=np.float32).copy()
    gaussians["sh"] = gaussians["sh"].reshape(-1, 16, 3)
    for key in field.GetAttribute("berry:arrayKeys").Get():
        gaussians[key] = _read_array(field, "berry:arrays:" + key)
    tissue = stage.GetPrimAtPath("/Berry/Tissue")
    particles = {
        key: _read_array(tissue, "berry:arrays:" + key) for key in tissue.GetAttribute("berry:arrayKeys").Get()
    }
    particles["xyz"] = np.asarray(tissue.GetAttribute("points").Get(), dtype=np.float32).copy()
    return stage, gaussians, particles, _read_data(stage, "/Berry/TaskData/Profile")


def _sh_basis(directions: np.ndarray) -> np.ndarray:
    """Evaluate all 16 real SH basis functions at unit directions, ordered by l,m."""
    x, y, z = np.moveaxis(directions, -1, 0)
    return np.stack(
        [
            x * 0 + 0.28209479177387814,
            -0.4886025119029199 * y,
            0.4886025119029199 * z,
            -0.4886025119029199 * x,
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (2 * z * z - x * x - y * y),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (x * x - y * y),
            -0.5900435899266435 * y * (3 * x * x - y * y),
            2.890611442640554 * x * y * z,
            -0.4570457994644658 * y * (4 * z * z - x * x - y * y),
            0.3731763325901154 * z * (2 * z * z - 3 * x * x - 3 * y * y),
            -0.4570457994644658 * x * (4 * z * z - x * x - y * y),
            1.445305721320277 * z * (x * x - y * y),
            -0.5900435899266435 * x * (x * x - 3 * y * y),
        ],
        axis=-1,
    )


def rotate_berry_sh(coefficients: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate a berry's degree-three spherical harmonics, in the Graphdeco real-SH basis of its MDL shader.

    Applies active rotations: new radiance(d) = old radiance(R.T @ d).

    Rotation is a single 3x3 matrix or one per Gaussian. A 4x8 spherical
    quadrature integrates degree-six products exactly; no fitting or SH loss.
    """
    z, weights = np.polynomial.legendre.leggauss(4)
    azimuth = np.arange(8) * np.pi / 4
    directions = np.stack(
        [
            np.repeat(np.sqrt(1 - z * z), 8) * np.tile(np.cos(azimuth), 4),
            np.repeat(np.sqrt(1 - z * z), 8) * np.tile(np.sin(azimuth), 4),
            np.repeat(z, 8),
        ],
        axis=-1,
    )
    projection = _sh_basis(directions).T * np.repeat(weights * np.pi / 4, 8)
    result = np.empty_like(coefficients)
    for start in range(0, len(coefficients), 2048):
        stop = start + 2048
        r = rotation if rotation.ndim == 2 else rotation[start:stop]
        local = directions @ r if r.ndim == 2 else np.einsum("dj,njk->ndk", directions, r)
        transform = projection @ _sh_basis(local)
        result[start:stop] = transform @ coefficients[start:stop]
    return result
