# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rebuild raspberry SH from the raw scan and repair the omitted settling rotation.

Authoring only: the deployed task does not require the POC or its NPZ files.
Preserves the existing merged Gaussian geometry, interiors, and physics exactly.
"""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from isaaclab_tasks.contrib.franka_pick_berries.assets.sh_rotation import proper_rotation, rotate_sh, sh_basis
from isaaclab_tasks.contrib.franka_pick_berries.physics.mpm.binding import make_binding


def read_ply(path: Path) -> np.ndarray:
    """Read the original float-only binary Gaussian PLY as a read-only map."""
    names = []
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply" or stream.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError("Expected binary little-endian Gaussian PLY")
        while True:
            line = stream.readline().decode("ascii").strip()
            if line.startswith("element vertex "):
                count = int(line.split()[-1])
            elif line.startswith("property "):
                _, kind, name = line.split()
                if kind != "float":
                    raise ValueError(f"Unsupported PLY property: {line}")
                names.append((name, "<f4"))
            elif line == "end_header":
                offset = stream.tell()
                break
            elif not line:
                raise ValueError("Incomplete PLY header")
    return np.memmap(path, dtype=np.dtype(names), mode="r", offset=offset, shape=(count,))


def rebuild_raspberry(poc: Path) -> tuple[np.ndarray, dict]:
    """Return corrected SH and diagnostics without modifying source files."""
    root = poc / "assets/processed"
    original = dict(np.load(root / "raspberry-explicit-fine-sharp/splats.npz"))
    proxy = dict(np.load(root / "raspberry-explicit-fine-sharp/proxy.npz"))
    settled = dict(np.load(root / "raspberry-squash/splats.npz"))
    settled_proxy = dict(np.load(root / "raspberry-squash/proxy.npz"))
    meta = json.loads((root / "raspberry-squash/profile.json").read_text())
    raw = read_ply(poc / "assets/raw/raspberry/scene.ply")

    def columns(names):
        return np.column_stack([raw[n] for n in names])

    xyz = ((columns(("x", "y", "z")) - np.array(meta["source_origin"])) * meta["metric_scale"]).astype(np.float32)
    xyz[:, 2] += 0.0007
    alpha = 1 / (1 + np.exp(-np.clip(raw["opacity"], -30, 30)))
    scales = np.exp(np.clip(columns([f"scale_{i}" for i in range(3)]), -30, 20))
    scales *= meta["metric_scale"]
    keep = (
        (alpha > 0.025)
        & (xyz[:, 2] > -0.001)
        & (xyz[:, 2] < 0.030)
        & (np.abs(xyz[:, :2]).max(axis=1) < 0.013)
        & (scales.max(axis=1) <= meta["max_source_sigma_m"])
    )
    ids = np.flatnonzero(keep)
    turn = Rotation.from_euler("x", 90, degrees=True)
    positions = (turn.apply(xyz[ids]) + np.array(meta["final_translation_m"])).astype(np.float32)
    cells = np.floor(positions / meta["merge_voxel_m"]).astype(np.int32)
    _, groups = np.unique(cells, axis=0, return_inverse=True)
    _, first = np.unique(groups, return_index=True)
    # Reproduce optical-footprint weighted merging, not the representative PLY
    # sample alone: source_ids identify cells but do not encode their average.
    s = scales[ids]
    weights = (
        -np.log1p(-np.minimum(alpha[ids], 0.999)) * (s[:, 0] * s[:, 1] + s[:, 1] * s[:, 2] + s[:, 2] * s[:, 0]) / 3
    )
    totals = np.bincount(groups, weights=weights)
    merged = np.empty((len(first), 16, 3), np.float32)
    for channel in range(3):
        for term in range(16):
            field = f"f_dc_{channel}" if term == 0 else f"f_rest_{channel * 15 + term - 1}"
            merged[:, term, channel] = np.bincount(groups, weights=weights * raw[field][ids]) / totals
    lookup = {int(source): i for i, source in enumerate(ids[first])}
    exterior = original["source_ids"] >= 0
    selected = np.array([lookup[int(i)] for i in original["source_ids"][exterior]])
    reconstructed = original["sh"].copy()
    reconstructed[exterior] = rotate_sh(merged[selected], turn.as_matrix())
    before_error = float(np.max(np.abs(reconstructed - original["sh"])))
    if before_error > 2.0e-5:
        raise ValueError(f"Raw-scan preparation recipe no longer matches input: SH error {before_error}")
    # Recover the precise per-Gaussian deformation used by prepare_squash;
    # rigid recentering has no effect on the material rotation.
    binding = make_binding(original, proxy["xyz"], proxy["regions"], "mls32-coherent")
    positions = settled_proxy["xyz"] - np.asarray(meta["pose_translation_m"], np.float32)
    xyz, factors = binding.factors_gpu(positions)
    np.testing.assert_allclose(xyz + np.asarray(meta["pose_translation_m"], np.float32), settled["xyz"], atol=2.0e-7)
    rotation = proper_rotation(factors.astype(float) @ np.linalg.inv(binding.base.astype(float)))
    corrected = rotate_sh(reconstructed, rotation)
    # Hold-out directions, distinct from projection quadrature.
    indices = np.arange(0, len(corrected), 151)
    directions = np.random.default_rng(83).normal(size=(17, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    expected = np.einsum(
        "ndk,nkc->ndc", sh_basis(np.einsum("dj,njk->ndk", directions, rotation[indices])), reconstructed[indices]
    )
    actual = np.einsum("dk,nkc->ndc", sh_basis(directions), corrected[indices])
    error = float(np.max(np.abs(expected - actual)))
    if error > 2.0e-6:
        raise ValueError(f"SH rotation failed independent-direction check: {error}")
    angles = np.degrees(Rotation.from_matrix(rotation).magnitude())
    return corrected, {
        "source": "raw scene.ply; original optical-footprint merging; Rx90; settled material polar rotation",
        "raw_reconstruction_max_error": before_error,
        "transport_max_error": error,
        "material_rotation_degrees_p50_p90_p99": np.percentile(angles, [50, 90, 99]).tolist(),
        "changed_sh_max": float(np.max(np.abs(corrected - settled["sh"]))),
        "geometry_and_physics_unchanged": True,
    }
