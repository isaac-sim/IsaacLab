#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build self-contained ManipulationNet board and F1 assets from the upstream STLs.

The upstream binary STLs are visual sources only. This authoring step normalizes
their centimeter-space pivots into a meter, Z-up board frame and adds explicit
primitive collision proxies. It intentionally does not add collision APIs to the
triangle meshes, so Newton never cooks or imports the dense board geometry.

Run from an Isaac Lab source checkout::

    uv run python scripts/tools/generate_manipulationnet_cable_assets.py \
        /path/to/mnet_client/assets/cable_management/cad_files
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from pxr import Gf, Kind, Usd, UsdGeom, UsdPhysics, Vt

SOURCE_REPOSITORY = "https://github.com/ManipulationNet/mnet_client"
SOURCE_BRANCH = "ros_2"
SOURCE_COMMIT = "2745ccc6099fb3b65e89cbdbaf7af6521bf8dd29"
SOURCE_UNITS_TO_METERS = 0.01

BOARD_ACTIVE_SIZE = (0.30, 0.40)
BOARD_THICKNESS = 0.00635
BOARD_COLOR = (0.32, 0.36, 0.42)
PEG_COLOR = (0.15, 0.35, 0.85)

PEG_HEIGHT = 0.0235
PEG_FLANGE_RADIUS = 0.0125
PEG_SHAFT_RADIUS = 0.00475


@dataclass(frozen=True)
class BoardSegment:
    """One normalized quadrant of the assembled 400 x 300 mm active board."""

    file_name: str
    prim_name: str
    expected_sha256: str
    row_center_x: float
    column_center_y: float
    has_front_label_strip: bool


BOARD_SEGMENTS = (
    BoardSegment(
        "board_segment_upper_left.stl",
        "UpperLeft",
        "fa90f8e015401c743b9dd967166023e66c14b8883d9808e0675a915072a9442f",
        0.075,
        0.10,
        False,
    ),
    BoardSegment(
        "board_segment_upper_right.stl",
        "UpperRight",
        "fa90f8e015401c743b9dd967166023e66c14b8883d9808e0675a915072a9442f",
        0.075,
        -0.10,
        False,
    ),
    BoardSegment(
        "board_segment_bottom_left.stl",
        "LowerLeft",
        "1256f953cd5a9e18000f107310b265ed63b6a984252413c4be5a427f9a097585",
        -0.075,
        0.10,
        True,
    ),
    BoardSegment(
        "board_segment_bottom_right.stl",
        "LowerRight",
        "6de8c5362d04f6a99a15b00f7655c5d706112103e9a5c8546f0e5306253be62c",
        -0.075,
        -0.10,
        True,
    ),
)

PEG_FILE_NAME = "round_peg.stl"
PEG_SHA256 = "29d8169aaf13374e7f3ebcbba5f85ef95592408498315686483a9c62b87230e7"


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one source or generated asset."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_mesh(path: Path, expected_sha256: str) -> trimesh.Trimesh:
    """Validate and load one pinned upstream STL without retaining its CAD pivot."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing ManipulationNet CAD source: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Unexpected SHA-256 for {path.name}: expected {expected_sha256}, got {actual_sha256}. "
            f"Use mnet_client branch {SOURCE_BRANCH} at {SOURCE_COMMIT}."
        )

    loaded = trimesh.load_mesh(path, process=True)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_mesh()
    if not isinstance(loaded, trimesh.Trimesh) or loaded.faces.shape[1] != 3:
        raise ValueError(f"Expected one triangular STL mesh at {path}.")
    return loaded


def _new_stage(path: Path, root_path: str) -> tuple[Usd.Stage, Usd.Prim]:
    """Create a meter, Z-up component stage with one kinematic rigid root."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(60.0)

    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root)
    Usd.ModelAPI(root).SetKind(Kind.Tokens.component)
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_api.CreateRigidBodyEnabledAttr(True)
    rigid_api.CreateKinematicEnabledAttr(True)
    return stage, root


def _set_provenance(stage: Usd.Stage, source_hashes: dict[str, str]) -> None:
    """Record enough source metadata to reproduce and audit the conversion."""
    stage.GetRootLayer().customLayerData = {
        "generator": "scripts/tools/generate_manipulationnet_cable_assets.py",
        "sourceBranch": SOURCE_BRANCH,
        "sourceCommit": SOURCE_COMMIT,
        "sourceHashes": source_hashes,
        "sourceRepository": SOURCE_REPOSITORY,
        "sourceUnitsToMeters": SOURCE_UNITS_TO_METERS,
    }


def _author_visual_mesh(
    stage: Usd.Stage,
    prim_path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: tuple[float, float, float],
    source_file: str,
    source_sha256: str,
) -> UsdGeom.Mesh:
    """Author one render mesh with no physics or mesh-collision schema."""
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.int32)
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    mesh.CreateExtentAttr(
        [
            Gf.Vec3f(*vertices.min(axis=0).tolist()),
            Gf.Vec3f(*vertices.max(axis=0).tolist()),
        ]
    )
    mesh.GetPrim().SetCustomDataByKey("sourceFile", source_file)
    mesh.GetPrim().SetCustomDataByKey("sourceSha256", source_sha256)
    return mesh


def _author_collision_cube(stage: Usd.Stage, prim_path: str, size: tuple[float, float, float]) -> None:
    """Author one invisible box collider with exact world-meter dimensions."""
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3d(*size))
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    collision_api = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    collision_api.CreateCollisionEnabledAttr(True)


def _author_collision_cylinder(
    stage: Usd.Stage,
    prim_path: str,
    radius: float,
    height: float,
    center_z: float,
) -> None:
    """Author one invisible Z-axis cylinder collider."""
    cylinder = UsdGeom.Cylinder.Define(stage, prim_path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, center_z))
    cylinder.CreatePurposeAttr(UsdGeom.Tokens.guide)
    collision_api = UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    collision_api.CreateCollisionEnabledAttr(True)


def _normalized_board_vertices(mesh: trimesh.Trimesh, segment: BoardSegment) -> np.ndarray:
    """Map an arbitrary CAD pivot into the simulator's centered board frame."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bounds_min, bounds_max = np.asarray(mesh.bounds, dtype=np.float64)
    source_center_x = 0.5 * (bounds_min[0] + bounds_max[0])

    # The bottom pieces contain a 30 mm branding strip in front of their
    # 150 mm active panel. Their active region therefore ends at source Y max
    # and has its center 75 mm below it. The upper pieces are exactly 150 mm.
    if segment.has_front_label_strip:
        source_active_center_y = bounds_max[1] - 7.5
    else:
        source_active_center_y = 0.5 * (bounds_min[1] + bounds_max[1])

    transformed = np.empty_like(vertices)
    transformed[:, 0] = (vertices[:, 1] - source_active_center_y) * SOURCE_UNITS_TO_METERS + segment.row_center_x
    transformed[:, 1] = -(vertices[:, 0] - source_center_x) * SOURCE_UNITS_TO_METERS + segment.column_center_y
    transformed[:, 2] = (vertices[:, 2] - bounds_min[2]) * SOURCE_UNITS_TO_METERS - 0.5 * BOARD_THICKNESS
    return transformed


def _build_board(cad_dir: Path, output_path: Path) -> None:
    """Build the four-piece board visual and its single active-workspace collider."""
    stage, _ = _new_stage(output_path, "/MNetBoard")
    UsdGeom.Scope.Define(stage, "/MNetBoard/Visuals")
    UsdGeom.Scope.Define(stage, "/MNetBoard/Collisions")

    source_hashes: dict[str, str] = {}
    for segment in BOARD_SEGMENTS:
        source_path = cad_dir / segment.file_name
        source_mesh = _load_source_mesh(source_path, segment.expected_sha256)
        source_hashes[segment.file_name] = segment.expected_sha256
        _author_visual_mesh(
            stage,
            f"/MNetBoard/Visuals/{segment.prim_name}",
            _normalized_board_vertices(source_mesh, segment),
            np.asarray(source_mesh.faces),
            BOARD_COLOR,
            segment.file_name,
            segment.expected_sha256,
        )

    # The dense mesh retains holes, seams, and front branding. Physics uses one
    # stable box for the documented 400 x 300 mm active workspace only.
    _author_collision_cube(
        stage,
        "/MNetBoard/Collisions/ActiveWorkspace",
        (BOARD_ACTIVE_SIZE[0], BOARD_ACTIVE_SIZE[1], BOARD_THICKNESS),
    )
    _set_provenance(stage, source_hashes)
    stage.GetRootLayer().Save()


def _build_round_peg(cad_dir: Path, output_path: Path) -> None:
    """Build the F1 spool visual and a three-cylinder compound collider."""
    source_path = cad_dir / PEG_FILE_NAME
    source_mesh = _load_source_mesh(source_path, PEG_SHA256)
    stage, _ = _new_stage(output_path, "/RoundPeg")
    UsdGeom.Scope.Define(stage, "/RoundPeg/Visuals")
    UsdGeom.Scope.Define(stage, "/RoundPeg/Collisions")

    vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    bounds_min, bounds_max = np.asarray(source_mesh.bounds, dtype=np.float64)
    center_xy = 0.5 * (bounds_min[:2] + bounds_max[:2])
    transformed = np.empty_like(vertices)
    transformed[:, :2] = (vertices[:, :2] - center_xy) * SOURCE_UNITS_TO_METERS
    transformed[:, 2] = (vertices[:, 2] - bounds_min[2]) * SOURCE_UNITS_TO_METERS - 0.5 * PEG_HEIGHT
    _author_visual_mesh(
        stage,
        "/RoundPeg/Visuals/F1",
        transformed,
        np.asarray(source_mesh.faces),
        PEG_COLOR,
        PEG_FILE_NAME,
        PEG_SHA256,
    )

    # These three primitives preserve the cable-bearing 9.5 mm waist and both
    # 25 mm retention flanges without importing the 2,592-triangle STL into
    # either MJWarp or VBD collision detection.
    _author_collision_cylinder(
        stage,
        "/RoundPeg/Collisions/LowerFlange",
        PEG_FLANGE_RADIUS,
        0.0025,
        -0.5 * PEG_HEIGHT + 0.00125,
    )
    _author_collision_cylinder(
        stage,
        "/RoundPeg/Collisions/Shaft",
        PEG_SHAFT_RADIUS,
        0.0200,
        -0.5 * PEG_HEIGHT + 0.0125,
    )
    _author_collision_cylinder(
        stage,
        "/RoundPeg/Collisions/UpperFlange",
        PEG_FLANGE_RADIUS,
        0.0010,
        0.5 * PEG_HEIGHT - 0.0005,
    )
    _set_provenance(stage, {PEG_FILE_NAME: PEG_SHA256})
    stage.GetRootLayer().Save()


def _build_assets(cad_dir: Path, output_dir: Path) -> None:
    """Generate both assets atomically so a failed conversion leaves no partial files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mnet-cable-assets-", dir=output_dir) as temporary_dir:
        temporary_path = Path(temporary_dir)
        board_path = temporary_path / "board.usdc"
        peg_path = temporary_path / "round_peg.usdc"
        _build_board(cad_dir, board_path)
        _build_round_peg(cad_dir, peg_path)
        os.replace(board_path, output_dir / board_path.name)
        os.replace(peg_path, output_dir / peg_path.name)

    for name in ("board.usdc", "round_peg.usdc"):
        path = output_dir / name
        print(f"{path}: {_sha256(path)}")


def main() -> None:
    """Parse command-line arguments and build the pinned USD package."""
    default_output = (
        Path(__file__).resolve().parents[2]
        / "source"
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "contrib"
        / "cable_routing"
        / "assets"
        / "manipulationnet"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cad_dir", type=Path, help="Directory containing the upstream cable-management STLs.")
    parser.add_argument(
        "--output-dir", type=Path, default=default_output, help="Destination for board.usdc and round_peg.usdc."
    )
    args = parser.parse_args()
    _build_assets(args.cad_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
