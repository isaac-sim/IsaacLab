"""Turn the YCB 012_strawberry scan into a USD mesh suitable for a Newton volume deformable.

The raw scan is a 16k-triangle shell that is not guaranteed watertight, and fTetWild
needs a closed surface. This voxel-remeshes it to a watertight manifold, decimates to
something a tetrahedraliser can chew on quickly, recentres it on the origin and drops
it on z=0, then writes a .usda containing a single UsdGeom.Mesh.

Run with the project venv python (it has trimesh + usd-core). Decimation additionally needs
``pip install fast-simplification``; without it the script keeps full resolution and
tetrahedralisation gets slow.

The source scan is the YCB 012_strawberry laser scan and is not redistributed here:

.. code-block:: bash

    curl -L -o 012_strawberry_google_16k.tgz \
        https://ycb-benchmarks.s3.amazonaws.com/data/google/012_strawberry_google_16k.tgz
    tar xzf 012_strawberry_google_16k.tgz

The generated ``.usda`` is deliberately not committed -- IsaacLab's .gitignore excludes
``**/*.usda`` -- so regenerate it with this script after fetching the scan.
"""

import os

import numpy as np
import trimesh

SRC = r"C:\isaac_soft\assets\012_strawberry\google_16k\nontextured.ply"
OUT = r"C:\isaac_soft\assets\strawberry_deformable.usda"
TARGET_FACES = 1200


def main():
    mesh = trimesh.load(SRC, force="mesh")
    print(f"loaded      : {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    print(f"watertight  : {mesh.is_watertight}, volume={mesh.volume:.3e} m^3")
    print(f"extents [m] : {np.round(mesh.extents, 4)}")

    # voxel remesh -> guaranteed closed manifold for the tetrahedraliser
    if not mesh.is_watertight:
        pitch = float(max(mesh.extents)) / 48.0
        print(f"remeshing (not watertight), pitch={pitch:.5f}")
        vox = mesh.voxelized(pitch=pitch).fill()
        mesh = vox.marching_cubes
        mesh.merge_vertices()
        mesh.fix_normals()
        print(f"remeshed    : {len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight}")

    # decimate: tetrahedralisation cost scales badly with surface triangle count
    if len(mesh.faces) > TARGET_FACES:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=TARGET_FACES)
            mesh.merge_vertices()
            mesh.fix_normals()
            print(f"decimated   : {len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight}")
        except Exception as exc:
            print(f"decimation unavailable ({type(exc).__name__}), keeping full resolution")

    # light smoothing to take the staircase off the voxel remesh
    try:
        trimesh.smoothing.filter_taubin(mesh, iterations=8)
        mesh.fix_normals()
    except Exception as exc:
        print(f"smoothing skipped: {type(exc).__name__}")

    # recentre in xy, rest on z=0
    mesh.vertices -= mesh.bounding_box.centroid
    mesh.vertices[:, 2] -= mesh.vertices[:, 2].min()

    ext = mesh.extents
    print(f"final       : {len(mesh.vertices)} verts, {len(mesh.faces)} faces, watertight={mesh.is_watertight}")
    print(f"final extents [m]: {np.round(ext, 4)}  height={ext[2]:.4f}  width_y={ext[1]:.4f}")

    from pxr import Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(OUT) if not os.path.exists(OUT) else Usd.Stage.Open(OUT)
    if stage.GetPrimAtPath("/Strawberry"):
        stage.RemovePrim("/Strawberry")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    xform = UsdGeom.Xform.Define(stage, "/Strawberry")
    stage.SetDefaultPrim(xform.GetPrim())
    geom = UsdGeom.Mesh.Define(stage, "/Strawberry/mesh")
    geom.CreatePointsAttr(Vt.Vec3fArray([tuple(map(float, v)) for v in mesh.vertices]))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in mesh.faces.reshape(-1)]))
    geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    geom.CreateDisplayColorAttr(Vt.Vec3fArray([(0.78, 0.05, 0.12)]))
    geom.CreateExtentAttr(
        Vt.Vec3fArray(
            [
                tuple(map(float, mesh.vertices.min(axis=0))),
                tuple(map(float, mesh.vertices.max(axis=0))),
            ]
        )
    )
    stage.GetRootLayer().Save()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
