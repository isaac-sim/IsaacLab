"""Fix plug/socket USDs so the rigid-body origin coincides with the visible mesh centroid.

Both ``plug_A_no_snapfit_latch_transformed.usd`` and
``socket_A_simplified_minimal_transformed.usd`` were authored with the rigid-body prim
at the asset origin and child Xforms / Mesh prims with large baked-in **translates**
(e.g. ``(0.025, 0.564, 0.053)`` on the plug), plus rotations that orient the geometry
correctly. The convention these USDs follow is "geometry is centered at mesh-local
origin, the xform translates the geometry to its desired place in the source scene".

The translate offsets put the visible/collidable mesh ~56 cm away from the rigid body
that IsaacLab tracks, so the gripper IK reaches the rigid-body origin while the actual
plug sits half a meter elsewhere. This script fixes that by:

1. Walking from each leaf ``Mesh`` up to the rigid-body root, accumulating the chain
   of local transforms into a single 4x4.
2. Splitting that 4x4 into a rotation ``R`` and a translation ``t``.
3. Baking only ``R`` into the mesh's ``points`` (and ``normals`` if authored).
4. Resetting all intermediate ``xformOps`` to identity.

End state: the mesh has identity xform and rotated vertices, so ``v_world = body_world
· R · v_local``. The mesh-local origin (which is the visual centroid by CAD-export
convention) maps to the rigid-body origin in world space, and the original orientation
is preserved.

Run via the helper that injects pxr's runtime libs onto the path:

    USD_LIBS=_isaac_sim/extscache/omni.usd.libs-1.0.1+4a5123f4.lx64.r.cp312
    LD_LIBRARY_PATH="$(pwd)/$USD_LIBS/bin:${LD_LIBRARY_PATH}" \\
    PYTHONPATH="$(pwd)/$USD_LIBS:${PYTHONPATH}" \\
    ./isaaclab.sh -p scripts/tools/flatten_cable_usd.py \\
        -o source/.../cable_insertion_assets/flattened
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

DEFAULT_ASSETS = [
    Path(
        "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/"
        "cable_insertion/cable_insertion_assets/plug_A_no_snapfit_latch_transformed.usd"
    ),
    Path(
        "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/"
        "cable_insertion/cable_insertion_assets/socket_A_simplified_minimal_transformed.usd"
    ),
]


def find_rigid_body_root(stage: Usd.Stage) -> Usd.Prim | None:
    """Return the first prim with ``RigidBodyAPI`` applied, depth-first."""
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    return None


def local_matrix(prim: Usd.Prim) -> Gf.Matrix4d:
    """Return the 4x4 local transform for a prim (identity if not Xformable)."""
    if not prim.IsA(UsdGeom.Xformable):
        return Gf.Matrix4d(1.0)
    return UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())


def reset_xform_ops(prim: Usd.Prim) -> None:
    """Clear all xformOps on a prim (it becomes identity in its parent's frame)."""
    if not prim.IsA(UsdGeom.Xformable):
        return
    UsdGeom.Xformable(prim).ClearXformOpOrder()
    for attr in list(prim.GetAttributes()):
        if attr.GetName().startswith("xformOp:"):
            prim.RemoveProperty(attr.GetName())


def rotation_only_4x4(matrix: Gf.Matrix4d) -> Gf.Matrix4d:
    """Strip the translation row from a 4x4, leaving rotation+scale only."""
    rot_only = Gf.Matrix4d(matrix)
    rot_only.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, 0.0))
    return rot_only


def apply_rotation_and_center(mesh: UsdGeom.Mesh, rot4x4: Gf.Matrix4d) -> Gf.Vec3d:
    """Rotate mesh points by ``rot4x4`` and shift so the AABB centre is at origin.

    Returns:
        The bbox-centre offset (in body-local coords) that was subtracted from each
        vertex, for diagnostic logging.
    """
    points_attr = mesh.GetPointsAttr()
    pts = points_attr.Get()
    if pts is None:
        return Gf.Vec3d(0.0, 0.0, 0.0)

    rotated = [rot4x4.Transform(p) for p in pts]

    if rotated:
        min_p = Gf.Vec3d(rotated[0])
        max_p = Gf.Vec3d(rotated[0])
        for p in rotated[1:]:
            for i in range(3):
                if p[i] < min_p[i]:
                    min_p[i] = p[i]
                if p[i] > max_p[i]:
                    max_p[i] = p[i]
        center = Gf.Vec3d(
            (min_p[0] + max_p[0]) / 2,
            (min_p[1] + max_p[1]) / 2,
            (min_p[2] + max_p[2]) / 2,
        )
    else:
        center = Gf.Vec3d(0.0, 0.0, 0.0)

    centered = [Gf.Vec3f(p[0] - center[0], p[1] - center[1], p[2] - center[2]) for p in rotated]
    points_attr.Set(centered)

    normals_attr = mesh.GetNormalsAttr()
    if normals_attr.HasAuthoredValue():
        rot3x3 = rot4x4.ExtractRotationMatrix()
        normals = normals_attr.Get()
        if normals is not None:
            new_normals = [rot3x3 * n for n in normals]
            normals_attr.Set(new_normals)

    return center


def extra_rotation_4x4(extra_rot_z_deg: float) -> Gf.Matrix4d:
    """Build a rotation-only 4x4 for the requested body-local Z rotation."""
    if extra_rot_z_deg == 0.0:
        return Gf.Matrix4d(1.0)
    rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), extra_rot_z_deg)
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotateOnly(rot)
    return matrix


def fix_descendants(rigid_root: Usd.Prim, extra_rot_z_deg: float = 0.0) -> tuple[int, int]:
    """Bake rotation, drop translation for every chain from ``rigid_root`` to a Mesh.

    Args:
        rigid_root: The prim with ``RigidBodyAPI`` to start the descent from.
        extra_rot_z_deg: Optional additional rotation (degrees) about the body-local Z
            axis applied after bbox-centring. Use this to align a new asset's geometry
            orientation with an existing convention (e.g. when a replacement plug has
            its long axis along X but the rest of the pipeline expects it along Y).

    Returns:
        A ``(num_meshes_processed, num_xforms_reset)`` tuple for reporting.
    """
    meshes_processed = 0
    xforms_to_reset: dict[Sdf.Path, Usd.Prim] = {}

    extra_rot = extra_rotation_4x4(extra_rot_z_deg)

    for prim in Usd.PrimRange(rigid_root):
        if prim == rigid_root:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)

        accumulated = Gf.Matrix4d(1.0)
        cursor = prim
        while cursor != rigid_root and cursor.IsValid():
            accumulated = local_matrix(cursor) * accumulated
            xforms_to_reset[cursor.GetPath()] = cursor
            cursor = cursor.GetParent()

        # Always run bbox-centring even when the chain is identity: some assets
        # ship with the geometry offset baked into the vertex coordinates
        # themselves (no xformOps), and we still want body origin == bbox centre.
        translation = accumulated.ExtractTranslation()
        rotation_only = rotation_only_4x4(accumulated)
        # Compose the asset's own rotation with the optional extra Z rotation; both
        # are applied in body-local space so geometry ends up centred and re-aligned.
        composed = extra_rot * rotation_only
        bbox_offset = apply_rotation_and_center(mesh, composed)
        meshes_processed += 1
        print(f"    mesh={prim.GetPath()}")
        print(f"      dropped translation={tuple(translation)}")
        print(f"      shifted bbox centre by={tuple(bbox_offset)} so geometry is centred at body origin")
        if extra_rot_z_deg != 0.0:
            print(f"      applied extra body-local Z rotation={extra_rot_z_deg} deg")

    for prim in xforms_to_reset.values():
        reset_xform_ops(prim)

    return meshes_processed, len(xforms_to_reset)


def process_asset(usd_path: Path, out_path: Path | None = None, extra_rot_z_deg: float = 0.0) -> None:
    print(f"--- {usd_path}")
    if not usd_path.exists():
        print("  ! file not found, skipping")
        return

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        print("  ! failed to open")
        return

    rigid_root = find_rigid_body_root(stage)
    if rigid_root is None:
        print("  ! no RigidBodyAPI prim found, skipping")
        return
    print(f"  rigid body prim: {rigid_root.GetPath()}")

    meshes, xforms = fix_descendants(rigid_root, extra_rot_z_deg=extra_rot_z_deg)
    print(f"  baked rotation into {meshes} mesh(es); reset {xforms} intermediate prim(s)")

    target = out_path if out_path is not None else usd_path
    if out_path is None:
        backup = usd_path.with_suffix(usd_path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(usd_path, backup)
            print(f"  wrote backup -> {backup}")

    stage.GetRootLayer().Export(str(target))
    print(f"  saved -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="If set, write fixed USDs into this dir instead of overwriting (no backup is made).",
    )
    parser.add_argument(
        "--rotate-z-deg",
        type=float,
        default=0.0,
        help=(
            "Apply this additional body-local Z rotation (degrees) to vertices after "
            "bbox-centring. Use to align a replacement asset's long axis with the "
            "convention of the rest of the pipeline (e.g. 90 deg to swap X and Y)."
        ),
    )
    args = parser.parse_args()

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for usd_path in args.paths:
        out = args.out_dir / usd_path.name if args.out_dir is not None else None
        process_asset(usd_path, out, extra_rot_z_deg=args.rotate_z_deg)


if __name__ == "__main__":
    main()
