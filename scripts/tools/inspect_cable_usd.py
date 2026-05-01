"""Quick inspector to find any baked-in xform offsets in plug/socket USDs."""

import sys

from pxr import Usd, UsdGeom, UsdPhysics

"""Set ``ASSETS`` from the command line, or edit this list to inspect other USDs."""
ASSETS = sys.argv[1:] or [
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/cable_insertion/cable_insertion_assets/plug_A_no_snapfit_latch_transformed.usd",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/deploy/cable_insertion/cable_insertion_assets/socket_A_simplified_minimal_transformed.usd",
]


def inspect_xform_ops(prim, depth=0, max_depth=8, rb_world_to_local=None):
    """Recurse and print xform ops, RigidBody/Mass/Collision markers, and Mesh bbox in body-local frame."""
    if depth > max_depth:
        return
    indent = "  " * depth
    name = prim.GetName() or "/"
    typename = prim.GetTypeName()

    has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    has_mass = prim.HasAPI(UsdPhysics.MassAPI)
    has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)

    msg_extras = []
    if has_rb:
        msg_extras.append("RigidBodyAPI")
        rb = UsdPhysics.RigidBodyAPI(prim)
        if rb.GetRigidBodyEnabledAttr().IsAuthored():
            msg_extras.append(f"rb_enabled={rb.GetRigidBodyEnabledAttr().Get()}")
        if rb.GetKinematicEnabledAttr().IsAuthored():
            msg_extras.append(f"kinematic={rb.GetKinematicEnabledAttr().Get()}")
        # disable_gravity is on PhysxRigidBodyAPI, but we can check via authored attrs
        for a in prim.GetAttributes():
            if "disableGravity" in a.GetName() or "gravity" in a.GetName().lower():
                msg_extras.append(f"{a.GetName()}={a.Get()}")
    if has_mass:
        msg_extras.append("MassAPI")
    if has_collision:
        msg_extras.append("CollisionAPI")

    if prim.IsA(UsdGeom.Xformable):
        x = UsdGeom.Xformable(prim)
        ops = x.GetOrderedXformOps()
        op_strs = []
        for op in ops:
            try:
                val = op.Get()
            except Exception:
                val = "?"
            op_strs.append(f"{op.GetOpName()}={val}")
        if op_strs:
            msg_extras.append("ops=[" + ", ".join(op_strs) + "]")

    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if pts is not None and len(pts) > 0:
            from pxr import Gf as _Gf
            min_p = _Gf.Vec3d(float("inf"), float("inf"), float("inf"))
            max_p = _Gf.Vec3d(float("-inf"), float("-inf"), float("-inf"))
            for p in pts:
                for i in range(3):
                    min_p[i] = min(min_p[i], p[i])
                    max_p[i] = max(max_p[i], p[i])
            center = ((min_p[0] + max_p[0]) / 2, (min_p[1] + max_p[1]) / 2, (min_p[2] + max_p[2]) / 2)
            size = (max_p[0] - min_p[0], max_p[1] - min_p[1], max_p[2] - min_p[2])
            msg_extras.append(f"bbox_center_local=({center[0]:.4f},{center[1]:.4f},{center[2]:.4f})")
            msg_extras.append(f"bbox_size=({size[0]:.4f},{size[1]:.4f},{size[2]:.4f})")

    extras = "  ".join(msg_extras)
    print(f"{indent}- {name} <{typename}>{'  ' + extras if extras else ''}")

    for child in prim.GetChildren():
        inspect_xform_ops(child, depth + 1, max_depth)


for asset in ASSETS:
    print("=" * 100)
    print(f"INSPECTING: {asset}")
    print("=" * 100)
    stage = Usd.Stage.Open(asset)
    if stage is None:
        print(f"  Failed to open {asset}")
        continue
    print(f"  Default prim: {stage.GetDefaultPrim().GetPath()}")
    print(f"  Up axis: {UsdGeom.GetStageUpAxis(stage)}")
    print(f"  Meters per unit: {UsdGeom.GetStageMetersPerUnit(stage)}")
    print()
    inspect_xform_ops(stage.GetPseudoRoot(), depth=0, max_depth=10)
    print()

sys.exit(0)
