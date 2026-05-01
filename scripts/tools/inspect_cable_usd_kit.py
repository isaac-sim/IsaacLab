# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quick USD inspector that runs through IsaacLab's app launcher (so pxr is importable).

Usage:
    ./isaaclab.sh -p scripts/tools/inspect_cable_usd_kit.py path/to/asset.usd [more.usd...]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("usd_paths", nargs="+", help="USD files to inspect.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

print("[INSPECT] AppLauncher up, importing pxr...", flush=True)

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

print("[INSPECT] pxr imported", flush=True)
print(f"[INSPECT] usd_paths={args.usd_paths!r}", flush=True)


def walk(prim, depth=0, max_depth=12):
    if depth > max_depth:
        return
    indent = "  " * depth
    name = prim.GetName() or "/"
    typename = prim.GetTypeName()

    extras = []
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        extras.append("RigidBodyAPI")
        rb = UsdPhysics.RigidBodyAPI(prim)
        if rb.GetKinematicEnabledAttr().IsAuthored():
            extras.append(f"kinematic={rb.GetKinematicEnabledAttr().Get()}")
    if prim.HasAPI(UsdPhysics.MassAPI):
        extras.append("MassAPI")
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        extras.append("CollisionAPI")

    if prim.IsA(UsdGeom.Xformable):
        x = UsdGeom.Xformable(prim)
        ops = x.GetOrderedXformOps()
        for op in ops:
            try:
                val = op.Get()
            except Exception:
                val = "?"
            extras.append(f"{op.GetOpName()}={val}")

    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if pts is not None and len(pts) > 0:
            min_p = Gf.Vec3d(float("inf"), float("inf"), float("inf"))
            max_p = Gf.Vec3d(float("-inf"), float("-inf"), float("-inf"))
            for p in pts:
                for i in range(3):
                    min_p[i] = min(min_p[i], p[i])
                    max_p[i] = max(max_p[i], p[i])
            center = ((min_p[0] + max_p[0]) / 2, (min_p[1] + max_p[1]) / 2, (min_p[2] + max_p[2]) / 2)
            size = (max_p[0] - min_p[0], max_p[1] - min_p[1], max_p[2] - min_p[2])
            extras.append(f"bbox_center=({center[0]:.4f},{center[1]:.4f},{center[2]:.4f})")
            extras.append(f"bbox_size=({size[0]:.4f},{size[1]:.4f},{size[2]:.4f})")

    extras_str = ("  " + " | ".join(extras)) if extras else ""
    print(f"{indent}- {name} <{typename}>{extras_str}")

    for child in prim.GetChildren():
        walk(child, depth + 1, max_depth)


for asset in args.usd_paths:
    try:
        print("=" * 100, flush=True)
        print(f"INSPECTING: {asset}", flush=True)
        print("=" * 100, flush=True)
        stage = Usd.Stage.Open(asset)
        if stage is None:
            print("  Failed to open stage", flush=True)
            continue
        default_prim = stage.GetDefaultPrim()
        print(f"  Default prim: {default_prim.GetPath() if default_prim else '<none>'}", flush=True)
        print(f"  Up axis: {UsdGeom.GetStageUpAxis(stage)}", flush=True)
        print(f"  Meters per unit: {UsdGeom.GetStageMetersPerUnit(stage)}", flush=True)
        print(flush=True)
        walk(stage.GetPseudoRoot(), 0)
        print(flush=True)
    except Exception as exc:
        print(f"[INSPECT] EXCEPTION while inspecting {asset}: {exc!r}", flush=True)
        import traceback

        traceback.print_exc()

simulation_app.close()
