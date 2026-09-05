# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a ladder of USD override layers that walk the shipped G1 towards ``g1_minimal.usd``.

The two assets behave very differently on rough terrain -- the stock velocity task scores 0.000 on
the shipped 43-joint robot and 1.000 on the superseded 37-joint one -- and a single-variable run
already showed the feet carry most of that. This builds the rest of the ladder so the remaining
differences can be attributed one rung at a time instead of argued about.

Each rung is cumulative: rung *n* carries every change from rungs 1..n, so training the ladder
answers "after which change does it behave like the old robot" directly.

    a1_feet       four 5 mm spheres per foot -> g1_minimal's 203 x 65.5 x 18.5 mm sole plate
    a2_colliders  every collider except the feet and torso deactivated
    a3_limits     leg joint limits taken from the old asset
    a4_masses     shared link masses, centres of mass and inertias taken from the old asset
    a5_origins    leg joint origins taken from the old asset
    a6_lockjoints the six degrees of freedom the old robot does not have, pinned shut

The joint *count* is the one difference an override cannot remove -- a1..a5 leave it at 43 against
37. Rung a6 gets at it the only way a layer can: it clamps the six extra joints to a range of
nothing, so the robot is kinematically equivalent to the old one while still presenting 43 joints
to the policy. ``waist_yaw``, ``elbow`` and ``wrist_roll`` are *not* among them -- they are the old
``torso_joint``, ``elbow_pitch`` and ``elbow_roll`` under new names.

Every value is read out of the two stages rather than hard-coded, so the layers cannot drift from
the assets they describe. Composing ``a5`` and diffing it against the old asset should leave the
limits, masses and origins sections empty; that is the check this file exists to pass.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

from pxr import Gf, Usd, UsdGeom, UsdPhysics

_LEG_KEYS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
_TRIM_LINKS = ("pelvis", "left_knee_link", "right_knee_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
_TRIM_FINGERS = ("palm", "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")

_EXTRA_JOINTS = (
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
"""The six degrees of freedom the 43-joint robot has and the 37-joint one does not."""

_LOCK_DEG = 0.1
"""Half-range left on a pinned joint [deg]. Exactly zero makes for a degenerate limit pair."""

Body = Callable[[int], str]
"""An override body, rendered at the indent its prim ends up at."""


class Tree:
    """Accumulates per-prim override bodies and renders them as one nested block.

    Emitting one ``over`` chain per edit would declare the same parent prim several times -- eight
    finger edits under ``left_hand`` would write eight ``over "left_hand"`` blocks -- and USD
    rejects that as a duplicate prim. Merging onto a path tree writes each prim exactly once.
    """

    def __init__(self) -> None:
        self.children: dict[str, Tree] = {}
        self.bodies: list[Body] = []

    def add(self, path_parts: list[str], body: Body) -> None:
        node = self
        for part in path_parts:
            node = node.children.setdefault(part, Tree())
        node.bodies.append(body)

    def merge(self, other: Tree) -> None:
        self.bodies.extend(other.bodies)
        for name, child in other.children.items():
            self.children.setdefault(name, Tree()).merge(child)

    def render(self, indent: int = 4) -> str:
        pad = " " * indent
        out = "".join(body(indent) for body in self.bodies)
        for name, child in self.children.items():
            out += f'{pad}over "{name}"\n{pad}{{\n{child.render(indent + 4)}{pad}}}\n'
        return out


def _v3(v) -> str:
    return f"({v[0]:.9g}, {v[1]:.9g}, {v[2]:.9g})"


def _quat(q) -> str:
    """USD writes quaternions real-first; ``Gf.Quatf`` exposes real and imaginary separately."""
    i = q.GetImaginary()
    return f"({q.GetReal():.9g}, {i[0]:.9g}, {i[1]:.9g}, {i[2]:.9g})"


def _lines(*lines: str) -> Body:
    """An override body of plain attribute assignments."""

    def render(indent: int) -> str:
        pad = " " * indent
        return "".join(f"{pad}{line}\n" for line in lines)

    return render


def _off_collisions(indent: int) -> str:
    pad = " " * indent
    return f'{pad}over "collisions" (\n{pad}    active = false\n{pad})\n{pad}{{\n{pad}}}\n'


def _prims(path: str) -> tuple[Usd.Stage, list[Usd.Prim]]:
    stage = Usd.Stage.Open(path, Usd.Stage.LoadAll)
    return stage, list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)))


def _path_parts(prims: list[Usd.Prim], name: str) -> list[str] | None:
    """Path of a named prim below the default prim, e.g. ``["left_hand", "left_hand_palm_link"]``."""
    for prim in prims:
        if prim.GetName() == name:
            return prim.GetPath().pathString.strip("/").split("/")[1:]
    return None


def _foot_plate(old_prims: list[Usd.Prim]) -> tuple[list[float], list[float]]:
    """Size and centre of the old asset's sole plate, in the ankle-roll body frame [m]."""
    cache = UsdGeom.XformCache()
    body = next(p for p in old_prims if p.GetName() == "left_ankle_roll_link" and p.HasAPI(UsdPhysics.RigidBodyAPI))
    body_m = cache.GetLocalToWorldTransform(body)
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        m = cache.GetLocalToWorldTransform(prim) * body_m.GetInverse()
        t = m.ExtractTranslation()
        scale = [Gf.Vec3d(m[i][0], m[i][1], m[i][2]).GetLength() for i in range(3)]
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        lo = [min(q[i] for q in pts) for i in range(3)]
        hi = [max(q[i] for q in pts) for i in range(3)]
        return (
            [(hi[i] - lo[i]) * scale[i] for i in range(3)],
            [t[i] + (lo[i] + hi[i]) / 2 * scale[i] for i in range(3)],
        )
    raise RuntimeError("the old asset's left ankle carries no collider")


def _leg_joints(prims: list[Usd.Prim]) -> dict[str, UsdPhysics.RevoluteJoint]:
    return {
        p.GetName(): UsdPhysics.RevoluteJoint(p)
        for p in prims
        if UsdPhysics.RevoluteJoint(p) and any(k in p.GetName() for k in _LEG_KEYS)
    }


def _mass_props(prims: list[Usd.Prim]) -> dict[str, UsdPhysics.MassAPI]:
    return {
        p.GetName(): UsdPhysics.MassAPI(p)
        for p in prims
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.HasAPI(UsdPhysics.MassAPI)
    }


def build(old_path: str, new_path: str, base_ref: str, out_dir: str) -> list[str]:
    """Write the cumulative rungs and return their paths."""
    _, old = _prims(old_path)
    new_stage, new = _prims(new_path)
    default_prim = new_stage.GetDefaultPrim().GetName()

    size, centre = _foot_plate(old)
    old_joints, new_joints = _leg_joints(old), _leg_joints(new)
    old_mass, new_mass = _mass_props(old), _mass_props(new)

    rungs: list[tuple[str, Tree, str]] = []

    plate = _lines(
        "double size = 1",
        f"float3 xformOp:translate = {_v3(centre)}",
        f"float3 xformOp:scale = {_v3(size)}",
        'uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
    )

    def foot_body(indent: int) -> str:
        pad = " " * indent
        return (
            _off_collisions(indent)
            + f'{pad}def Cube "foot_plate" (\n{pad}    prepend apiSchemas = ["PhysicsCollisionAPI"]\n{pad})\n'
            + f"{pad}{{\n{plate(indent + 4)}{pad}}}\n"
        )

    feet = Tree()
    for side in ("left", "right"):
        feet.add([f"{side}_ankle_roll_link"], foot_body)
    rungs.append(("a1_feet", feet, "sole plate instead of four spheres per foot"))

    trim = Tree()
    for link in _TRIM_LINKS:
        trim.add([link], _off_collisions)
    for side in ("left", "right"):
        for finger in _TRIM_FINGERS:
            parts = _path_parts(new, f"{side}_hand_{finger}_link")
            if parts:
                trim.add(parts, _off_collisions)
    rungs.append(("a2_colliders", trim, "every collider but the feet and torso deactivated"))

    limits = Tree()
    for name, joint in sorted(old_joints.items()):
        parts = _path_parts(new, name)
        if name not in new_joints or parts is None:
            continue
        limits.add(
            parts,
            _lines(
                f"float physics:lowerLimit = {joint.GetLowerLimitAttr().Get():.9g}",
                f"float physics:upperLimit = {joint.GetUpperLimitAttr().Get():.9g}",
            ),
        )
    rungs.append(("a3_limits", limits, "leg joint limits from the old asset"))

    masses = Tree()
    for name, api in sorted(old_mass.items()):
        parts = _path_parts(new, name)
        if name not in new_mass or parts is None:
            continue
        mass, com = api.GetMassAttr().Get(), api.GetCenterOfMassAttr().Get()
        inertia, axes = api.GetDiagonalInertiaAttr().Get(), api.GetPrincipalAxesAttr().Get()
        if mass is None:
            continue
        entries = [f"float physics:mass = {mass:.9g}"]
        if com is not None:
            entries.append(f"point3f physics:centerOfMass = {_v3(com)}")
        if inertia is not None:
            entries.append(f"float3 physics:diagonalInertia = {_v3(inertia)}")
        if axes is not None:
            entries.append(f"quatf physics:principalAxes = {_quat(axes)}")
        masses.add(parts, _lines(*entries))
    rungs.append(("a4_masses", masses, "shared link mass, centre of mass and inertia from the old asset"))

    origins = Tree()
    for name, joint in sorted(old_joints.items()):
        parts = _path_parts(new, name)
        if name not in new_joints or parts is None:
            continue
        p0, p1 = joint.GetLocalPos0Attr().Get(), joint.GetLocalPos1Attr().Get()
        entries = []
        if p0 is not None:
            entries.append(f"point3f physics:localPos0 = {_v3(p0)}")
        if p1 is not None:
            entries.append(f"point3f physics:localPos1 = {_v3(p1)}")
        if entries:
            origins.add(parts, _lines(*entries))
    rungs.append(("a5_origins", origins, "leg joint origins from the old asset"))

    locked = Tree()
    for name in _EXTRA_JOINTS:
        parts = _path_parts(new, name)
        if parts is None:
            continue
        locked.add(
            parts,
            _lines(f"float physics:lowerLimit = {-_LOCK_DEG:.9g}", f"float physics:upperLimit = {_LOCK_DEG:.9g}"),
        )
    rungs.append(("a6_lockjoints", locked, "the six extra degrees of freedom pinned shut"))

    os.makedirs(out_dir, exist_ok=True)
    written: list[str] = []
    cumulative = Tree()
    for name, tree, note in rungs:
        cumulative.merge(tree)
        path = os.path.join(out_dir, f"g1_{name}.usda")
        with open(path, "w") as handle:
            handle.write(
                f'#usda 1.0\n(\n    defaultPrim = "{default_prim}"\n)\n\n'
                f"# Cumulative rung '{name}': {note}.\n"
                f"# Generated by scripts/make_g1_ladder.py from the two assets themselves.\n"
                f'over "{default_prim}" (\n    prepend references = @{base_ref}@\n)\n'
                f"{{\n{cumulative.render()}}}\n"
            )
        written.append(path)
        print(f"[ladder] {name}: {note}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, help="g1_minimal.usd, on a path plain USD can open.")
    parser.add_argument("--new", required=True, help="The shipped g1.usd, on a path plain USD can open.")
    parser.add_argument(
        "--base_ref",
        default=None,
        help="Path the layers reference. Defaults to --new; set it when the layers will be composed"
        " somewhere this machine's path does not exist.",
    )
    parser.add_argument("--out_dir", required=True, help="Directory to write the layers into.")
    args = parser.parse_args()
    build(args.old, args.new, args.base_ref or args.new, args.out_dir)


if __name__ == "__main__":
    main()
