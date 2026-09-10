# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rebuild the G1's collision set as mjlab authors it, as a USD override layer.

mjlab does not collide the robot's meshes at all: every STL in its ``g1.xml`` is
``class="visual"`` with ``contype=0 conaffinity=0 density=0``, and collision is 33 hand-placed
primitives -- 31 capsules and 2 spheres. The feet are seven thin capsules each, a rake across the
sole, where the shipped Isaac asset has four 5 mm spheres.

This reads those geoms out of mjlab's MJCF and writes them into an override layer over the shipped
``g1.usd``: every existing ``collisions`` scope is deactivated, including the sixteen on the hand
links that mjlab's 29-DoF robot does not have at all, and the primitives are added in their place.

The two assets share link frames -- both are ``g1_29dof_rev_1_0`` -- which is what makes the MJCF's
body-frame coordinates directly usable. That is checked rather than assumed: the foot capsules sit
at z = -0.025 with radius 0.01, putting the sole at -0.035, and the sole plate measured on the
Isaac asset independently is at -0.0345.

Usage:
    ./isaaclab.sh -p scripts/make_g1_mjlab_colliders.py \\
        --mjcf ~/workspace/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml \\
        --usd <shipped g1.usd> --out /tmp/ladder/g1_mjlab_colliders.usda
"""

import argparse
import math
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--mjcf",
    default=None,
    help="mjlab's g1.xml. Optional: the table below is the same data, embedded so the"
    " generator runs where mjlab is not checked out. Pass it to re-verify.",
)
parser.add_argument("--usd", required=True, help="The shipped g1.usd the layer references.")
parser.add_argument("--out", required=True, help="Output .usda path.")
args = parser.parse_args()

DEFAULT_PRIM = "g1_29dof_with_hand_rev_1_0"
FOOT_CAPSULE_RADIUS = 0.01
"""``foot_capsule`` class default in mjlab's MJCF; the seven sole capsules carry no explicit size."""

NESTED_PARENT = {"hand": ("left_hand", "right_hand")}
"""Hand links sit under a ``left_hand``/``right_hand`` Xform rather than directly under the root."""


_GEOMS = [
    # (parent body, geom name, type, size, fromto, pos) -- mjlab's 33 collision geoms, verbatim.
    # A ``size`` of None means the ``foot_capsule`` class default, 0.01.
    ("pelvis", "pelvis_collision", "sphere", 0.07, None, "0 0 -0.08"),
    ("left_hip_roll_link", "left_hip_collision", "capsule", 0.06, "0.02 0 0 0.02 0 -0.08", None),
    ("left_hip_yaw_link", "left_thigh_collision", "capsule", 0.055, "-0.0 0 -0.03 -0.06 0 -0.17", None),
    ("left_knee_link", "left_shin_collision", "capsule", 0.045, "0.01 0 0 0.01 0 -0.15", None),
    ("left_knee_link", "left_linkage_brace_collision", "capsule", 0.03, "0.01 0 -0.2 0.01 0 -0.28", None),
    ("left_ankle_roll_link", "left_foot1_collision", "capsule", None, "0.1 -0.026 -0.025 0.05 -0.027 -0.025", None),
    ("left_ankle_roll_link", "left_foot2_collision", "capsule", None, "-0.044 -0.018 -0.025 0.123 -0.018 -0.025", None),
    ("left_ankle_roll_link", "left_foot3_collision", "capsule", None, "-0.052 -0.01 -0.025 0.13 -0.01 -0.025", None),
    ("left_ankle_roll_link", "left_foot4_collision", "capsule", None, "-0.054 0 -0.025 0.132 0 -0.025", None),
    ("left_ankle_roll_link", "left_foot5_collision", "capsule", None, "-0.052 0.01 -0.025 0.13 0.01 -0.025", None),
    ("left_ankle_roll_link", "left_foot6_collision", "capsule", None, "-0.044 0.018 -0.025 0.123 0.018 -0.025", None),
    ("left_ankle_roll_link", "left_foot7_collision", "capsule", None, "0.1 0.026 -0.025 0.05 0.026 -0.025", None),
    ("right_hip_roll_link", "right_hip_collision", "capsule", 0.06, "0.02 0 0 0.02 0 -0.08", None),
    ("right_hip_yaw_link", "right_thigh_collision", "capsule", 0.055, "-0.0 0 -0.03 -0.06 0 -0.17", None),
    ("right_knee_link", "right_shin_collision", "capsule", 0.045, "0.01 0 0 0.01 0 -0.15", None),
    ("right_knee_link", "right_linkage_brace_collision", "capsule", 0.03, "0.01 0 -0.2 0.01 0 -0.28", None),
    ("right_ankle_roll_link", "right_foot1_collision", "capsule", None, "0.1 -0.026 -0.025 0.05 -0.026 -0.025", None),
    (
        "right_ankle_roll_link",
        "right_foot2_collision",
        "capsule",
        None,
        "-0.044 -0.018 -0.025 0.123 -0.018 -0.025",
        None,
    ),
    ("right_ankle_roll_link", "right_foot3_collision", "capsule", None, "-0.052 -0.01 -0.025 0.13 -0.01 -0.025", None),
    ("right_ankle_roll_link", "right_foot4_collision", "capsule", None, "-0.054 0 -0.025 0.132 0 -0.025", None),
    ("right_ankle_roll_link", "right_foot5_collision", "capsule", None, "-0.052 0.01 -0.025 0.13 0.01 -0.025", None),
    ("right_ankle_roll_link", "right_foot6_collision", "capsule", None, "-0.044 0.018 -0.025 0.123 0.018 -0.025", None),
    ("right_ankle_roll_link", "right_foot7_collision", "capsule", None, "0.1 0.026 -0.025 0.05 0.026 -0.025", None),
    ("torso_link", "torso_collision", "capsule", 0.09, "0.01 0 0.08 0.01 0 0.2", None),
    ("torso_link", "head_collision", "sphere", 0.06, None, "0 0 .43"),
    ("left_shoulder_yaw_link", "left_shoulder_yaw_collision", "capsule", 0.035, "0 0 -0.08 0 0 0.05", None),
    ("left_elbow_link", "left_elbow_yaw_collision", "capsule", 0.035, "-0.01 0 -0.01 0.08 0 -0.01", None),
    ("left_wrist_pitch_link", "left_wrist_collision", "capsule", 0.035, "-0.01 0 0 0.06 0 0", None),
    ("left_wrist_yaw_link", "left_hand_collision", "capsule", 0.035, "0.07 0 0 0.15 -0.02 0", None),
    ("right_shoulder_yaw_link", "right_shoulder_yaw_collision", "capsule", 0.035, "0 0 -0.08 0 0 0.05", None),
    ("right_elbow_link", "right_elbow_yaw_collision", "capsule", 0.035, "-0.01 0 -0.01 0.08 0 -0.01", None),
    ("right_wrist_pitch_link", "right_wrist_collision", "capsule", 0.035, "-0.01 0 0 0.06 0 0", None),
    ("right_wrist_yaw_link", "right_hand_collision", "capsule", 0.035, "0.07 0 0 0.15 0.02 0", None),
]
"""mjlab's collision set, embedded.

The container that runs the sweep has Isaac Lab and this repository, not mjlab, so the table lives
here rather than being read from their MJCF. ``--mjcf`` re-derives it and fails on any difference,
which is how this stays honest if mjlab changes the asset.
"""


def _collect(node, body, out):
    """Walk the MJCF and collect ``(body_name, geom_attrs)`` for every ``*_collision`` geom."""
    for child in node:
        if child.tag == "body":
            _collect(child, child.get("name"), out)
        elif child.tag == "geom" and (child.get("name") or "").endswith("_collision"):
            out.append((body, dict(child.attrib)))
        else:
            _collect(child, body, out)
    return out


def _quat_z_to(direction):
    """Quaternion (w, x, y, z) rotating +Z onto ``direction``."""
    length = math.sqrt(sum(component * component for component in direction))
    unit = [component / length for component in direction]
    dot = unit[2]
    if dot > 1.0 - 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    if dot < -1.0 + 1e-9:
        return (0.0, 1.0, 0.0, 0.0)
    axis = (-unit[1], unit[0], 0.0)  # cross((0,0,1), unit)
    norm = math.sqrt(axis[0] ** 2 + axis[1] ** 2)
    axis = (axis[0] / norm, axis[1] / norm, 0.0)
    angle = math.acos(max(-1.0, min(1.0, dot)))
    s = math.sin(angle / 2.0)
    return (math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s)


def _prim(name, attrs):
    """Emit the USD prim text for one MJCF collision geom, indented for a link body."""
    kind = attrs.get("type", "capsule")
    if kind == "sphere":
        pos = [float(v) for v in attrs["pos"].split()]
        radius = float(attrs["size"])
        return (
            f'        def Sphere "mj_{name}" (\n'
            f'            prepend apiSchemas = ["PhysicsCollisionAPI"]\n'
            f"        )\n        {{\n"
            f"            double radius = {radius}\n"
            f"            float3 xformOp:translate = ({pos[0]}, {pos[1]}, {pos[2]})\n"
            f'            uniform token[] xformOpOrder = ["xformOp:translate"]\n'
            f"        }}\n"
        )
    if kind != "capsule":
        raise ValueError(f"{name}: unsupported geom type {kind!r}")
    a = [float(v) for v in attrs["fromto"].split()[:3]]
    b = [float(v) for v in attrs["fromto"].split()[3:]]
    radius = float(attrs["size"]) if attrs.get("size") else FOOT_CAPSULE_RADIUS
    direction = [b[i] - a[i] for i in range(3)]
    height = math.sqrt(sum(component * component for component in direction))
    mid = [(a[i] + b[i]) / 2.0 for i in range(3)]
    w, x, y, z = _quat_z_to(direction)
    return (
        f'        def Capsule "mj_{name}" (\n'
        f'            prepend apiSchemas = ["PhysicsCollisionAPI"]\n'
        f"        )\n        {{\n"
        f'            uniform token axis = "Z"\n'
        f"            double height = {height}\n"
        f"            double radius = {radius}\n"
        f"            float3 xformOp:translate = ({mid[0]}, {mid[1]}, {mid[2]})\n"
        f"            quatf xformOp:orient = ({w}, {x}, {y}, {z})\n"
        f'            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]\n'
        f"        }}\n"
    )


def _embedded():
    """The embedded table in the same shape ``_collect`` returns."""
    out = []
    for body, name, kind, size, fromto, pos in _GEOMS:
        attrs = {"name": name, "type": kind}
        if size is not None:
            attrs["size"] = str(size)
        if fromto is not None:
            attrs["fromto"] = fromto
        if pos is not None:
            attrs["pos"] = pos
        out.append((body, attrs))
    return out


def main() -> None:
    geoms = _embedded()
    if args.mjcf:
        # Compare the geometry, not the raw attribute dicts: the MJCF carries a ``class`` the table
        # does not, and writes sizes as strings.
        def _norm(rows):
            out = []
            for body, a in rows:
                size = float(a["size"]) if a.get("size") else None
                out.append((body, a["name"], a.get("type", "capsule"), size, a.get("fromto"), a.get("pos")))
            return sorted(out)

        if _norm(_collect(ET.parse(args.mjcf).getroot(), None, [])) != _norm(geoms):
            raise SystemExit(
                f"the embedded table disagrees with {args.mjcf}. Regenerate it rather than trusting either."
            )
        print(f"[colliders] embedded table verified against {args.mjcf}")
    by_link: dict[str, list[str]] = {}
    for body, attrs in geoms:
        by_link.setdefault(body, []).append(_prim(attrs["name"], attrs))

    from pxr import Usd  # noqa: PLC0415

    stage = Usd.Stage.Open(args.usd)
    root = stage.GetDefaultPrim()
    top_scopes, nested_scopes = [], {}
    for child in root.GetChildren():
        names = [g.GetName() for g in child.GetChildren()]
        if "collisions" in names:
            top_scopes.append(child.GetName())
        for grandchild in child.GetChildren():
            if "collisions" in [x.GetName() for x in grandchild.GetChildren()]:
                nested_scopes.setdefault(child.GetName(), []).append(grandchild.GetName())

    lines = [
        "#usda 1.0\n(\n"
        f'    defaultPrim = "{DEFAULT_PRIM}"\n)\n\n'
        "# mjlab's collision set: 31 capsules and 2 spheres, feet as seven capsules each.\n"
        "# Generated by scripts/make_g1_mjlab_colliders.py from mjlab's g1.xml.\n"
        f'over "{DEFAULT_PRIM}" (\n    prepend references = @{args.usd}@\n)\n{{\n'
    ]
    for link in sorted(set(top_scopes) | set(by_link)):
        lines.append(f'    over "{link}"\n    {{\n')
        if link in top_scopes:
            lines.append('        over "collisions" (\n            active = false\n        )\n        {\n        }\n')
        lines.extend(by_link.get(link, []))
        lines.append("    }\n")
    for parent, children in sorted(nested_scopes.items()):
        lines.append(f'    over "{parent}"\n    {{\n')
        for child in sorted(children):
            lines.append(
                f'        over "{child}"\n        {{\n'
                '            over "collisions" (\n                active = false\n            )\n'
                "            {\n            }\n        }\n"
            )
        lines.append("    }\n")
    lines.append("}\n")

    with open(args.out, "w") as handle:
        handle.write("".join(lines))
    n_add = sum(len(v) for v in by_link.values())
    n_off = len(top_scopes) + sum(len(v) for v in nested_scopes.values())
    print(f"[colliders] wrote {args.out}: {n_add} primitives added, {n_off} shipped scopes deactivated")
    for link, prims in sorted(by_link.items()):
        print(f"[colliders]   {link:24s} {len(prims)}")


if __name__ == "__main__":
    main()
