# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write the USD override layer that makes the training robot's inertials MuJoCo's.

The MuJoCo model in ``unitree_mujoco`` is the one the hardware agrees with -- a policy that behaves
the same in MuJoCo and on the robot behaves differently in Isaac Lab -- so the alignment moves Isaac
Lab toward it, not the other way.

Audited body by body on 2026-09-11 (``scratchpad/align_audit.py``). Comparing the principal moments
*sorted*, which is convention-independent, 25 of the 30 shared bodies already agree exactly. Only
five disagree, and they are the same five whose mass disagrees:

===================== ================ ================
body                  Isaac Lab        MuJoCo
===================== ================ ================
``torso_link``        7.817 kg         9.598 kg (+23%)
``waist_yaw_link``    0.214 kg         0.244 kg
``waist_roll_link``   0.086 kg         0.047 kg
``*_wrist_yaw_link``  0.085 kg         0.255 kg (+200%)
===================== ================ ================

The wrist is not a discrepancy in the same sense as the others: MuJoCo fuses a rigid rubber hand
into ``wrist_yaw_link`` where the USD carries an articulated Dex3 as separate bodies. Porting the
MuJoCo value there therefore double-counts unless the Dex3 bodies are also removed, so it is behind
``--wrists`` and off by default.

The plate foot is emitted too, identical to the ``a1_feet`` rung, so this layer is a drop-in
replacement for it rather than something to stack on top.

Usage::

    ./isaaclab.sh -p scripts/make_g1_mujoco_aligned.py \
        --usd ~/g1_assets/assets/G1_shipped/g1.usd --out ~/g1_assets/ladder/g1_mj_aligned.usda
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", required=True, help="The shipped g1.usd the layer references.")
parser.add_argument("--out", required=True, help="Output .usda path.")
parser.add_argument(
    "--mjcf",
    default=None,
    help="unitree_mujoco's scene_29dof.xml. Optional: the table below is the same data, embedded so"
    " the generator runs where unitree_mujoco is not checked out. Pass it to re-verify.",
)
parser.add_argument(
    "--wrists",
    action="store_true",
    help="Also port wrist_yaw_link, which in the MJCF carries a fused rubber hand the USD models as"
    " separate Dex3 bodies. Double-counts unless those bodies are removed too.",
)
args = parser.parse_args()

DEFAULT_PRIM = "g1_29dof_with_hand_rev_1_0"

# Mass [kg], principal moments [kg m^2], centre of mass [m] and principal-axis quaternion (w,x,y,z),
# read off the compiled MuJoCo model. Regenerate with --mjcf to check them.
_MJCF_INERTIAL = {
    "torso_link": dict(
        mass=9.598,
        inertia=(0.12407, 0.111951, 0.0325382),
        com=(0.00331658, 0.000261533, 0.179856),
        axes=(0.9998309708889911, 0.00037620398904647084, 0.017989499476218983, -0.00377703989002797),
    ),
    "waist_yaw_link": dict(
        mass=0.244,
        inertia=(0.000158561, 0.000124229, 9.67669e-05),
        com=(0.003964, 0.0, 0.018769),
        axes=(-0.01782910157428995, 0.6284640554926809, 0.028247102494187906, 0.7771210686189307),
    ),
    "waist_roll_link": dict(
        mass=0.047,
        inertia=(7.515e-06, 6.40206e-06, 3.98394e-06),
        com=(0.0, -0.000236, 0.010111),
        axes=(0.9997900169144477, 0.020492000346683668, 0.0, 0.0),
    ),
    "left_wrist_yaw_link": dict(
        mass=0.254576,
        inertia=(0.000646113, 0.000559993, 0.000147566),
        com=(0.0708244, 0.000191745, 0.00161742),
        axes=(0.5105708492555107, 0.5262948446130489, 0.4680778618014359, 0.49318785438778706),
    ),
    "right_wrist_yaw_link": dict(
        mass=0.254576,
        inertia=(0.000646113, 0.000559993, 0.000147566),
        com=(0.0708244, -0.000191745, 0.00161742),
        axes=(0.49318785438778706, 0.4680778618014359, 0.5262948446130489, 0.5105708492555107),
    ),
}

# The a1_feet rung, verbatim: the sole plate that replaces the four contact spheres. Sized and
# placed from the spheres themselves, and identical to what the MuJoCo deploy loop's
# foot_plate_override builds, so the two simulators stand at the same height on the same surface.
_PLATE_TRANSLATE = (0.0359170487, 2.22044605e-16, -0.0251700647)
_PLATE_SCALE = (0.203109218, 0.065469244, 0.0185078794)
_FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def _verify(path: str) -> None:
    """Recompute the table from the MJCF and complain if it moved."""
    import sys  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    sys.path.insert(0, str(Path.home() / "workspace/g1_deploy"))
    import mujoco  # noqa: PLC0415
    from g1_deploy.sim.mjcf import compile_model, resolve_includes  # noqa: PLC0415

    model = compile_model(resolve_includes(Path(path)), path)
    bad = []
    for name, want in _MJCF_INERTIAL.items():
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            bad.append(f"{name}: not in the MJCF")
            continue
        if abs(float(model.body_mass[i]) - want["mass"]) > 1e-6:
            bad.append(f"{name}: mass {model.body_mass[i]} against the table's {want['mass']}")
        for k in range(3):
            if abs(float(model.body_inertia[i][k]) - want["inertia"][k]) > 1e-9:
                bad.append(f"{name}: inertia[{k}] moved")
    if bad:
        raise SystemExit("[mjalign] the embedded table no longer matches the MJCF:\n  " + "\n  ".join(bad))
    print(f"[mjalign] verified {len(_MJCF_INERTIAL)} bodies against {path}")


def main() -> None:
    if args.mjcf:
        _verify(args.mjcf)

    bodies = dict(_MJCF_INERTIAL)
    if not args.wrists:
        bodies.pop("left_wrist_yaw_link", None)
        bodies.pop("right_wrist_yaw_link", None)

    lines = [
        "#usda 1.0\n(\n"
        f'    defaultPrim = "{DEFAULT_PRIM}"\n)\n\n'
        "# The plate foot of the a1_feet rung, plus MuJoCo's inertials on the bodies where the two\n"
        "# models disagree. Generated by scripts/make_g1_mujoco_aligned.py.\n"
        f'over "{DEFAULT_PRIM}" (\n    prepend references = @{args.usd}@\n)\n{{\n'
    ]
    for link in _FEET:
        lines.append(
            f'    over "{link}"\n    {{\n'
            '        over "collisions" (\n            active = false\n        )\n        {\n        }\n'
            '        def Cube "foot_plate" (\n            prepend apiSchemas = ["PhysicsCollisionAPI"]\n        )\n'
            "        {\n            double size = 1\n"
            f"            float3 xformOp:translate = ({_PLATE_TRANSLATE[0]}, {_PLATE_TRANSLATE[1]}, {_PLATE_TRANSLATE[2]})\n"
            f"            float3 xformOp:scale = ({_PLATE_SCALE[0]}, {_PLATE_SCALE[1]}, {_PLATE_SCALE[2]})\n"
            '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n'
            "        }\n    }\n"
        )
    for link, v in bodies.items():
        i, c, a = v["inertia"], v["com"], v["axes"]
        lines.append(
            f'    over "{link}"\n    {{\n'
            f"        float physics:mass = {v['mass']!r}\n"
            f"        float3 physics:diagonalInertia = ({i[0]!r}, {i[1]!r}, {i[2]!r})\n"
            f"        point3f physics:centerOfMass = ({c[0]!r}, {c[1]!r}, {c[2]!r})\n"
            f"        quatf physics:principalAxes = ({a[0]!r}, {a[1]!r}, {a[2]!r}, {a[3]!r})\n"
            "    }\n"
        )
    lines.append("}\n")

    with open(args.out, "w") as handle:
        handle.write("".join(lines))
    print(f"[mjalign] wrote {args.out}: plate feet + {len(bodies)} inertials")
    for link, v in bodies.items():
        print(f"[mjalign]   {link:24s} mass {v['mass']:.4f} kg")


if __name__ == "__main__":
    main()
