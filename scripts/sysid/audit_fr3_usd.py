# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Executable audit of the converted FR3 USD (acceptance gate, no Kit needed).

Per-NAMED-joint checks against the ascii usda payloads: each fr3_joint1..7 must
carry an angular position drive with its URDF effort limit (87 Nm on joints
1-4, 12 Nm on joints 5-7), an explicit PhysicsFixedJoint must fix the base,
every link mass must be present AND positive, and the MuJoCo payload consumed
by the Newton backend must define an MjcActuator per joint. Prints the
aggregate content digest of all usda layers (the asset is gitignored — runtime
proofs pin this digest). Exits nonzero on any failure.

    python scripts/sysid/audit_fr3_usd.py [--usd-dir <assets/fr3.usd>]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

DEFAULT_USD_DIR = (
    Path(__file__).resolve().parents[2]
    / "source/isaaclab_tasks/isaaclab_tasks/contrib/sysid/config/franka_fr3/assets/fr3.usd"
)
EXPECTED_MAX_FORCE = {f"fr3_joint{i}": 87.0 for i in range(1, 5)} | {f"fr3_joint{i}": 12.0 for i in range(5, 8)}
EXPECTED_LINK_MASSES = 8  # base link0..link7 (fr3_link8 is massless/merged)


def joint_block(text: str, joint: str) -> str | None:
    """Extract the prim block for one named revolute joint."""
    m = re.search(rf'def PhysicsRevoluteJoint "{joint}".*?\n(.*?)\n\s*\}}', text, re.DOTALL)
    return m.group(1) if m else None


def actuator_block(text: str, joint: str) -> str | None:
    """Extract the MjcActuator prim block for one named joint."""
    m = re.search(rf'def MjcActuator "{joint}_actuator".*?\n(.*?)\n\s*\}}', text, re.DOTALL)
    return m.group(1) if m else None


def usd_digest(usd_dir: Path) -> str:
    digest = hashlib.sha256()
    for f in sorted(usd_dir.rglob("*.usda")):
        digest.update(f.name.encode())
        digest.update(f.read_bytes())
    return digest.hexdigest()[:16]


def run_audit(usd_dir: Path) -> int:
    failures: list[str] = []

    def check(condition: bool, ok_msg: str, fail_msg: str) -> None:
        print(f"[{'PASS' if condition else 'FAIL'}] {ok_msg if condition else fail_msg}")
        if not condition:
            failures.append(fail_msg)

    physics_files = list(usd_dir.rglob("payloads/Physics/physics.usda"))
    check(
        len(physics_files) == 1,
        f"physics payload found: {physics_files[0] if physics_files else ''}",
        f"expected exactly one payloads/Physics/physics.usda under {usd_dir}, found {len(physics_files)}",
    )
    if not physics_files:
        return 1
    physics = physics_files[0].read_text()
    payload_dir = physics_files[0].parent

    # Per-named-joint drive + effort limit checks.
    for joint, expected_force in EXPECTED_MAX_FORCE.items():
        block = joint_block(physics, joint)
        if block is None:
            check(False, "", f"{joint}: PhysicsRevoluteJoint prim missing")
            continue
        stiff = re.search(r"drive:angular:physics:stiffness = ([0-9.eE+-]+)", block)
        damp = re.search(r"drive:angular:physics:damping = ([0-9.eE+-]+)", block)
        force = re.search(r"drive:angular:physics:maxForce = ([0-9.eE+-]+)", block)
        check(
            stiff is not None and damp is not None and force is not None,
            f"{joint}: angular position drive present",
            f"{joint}: missing drive attrs (stiffness/damping/maxForce)",
        )
        if force is not None:
            check(
                float(force.group(1)) == expected_force,
                f"{joint}: maxForce {float(force.group(1)):.0f} Nm",
                f"{joint}: maxForce {float(force.group(1)):.0f} != expected {expected_force:.0f} Nm",
            )

    check(
        "PhysicsFixedJoint" in physics,
        "explicit PhysicsFixedJoint (fixed base) present",
        "no PhysicsFixedJoint found — base is not fixed",
    )

    robot_files = list(usd_dir.rglob("payloads/robot.usda"))
    mass_text = physics + (robot_files[0].read_text() if robot_files else "")
    masses = [float(v) for v in re.findall(r"physics:mass = ([0-9.eE+-]+)", mass_text)]
    check(
        len(masses) == EXPECTED_LINK_MASSES,
        f"{len(masses)} link masses present",
        f"expected {EXPECTED_LINK_MASSES} link masses, found {len(masses)}",
    )
    check(
        all(m > 0 for m in masses),
        "all link masses positive",
        f"non-positive link mass present: {[m for m in masses if m <= 0]}",
    )

    mujoco = payload_dir / "mujoco.usda"
    check(mujoco.exists(), "MuJoCo payload present (Newton backend input)", f"missing {mujoco}")
    if mujoco.exists():
        mtext = mujoco.read_text()
        default_prim = re.search(r'defaultPrim = "(\w+)"', mtext)
        check(
            default_prim is not None,
            f"MuJoCo payload defaultPrim: {default_prim.group(1) if default_prim else ''}",
            "MuJoCo payload has no defaultPrim — actuator targets cannot be resolved",
        )
        prim_root = default_prim.group(1) if default_prim else None
        for joint, expected_force in EXPECTED_MAX_FORCE.items():
            block = actuator_block(mtext, joint)
            if block is None:
                check(False, "", f"{joint}: MjcActuator missing from MuJoCo payload")
                continue
            target = re.search(r"rel mjc:target = <([^>]+)>", block)
            fmax = re.search(r"mjc:forceRange:max = ([0-9.eE+-]+)", block)
            fmin = re.search(r"mjc:forceRange:min = ([0-9.eE+-]+)", block)
            gain = re.search(r"mjc:gainPrm = \[([0-9.eE+-]+)", block)
            gain_type = re.search(r'mjc:gainType = "([^"]+)"', block)
            bias_type = re.search(r'mjc:biasType = "([^"]+)"', block)
            # Exact resolution, not endswith: the target must be the joint prim
            # under the payload's own defaultPrim scope chain.
            expected_target = f"/{prim_root}/Physics/{joint}" if prim_root else None
            check(
                target is not None and expected_target is not None and target.group(1) == expected_target,
                f"{joint}: actuator targets {expected_target}",
                f"{joint}: mjc:target {'absent' if target is None else target.group(1)} != {expected_target}",
            )
            check(
                fmax is not None
                and fmin is not None
                and float(fmax.group(1)) == expected_force
                and float(fmin.group(1)) == -expected_force,
                f"{joint}: forceRange ±{expected_force:.0f} Nm",
                f"{joint}: forceRange != ±{expected_force:.0f} Nm",
            )
            check(
                gain is not None and float(gain.group(1)) > 0,
                f"{joint}: gainPrm[0] positive",
                f"{joint}: gainPrm missing or non-positive",
            )
            check(
                gain_type is not None
                and gain_type.group(1) == "fixed"
                and bias_type is not None
                and bias_type.group(1) == "affine",
                f"{joint}: gainType=fixed, biasType=affine",
                f"{joint}: unexpected gain/bias type",
            )

    for link in [f"fr3_link{i}" for i in range(8)]:
        check(link in mass_text, f"{link}: named body present", f"{link}: named body missing from payloads")

    print(f"\nusd content digest: {usd_digest(usd_dir)}")
    print(f"{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd-dir", type=str, default=str(DEFAULT_USD_DIR))
    args = parser.parse_args()
    return run_audit(Path(args.usd_dir))


if __name__ == "__main__":
    sys.exit(main())
