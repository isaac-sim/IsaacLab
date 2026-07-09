# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mutation tests for the FR3 USD audit.

Runs against a SYNTHETIC fixture built here (so the suite never skips in a
clean checkout where the generated asset is absent), plus a bonus pass on the
real converted asset when present. Every corruption class the audit claims to
catch has a mutation that must FAIL it.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from audit_fr3_usd import DEFAULT_USD_DIR, EXPECTED_MAX_FORCE, run_audit  # noqa: E402


def make_fixture(tmp_path: Path) -> Path:
    """Minimal usda tree with the structure the audit validates."""
    usd_dir = tmp_path / "fr3.usd"
    root = usd_dir / "fr3_nomesh" / "payloads" / "Physics"
    root.mkdir(parents=True)

    physics = ['        def PhysicsFixedJoint "root_joint"\n        {\n        }']
    for joint, force in EXPECTED_MAX_FORCE.items():
        physics.append(
            f'        def PhysicsRevoluteJoint "{joint}" (\n'
            "        )\n"
            "        {\n"
            "            float drive:angular:physics:damping = 0.003\n"
            f"            float drive:angular:physics:maxForce = {force:g}\n"
            "            float drive:angular:physics:stiffness = 1.75\n"
            "        }"
        )
    for i in range(8):
        physics.append(f'        def Xform "fr3_link{i}"\n        {{\n            float physics:mass = 2.5\n        }}')
    (root / "physics.usda").write_text("\n".join(physics))

    mujoco = ['#usda 1.0\n(\n    defaultPrim = "fr3"\n)']
    for joint, force in EXPECTED_MAX_FORCE.items():
        mujoco.append(
            f'        def MjcActuator "{joint}_actuator"\n'
            "        {\n"
            "            uniform double[] mjc:biasPrm = [0, -1.75, -0.003]\n"
            '            uniform token mjc:biasType = "affine"\n'
            f"            uniform double mjc:forceRange:max = {force:g}\n"
            f"            uniform double mjc:forceRange:min = -{force:g}\n"
            "            uniform double[] mjc:gainPrm = [1.75, 0, 0]\n"
            '            uniform token mjc:gainType = "fixed"\n'
            f"            prepend rel mjc:target = </fr3/Physics/{joint}>\n"
            "        }"
        )
    (root / "mujoco.usda").write_text("\n".join(mujoco))
    return usd_dir


@pytest.fixture()
def usd_fixture(tmp_path):
    return make_fixture(tmp_path)


def _physics(usd_dir: Path) -> Path:
    return next(usd_dir.rglob("payloads/Physics/physics.usda"))


def _mujoco(usd_dir: Path) -> Path:
    return next(usd_dir.rglob("payloads/Physics/mujoco.usda"))


def test_clean_fixture_passes(usd_fixture):
    assert run_audit(usd_fixture) == 0


def test_swapped_effort_limit_fails(usd_fixture):
    p = _physics(usd_fixture)
    import re

    text = p.read_text()
    m = re.search(r'(def PhysicsRevoluteJoint "fr3_joint1".*?maxForce = )87', text, re.DOTALL)
    p.write_text(text[: m.end() - 2] + "12" + text[m.end() :])
    assert run_audit(usd_fixture) == 1


def test_deleted_fixed_joint_fails(usd_fixture):
    p = _physics(usd_fixture)
    p.write_text(p.read_text().replace("PhysicsFixedJoint", "PhysicsBrokenJoint"))
    assert run_audit(usd_fixture) == 1


def test_zero_mass_fails(usd_fixture):
    p = _physics(usd_fixture)
    p.write_text(p.read_text().replace("physics:mass = 2.5", "physics:mass = 0", 1))
    assert run_audit(usd_fixture) == 1


def test_missing_actuator_fails(usd_fixture):
    m = _mujoco(usd_fixture)
    m.write_text(m.read_text().replace('def MjcActuator "fr3_joint4_actuator"', 'def MjcActuator "gone"', 1))
    assert run_audit(usd_fixture) == 1


def test_missing_joint_prim_fails(usd_fixture):
    p = _physics(usd_fixture)
    p.write_text(p.read_text().replace('def PhysicsRevoluteJoint "fr3_joint7"', 'def PhysicsRevoluteJoint "zz"', 1))
    assert run_audit(usd_fixture) == 1


def test_corrupt_force_range_fails(usd_fixture):
    m = _mujoco(usd_fixture)
    m.write_text(m.read_text().replace("mjc:forceRange:max = 12", "mjc:forceRange:max = 500", 1))
    assert run_audit(usd_fixture) == 1


def test_wrong_actuator_target_fails(usd_fixture):
    m = _mujoco(usd_fixture)
    m.write_text(m.read_text().replace("</fr3/Physics/fr3_joint2>", "</fr3/Physics/fr3_joint3>", 1))
    assert run_audit(usd_fixture) == 1


def test_wrong_gain_type_fails(usd_fixture):
    m = _mujoco(usd_fixture)
    m.write_text(m.read_text().replace('mjc:gainType = "fixed"', 'mjc:gainType = "muscle"', 1))
    assert run_audit(usd_fixture) == 1


@pytest.mark.skipif(not Path(DEFAULT_USD_DIR).exists(), reason="generated fr3.usd not present")
def test_real_asset_passes():
    assert run_audit(Path(DEFAULT_USD_DIR)) == 0


def test_unresolvable_target_root_fails(usd_fixture):
    # endswith('/joint') is not resolution: a foreign root must FAIL.
    m = _mujoco(usd_fixture)
    m.write_text(m.read_text().replace("</fr3/Physics/fr3_joint2>", "</nonexistent/fr3_joint2>", 1))
    assert run_audit(usd_fixture) == 1


def test_missing_default_prim_fails(usd_fixture):
    m = _mujoco(usd_fixture)
    m.write_text(m.read_text().replace('defaultPrim = "fr3"', "", 1))
    assert run_audit(usd_fixture) == 1
