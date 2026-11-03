# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Protect prebuilt foot-site geometry and the gearbox-play actuator contract.

The physical asset suites cover the original three model families. These checks retain
sole sites for terrain sensing and the native firmware settings of the backlash variants.
"""

from pathlib import Path

import pytest
from microduck_asset_test_utils import prebuilt_usd_path

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab_tasks.contrib.microduck import assets as microduck


@pytest.mark.parametrize("name", ["FULLCONTACT", "WALK", "GROUNDCONTACT", "ROLLERS"])
def test_backlash_is_in_the_native_firmware_loop(name):
    cfg = getattr(microduck, f"MICRODUCK_{name}_BACKLASH_CFG")
    servo = cfg.actuators["servos"]
    assert servo.backlash_joint_template == "passive_{}_backlash"
    assert cfg.init_state.joint_pos["passive_.*"] == 0.0
    assert servo.joint_names_expr == ["^(?!passive_).*"]
    assert servo.motor.max_current == (0.0 if name == "FULLCONTACT" else 1.75)
    if name != "FULLCONTACT":
        assert servo.joint_effort_limit == pytest.approx(0.96)


def test_fullcontact_has_its_own_unrestricted_electrical_limit():
    cfg = microduck.MICRODUCK_FULLCONTACT_CFG
    servo = cfg.actuators["servos"]
    stall = 8.2 * servo.motor.kt / servo.motor.resistance
    assert servo.motor.max_current == 0.0
    assert servo.actuator_effort_limit == pytest.approx(stall)
    assert servo.joint_effort_limit == pytest.approx(stall)
    assert microduck.MICRODUCK_CFG.actuators["servos"].motor.max_current == 1.75
    assert microduck.MICRODUCK_CFG.actuators["servos"].joint_effort_limit is None


@pytest.mark.integration
@pytest.mark.parametrize(
    "cfg",
    [
        microduck.MICRODUCK_CFG,
        microduck.MICRODUCK_ALLCOLLISIONS_CFG,
        microduck.MICRODUCK_ROLLERS_CFG,
        microduck.MICRODUCK_WALK_BACKLASH_CFG,
        microduck.MICRODUCK_GROUNDCONTACT_BACKLASH_CFG,
        microduck.MICRODUCK_ROLLERS_BACKLASH_CFG,
        microduck.MICRODUCK_FULLCONTACT_CFG,
        microduck.MICRODUCK_FULLCONTACT_BACKLASH_CFG,
    ],
    ids=lambda cfg: Path(cfg.spawn.usd_path).stem,
)
def test_prebuilt_foot_sites_are_retained_without_becoming_colliders(cfg):
    stage = Usd.Stage.Open(prebuilt_usd_path(cfg.spawn.usd_path))
    sites = {p.GetName(): p for p in stage.Traverse() if p.GetName() in {"left_foot", "right_foot"}}
    assert set(sites) == {"left_foot", "right_foot"}
    for name, prim in sites.items():
        assert prim.IsA(UsdGeom.Sphere)
        assert not prim.HasAPI(UsdPhysics.CollisionAPI)
        if "fullcontact" in Path(cfg.spawn.usd_path).name:
            offset = microduck.MICRODUCK_FULLCONTACT_FOOT_SITE_OFFSETS[name.removesuffix("_foot")]
            assert prim.GetParent().GetName() == offset["body"]
            translation = UsdGeom.Xformable(prim).GetLocalTransformation().ExtractTranslation()
            assert tuple(translation) == pytest.approx(offset["pos"], abs=1e-7)
