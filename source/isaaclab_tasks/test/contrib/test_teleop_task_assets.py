# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check hard-coded teleoperation task paths against the robot USD assets the tasks spawn.

These tests read the published assets without a simulator. They skip when the asset server is
unreachable.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pxr import Usd, UsdPhysics

from isaaclab.utils.assets import retrieve_file_path

_ENV_ROBOT = "{ENV_REGEX_NS}/Robot"


def _open_asset(usd_path: str) -> Usd.Stage:
    try:
        local_path = retrieve_file_path(usd_path)
    except Exception as exc:  # The asset server is optional for local test runs.
        pytest.skip(f"Robot asset is unavailable: {exc}")
    stage = Usd.Stage.Open(local_path)
    assert stage is not None, local_path
    return stage


def _robot_paths(stage: Usd.Stage, pattern: str) -> list[Usd.Prim]:
    """Return prims of the asset's default prim (the spawned ``Robot``) that fully match ``pattern``."""
    root = stage.GetDefaultPrim()
    assert pattern.startswith(_ENV_ROBOT), pattern
    expression = re.compile(re.escape(str(root.GetPath())) + pattern[len(_ENV_ROBOT) :])
    return [
        prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if expression.fullmatch(str(prim.GetPath()))
    ]


@pytest.mark.parametrize("task", ["fixed_base", "locomanipulation"])
def test_g1_hand_contact_sensors_resolve_to_rigid_hand_links(task):
    """Newton matches whole paths, so the patterns must name the ``left_hand``/``right_hand`` parent prims."""
    from isaaclab_tasks.contrib.locomanip_pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
        FixedBaseUpperBodyIKG1EnvCfg,
    )
    from isaaclab_tasks.contrib.locomanip_pick_place.locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg

    cfg = FixedBaseUpperBodyIKG1EnvCfg() if task == "fixed_base" else LocomanipulationG1EnvCfg()
    stage = _open_asset(cfg.scene.robot.spawn.usd_path)
    matched = {}
    for side in ("left", "right"):
        prims = _robot_paths(stage, getattr(cfg.scene, f"{side}_hand_contact").prim_path)
        assert prims, side
        assert all(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in prims)
        matched[side] = {str(prim.GetPath()) for prim in prims}
    assert matched["left"].isdisjoint(matched["right"])


def test_gr1t2_kitless_kinematics_urdf_matches_the_spawned_asset():
    """The vendored Pink URDF must keep the spawned USD's movable joints and limits [rad]."""
    from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import PickPlaceGR1T2EnvCfg

    cfg = PickPlaceGR1T2EnvCfg()
    stage = _open_asset(cfg.scene.robot.spawn.usd_path)
    usd_limits = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            usd_limits[prim.GetName()] = (
                math.radians(joint.GetLowerLimitAttr().Get()),
                math.radians(joint.GetUpperLimitAttr().Get()),
            )
        elif prim.IsA(UsdPhysics.Joint) and not prim.IsA(UsdPhysics.FixedJoint):
            pytest.fail(f"Unexpected non-revolute movable joint {prim.GetPath()}")

    urdf_path = (
        Path(__import__("isaaclab_tasks").__file__).parent
        / "contrib/pick_place/assets/GR1T2_fourier_hand_6dof_kinematics.urdf"
    )
    urdf_limits = {}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        urdf_limits[joint.get("name")] = (float(limit.get("lower")), float(limit.get("upper")))

    assert set(urdf_limits) == set(usd_limits)
    for name, (lower, upper) in usd_limits.items():
        assert urdf_limits[name] == pytest.approx((lower, upper), abs=1.0e-4), name
