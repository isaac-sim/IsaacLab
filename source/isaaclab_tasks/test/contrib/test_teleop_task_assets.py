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


def _retrieve_asset(usd_path: str) -> str:
    try:
        return retrieve_file_path(usd_path)
    except (FileNotFoundError, RuntimeError) as exc:
        # The asset server is optional for local test runs: it may be unreachable or fail the download.
        pytest.skip(f"Robot asset is unavailable: {exc}")


def _open_asset(usd_path: str) -> Usd.Stage:
    local_path = _retrieve_asset(usd_path)
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


def test_gr1t2_kitless_kinematics_urdf_reproduces_the_asset_forward_kinematics():
    """Every URDF link must sit where the spawned USD's joint frames place it at an asymmetric pose [m, rad].

    Matching joint names and limits would not catch a moved joint origin or a reversed joint axis.
    """
    import numpy as np

    pin = pytest.importorskip("pinocchio")
    from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import PickPlaceGR1T2EnvCfg

    cfg = PickPlaceGR1T2EnvCfg()
    stage = _open_asset(cfg.scene.robot.spawn.usd_path)

    def joint_frame(position, orientation) -> np.ndarray:
        transform = np.eye(4)
        quaternion = pin.Quaternion(orientation.GetReal(), *orientation.GetImaginary())
        transform[:3, :3] = quaternion.normalized().toRotationMatrix()
        transform[:3, 3] = position
        return transform

    # Child link -> (joint name, parent link, joint frame on parent, joint frame on child, joint axis).
    joints, limits = {}, {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            joints[joint.GetBody1Rel().GetTargets()[0].name] = (
                prim.GetName(),
                joint.GetBody0Rel().GetTargets()[0].name,
                joint_frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                joint_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
                np.eye(3)["XYZ".index(joint.GetAxisAttr().Get())],
            )
            limits[prim.GetName()] = (
                math.radians(joint.GetLowerLimitAttr().Get()),
                math.radians(joint.GetUpperLimitAttr().Get()),
            )

    def usd_link_poses(angles: dict[str, float]) -> dict[str, np.ndarray]:
        """Compose ``parent @ frame0 @ R(axis, angle) @ inv(frame1)`` from ``base_link``."""
        poses = {"base_link": np.eye(4)}

        def pose(link: str) -> np.ndarray:
            if link not in poses:
                name, parent, frame0, frame1, axis = joints[link]
                rotation = np.eye(4)
                rotation[:3, :3] = pin.AngleAxis(angles[name], axis).toRotationMatrix()
                poses[link] = pose(parent) @ frame0 @ rotation @ np.linalg.inv(frame1)
            return poses[link]

        for link in joints:
            pose(link)
        return poses

    def pose_change(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:3, 3] - b[:3, 3]) + np.linalg.norm(pin.log3(a[:3, :3].T @ b[:3, :3])))

    # Spread the joints over their ranges at golden-ratio fractions, so neighbouring joints and the
    # two sides take different angles.
    angles = {
        name: lower + (upper - lower) * (0.2 + 0.6 * ((index * 0.618034) % 1.0))
        for index, (name, (lower, upper)) in enumerate(sorted(limits.items()))
    }
    expected = usd_link_poses(angles)

    urdf_path = (
        Path(__import__("isaaclab_tasks").__file__).parent
        / "contrib/pick_place/assets/GR1T2_fourier_hand_6dof_kinematics.urdf"
    )
    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    configuration = pin.neutral(model)
    for name, angle in angles.items():
        assert model.existJointName(name), name
        joint = model.joints[model.getJointId(name)]
        assert joint.nq == 1, name
        configuration[joint.idx_q] = angle
    pin.framesForwardKinematics(model, data, configuration)
    base_inverse = data.oMf[model.getFrameId("base_link")].inverse()
    for link, usd_pose in expected.items():
        assert model.existFrame(link), link
        urdf_pose = (base_inverse * data.oMf[model.getFrameId(link)]).homogeneous
        assert urdf_pose[:3, 3] == pytest.approx(usd_pose[:3, 3], abs=1.0e-5), link
        assert np.linalg.norm(pin.log3(urdf_pose[:3, :3].T @ usd_pose[:3, :3])) < 5.0e-4, link

    # The pose must expose a reversed arm axis: negating any one arm joint moves its Pink wrist frame.
    wrists = cfg.actions.upper_body_ik.target_eef_link_names
    for side in ("left", "right"):
        wrist = wrists[f"{side}_wrist"]
        arm_joints = [name for name in angles if re.fullmatch(rf"{side}_(shoulder|elbow|wrist)_\w+_joint", name)]
        assert len(arm_joints) == 7, arm_joints
        for name in arm_joints:
            reversed_pose = usd_link_poses(angles | {name: -angles[name]})[wrist]
            assert pose_change(reversed_pose, expected[wrist]) > 1.0e-2, name


@pytest.mark.parametrize("task", ["fixed_base_g1", "gr1t2", "gr1t2_waist"])
def test_xr_head_exclusions_partition_only_visual_head_meshes(task):
    """OVXR attach fails closed when an exclusion matches nothing; partition the real spawned asset."""
    from pxr import UsdGeom

    from isaaclab_tasks.contrib.locomanip_pick_place.fixed_base_upper_body_ik_g1_env_cfg import (
        FixedBaseUpperBodyIKG1EnvCfg,
    )
    from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import PickPlaceGR1T2EnvCfg
    from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_waist_enabled_env_cfg import (
        PickPlaceGR1T2WaistEnabledEnvCfg,
    )

    cfg_type = {
        "fixed_base_g1": FixedBaseUpperBodyIKG1EnvCfg,
        "gr1t2": PickPlaceGR1T2EnvCfg,
        "gr1t2_waist": PickPlaceGR1T2WaistEnabledEnvCfg,
    }[task]
    cfg = cfg_type()
    if cfg.xr is None or not cfg.xr.excluded_visual_meshes:
        pytest.skip("The teleoperation configuration is unavailable")
    from isaaclab_teleop.xr_scene_partition import partition_visual_meshes

    # Reference the robot where the single-environment scene spawns it.
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/envs/env_0/Robot").GetReferences().AddReference(
        _retrieve_asset(cfg.scene.robot.spawn.usd_path)
    )
    session_sublayers = list(stage.GetSessionLayer().subLayerPaths)
    with partition_visual_meshes(stage, cfg.xr.excluded_visual_meshes) as paths:
        assert paths
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            assert prim.IsA(UsdGeom.Mesh), path
            assert "head" in path and "collisions" not in path.split("/")
            assert not prim.HasAPI(UsdPhysics.CollisionAPI)
            assert UsdGeom.PrimvarsAPI(prim).GetPrimvar("omni:scenePartition").Get() == "xr_excluded"
    # The partition is temporary: exiting removes its session sublayer and the authored primvars.
    assert list(stage.GetSessionLayer().subLayerPaths) == session_sublayers
    for path in paths:
        assert not UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).HasPrimvar("omni:scenePartition")
