# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build a fixed tap with a physical spring button and a separate cup station floor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

HERE = Path(__file__).resolve().parent
ASSET_PATH = HERE / "assets/tap.usda"
TAP_ORIGIN_M = (0.28, -0.30, 0.0)
CUP_STATION_POSITION_M = (0.38, -0.30, 0.013)
NOZZLE_POSITION_M = (0.38, -0.30, 0.290)
BUTTON_TOP_POSITION_M = (0.28, -0.30, 0.100)
BUTTON_PRESSED_POSITION_M = (0.28, -0.30, 0.096)
BUTTON_BODY_CENTER_LOCAL_M = (0.0, 0.0, 0.096)
BUTTON_TRAVEL_M = 0.004
BUTTON_PRESS_THRESHOLD_M = 0.002
BUTTON_RELEASE_THRESHOLD_M = 0.001
BUTTON_RADIUS_M = 0.022
BUTTON_MASS_KG = 0.02
BUTTON_STIFFNESS_N_M = 300.0
BUTTON_DAMPING_N_S_M = 3.0
BUTTON_EFFORT_LIMIT_N = 10.0
STATION_FLOOR_RADIUS_M = 0.067
STATION_FLOOR_TOP_M = 0.012
BASE_BODY_PATH = "/Asset/Base"
BUTTON_BODY_PATH = "/Asset/Button"
BUTTON_JOINT_PATH = "/Asset/TapButton"
BUTTON_JOINT_NAME = "TapButton"

# Each part contributes a simple primitive collider and its component mass.
# Overlapping fixed fittings are intentional; the spring button is a separate link.
PARTS = (
    ("Foot", "box", (-0.005, 0.045, 0.009), (0.070, 0.180, 0.018), "charcoal"),
    ("CupStationFloor", "cylinder", (0.100, 0.0, 0.006), (0.067, 0.012, "Z"), "charcoal"),
    ("Mast", "cylinder", (0.0, 0.105, 0.172), (0.019, 0.308, "Z"), "steel"),
    ("ArmX", "cylinder", (0.050, 0.105, 0.326), (0.015, 0.100, "X"), "steel"),
    ("ArmY", "cylinder", (0.100, 0.0525, 0.326), (0.015, 0.105, "Y"), "steel"),
    ("Nozzle", "cylinder", (0.100, 0.0, 0.308), (0.015, 0.036, "Z"), "steel"),
    ("ControlStand", "cylinder", (0.0, 0.0, 0.053), (0.026, 0.070, "Z"), "charcoal"),
)
COLORS = {
    "steel": (0.62, 0.69, 0.76),
    "charcoal": (0.065, 0.085, 0.105),
    "button": (0.075, 0.50, 0.78),
    "white": (0.88, 0.95, 1.0),
    "outlet": (0.018, 0.024, 0.030),
}


def _material(stage: Usd.Stage, name: str) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, f"/Asset/Materials/{name}")
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("Surface"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*COLORS[name]))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.23 if name == "steel" else 0.38)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.85 if name == "steel" else 0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _primitive(stage, path, kind, center, dimensions, material, *, collision=True):
    if kind == "box":
        shape = UsdGeom.Cube.Define(stage, path)
        shape.CreateSizeAttr(1.0)
        shape.AddTranslateOp().Set(Gf.Vec3d(*center))
        shape.AddScaleOp().Set(Gf.Vec3f(*dimensions))
    else:
        radius, height, axis = dimensions
        shape = UsdGeom.Cylinder.Define(stage, path)
        shape.CreateRadiusAttr(radius)
        shape.CreateHeightAttr(height)
        shape.CreateAxisAttr(axis)
        shape.AddTranslateOp().Set(Gf.Vec3d(*center))
    shape.CreateDisplayColorAttr([Gf.Vec3f(*COLORS[material])])
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(
        UsdShade.Material(stage.GetPrimAtPath(f"/Asset/Materials/{material}"))
    )
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    return shape


def _base_mass_properties() -> tuple[float, np.ndarray, np.ndarray]:
    """Return additive component mass [kg], COM [m], and inertia tensor [kg m²]."""
    total_mass = 1.2
    volumes, inertias, centers = [], [], []
    for _, kind, center, dimensions, _ in PARTS:
        centers.append(center)
        if kind == "box":
            x, y, z = dimensions
            volumes.append(x * y * z)
            inertias.append(np.array([y * y + z * z, x * x + z * z, x * x + y * y]) / 12)
        else:
            radius, height, axis = dimensions
            volumes.append(math.pi * radius * radius * height)
            inertia = np.full(3, (3 * radius * radius + height * height) / 12)
            inertia["XYZ".index(axis)] = radius * radius / 2
            inertias.append(inertia)
    masses = total_mass * np.asarray(volumes) / sum(volumes)
    centers = np.asarray(centers)
    com = np.sum(masses[:, None] * centers, axis=0) / total_mass
    tensor = np.zeros((3, 3))
    for mass, center, inertia in zip(masses, centers, inertias, strict=True):
        delta = center - com
        tensor += mass * (np.diag(inertia) + np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
    return total_mass, com, tensor


def _body(stage, path, mass, center, tensor):
    body = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    values, axes = np.linalg.eigh(tensor)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    rotation = Gf.Matrix3d(*axes.T.reshape(-1).tolist()).ExtractRotation().GetQuat()
    physics = UsdPhysics.MassAPI.Apply(body.GetPrim())
    physics.CreateMassAttr(mass)
    physics.CreateCenterOfMassAttr(Gf.Vec3f(*center))
    physics.CreateDiagonalInertiaAttr(Gf.Vec3f(*values))
    physics.CreatePrincipalAxesAttr(Gf.Quatf(rotation))
    return body


def build_tap_asset(*, output: Path = ASSET_PATH) -> dict:
    """Write the tap articulation in SI units and verify its authored structure.

    The root is placed at :data:`TAP_ORIGIN_M`. The button travels downward with
    negative joint displacement [m], and a force drive returns it to zero.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Asset")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    for name in COLORS:
        _material(stage, name)
    mass, com, inertia = _base_mass_properties()
    _body(stage, BASE_BODY_PATH, mass, com, inertia)
    for name, kind, center, dimensions, material in PARTS:
        _primitive(stage, f"{BASE_BODY_PATH}/{name}", kind, center, dimensions, material)
    # Dark outlet and light accents are display-only, with no hidden colliders.
    _primitive(
        stage,
        f"{BASE_BODY_PATH}/Outlet",
        "cylinder",
        (0.100, 0.0, 0.2898),
        (0.010, 0.0003, "Z"),
        "outlet",
        collision=False,
    )
    _primitive(
        stage,
        f"{BASE_BODY_PATH}/StationMark",
        "cylinder",
        (0.100, 0.0, 0.0122),
        (0.052, 0.0003, "Z"),
        "steel",
        collision=False,
    )
    mount = UsdPhysics.FixedJoint.Define(stage, "/Asset/Mount")
    mount.CreateBody1Rel().SetTargets([Sdf.Path(BASE_BODY_PATH)])
    radius, height, mass = BUTTON_RADIUS_M, 0.008, BUTTON_MASS_KG
    button_tensor = np.diag([mass * (3 * radius * radius + height * height) / 12] * 2 + [mass * radius * radius / 2])
    button = _body(stage, BUTTON_BODY_PATH, mass, (0.0, 0.0, 0.0), button_tensor)
    button.AddTranslateOp().Set(Gf.Vec3d(*BUTTON_BODY_CENTER_LOCAL_M))
    _primitive(stage, f"{BUTTON_BODY_PATH}/Face", "cylinder", (0.0, 0.0, 0.0), (radius, height, "Z"), "button")
    _primitive(
        stage,
        f"{BUTTON_BODY_PATH}/Indicator",
        "cylinder",
        (0.0, 0.0, 0.00415),
        (0.004, 0.0002, "Z"),
        "white",
        collision=False,
    )
    joint = UsdPhysics.PrismaticJoint.Define(stage, BUTTON_JOINT_PATH)
    joint.CreateAxisAttr("Z")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(BASE_BODY_PATH)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(BUTTON_BODY_PATH)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*BUTTON_BODY_CENTER_LOCAL_M))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    joint.CreateLowerLimitAttr(-BUTTON_TRAVEL_M)
    joint.CreateUpperLimitAttr(0.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(BUTTON_STIFFNESS_N_M)
    drive.CreateDampingAttr(BUTTON_DAMPING_N_S_M)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateMaxForceAttr(BUTTON_EFFORT_LIMIT_N)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    content = stage.GetRootLayer().ExportToString()
    if not output.is_file() or output.read_text() != content:
        temporary = output.with_name(".tap.tmp.usda")
        temporary.write_text(content)
        temporary.replace(output)
    return inspect_tap_asset(output)


def inspect_tap_asset(path: Path = ASSET_PATH) -> dict:
    """Check body/joint identities, positive mass/inertia, and button limits [m]."""
    stage = Usd.Stage.Open(str(path))
    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert {str(prim.GetPath()) for prim in bodies} == {BASE_BODY_PATH, BUTTON_BODY_PATH}
    for prim in bodies:
        physics = UsdPhysics.MassAPI(prim)
        assert physics.GetMassAttr().Get() > 0
        assert min(physics.GetDiagonalInertiaAttr().Get()) > 0
    joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(BUTTON_JOINT_PATH))
    assert joint and joint.GetAxisAttr().Get() == "Z"
    assert math.isclose(joint.GetLowerLimitAttr().Get(), -BUTTON_TRAVEL_M, abs_tol=1e-9)
    assert joint.GetUpperLimitAttr().Get() == 0.0
    collisions = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(collisions) == len(PARTS) + 1
    return {
        "asset": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "origin_m": TAP_ORIGIN_M,
        "body_paths": [BASE_BODY_PATH, BUTTON_BODY_PATH],
        "joint_path": BUTTON_JOINT_PATH,
        "joint_limits_m": [-BUTTON_TRAVEL_M, 0.0],
        "button_top_world_m": BUTTON_TOP_POSITION_M,
        "button_pressed_top_world_m": BUTTON_PRESSED_POSITION_M,
        "press_threshold_depression_m": BUTTON_PRESS_THRESHOLD_M,
        "release_threshold_depression_m": BUTTON_RELEASE_THRESHOLD_M,
        "nozzle_outlet_world_m": NOZZLE_POSITION_M,
        "cup_station_pose_m": CUP_STATION_POSITION_M,
        "station_floor_top_m": STATION_FLOOR_TOP_M,
        "station_floor_radius_m": STATION_FLOOR_RADIUS_M,
        "collision_paths": collisions,
        "rigid_body_count": 2,
        "particle_count": 0,
        "physical_rollout_performed": False,
    }


def tap_articulation_cfg(prim_path: str = "{ENV_REGEX_NS}/Tap"):
    """Return the fixed tap articulation with its spring drive [N/m, N s/m]."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim import UsdFileCfg

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=UsdFileCfg(usd_path=str(ASSET_PATH)),
        init_state=ArticulationCfg.InitialStateCfg(pos=TAP_ORIGIN_M, joint_pos={BUTTON_JOINT_NAME: 0.0}),
        actuators={
            "button": ImplicitActuatorCfg(
                joint_names_expr=[BUTTON_JOINT_NAME],
                stiffness=BUTTON_STIFFNESS_N_M,
                damping=BUTTON_DAMPING_N_S_M,
                joint_effort_limit=BUTTON_EFFORT_LIMIT_N,
            )
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(inspect_tap_asset() if args.check else build_tap_asset(), indent=2))
