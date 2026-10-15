# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A motor-driven, two-axis sand tray. Particle motion remains solver-owned."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import MISSING

from pxr import Gf, Sdf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass


@sim_utils.clone
def spawn_tilting_testbed(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Author a fixed base, roll cradle, and pitch tray with five collision boxes."""
    stage = sim_utils.get_current_stage()
    root = sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    root.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    for name, mass, inertia in (
        ("base", 100.0, (100.0, 100.0, 100.0)),
        ("cradle", 100.0, (100.0, 100.0, 100.0)),
        ("tray", cfg.mass, (cfg.mass, 3.0 * cfg.mass, 3.0 * cfg.mass)),
    ):
        body = UsdGeom.Xform.Define(stage, f"{prim_path}/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mass_api = UsdPhysics.MassAPI.Apply(body)
        mass_api.CreateMassAttr(mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    fixed = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/base_fixed_joint")
    fixed.CreateBody1Rel().SetTargets([f"{prim_path}/base"])
    for name, parent, child, axis in (("roll", "base", "cradle", "X"), ("pitch", "cradle", "tray", "Y")):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/{name}")
        joint.CreateBody0Rel().SetTargets([f"{prim_path}/{parent}"])
        joint.CreateBody1Rel().SetTargets([f"{prim_path}/{child}"])
        joint.CreateAxisAttr(axis)
        joint.CreateLowerLimitAttr(-cfg.joint_limit_deg)
        joint.CreateUpperLimitAttr(cfg.joint_limit_deg)
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)

    length, width, depth = cfg.sand_bed_size
    wall = cfg.wall_thickness
    # The joint pivot is at the rigid supporting floor's upper surface.
    geometry = [
        ("floor", (length + 2 * wall, width + 2 * wall, cfg.floor_thickness), (0.0, 0.0, -cfg.floor_thickness / 2)),
        ("front", (wall, width + 2 * wall, depth), ((length + wall) / 2, 0.0, depth / 2)),
        ("back", (wall, width + 2 * wall, depth), (-(length + wall) / 2, 0.0, depth / 2)),
        ("left", (length, wall, depth), (0.0, (width + wall) / 2, depth / 2)),
        ("right", (length, wall, depth), (0.0, -(width + wall) / 2, depth / 2)),
    ]
    for name, size, position in geometry:
        box = sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.37), roughness=0.8),
        )
        box.func(f"{prim_path}/tray/{name}", box, translation=position)
    return root


@configclass
class TiltingTestbedSpawnerCfg(SpawnerCfg):
    """Geometry and mechanical limits for the powered tray."""

    func: Callable = spawn_tilting_testbed
    sand_bed_size: tuple[float, float, float] = MISSING
    """Interior sand-bed length, width, and depth [m], supplied by the scene config."""
    wall_thickness: float = MISSING
    """Tray wall thickness [m]."""
    floor_thickness: float = MISSING
    """Tray floor thickness [m]."""
    mass: float = 250.0
    """Tray mass [kg], excluding the simulated sand."""
    joint_limit_deg: float = 45.0
    """Symmetric mechanical joint limit [deg]."""


@configclass
class TestbedMotionCfg:
    """Smooth target motion; the joints, tray, and particles are integrated by physics."""

    roll_range: tuple[float, float] = (-math.radians(10), math.radians(10))
    pitch_range: tuple[float, float] = (-math.radians(10), math.radians(10))
    """Joint target ranges [rad], each containing the level orientation (zero)."""
    periods: tuple[float, float] = (12.0, 17.0)
    """Roll and pitch periods [s]."""
    settling_time: float = 2.0
    """Hold the tray level after reset [s]."""
    ramp_time: float = 3.0
    """Smoothly increase oscillation amplitude from zero [s]."""
    debug_vis: bool = True
    """Show the tray normal (blue) against the direction opposite gravity (green)."""
    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TestbedTilt",
        markers={
            "tray_normal": BLUE_ARROW_X_MARKER_CFG.markers["arrow"].replace(scale=(0.8, 0.12, 0.12)),
            "opposite_gravity": GREEN_ARROW_X_MARKER_CFG.markers["arrow"].replace(scale=(1.0, 0.08, 0.08)),
        },
    )

    def validate(self):
        for bounds in (self.roll_range, self.pitch_range):
            if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or not bounds[0] <= 0 <= bounds[1]:
                raise ValueError("Testbed angle ranges must be finite, ordered, and contain zero.")
        if len(self.periods) != 2 or any(not math.isfinite(value) or value <= 0 for value in self.periods):
            raise ValueError("Testbed periods must contain two positive finite durations.")
        if not math.isfinite(self.settling_time) or self.settling_time < 0:
            raise ValueError("Testbed settling_time must be finite and non-negative.")
        if not math.isfinite(self.ramp_time) or self.ramp_time <= 0:
            raise ValueError("Testbed ramp_time must be finite and positive.")
