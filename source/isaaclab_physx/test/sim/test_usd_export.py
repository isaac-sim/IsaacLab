# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from isaaclab.test.utils import DeviceScope, resolve_test_sim_device, test_devices

simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

"""Rest everything follows."""

import math

import pytest
import torch
from isaaclab_physx.assets import Articulation
from isaaclab_physx.sim.usd_export import export_articulation_to_usd

from pxr import Usd, UsdPhysics

from isaaclab.sim import build_simulation_context

from isaaclab_assets import FRANKA_PANDA_CFG

# Values written into the simulation after the stage is parsed. They are deliberately unlike any
# plausible authored default, so a stage still carrying the spawn-time value is unmistakable.
OVERRIDE_STIFFNESS = 1234.5
OVERRIDE_DAMPING = 67.25
OVERRIDE_MASS = 9.75
# Static effort, dynamic effort and viscous coefficient of the friction model; distinct and non-zero
# so a component landing in the wrong attribute is caught.
OVERRIDE_FRICTION = (0.375, 0.25, 0.125)


@pytest.fixture
def sim(request):
    """Simulation context for the requested device."""
    with build_simulation_context(
        device=request.getfixturevalue("device"), auto_add_lighting=True, gravity_enabled=False
    ) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


def _spawn(sim) -> Articulation:
    """Spawn a single articulation whose stage authors drive gains and limits to be overridden."""
    articulation = Articulation(FRANKA_PANDA_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    assert articulation.is_initialized, "articulation failed to initialize; the test would be vacuous"
    return articulation


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_export_captures_overrides_the_stage_never_saw(sim, device, tmp_path):
    """Values written through the tensor API reach the exported stage.

    This is the property the exporter exists for. Drive gains and masses written after the stage is
    parsed live only in PhysX buffers, so an exporter that simply saved the stage would emit the
    spawn-time values and look correct while describing a simulation that is not running.
    """
    articulation = _spawn(sim)

    stiffness = torch.full_like(articulation.data.joint_stiffness.torch, OVERRIDE_STIFFNESS)
    damping = torch.full_like(articulation.data.joint_damping.torch, OVERRIDE_DAMPING)
    articulation.write_joint_stiffness_to_sim_index(stiffness=stiffness, full_data=True)
    articulation.write_joint_damping_to_sim_index(damping=damping, full_data=True)
    masses = articulation.data.body_mass.torch.clone()
    masses[:] = OVERRIDE_MASS
    articulation.set_masses_index(masses=masses)
    sim.step()
    articulation.update(sim.get_physics_dt())

    out = tmp_path / "exported.usda"
    export_articulation_to_usd(articulation, str(out))

    exported = Usd.Stage.Open(str(out))
    joint_paths = [str(path) for path in articulation.root_view.dof_paths[0]]
    checked = 0
    for path in joint_paths:
        prim = exported.GetPrimAtPath(path)
        assert prim.IsValid(), f"joint prim {path} missing from the exported stage"
        token = {"PhysicsPrismaticJoint": "linear", "PhysicsRevoluteJoint": "angular"}.get(prim.GetTypeName())
        if token is None:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, token)
        assert drive, f"joint {path} carries no drive in the export"
        # Angular drive gains are per degree on the stage and per radian in the simulation.
        gain_scale = math.pi / 180.0 if token == "angular" else 1.0
        assert drive.GetStiffnessAttr().Get() == pytest.approx(OVERRIDE_STIFFNESS * gain_scale, rel=1e-4), (
            f"joint {path} exported the stage's spawn-time stiffness instead of the simulated one"
        )
        assert drive.GetDampingAttr().Get() == pytest.approx(OVERRIDE_DAMPING * gain_scale, rel=1e-4)
        checked += 1
    assert checked > 0, "fixture produced no drivable joints; the test would be vacuous"

    body_paths = [str(path) for path in articulation.root_view.link_paths[0]]
    for path in body_paths:
        mass_api = UsdPhysics.MassAPI(exported.GetPrimAtPath(path))
        assert mass_api.GetMassAttr().Get() == pytest.approx(OVERRIDE_MASS, rel=1e-4), (
            f"body {path} exported the stage's spawn-time mass instead of the simulated one"
        )


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_export_writes_joint_limits_in_stage_units(sim, device, tmp_path):
    """Joint limits are authored in the unit the joint schema declares, not the simulation's.

    The simulation reports revolute limits in radians while USD authors them in degrees, so an
    export that copies the number across unconverted narrows every limit by a factor of 57.
    """
    articulation = _spawn(sim)
    out = tmp_path / "exported.usda"
    export_articulation_to_usd(articulation, str(out))

    exported = Usd.Stage.Open(str(out))
    limits = articulation.data.joint_pos_limits.torch[0]
    joint_row = {name: index for index, name in enumerate(articulation.joint_names)}

    checked = 0
    for backend_index, path in enumerate([str(p) for p in articulation.root_view.dof_paths[0]]):
        prim = exported.GetPrimAtPath(path)
        if prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue
        row = joint_row.get(articulation.backend_joint_names[backend_index])
        if row is None:
            continue
        lower = prim.GetAttribute("physics:lowerLimit").Get()
        if lower is None or not math.isfinite(float(limits[row][0])):
            continue
        assert lower == pytest.approx(math.degrees(float(limits[row][0])), abs=1e-3), (
            f"joint {path} exported its lower limit in radians rather than degrees"
        )
        checked += 1
    assert checked > 0, "fixture produced no revolute joints with finite limits; the test would be vacuous"


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_export_rejects_an_environment_the_view_does_not_have(sim, device, tmp_path):
    """Selecting a missing environment fails instead of exporting another one's state."""
    articulation = _spawn(sim)
    with pytest.raises(ValueError, match="out of range"):
        export_articulation_to_usd(articulation, str(tmp_path / "out.usda"), env_index=99)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_exported_armature_is_readable_by_physx(sim, device, tmp_path):
    """Armature is authored where PhysX reads it, since UsdPhysics has no home for it."""
    articulation = _spawn(sim)
    out = tmp_path / "exported.usda"
    export_articulation_to_usd(articulation, str(out))

    exported = Usd.Stage.Open(str(out))
    armature = articulation.data.joint_armature.torch[0]
    joint_row = {name: index for index, name in enumerate(articulation.joint_names)}

    checked = 0
    for backend_index, path in enumerate([str(p) for p in articulation.root_view.dof_paths[0]]):
        prim = exported.GetPrimAtPath(path)
        row = joint_row.get(articulation.backend_joint_names[backend_index])
        if row is None:
            continue
        attribute = prim.GetAttribute("physxJoint:armature")
        assert attribute, f"joint {path} carries no physxJoint:armature in the export"
        assert attribute.Get() == pytest.approx(float(armature[row]), abs=1e-5)
        # the per-axis schema, applied for friction, shadows the joint-level value once present
        token = {"PhysicsPrismaticJoint": "linear", "PhysicsRevoluteJoint": "angular"}.get(prim.GetTypeName())
        if token is not None:
            axis_attribute = prim.GetAttribute(f"physxJointAxis:{token}:armature")
            assert axis_attribute, f"joint {path} carries no physxJointAxis:{token}:armature in the export"
            assert axis_attribute.Get() == pytest.approx(float(armature[row]), abs=1e-5)
        checked += 1
    assert checked > 0, "fixture produced no joints; the test would be vacuous"


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_exported_friction_is_authored_on_the_drive_axis(sim, device, tmp_path):
    """Friction lands on the drive axis' per-axis PhysX schema, the only place the runtimes read it from.

    The legacy ``physxJoint:jointFriction`` scalar parses into nothing on current runtimes, so a stage
    carrying only that reimports with zero friction.
    """
    articulation = _spawn(sim)
    static, dynamic, viscous = (
        torch.full_like(articulation.data.joint_friction_coeff.torch, value) for value in OVERRIDE_FRICTION
    )
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=static,
        joint_dynamic_friction_coeff=dynamic,
        joint_viscous_friction_coeff=viscous,
        full_data=True,
    )
    sim.step()
    articulation.update(sim.get_physics_dt())

    out = tmp_path / "exported.usda"
    export_articulation_to_usd(articulation, str(out))
    exported = Usd.Stage.Open(str(out))

    names = ("staticFrictionEffort", "dynamicFrictionEffort", "viscousFrictionCoefficient")
    checked = 0
    for path in [str(p) for p in articulation.root_view.dof_paths[0]]:
        prim = exported.GetPrimAtPath(path)
        token = {"PhysicsPrismaticJoint": "linear", "PhysicsRevoluteJoint": "angular"}.get(prim.GetTypeName())
        if token is None:
            continue
        assert f"PhysxJointAxisAPI:{token}" in prim.GetAppliedSchemas(), f"joint {path} lacks PhysxJointAxisAPI:{token}"
        # the viscous coefficient is per unit angular rate, so it follows the drive gains onto degrees
        gain_scale = math.pi / 180.0 if token == "angular" else 1.0
        expected = (*OVERRIDE_FRICTION[:2], OVERRIDE_FRICTION[2] * gain_scale)
        for name, value in zip(names, expected):
            attribute = prim.GetAttribute(f"physxJointAxis:{token}:{name}")
            assert attribute, f"joint {path} carries no physxJointAxis:{token}:{name} in the export"
            assert attribute.Get() == pytest.approx(value, abs=1e-5)
        checked += 1
    assert checked > 0, "fixture produced no drivable joints; the test would be vacuous"


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_export_leaves_the_live_stage_untouched(sim, device, tmp_path):
    """The export authors onto a flattened snapshot, never onto the stage the simulation reads.

    On PhysX, applying a schema to a prim that is an articulation root invalidates every articulation
    view on the stage for the rest of the session, so an in-place export would break the very
    simulation it describes. The snapshot carries the new schema; the live prim does not.
    """
    articulation = _spawn(sim)
    out = tmp_path / "exported.usda"
    export_articulation_to_usd(articulation, str(out))
    exported = Usd.Stage.Open(str(out))

    checked = 0
    for path in [str(p) for p in articulation.root_view.dof_paths[0]]:
        live = articulation.stage.GetPrimAtPath(path)
        token = {"PhysicsPrismaticJoint": "linear", "PhysicsRevoluteJoint": "angular"}.get(live.GetTypeName())
        if token is None:
            continue
        schema = f"PhysxJointAxisAPI:{token}"
        assert schema in exported.GetPrimAtPath(path).GetAppliedSchemas(), f"snapshot of {path} lacks {schema}"
        assert schema not in live.GetAppliedSchemas(), f"live prim {path} was edited by the export"
        checked += 1
    assert checked > 0, "fixture produced no drivable joints; the test would be vacuous"
    assert articulation.root_view is not None and list(articulation.joint_names), "view unreadable after export"
