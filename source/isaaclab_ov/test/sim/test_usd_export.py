# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export of a running OVPhysX articulation back to USD.

Mirrors ``isaaclab_physx/test/sim/test_usd_export.py``. Both backends share the authoring layer in
:mod:`isaaclab.sim.usd_export`; what differs, and what these tests cover, is how each recovers the
prim path of every body and joint. OVPhysX resolves them from the stage rather than from the view.

Runs kitless -- :class:`~isaaclab.sim.SimulationContext` is driven directly with
``physics=OvPhysxCfg()``, with no ``AppLauncher`` boot.
"""

from __future__ import annotations

import math

import pytest
import torch
from isaaclab_ov.assets import Articulation
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.sim.usd_export import export_articulation_to_usd, resolve_articulation_prim_paths

from pxr import Usd, UsdPhysics

from isaaclab.sim import SimulationCfg, build_simulation_context

from isaaclab_assets import FRANKA_PANDA_CFG

# Values written into the simulation after the stage is parsed. They are deliberately unlike any
# plausible authored default, so a stage still carrying the spawn-time value is unmistakable.
OVERRIDE_STIFFNESS = 1234.5
OVERRIDE_DAMPING = 67.25
OVERRIDE_MASS = 9.75


@pytest.fixture
def sim(request):
    """Simulation context backed by OVPhysX rather than PhysX."""
    device = request.getfixturevalue("device")
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, dt=1.0 / 60.0, gravity=(0.0, 0.0, 0.0))
    with build_simulation_context(device=device, sim_cfg=sim_cfg, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


def _spawn(sim) -> Articulation:
    """Spawn a single articulation whose stage authors drive gains and limits to be overridden."""
    articulation = Articulation(FRANKA_PANDA_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    assert articulation.is_initialized, "articulation failed to initialize; the test would be vacuous"
    return articulation


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_resolved_prim_paths_exist_and_cover_the_articulation(sim, device):
    """Every body and joint resolves to a real prim under the articulation root.

    OVPhysX does not record provenance on its view, so the paths are recovered from the stage. If
    that resolution drifts, the export silently describes a different prim -- or none.
    """
    articulation = _spawn(sim)
    paths = resolve_articulation_prim_paths(articulation)

    assert len(paths.bodies) == articulation.num_bodies
    assert len(paths.joints) == articulation.num_joints
    for path in paths.bodies + paths.joints:
        assert articulation.stage.GetPrimAtPath(path).IsValid(), f"resolved path {path} is not a prim"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_export_captures_overrides_the_stage_never_saw(sim, device, tmp_path):
    """Values written through the backend reach the exported stage.

    This is the property the exporter exists for. Drive gains and masses written after the stage is
    parsed live only in the backend's buffers, so an exporter that simply saved the stage would emit
    the spawn-time values and look correct while describing a simulation that is not running.
    """
    articulation = _spawn(sim)

    stiffness = torch.full_like(articulation.data.joint_stiffness.torch, OVERRIDE_STIFFNESS)
    damping = torch.full_like(articulation.data.joint_damping.torch, OVERRIDE_DAMPING)
    articulation.write_joint_stiffness_to_sim_index(stiffness=stiffness)
    articulation.write_joint_damping_to_sim_index(damping=damping)
    masses = articulation.data.body_mass.torch.clone()
    masses[:] = OVERRIDE_MASS
    articulation.set_masses_index(masses=masses)
    sim.step()
    articulation.update(sim.get_physics_dt())

    out = tmp_path / "exported.usda"
    export_articulation_to_usd(articulation, str(out))
    exported = Usd.Stage.Open(str(out))
    paths = resolve_articulation_prim_paths(articulation)

    checked = 0
    for path in paths.joints:
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

    for path in paths.bodies:
        mass_api = UsdPhysics.MassAPI(exported.GetPrimAtPath(path))
        assert mass_api.GetMassAttr().Get() == pytest.approx(OVERRIDE_MASS, rel=1e-4), (
            f"body {path} exported the stage's spawn-time mass instead of the simulated one"
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
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
    paths = resolve_articulation_prim_paths(articulation)

    checked = 0
    for backend_index, path in enumerate(paths.joints):
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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_export_rejects_an_environment_the_view_does_not_have(sim, device, tmp_path):
    """Selecting a missing environment fails instead of exporting another one's state."""
    articulation = _spawn(sim)
    with pytest.raises(ValueError, match="out of range"):
        export_articulation_to_usd(articulation, str(tmp_path / "out.usda"), env_index=99)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_exported_armature_is_readable_by_physx(sim, device, tmp_path):
    """Armature is authored where the solver reads it, since UsdPhysics has no home for it."""
    articulation = _spawn(sim)
    out = tmp_path / "exported.usda"
    export_articulation_to_usd(articulation, str(out))

    exported = Usd.Stage.Open(str(out))
    armature = articulation.data.joint_armature.torch[0]
    joint_row = {name: index for index, name in enumerate(articulation.joint_names)}
    paths = resolve_articulation_prim_paths(articulation)

    checked = 0
    for backend_index, path in enumerate(paths.joints):
        row = joint_row.get(articulation.backend_joint_names[backend_index])
        if row is None:
            continue
        attribute = exported.GetPrimAtPath(path).GetAttribute("physxJoint:armature")
        assert attribute, f"joint {path} carries no physxJoint:armature in the export"
        assert attribute.Get() == pytest.approx(float(armature[row]), abs=1e-5)
        checked += 1
    assert checked > 0, "fixture produced no joints; the test would be vacuous"
