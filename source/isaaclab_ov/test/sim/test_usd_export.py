# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX USD export: an OVPhysX import of the exported file reproduces the initialized engine."""

import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from pxr import Usd, UsdPhysics

# The file loader runs in a spawned process and must import by module name.
sys.path.insert(0, str(Path(__file__).parent))


@pytest.mark.parametrize("num_envs,env_id", [(1, 0), (2, 1)])
def test_importing_the_export_reproduces_the_initialized_engine(tmp_path, num_envs, env_id):
    """Every exported value of a fresh OVPhysX import equals the live engine's, including config overrides."""
    from isaaclab_ov.physics import OvPhysxCfg, OvPhysxManager
    from isaaclab_ov.sim.usd_export.usd_exporter import read, readings
    from usd_export_read_import import read_import

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.sim.usd_export.physx_usd_exporter import physics_prims
    from isaaclab.test.utils.usd_export import make_export_scene_cfg

    cfg = make_export_scene_cfg(tmp_path, num_envs=num_envs)
    sim_cfg = SimulationCfg(device="cpu", dt=1 / 120, gravity=(0.2, -0.1, -4.0), physics=OvPhysxCfg())
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        output = scene.export_to_usd(str(tmp_path / "exported.usda"), env_id=env_id)
        stage = Usd.Stage.Open(output)
        articulations, bodies = physics_prims(stage)
        live = readings(stage, read(OvPhysxManager.get_physx_instance(), articulations, bodies, "cpu"))
        gravity = np.asarray(OvPhysxManager.get_gravity())
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        imported = pool.submit(read_import, stage.Flatten().ExportToString(), articulations, bodies).result()
    loaded = readings(stage, imported)

    hinge = live[f"/World/envs/env_{env_id}/Robot/Hinge", "angular"]
    assert hinge["stiffness"] == pytest.approx(83.0)
    assert hinge["armature"] == pytest.approx(0.023)
    exported_hinge = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot/Hinge")
    assert exported_hinge.GetAttribute("physxJoint:armature").HasAuthoredValue()
    # Zero joint friction equals the schema default, so the export leaves it unset.
    assert not exported_hinge.GetAttribute("physxJointAxis:angular:staticFrictionEffort").HasAuthoredValue()
    assert live.keys() == loaded.keys()
    for key, values in live.items():
        for name, value in values.items():
            expected = loaded[key][name]
            np.testing.assert_allclose(expected, value, rtol=1e-4, atol=1e-6, err_msg=f"{key} {name}")
    scene_prim = UsdPhysics.Scene(stage.GetPrimAtPath("/physicsScene"))
    exported = np.asarray(scene_prim.GetGravityDirectionAttr().Get()) * scene_prim.GetGravityMagnitudeAttr().Get()
    np.testing.assert_allclose(exported, gravity, rtol=1e-5)


def test_read_bindings_cover_ovphysx_properties():
    """Every OVPhysX articulation and rigid-body binding is exported, compared, or deliberately excluded."""
    from isaaclab_ov import tensor_types as TT
    from isaaclab_ov.sim.usd_export.usd_exporter import ARTICULATION_TENSORS, RIGID_BODY_TENSORS

    read = {*ARTICULATION_TENSORS.values(), *RIGID_BODY_TENSORS.values()}
    excluded = {
        # state, targets and forces are not part of the export
        TT.ROOT_POSE,
        TT.ROOT_VELOCITY,
        TT.LINK_POSE,
        TT.LINK_VELOCITY,
        TT.LINK_ACCELERATION,
        TT.DOF_POSITION,
        TT.DOF_VELOCITY,
        TT.DOF_POSITION_TARGET,
        TT.DOF_VELOCITY_TARGET,
        TT.DOF_ACTUATION_FORCE,
        TT.LINK_WRENCH,
        TT.RIGID_BODY_POSE,
        TT.RIGID_BODY_VELOCITY,
        TT.RIGID_BODY_ACCELERATION,
        TT.RIGID_BODY_WRENCH,
        # derived from exported properties
        TT.BODY_INV_MASS,
        TT.BODY_INV_INERTIA,
        TT.RIGID_BODY_INV_MASS,
        TT.RIGID_BODY_INV_INERTIA,
        TT.JACOBIAN,
        TT.MASS_MATRIX,
        TT.CORIOLIS,
        TT.GRAVITY_FORCE,
        TT.LINK_INCOMING_JOINT_FORCE,
        TT.DOF_PROJECTED_JOINT_FORCE,
        # drive type and model come from the USD drive schema and are not written at runtime
        TT.DOF_DRIVE_TYPE,
        TT.DOF_DRIVE_MODEL,
        # articulation shapes and tendons have no per-shape or per-tendon USD identity to write
        TT.CONTACT_OFFSET,
        TT.REST_OFFSET,
        TT.SHAPE_FRICTION_AND_RESTITUTION,
        TT.FIXED_TENDON_STIFFNESS,
        TT.FIXED_TENDON_DAMPING,
        TT.FIXED_TENDON_LIMIT_STIFFNESS,
        TT.FIXED_TENDON_LIMIT,
        TT.FIXED_TENDON_REST_LENGTH,
        TT.FIXED_TENDON_OFFSET,
        TT.SPATIAL_TENDON_STIFFNESS,
        TT.SPATIAL_TENDON_DAMPING,
        TT.SPATIAL_TENDON_LIMIT_STIFFNESS,
        TT.SPATIAL_TENDON_OFFSET,
    }
    bindings = {
        value
        for name, value in vars(TT).items()
        if name.isupper() and "DEFORMABLE" not in name and isinstance(value, type(TT.DOF_STIFFNESS))
    }
    unclassified = sorted(str(binding) for binding in bindings - read - excluded)
    assert not unclassified, f"Unclassified bindings: {unclassified}"
