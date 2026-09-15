# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fresh OVPhysX load of a complete fixed single-environment scene."""

import numpy as np
import ovphysx
import ovstage
import pytest
import torch
from isaaclab_ov import tensor_types as TT
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.sim.views import OvPhysxView
from isaaclab_ov.stage import create_ovstage

from pxr import Usd, UsdPhysics

from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils.usd_export import (
    assert_physics_structure_equal,
    capture_physics_structure,
    make_fixed_scene_cfg,
)


@pytest.mark.parametrize("env_id,num_envs", [(0, 1), (37, 64)])
def test_fixed_environment_round_trip(tmp_path, env_id, num_envs):
    cfg = make_fixed_scene_cfg(tmp_path)
    cfg.num_envs = num_envs
    expected, structure = {}, {}
    art_props = (
        TT.BODY_MASS,
        TT.BODY_INERTIA,
        TT.BODY_COM_POSE,
        TT.BODY_DISABLE_GRAVITY,
        TT.DOF_STIFFNESS,
        TT.DOF_DAMPING,
        TT.DOF_LIMIT,
        TT.DOF_MAX_VELOCITY,
        TT.DOF_MAX_FORCE,
        TT.DOF_ARMATURE,
        TT.DOF_FRICTION_PROPERTIES,
        TT.SHAPE_FRICTION_AND_RESTITUTION,
        TT.CONTACT_OFFSET,
        TT.REST_OFFSET,
    )
    rigid_props = (
        TT.RIGID_BODY_MASS,
        TT.RIGID_BODY_INERTIA,
        TT.RIGID_BODY_COM_POSE,
        TT.RIGID_BODY_DISABLE_GRAVITY,
        TT.RIGID_BODY_SHAPE_FRICTION_AND_RESTITUTION,
        TT.RIGID_BODY_CONTACT_OFFSET,
        TT.RIGID_BODY_REST_OFFSET,
    )

    output = tmp_path / "fixed.usda"
    simulation_cfg = SimulationCfg(device="cpu", physics=OvPhysxCfg(), dt=1 / 120, gravity=(0.2, -0.1, -4.0))
    with build_simulation_context(sim_cfg=simulation_cfg) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        scene.reset_to_default()
        sim.forward()
        scene.update(0.0)
        for group in (scene.articulations, scene.rigid_objects, scene.rigid_object_collections):
            for asset in group.values():
                masses = asset.data.body_mass.torch.clone()
                inertias = asset.data.body_inertia.torch.clone()
                factors = 1.1 + torch.arange(num_envs, device=asset.device) / 100
                asset.set_masses_index(masses=masses * factors.reshape((-1,) + (1,) * (masses.ndim - 1)))
                asset.set_inertias_index(inertias=inertias * factors.reshape((-1,) + (1,) * (inertias.ndim - 1)))
        if num_envs == 1:
            structure.update(capture_physics_structure(scene.sim.stage))
        structure["/World/envs/env_0/Robot/FixedRoot", "localPose0Position"] = np.array(cfg.robot.init_state.pos)
        for group in (scene.articulations, scene.rigid_objects, scene.rigid_object_collections):
            for asset in group.values():
                articulation = asset in scene.articulations.values()
                view = asset.root_view
                props = art_props if articulation else rigid_props
                for row, root in enumerate(view.prim_paths):
                    if f"/World/envs/env_{env_id}/" not in root:
                        continue
                    expected[root] = (
                        {token: view.get_attribute(token).numpy()[row].copy() for token in props},
                        list(view.body_names) if articulation else [],
                        list(view.dof_names) if articulation else [],
                    )
        before = scene.sim.stage.GetRootLayer().ExportToString()
        scene.export_to_usd(str(output), env_id=env_id, preserve_source_contacts=True)
        assert scene.sim.stage.GetRootLayer().ExportToString() == before
    stage = Usd.Stage.Open(str(output))
    if num_envs == 1:
        assert_physics_structure_equal(structure, capture_physics_structure(stage))
    assert all(not stage.GetPrimAtPath(f"/World/envs/env_{i}") for i in range(num_envs) if i != env_id)
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]) == 5
    assert stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Table") and stage.GetPrimAtPath("/World/Ground")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            np.testing.assert_array_equal(UsdPhysics.RigidBodyAPI(prim).GetVelocityAttr().Get(), [0, 0, 0])
            np.testing.assert_array_equal(UsdPhysics.RigidBodyAPI(prim).GetAngularVelocityAttr().Get(), [0, 0, 0])
    assert stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:timeStepsPerSecond").Get() == 120
    physics_scene = UsdPhysics.Scene(stage.GetPrimAtPath("/physicsScene"))
    np.testing.assert_allclose(
        np.array(physics_scene.GetGravityDirectionAttr().Get()) * physics_scene.GetGravityMagnitudeAttr().Get(),
        [0.2, -0.1, -4],
        atol=1e-6,
    )
    fresh_stage = create_ovstage("fixed_export_round_trip")
    fresh = ovphysx.PhysX()
    try:
        ovstage.population.open_usd_from_string(
            fresh_stage, stage.ExportToString(), ordinal=1, domains=ovstage.PopulationDomain.ALL
        )
        fresh_stage.advance_write_floor(ordinal=1).wait()
        fresh.attach_ovstage(fresh_stage, read_ordinal=1)
        for path, (props, body_names, dof_names) in expected.items():
            view = OvPhysxView(fresh, prim_paths=[path], device="cpu", tensor_types=list(props), eager=True)
            try:
                assert view.count == 1
                for token, value in props.items():
                    actual = view.get_attribute(token).numpy()[0]
                    if token.name.startswith(("ARTICULATION_BODY_", "ARTICULATION_DOF_")):
                        names = view.body_names if token.name.startswith("ARTICULATION_BODY_") else view.dof_names
                        reference = body_names if token.name.startswith("ARTICULATION_BODY_") else dof_names
                        assert set(names) == set(reference)
                        actual = actual[[names.index(name) for name in reference]]
                    if token in (TT.BODY_COM_POSE, TT.RIGID_BODY_COM_POSE):
                        actual, value = actual[..., :3], value[..., :3]
                    if value.dtype.kind in "biu":
                        np.testing.assert_array_equal(actual, value, err_msg=f"{path}: {token.name}")
                    else:
                        np.testing.assert_allclose(actual, value, rtol=3e-4, atol=1e-5, err_msg=f"{path}: {token.name}")
            finally:
                view.close()
    finally:
        fresh.release()
        fresh_stage.destroy()
