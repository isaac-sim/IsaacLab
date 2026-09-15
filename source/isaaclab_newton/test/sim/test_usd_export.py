# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent fresh-Newton validation of fixed scene export."""

import newton
import numpy as np
import pytest

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors


def _load(path: str) -> tuple[newton.Model, dict]:
    """Import ``path`` the way Isaac Lab does, returning the model and the importer's result maps.

    Newton's importer colours a shape only from a bound visual material; Isaac Lab then aligns the
    builder's colours with the stage's preserved ``primvars:displayColor``.
    """
    builder = newton.ModelBuilder()
    stage = Usd.Stage.Open(str(path))
    options = stage.GetRootLayer().customLayerData.get("isaaclab:newtonImportOptions", {})
    from newton.usd import SchemaResolverNewton, SchemaResolverPhysx

    stage_info = builder.add_usd(str(path), schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()], **options)
    replace_newton_builder_shape_colors(builder, stage)
    return builder.finalize(), stage_info


def _capture_environment_physics(model, world, contact_pairs=None):
    """Canonical physical configuration, keyed by entity identity rather than backend ordering."""
    result = {}
    indices = {}
    for kind in ("body", "joint", "shape"):
        worlds = getattr(model, f"{kind}_world").numpy()
        indices[kind] = [i for i, w in enumerate(worlds) if w < 0 or w == world]
    names = {kind: getattr(model, f"{kind}_label") for kind in indices}
    fields = {
        "body": ("body_mass", "body_inertia", "body_com", "body_flags"),
        "joint": ("joint_type", "joint_X_p", "joint_X_c", "joint_enabled"),
        "shape": (
            "shape_type",
            "shape_transform",
            "shape_scale",
            "shape_margin",
            "shape_gap",
            "shape_material_mu",
            "shape_material_restitution",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_kf",
            "shape_material_ka",
            "shape_material_mu_torsional",
            "shape_material_mu_rolling",
        ),
    }
    for kind, selected in indices.items():
        for index in selected:
            if kind == "joint" and int(model.joint_type.numpy()[index]) == int(newton.JointType.FREE):
                continue
            if kind == "shape" and not int(model.shape_flags.numpy()[index]) & int(newton.ShapeFlags.COLLIDE_SHAPES):
                continue
            path = names[kind][index]
            for field in fields[kind]:
                value = getattr(model, field).numpy()[index]
                result[kind, path, field] = np.asarray(value).copy()
            if kind == "joint":
                for field in ("joint_parent", "joint_child"):
                    body = int(getattr(model, field).numpy()[index])
                    result[kind, path, field] = names["body"][body] if body >= 0 else "world"
                start, end = model.joint_qd_start.numpy()[index : index + 2]
                for field in (
                    "joint_axis",
                    "joint_target_ke",
                    "joint_target_kd",
                    "joint_limit_lower",
                    "joint_limit_upper",
                    "joint_limit_ke",
                    "joint_limit_kd",
                    "joint_armature",
                    "joint_friction",
                    "joint_effort_limit",
                    "joint_velocity_limit",
                    "joint_target_mode",
                ):
                    result[kind, path, field] = getattr(model, field).numpy()[start:end].copy()
            if kind == "shape":
                body = int(model.shape_body.numpy()[index])
                result[kind, path, "body"] = names["body"][body] if body >= 0 else "world"
                flags = int(model.shape_flags.numpy()[index])
                result[kind, path, "collision"] = bool(flags & int(newton.ShapeFlags.COLLIDE_SHAPES))
                source = model.shape_source[index]
                if source is not None and hasattr(source, "vertices"):
                    result[kind, path, "vertices"] = np.asarray(source.vertices).copy()
                    result[kind, path, "indices"] = np.asarray(source.indices).copy()
    included = {
        i for i in indices["shape"] if int(model.shape_flags.numpy()[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES)
    }
    if contact_pairs is None:
        contact_pairs = model.shape_contact_pairs.numpy()
    allowed = {tuple(sorted(map(int, pair))) for pair in contact_pairs}
    pairs = set()
    selected = sorted(included)
    for offset, first in enumerate(selected):
        for second in selected[offset + 1 :]:
            if (first, second) not in allowed or (
                model.shape_body.numpy()[first] < 0 and model.shape_body.numpy()[second] < 0
            ):
                pairs.add(tuple(sorted((names["shape"][first], names["shape"][second]))))
    result["filters"] = pairs
    result["gravity"] = model.gravity.numpy()[world if model.world_count else -1].copy()
    return result


@pytest.mark.parametrize("env_id,num_envs", [(0, 1), (37, 64)])
def test_fixed_scene_configuration_uses_shared_export(tmp_path, env_id, num_envs):
    """Normal cfg initialization exports every body, fixed actuator property and authored collider."""
    from isaaclab_newton.physics import NewtonCfg, XPBDSolverCfg

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.test.utils.usd_export import make_fixed_scene_cfg

    cfg = make_fixed_scene_cfg(tmp_path)
    cfg.num_envs = num_envs
    simulation_cfg = SimulationCfg(
        device="cpu", dt=1 / 120, gravity=(0.2, -0.1, -4.0), physics=NewtonCfg(solver_cfg=XPBDSolverCfg(iterations=13))
    )
    output = tmp_path / "fixed_scene.usda"
    expected = {}
    expected_state = {}

    with build_simulation_context(sim_cfg=simulation_cfg) as sim:
        scene = InteractiveScene(cfg)
        sim.reset()
        scene.reset_to_default()
        sim.forward()
        scene.update(0.0)
        if num_envs > 1:
            import torch

            box = scene.rigid_objects["box"]
            factors = 1 + torch.arange(num_envs, device=box.device)[:, None] / 100
            box.set_masses_index(masses=box.data.body_mass.torch.clone() * factors)
            box.set_inertias_index(inertias=box.data.body_inertia.torch.clone() * factors[..., None])
        manager = scene.sim.physics_manager
        pairs = manager._collision_pipeline.shape_pairs_filtered.numpy()
        expected.update(_capture_environment_physics(manager.get_model(), env_id, pairs))
        state = manager.get_state_0()
        for index, path in enumerate(manager.get_model().body_label):
            if manager.get_model().body_world.numpy()[index] not in (-1, env_id):
                continue
            expected_state[path] = (state.body_q.numpy()[index].copy(), state.body_qd.numpy()[index].copy())
        scene.export_to_usd(str(output), env_id=env_id)
    stage = Usd.Stage.Open(str(output))
    bodies = {str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    assert bodies == {
        f"/World/envs/env_{env_id}/Robot/Base",
        f"/World/envs/env_{env_id}/Robot/Link",
        f"/World/envs/env_{env_id}/Box",
        f"/World/envs/env_{env_id}/CollectedFirst",
        f"/World/envs/env_{env_id}/CollectedSecond",
    }
    joint = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Robot/Hinge")
    assert UsdPhysics.DriveAPI(joint, "angular").GetStiffnessAttr().Get() == pytest.approx(83 * np.pi / 180)
    assert joint.GetAttribute("state:angular:physics:position").Get() == pytest.approx(np.degrees(0.21))
    for name, mass in (("Box", 2.5), ("CollectedFirst", 1.5), ("CollectedSecond", 3.5)):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/{name}")
        assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == pytest.approx(
            mass * (1 + env_id / 100 if name == "Box" else 1)
        )
    for path in ("/World/Ground", "/World/Light", f"/World/envs/env_{env_id}/Table"):
        assert stage.GetPrimAtPath(path)
    fresh, info = _load(str(output))
    # Construct the fresh driver exclusively from export metadata, not the source cfg.
    driver = dict(stage.GetRootLayer().customLayerData["isaaclab:newtonDriver"])
    assert driver.pop("solver") == "xpbd"
    solver = newton.solvers.SolverXPBD(fresh, **driver)
    assert solver.iterations == 13
    assert stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:timeStepsPerSecond").Get() == 120
    actual = _capture_environment_physics(fresh, 0)
    assert expected.keys() == actual.keys()
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            if value.dtype.kind == "f":
                np.testing.assert_allclose(actual[key], value, rtol=3e-5, atol=1e-6, err_msg=str(key))
            else:
                np.testing.assert_array_equal(actual[key], value, err_msg=str(key))
        else:
            assert actual[key] == value, key
    state = fresh.state()
    newton.eval_fk(fresh, fresh.joint_q, fresh.joint_qd, state)
    assert set(fresh.body_label) == set(expected_state)
    for index, path in enumerate(fresh.body_label):
        pose, velocity = expected_state[path]
        np.testing.assert_allclose(state.body_q.numpy()[index], pose, rtol=3e-5, atol=1e-6, err_msg=path)
        np.testing.assert_allclose(state.body_qd.numpy()[index], velocity, rtol=3e-5, atol=1e-6, err_msg=path)


@pytest.mark.parametrize("bound", [True, False, "complete"])
def test_fixed_contact_materials_preserve_distinct_collider_values(bound, monkeypatch):
    from types import SimpleNamespace

    import warp as wp
    from isaaclab_newton.physics import NewtonCfg, NewtonManager, XPBDSolverCfg

    from pxr import UsdShade

    from isaaclab.sim.usd_export import UsdWriter

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    shared = UsdShade.Material.Define(stage, "/Shared")
    shared.GetPrim().CreateAttribute("newton:contactStiffness", Sdf.ValueTypeNames.Float).Set(17)
    shapes = [UsdGeom.Cube.Define(stage, "/" + name).GetPrim() for name in ("A", "B")]
    for prim in shapes:
        UsdPhysics.CollisionAPI.Apply(prim)
        if bound:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(shared, materialPurpose="physics")
    values = {
        name: np.array([a, b])
        for name, a, b in (
            ("shape_gap", 0.01, 0.02),
            ("shape_margin", 0.03, 0.04),
            ("shape_material_ke", 1.0, 2.0),
            ("shape_material_kd", 3.0, 4.0),
            ("shape_material_kf", 5.0, 6.0),
            ("shape_material_ka", 7.0, 8.0),
            ("shape_material_mu_torsional", 0.1, 0.2),
            ("shape_material_mu_rolling", 0.3, 0.4),
            ("shape_material_mu", 0.5, 0.6),
            ("shape_material_restitution", 0.7, 0.8),
        )
    }
    if bound == "complete":
        physics = UsdPhysics.MaterialAPI.Apply(shared.GetPrim())
        physics.CreateStaticFrictionAttr().Set(0.5)
        physics.CreateDynamicFrictionAttr().Set(0.5)
        physics.CreateRestitutionAttr().Set(0.7)
        for name, value in values.items():
            if name.startswith("shape_material_"):
                value[1] = value[0]
        values["shape_material_ke"][:] = 17.0
        for name, value in (
            ("contactStiffness", 17.0),
            ("contactDamping", 3.0),
            ("contactFrictionGain", 5.0),
            ("contactAdhesion", 7.0),
            ("torsionalFriction", 0.1),
            ("rollingFriction", 0.3),
        ):
            shared.GetPrim().CreateAttribute("newton:" + name, Sdf.ValueTypeNames.Float).Set(value)
    model = SimpleNamespace(**{name: wp.array(value, dtype=wp.float32, device="cpu") for name, value in values.items()})
    model.shape_label = [str(prim.GetPath()) for prim in shapes]
    model.joint_qd_start = wp.array([0], dtype=wp.int32, device="cpu")
    model.joint_world = wp.array([], dtype=wp.int32, device="cpu")
    for name in ("joint_target_ke", "joint_target_kd", "joint_target_mode"):
        setattr(model, name, wp.array([], dtype=wp.float32, device="cpu"))
    monkeypatch.setattr(NewtonManager, "get_model", lambda: model)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene = SimpleNamespace(
        physics_scene_path="/physicsScene",
        sim=SimpleNamespace(
            get_physics_dt=lambda: 1 / 60, cfg=SimpleNamespace(physics=NewtonCfg(solver_cfg=XPBDSolverCfg()))
        ),
    )
    NewtonManager.author_fixed_configuration(UsdWriter(stage), scene)
    for i, prim in enumerate(shapes):
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        assert material.GetPrim().GetAttribute("newton:contactStiffness").Get() == values["shape_material_ke"][i]
        if bound == "complete":
            assert material.GetPath() == shared.GetPath()
            assert not prim.GetChild("ExportPhysicsMaterial")
        if not bound:
            assert UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Get() == pytest.approx(
                0.5 + 0.1 * i
            )
    assert shared.GetPrim().GetAttribute("newton:contactStiffness").Get() == 17
    from newton.usd import SchemaResolverNewton, SchemaResolverPhysx

    builder = newton.ModelBuilder()
    info = builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
    model = builder.finalize(device="cpu")
    for index, name in enumerate(("A", "B")):
        row = info["path_shape_map"]["/" + name]
        assert model.shape_material_ke.numpy()[row] == values["shape_material_ke"][index]
