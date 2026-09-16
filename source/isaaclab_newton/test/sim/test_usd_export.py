# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent fresh-Newton validation of fixed scene export."""

import newton
import numpy as np
import pytest

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


def _load(path: str, device="cpu", solver_name="mujoco") -> tuple[newton.Model, dict]:
    """Use native USD import; cable supplements are read by their existing asset owner."""
    from isaaclab_newton.assets.cable_object.cable_object import CableObject
    from isaaclab_newton.assets.physics_properties import NewtonContactData
    from isaaclab_newton.physics import NewtonManager
    from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

    builder = newton.ModelBuilder()
    solver_type = {
        "mujoco": newton.solvers.SolverMuJoCo,
        "xpbd": newton.solvers.SolverXPBD,
        "kamino": newton.solvers.SolverKamino,
        "vbd": newton.solvers.SolverVBD,
    }[solver_name]
    solver_type.register_custom_attributes(builder)
    stage = Usd.Stage.Open(path)
    # Reuse the normal terrain adapter where native USD import has no heightfield support.
    terrain_paths = NewtonManager._inject_terrain_heightfields(stage, builder, root_paths=("/",))
    for row, shape_path in enumerate(builder.shape_label):
        NewtonContactData.restore_fixed_configuration(stage.GetPrimAtPath(shape_path), builder, row)
    info = builder.add_usd(
        path,
        schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton(), SchemaResolverPhysx()],
        return_deformable_results=True,
        ignore_paths=terrain_paths,
        **stage.GetRootLayer().customLayerData.get("isaaclab:newtonImportOptions", {}),
    )
    CableObject.restore_fixed_configuration(stage, builder, info.get("path_cable_map", {}))
    if solver_name == "vbd":
        builder.color()
    return builder.finalize(device=device), info


def _make_driver(model, name, driver=None):
    """Construct a native solver; only optional MJWarp settings are restored."""
    import json

    if name == "mujoco":
        return newton.solvers.SolverMuJoCo(model, **json.loads((driver or {}).get("options", "{}")))
    if name == "kamino":
        return newton.solvers.SolverKamino(model)
    return {"xpbd": newton.solvers.SolverXPBD, "vbd": newton.solvers.SolverVBD}[name](model)


def _capture_mujoco_physics(solver, world):
    """Capture native solver properties using Newton lineage, including material and DOF buffers."""
    import warp as wp

    model = solver.model
    native = solver.mj_model if solver.use_mujoco_cpu else solver.mjw_model
    world = world if solver.mjc_body_to_newton.shape[0] > 1 else 0
    result = {}

    def values(owner, name):
        value = getattr(owner, name)
        if isinstance(value, wp.array):
            value = value.numpy()
            template = solver.mj_model.opt if owner is native.opt else solver.mj_model
            if value.ndim > np.asarray(getattr(template, name, 0.0)).ndim:
                value = value[world if len(value) > 1 else 0]
        return np.asarray(value).copy()

    properties = {
        "body": ("mass", "inertia", "ipos", "iquat", "gravcomp"),
        "geom": (
            "type",
            "size",
            "pos",
            "quat",
            "condim",
            "priority",
            "solmix",
            "solref",
            "solimp",
            "friction",
            "margin",
            "gap",
        ),
        "jnt": ("type", "axis", "pos", "range", "stiffness", "margin", "solref", "solimp", "actfrcrange"),
        "dof": ("armature", "frictionloss", "damping", "solref", "solimp"),
    }
    maps = {
        "body": solver.mjc_body_to_newton,
        "geom": solver.mjc_geom_to_newton_shape,
        "jnt": solver.mjc_jnt_to_newton_jnt,
        "dof": solver.mjc_dof_to_newton_dof,
    }
    starts = model.joint_qd_start.numpy()

    def joint_identity(index):
        if int(model.joint_type.numpy()[index]) == int(newton.JointType.FREE):
            return model.body_label[int(model.joint_child.numpy()[index])] + "/__free"
        return model.joint_label[index]

    for kind, fields in properties.items():
        mapping = maps[kind].numpy()[world]
        for name in fields:
            field = kind + "_" + name
            if not hasattr(native, field):
                continue
            array = values(native, field)
            for index, source in enumerate(mapping):
                if source < 0:
                    continue
                if kind == "dof":
                    joint = int(np.searchsorted(starts, source, side="right") - 1)
                    identity = (joint_identity(joint), int(source - starts[joint]))
                else:
                    labels = getattr(model, {"body": "body_label", "geom": "shape_label", "jnt": "joint_label"}[kind])
                    identity = joint_identity(source) if kind == "jnt" else labels[source]
                    if kind == "jnt":
                        # A D6 USD joint becomes several native joints; never overwrite another axis's reference.
                        dof = int(solver.mjc_jnt_to_newton_dof.numpy()[world, index])
                        identity = (identity, dof - int(starts[source]))
                result[kind, identity, field] = array[index].copy()
    # Principal axes are non-unique for repeated moments; compare link-frame inertia.
    from scipy.spatial.transform import Rotation

    for key in list(result):
        if key[0] == "body" and key[2] == "body_iquat":
            quat = result.pop(key)
            inertia_key = ("body", key[1], "body_inertia")
            rotation = Rotation.from_quat(np.roll(quat, -1)).as_matrix()
            result[inertia_key] = rotation @ np.diag(result[inertia_key]) @ rotation.T
    # Bit assignments can change after pruning worlds; compare the resulting relationships.
    masks = values(native, "geom_contype"), values(native, "geom_conaffinity")
    bodies = values(native, "geom_bodyid")
    excludes = set(map(int, np.asarray(solver.mj_model.exclude_signature)))
    mapping = solver.mjc_geom_to_newton_shape.numpy()[world]
    for first, source in enumerate(mapping):
        if source < 0:
            continue
        for second in range(first + 1, len(mapping)):
            if mapping[second] < 0:
                continue
            body0, body1 = sorted((int(bodies[first]), int(bodies[second])))
            enabled = bool(
                (int(masks[0][first]) & int(masks[1][second])) or (int(masks[0][second]) & int(masks[1][first]))
            )
            enabled &= body0 != body1 and ((body0 << 16) + body1) not in excludes
            pair = tuple(sorted((model.shape_label[source], model.shape_label[mapping[second]])))
            result["collision_pair", pair] = np.asarray(enabled)
    for name in (
        "iterations",
        "ls_iterations",
        "solver",
        "integrator",
        "cone",
        "tolerance",
        "ls_tolerance",
        "ccd_tolerance",
        "gravity",
        "density",
        "viscosity",
        "wind",
        "magnetic",
        "disableflags",
        "enableflags",
        "impratio_invsqrt",
    ):
        if hasattr(native.opt, name):
            result["option", name] = values(native.opt, name)
    return result


def _capture_kamino_physics(solver, world):
    """Capture Kamino's converted body, joint, geometry and material-pair configuration."""
    native = solver._model_kamino
    colliders = {
        label
        for i, label in enumerate(solver.model.shape_label)
        if int(solver.model.shape_flags.numpy()[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES)
    }
    result = {}

    def joint_label(index):
        if native.joints.num_dofs.numpy()[index] == 6:
            child = int(native.joints.bid_F.numpy()[index])
            return native.bodies.label[child] + "/__free"
        return native.joints.label[index]

    for kind, fields in {
        "bodies": ("m_i", "i_I_i", "i_r_com_i", "is_immovable"),
        "geoms": ("type", "flags", "params", "offset"),
        "joints": ("dof_type", "act_type", "num_dofs", "num_coords"),
    }.items():
        container = getattr(native, kind)
        worlds = container.wid.numpy()
        for name in fields:
            data = getattr(container, name).numpy()
            for index, label in enumerate(container.label):
                if kind == "geoms" and label not in colliders:
                    continue
                if worlds[index] < 0 or worlds[index] == world:
                    result[kind, joint_label(index) if kind == "joints" else label, name] = data[index].copy()
    joints = native.joints
    starts = joints.dofs_offset.numpy()
    for index, label in enumerate(joints.label):
        label = joint_label(index)
        if joints.wid.numpy()[index] not in (-1, world):
            continue
        for name in ("bid_B", "bid_F"):
            body = int(getattr(joints, name).numpy()[index])
            result["joints", label, name] = native.bodies.label[body] if body >= 0 else "world"
        for name in (
            "q_j_min",
            "q_j_max",
            "dq_j_max",
            "tau_j_max",
            "a_j",
            "b_j",
            "f_j",
            "k_p_j",
            "k_d_j",
            "dof_act_types",
            "dof_act_paths",
        ):
            result["joints", label, name] = getattr(joints, name).numpy()[starts[index] : starts[index + 1]].copy()
    shapes = [
        (i, label)
        for i, label in enumerate(native.geoms.label)
        if native.geoms.wid.numpy()[i] in (-1, world) and label in colliders
    ]
    materials = native.geoms.material.numpy()
    for offset, (i, label) in enumerate(shapes):
        body = int(native.geoms.bid.numpy()[i])
        result["geoms", label, "body"] = native.bodies.label[body] if body >= 0 else "world"
        for j, other in shapes[offset:]:
            row, column = sorted((int(materials[i]), int(materials[j])), reverse=True)
            index = row * (row + 1) // 2 + column
            for name in ("static_friction", "dynamic_friction", "restitution"):
                if hasattr(native.material_pairs, name):
                    result["pair", tuple(sorted((label, other))), name] = (
                        getattr(native.material_pairs, name).numpy()[index].copy()
                    )
    return result


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
                elif isinstance(source, newton.Heightfield):
                    for field in ("nrow", "ncol", "hx", "hy", "min_z", "max_z", "_data"):
                        result[kind, path, "heightfield", field] = np.asarray(getattr(source, field)).copy()
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


def _assert_physical_value_equal(key, actual, expected):
    """Compare discrete identities exactly and physical tensors at their meaningful scale."""
    if isinstance(expected, (np.ndarray, np.generic)):
        if expected.dtype.kind == "f":
            if key[-1] in {"geom_quat", "shape_transform", "joint_X_p", "joint_X_c", "offset"}:
                from scipy.spatial.transform import Rotation

                a, b = np.asarray(actual), np.asarray(expected)
                if a.shape == (4,):
                    a, b = np.roll(a, -1), np.roll(b, -1)
                elif a.shape == (7,):
                    np.testing.assert_allclose(a[:3], b[:3], rtol=3e-5, atol=1e-6, err_msg=str(key))
                    a, b = a[3:], b[3:]
                else:
                    np.testing.assert_allclose(a, b, rtol=3e-5, atol=1e-6, err_msg=str(key))
                    return
                np.testing.assert_allclose(
                    Rotation.from_quat(a).as_matrix(), Rotation.from_quat(b).as_matrix(), atol=3e-5, err_msg=str(key)
                )
            elif key[-1] in {"body_inertia", "i_I_i"}:
                # Principal-frame float roundoff scales with the full tensor, including zero entries.
                error = np.linalg.norm(np.asarray(actual) - expected)
                assert error <= 1e-6 + 3e-5 * np.linalg.norm(expected), (key, error)
            else:
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-6, err_msg=str(key))
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=str(key))
    else:
        assert actual == expected, (key, actual, expected)


@pytest.mark.parametrize(
    "env_id,num_envs,solver_name",
    [(env_id, num_envs, solver) for solver in ("xpbd", "mujoco", "kamino") for env_id, num_envs in ((0, 1), (1, 2))]
    + [(0, 1, "xpbd_passive"), (1, 2, "xpbd_cartesian"), (1, 2, "vbd")],
)
def test_fixed_scene_configuration_uses_shared_export(tmp_path, env_id, num_envs, solver_name):
    """Normal cfg initialization exports every body, fixed actuator property and authored collider."""
    import warp as wp
    from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg, VBDSolverCfg, XPBDSolverCfg

    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, build_simulation_context
    from isaaclab.test.utils.usd_export import make_fixed_scene_cfg

    cfg = make_fixed_scene_cfg(tmp_path)
    if solver_name == "vbd":
        import isaaclab.sim as sim_utils
        from isaaclab.assets import CableObjectCfg

        cfg.cable = CableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cable",
            spawn=sim_utils.CableCfg(
                positions=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.4, 0.1, 0.0)],
                physics_material=sim_utils.CableMaterialCfg(thickness=0.02, density=1000.0),
            ),
            init_state=CableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)),
        )
    cartesian = solver_name == "xpbd_cartesian"
    if cartesian:
        source = Usd.Stage.Open(cfg.robot.spawn.usd_path)
        joint = UsdPhysics.Joint.Define(source, "/Robot/Hinge").GetPrim()
        joint.RemoveAPI(UsdPhysics.DriveAPI, "angular")
        for name in ("physics:lowerLimit", "physics:upperLimit", "physics:axis"):
            joint.RemoveProperty(name)
        for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
            limit = UsdPhysics.LimitAPI.Apply(joint, axis)
            angle = {"rotX": 0.2, "rotZ": 0.7}.get(axis)
            limit.CreateLowAttr().Set(-np.degrees(angle) if angle else 1)
            limit.CreateHighAttr().Set(np.degrees(angle) if angle else -1)
        source.GetRootLayer().Save()
        cfg.robot.init_state.joint_pos = {".*": 0.0}
        cfg.robot.init_state.joint_vel = {".*": 0.0}
        cfg.robot.actuators["hinge"].joint_names_expr = ["Hinge.*"]
        solver_name = "xpbd"
    passive = solver_name == "xpbd_passive"
    if passive or solver_name == "mujoco":
        source = Usd.Stage.Open(cfg.robot.spawn.usd_path)
        joint = source.GetPrimAtPath("/Robot/Hinge")
        if passive:
            cfg.robot.actuators = {}
            joint.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            solver_name = "xpbd"
        else:
            joint.CreateAttribute("mjc:armature", Sdf.ValueTypeNames.Double).Set(0.023)
            if num_envs > 1:
                joint.AddAppliedSchema("MjcJointAPI")  # Preserve implicit native limit semantics.
        source.GetRootLayer().Save()
    cfg.num_envs = num_envs
    device = "cpu" if solver_name == "xpbd" else "cuda:0"
    if device != "cpu" and not wp.is_cuda_available():
        pytest.skip("MJWarp and Kamino round-trips require CUDA")
    solver_cfg = {
        "xpbd": XPBDSolverCfg(iterations=13),
        "mujoco": MJWarpSolverCfg(iterations=13, ls_iterations=7),
        "kamino": KaminoPADMMSolverCfg(),
        "vbd": VBDSolverCfg(iterations=13, rigid_body_particle_contact_buffer_size=513),
    }[solver_name]
    simulation_cfg = SimulationCfg(
        device=device,
        dt=0.007 if cartesian else 1 / 120,
        gravity=(0.2, -0.1, -4.0),
        physics=NewtonCfg(solver_cfg=solver_cfg),
    )
    output = tmp_path / "fixed_scene.usda"
    expected = {}
    expected_native = {}

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
        manager.synchronize_model_changes()
        pairs = (
            manager._collision_pipeline.shape_pairs_filtered.numpy()
            if manager._collision_pipeline is not None
            else None
        )
        expected.update(_capture_environment_physics(manager.get_model(), env_id, pairs))
        if solver_name == "mujoco":
            expected_native = _capture_mujoco_physics(manager._solver, env_id)
            expected["filters"].update(
                key[1] for key, value in expected_native.items() if key[0] == "collision_pair" and not value
            )

        elif solver_name == "kamino":
            expected_native = _capture_kamino_physics(manager._solver, env_id)
        source_layer = scene.stage.GetRootLayer().ExportToString()
        scene.export_to_usd(str(output), env_id=env_id, include_solver_settings=solver_name == "mujoco")
        assert scene.stage.GetRootLayer().ExportToString() == source_layer
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
    if passive:
        assert not joint.HasAPI(UsdPhysics.DriveAPI, "angular")
    elif not cartesian:
        assert UsdPhysics.DriveAPI(joint, "angular").GetStiffnessAttr().Get() == pytest.approx(83 * np.pi / 180)
    assert not joint.GetAttribute("state:angular:physics:position").HasAuthoredValueOpinion()
    for name, mass in (("Box", 2.5), ("CollectedFirst", 1.5), ("CollectedSecond", 3.5)):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/{name}")
        assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == pytest.approx(
            mass * (1 + env_id / 100 if name == "Box" else 1)
        )
    for path in ("/World/Ground", "/World/Light", f"/World/envs/env_{env_id}/Table"):
        assert stage.GetPrimAtPath(path)
    fresh, info = _load(str(output), device=device, solver_name=solver_name)
    driver = dict(stage.GetRootLayer().customLayerData.get("isaaclab:newtonDriver", {}))
    if solver_name != "mujoco":
        assert not driver  # Missing solver settings never prevent native scene/asset loading.
    else:
        import json

        options = json.loads(driver["options"])
        options["iterations"] = int(stage.GetPrimAtPath("/physicsScene").GetAttribute("mjc:option:iterations").Get())
        driver["options"] = json.dumps(options)
    solver = _make_driver(fresh, solver_name, driver)
    if solver_name in {"mujoco", "kamino"}:
        native = (_capture_mujoco_physics if solver_name == "mujoco" else _capture_kamino_physics)(solver, 0)
        assert native.keys() == expected_native.keys(), (
            native.keys() - expected_native.keys(),
            expected_native.keys() - native.keys(),
        )
        for key, value in expected_native.items():
            _assert_physical_value_equal(key, native[key], value)
    actual = _capture_environment_physics(fresh, 0)
    assert expected.keys() == actual.keys()
    for key, value in expected.items():
        _assert_physical_value_equal(key, actual[key], value)
    state = fresh.state()
    if solver_name not in {"kamino", "vbd"}:
        newton.eval_fk(fresh, fresh.joint_q, fresh.joint_qd, state)
    assert {path for path in fresh.body_label if "/Cable/" not in path} == bodies
    np.testing.assert_array_equal(state.body_qd.numpy(), 0)
    np.testing.assert_array_equal(fresh.joint_qd.numpy(), 0)


@pytest.mark.parametrize("bound", [True, False, "complete", "unmapped"])
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
    defaults = newton.ModelBuilder().finalize(device="cpu")
    for name in ("soft_contact_ke", "soft_contact_kd", "soft_contact_kf", "soft_contact_mu"):
        setattr(model, name, getattr(defaults, name))
    model.shape_label = [str(prim.GetPath()) for prim in shapes]
    model.shape_flags = wp.array([int(newton.ShapeFlags.COLLIDE_SHAPES)] * 2, dtype=wp.int32, device="cpu")
    if bound == "unmapped":
        model.shape_label[0] = "/NativeOnlyCollider"
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
    if bound == "unmapped":
        with pytest.raises(NotImplementedError, match="no authored USD collision identity"):
            NewtonManager.author_fixed_configuration(UsdWriter(stage), scene)
        return
    NewtonManager.author_fixed_configuration(UsdWriter(stage), scene)
    from isaaclab_newton.assets.physics_properties import NewtonContactData

    restored = SimpleNamespace(**{name: [float("nan")] * 2 for name in values})
    for row, prim in enumerate(shapes):
        NewtonContactData.restore_fixed_configuration(prim, restored, row)
    for name, expected in values.items():
        np.testing.assert_allclose(getattr(restored, name), expected, rtol=1e-6)
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
