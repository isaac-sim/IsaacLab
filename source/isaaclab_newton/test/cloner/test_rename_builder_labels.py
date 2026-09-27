# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Newton clone label rewriting and visualization clone-plan sources."""

import unittest
from types import SimpleNamespace
from unittest import mock

import newton
import numpy as np
import warp as wp
from isaaclab_newton.cloner import NewtonReplicateContext
from isaaclab_newton.cloner import newton_clone_utils as newton_clone_utils_module
from isaaclab_newton.cloner import replicate as replicate_module
from isaaclab_newton.cloner.newton_clone_utils import replicate_builder_mapping

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import ClonePlan, PrototypeWorldTopology, make_clone_plan
from isaaclab.cloner import path as cloner_path
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim import SimulationContext, SpawnerCfg
from isaaclab.sim.schemas import define_deformable_curve_properties


class TestReplicateBuilderMapping(unittest.TestCase):
    def test_world_prototypes_preserve_repeated_instances_and_default_poses(self):
        """Repeated instances preserve poses; empty worlds and unused prototypes add no bodies."""
        names = "Banana", "Franka", "Unused"
        cfgs = tuple(AssetBaseCfg(prim_path="/World/envs/env_[^/]+/" + name) for name in names)
        plan = make_clone_plan(
            cfgs, ((0, 1), (0, 1, 1), (0, 0, 1), (1,), ()), 20, positions=np.zeros((20, 3), dtype=np.float32)
        )
        assets = {}
        for source in cloner_path.get_asset_prototype_paths(plan)[:2]:
            asset = assets[source] = newton.ModelBuilder()
            asset.add_body(label=source, xform=wp.transform((1, 2, 3), wp.quat_identity()))
        assets["/World/envs/env_0/Unused"] = newton.ModelBuilder()
        assets["/World/envs/env_0/Unused"].add_body(label="/outside/the/plan")
        builder = newton.ModelBuilder()
        with mock.patch.object(builder, "replicate", wraps=builder.replicate) as replicate:
            replicate_builder_mapping(
                builder, plan, plan.positions, np.tile([0, 0, 0, 1], (20, 1)), assets, env_ids=np.arange(20)
            )
        self.assertEqual(replicate.call_count, 5)
        self.assertEqual(builder.world_count, 20)
        self.assertEqual(builder.body_count, 36)
        self.assertEqual(len(set(builder.body_label)), 36)
        np.testing.assert_array_equal(
            np.bincount(builder.body_world, minlength=20), [2] * 4 + [3] * 8 + [1] * 4 + [0] * 4
        )
        np.testing.assert_allclose(np.asarray(builder.body_q)[:, :3], np.tile([1, 2, 3], (36, 1)))
        self.assertIn("/World/envs/env_4/Franka_1", builder.body_label)
        self.assertIn("/World/envs/env_8/Banana_1", builder.body_label)

    def test_local_and_env_root_sites_keep_indices_labels_and_world_positions(self):
        source_path = "/World/envs/env_0/Robot"
        source = newton.ModelBuilder()
        source.add_body(xform=wp.transform((2.0, 0.0, 0.0), wp.quat_identity()), label=source_path)
        site_idx = source.add_site(body=0, xform=wp.transform(), label="ee")
        other_path = "/World/envs/env_0/Box"
        other = newton.ModelBuilder()
        other.add_body(xform=wp.transform((2.0, 0.0, 1.0), wp.quat_identity()), label=other_path)
        other.add_shape_box(body=0, label=other_path + "/shape")
        root_site_idx = source.shape_count + other.shape_count
        shape_stride = root_site_idx + 1

        # Non-zero base so site indices are not trivially zero-based.
        builder = newton.ModelBuilder()
        builder.add_body(xform=wp.transform())
        builder.add_shape(body=0, type=newton.GeoType.SPHERE)
        base_shape = builder.shape_count
        positions = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=np.float32)
        quaternions = np.array([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=np.float32)
        assets = {source_path: source, other_path: other}
        sites = dict(source_site_indices={id(source): {"ee": [site_idx]}})
        sites["env_root_sites"] = {"origin": wp.transform((0.1, 0.0, 0.0), wp.quat_identity())}

        plan = make_clone_plan(tuple(AssetBaseCfg(prim_path=path) for path in (source_path, other_path)), ((0, 1),), 3)
        with mock.patch.object(builder, "replicate", wraps=builder.replicate) as replicate:
            local_site_map, _ = replicate_builder_mapping(
                builder, plan, positions, quaternions, assets, env_ids=np.arange(3, dtype=np.int64), **sites
            )

        replicate.assert_called_once()
        for name, index in (("ee", site_idx), ("origin", root_site_idx)):
            self.assertEqual(local_site_map[name], [[base_shape + world * shape_stride + index] for world in range(3)])
            for world, (index,) in enumerate(local_site_map[name]):
                self.assertEqual(builder.shape_label[index], f"/World/envs/env_{world}/{name}")
        site_positions = [tuple(builder.shape_transform[index].p) for (index,) in local_site_map["origin"]]
        np.testing.assert_allclose(site_positions, positions + [0.1, 0.0, 0.0], atol=1e-6)
        self.assertEqual(source.shape_count + other.shape_count, root_site_idx)
        self.assertEqual(
            builder.body_label[1:],
            [f"/World/envs/env_{world}/{name}" for world in range(3) for name in ("Robot", "Box")],
        )


class TestVisualizationClonePlan(unittest.TestCase):
    def setUp(self):
        self.sim = object.__new__(SimulationContext)
        self.sim.cfg = SimpleNamespace(physics=object(), device="cpu")
        self.sim.physics_manager = SimpleNamespace(get_device=lambda: "cpu")
        self.sim.stage, self.sim._backend_registry = None, []
        patch = mock.patch.object(SimulationContext, "instance", return_value=self.sim)
        patch.start()
        self.addCleanup(patch.stop)

    def tearDown(self):
        for _, resource in tuple(self.sim._backend_registry):
            self.sim.close_backend(resource)

    @staticmethod
    def _define_xform(stage, path, translation=None):
        xform = UsdGeom.Xform.Define(stage, path)
        if translation is not None:
            xform.AddTranslateOp().Set(translation)

    def test_visualization_builder_imports_only_declared_global_roots(self):
        stage = Usd.Stage.CreateInMemory()
        self.sim.stage = stage
        self._define_xform(stage, "/World")
        for path in ("/World/Declared", "/World/Undeclared", "/World/Excluded"):
            body = UsdGeom.Cube.Define(stage, path)
            body.CreateSizeAttr(0.2)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        cfgs = AssetBaseCfg(prim_path="/World/Declared"), AssetBaseCfg(prim_path="/World/Excluded", cloning_contexts=())
        plan = make_clone_plan(cfgs, ((),), 1, shared_assets=(0, 1))
        builder, stage_info, site_index_map = NewtonReplicateContext(self.sim).replicate(plan, (0,))

        self.assertEqual(builder.body_label, ["/World/Declared"])
        self.assertIsNone(stage_info)
        self.assertEqual(site_index_map, {})
        for _, resource in tuple(self.sim._backend_registry):
            self.sim.close_backend(resource)

        # A non-cloning sensor must not exclude the body selected from its owner's subtree.
        cfgs = AssetBaseCfg(prim_path="/World"), SensorBaseCfg(prim_path="/World/Declared")
        plan = make_clone_plan(cfgs, ((),), 1, shared_assets=(0, 1))
        builder, _, _ = NewtonReplicateContext(self.sim).replicate(plan, (0,))
        self.assertCountEqual(builder.body_label, ["/World/Declared", "/World/Undeclared", "/World/Excluded"])
        for _, resource in tuple(self.sim._backend_registry):
            self.sim.close_backend(resource)

        assets = (
            AssetBaseCfg(prim_path="/Copies/env_[^/]+/Body", spawn=SpawnerCfg(spawn_path="/World/Declared")),
            AssetBaseCfg(prim_path="/World"),
        )
        for positions in (np.array([[3, 0, 0], [5, 0, 0]], dtype=np.float32), None):
            with self.subTest(positions=positions):
                plan = make_clone_plan(
                    assets, ((0,),), 2, shared_assets=(1,), positions=positions, env_template="/Copies/env_{}"
                )

                builder, _, _ = NewtonReplicateContext(self.sim).replicate(plan, (0, 1))
                self.assertCountEqual(
                    builder.body_label,
                    ["/World/Undeclared", "/World/Excluded", "/Copies/env_0/Body", "/Copies/env_1/Body"],
                )
                source_position = np.asarray(builder.body_q[builder.body_label.index("/Copies/env_0/Body")])[:3]
                target_position = np.asarray(builder.body_q[builder.body_label.index("/Copies/env_1/Body")])[:3]
                np.testing.assert_allclose(source_position, np.zeros(3) if positions is None else positions[0])
                offset = np.zeros(3) if positions is None else positions[1] - positions[0]
                np.testing.assert_allclose(target_position - source_position, offset)
                for _, resource in tuple(self.sim._backend_registry):
                    self.sim.close_backend(resource)

        # The same definition can also be shared and appear twice in each replicated world.
        plan = make_clone_plan((AssetBaseCfg(prim_path="/World/Declared"),), ((0, 0),), 2, shared_assets=(0,))
        with mock.patch.object(
            newton.ModelBuilder, "add_usd", autospec=True, side_effect=newton.ModelBuilder.add_usd
        ) as add_usd:
            builder, _, _ = NewtonReplicateContext(self.sim).replicate(plan, (0,))
        self.assertEqual(add_usd.call_count, 1)
        self.assertEqual(builder.body_world, [-1, 0, 0, 1, 1])
        self.assertEqual(len(set(builder.body_label)), 5)

    def test_cable_import_keeps_physics_bindings_without_destination_prims(self):
        stage = self.sim.stage = Usd.Stage.CreateInMemory()
        source = "/Scene/copy_7/Rope"
        shared = ("/Scene/SharedRope", "/Scene/PeriodicRope", "/Scene/MultiRope", "/Scene/CubicRope", "/Scene/OnePoint")
        for path in (source, "/Scene/copy_7/OtherRope", *shared):
            curve = UsdGeom.BasisCurves.Define(stage, path)
            points, counts = [(0, 0, 0), (0, 1, 0), (1, 1, 0)], [3]
            if path.endswith("MultiRope"):
                points, counts = points * 2, [3, 3]
            elif path.endswith("OnePoint"):
                points, counts = points[:1], [1]
            curve.CreatePointsAttr(points)
            curve.CreateCurveVertexCountsAttr(counts)
            curve.CreateTypeAttr(UsdGeom.Tokens.cubic if path.endswith("CubicRope") else UsdGeom.Tokens.linear)
            curve.CreateWrapAttr(
                UsdGeom.Tokens.periodic if path.endswith("PeriodicRope") else UsdGeom.Tokens.nonperiodic
            )
            curve.CreateWidthsAttr([0.02])
            curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            define_deformable_curve_properties(path, stage)
        assets = (
            AssetBaseCfg(prim_path="/Scene/copy_[^/]+/Rope", spawn=SpawnerCfg(spawn_path=source)),
            AssetBaseCfg(prim_path="/Scene/copy_[^/]+/OtherRope"),
        )
        assets += tuple(AssetBaseCfg(prim_path=path) for path in shared)
        plan = make_clone_plan(assets, ((0, 1),), 2, shared_assets=range(2, len(assets)), env_template="/Scene/copy_{}")
        bindings = {"/PhysicsOwned": [0]}
        with mock.patch.object(replicate_module.NewtonManager, "_cable_bindings", bindings):
            builder, _, _ = NewtonReplicateContext(self.sim).replicate(plan, (0, *range(2, len(shared) + 2)))
            self.assertIs(replicate_module.NewtonManager._cable_bindings, bindings)
        for path in ("/Scene/SharedRope", "/Scene/copy_0/Rope", "/Scene/copy_1/Rope"):
            self.assertTrue(all(f"{path}_edge_capsule_{index}" in builder.shape_label for index in range(2)))
        self.assertFalse(any("OtherRope" in path for path in builder.shape_label))
        self.assertFalse(stage.GetPrimAtPath("/Scene/copy_1/Rope"))

    def test_visualization_builder_disables_collision_pairs(self):
        stage = Usd.Stage.CreateInMemory()
        self.sim.stage = stage
        robot_path = "/World/envs/env_0/Robot"
        self._define_xform(stage, "/World")
        self._define_xform(stage, "/World/envs")
        self._define_xform(stage, "/World/envs/env_0")
        self._define_xform(stage, "/World/envs/env_1", (2.0, 0.0, 0.0))
        robot = UsdGeom.Xform.Define(stage, robot_path).GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(robot)
        robot.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
        for name, translation in (("A", 0.0), ("B", 1.0)):
            body_path = f"{robot_path}/{name}"
            body = UsdGeom.Xform.Define(stage, body_path)
            body.AddTranslateOp().Set((translation, 0.0, 0.0))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            collision = UsdGeom.Cube.Define(stage, f"{body_path}/Collision")
            collision.CreateSizeAttr(0.2)
            UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{robot_path}/Joint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{robot_path}/A")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"{robot_path}/B")])

        asset = AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Robot", spawn=SpawnerCfg(spawn_path=robot_path))
        plan = make_clone_plan((asset,), ((0,),), 2, positions=np.asarray(((0, 0, 0), (2, 0, 0)), dtype=np.float32))
        builder, _, _ = NewtonReplicateContext(self.sim).replicate(plan, (0,))
        model = builder.finalize(device="cpu")

        self.assertEqual(model.shape_count, 4)
        self.assertEqual(len(model.shape_collision_filter_pairs), 0)
        self.assertEqual(
            model.body_label, [f"/World/envs/env_{env}/Robot/{body}" for env in range(2) for body in ("A", "B")]
        )
        self.assertEqual(model.shape_contact_pair_count, 0)

    def test_visualization_builder_uses_clone_plan_sources_and_rewrites_labels(self):
        stage = Usd.Stage.CreateInMemory()
        self.sim.stage = stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        self._define_xform(stage, "/World")
        self._define_xform(stage, "/World/envs")
        material = UsdShade.Material.Define(stage, "/World/envs/env_0/Material")
        env_paths = [(env_id, f"/World/envs/env_{env_id}") for env_id in (0, 1)]
        for env_id, env_path in env_paths:
            self._define_xform(stage, env_path, (float(env_id) * 3.0, 0.0, 0.0))
            body = UsdGeom.Xform.Define(stage, f"{env_path}/Object")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(body.GetPrim()).Bind(material)
            UsdGeom.Cube.Define(stage, f"{env_path}/Object/source_{env_id}_visual").CreateSizeAttr(0.2)

        plan = ClonePlan(
            PrototypeWorldTopology(
                num_asset_prototypes=3,
                world_prototypes=np.array([0, 2, 1, 2]),
                world_prototype_starts=np.array([0, 0, 2, 4]),
                world_prototype_layout=np.array([0, 1, 0]),
            ),
            asset_cfgs=tuple(
                AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Object", spawn=SpawnerCfg(spawn_path=path + "/Object"))
                for _, path in env_paths
            )
            + (AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Material", cloning_contexts=()),),
            positions=np.asarray(((0, 0, 0), (3, 0, 0), (6, 0, 0)), dtype=np.float32),
        )
        builder, _, _ = NewtonReplicateContext(self.sim).replicate(plan, (0, 1))
        self.assertEqual(builder.body_label, [f"/World/envs/env_{i}/Object" for i in range(3)])
        self.assertEqual(
            builder.shape_label,
            [f"/World/envs/env_{i}/Object/source_{source}_visual" for i, source in enumerate((0, 1, 0))],
        )
        self.assertEqual(builder.body_world, [0, 1, 2])
        self.assertEqual(builder.shape_world, [0, 1, 2])
        self.assertEqual(
            builder.custom_attributes["isaaclab:visual_material_path"].values,
            {world: f"/World/envs/env_{world}/Material" for world in range(3)},
        )

    def test_render_deformables_import_once_then_follow_native_replication(self):
        stage = self.sim.stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        self._define_xform(stage, "/Scene/copy_7", (10.0, 0.0, 0.0))
        self._define_xform(stage, "/Scene/copy_7/Parent", (2.0, 0.0, 0.0))
        sources = "/Scene/copy_7/Parent", "/Sources/Volume", "/Sources/Tet", "/Shared"
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        visual = np.concatenate((vertices[:3], vertices[:3] + [0, 0, 1])).astype(np.float32)
        roots = sources[0] + "/Cloth", *sources[1:]
        for root, volume in zip(roots, (False, True, True, False), strict=True):
            prim = UsdGeom.Xform.Define(stage, root).GetPrim()
            prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(["OmniPhysicsDeformableBodyAPI"]))
            if root.endswith("Cloth"):
                UsdGeom.Xform(prim).AddTranslateOp().Set((3.0, 0.0, 0.0))
            if volume:
                mesh = UsdGeom.TetMesh.Define(stage, root + "/sim")
                mesh.CreatePointsAttr(vertices)
                mesh.CreateTetVertexIndicesAttr([(0, 1, 2, 3)])
                if root == sources[1]:
                    mesh = UsdGeom.Mesh.Define(stage, root + "/vis")
                    mesh.CreatePointsAttr(visual)
                    mesh.CreateFaceVertexCountsAttr([3, 3])
                    mesh.CreateFaceVertexIndicesAttr(np.arange(6))
            else:
                mesh = UsdGeom.Mesh.Define(stage, root + "/sim")
                mesh.CreatePointsAttr(vertices[:3])
                mesh.CreateFaceVertexCountsAttr([3])
                mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        assets = tuple(
            AssetBaseCfg(prim_path="/Scene/copy_[^/]+/" + name, spawn=SpawnerCfg(spawn_path=source))
            for name, source in zip(("Parent", "Volume", "Tet"), sources[:3], strict=True)
        ) + (AssetBaseCfg(prim_path="/Shared"),)
        env_ids = np.array([7, 9, 12])
        quaternions = np.asarray([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 2**-0.5, 2**-0.5]], dtype=np.float32)
        for positions in (np.array([[10, 0, 0], [20, 0, 0], [30, 0, 0]], dtype=np.float32), None):
            with self.subTest(positions=positions):
                for _, resource in tuple(self.sim._backend_registry):
                    self.sim.close_backend(resource)
                options = dict(shared_assets=(3,), env_template="/Scene/copy_{}")
                options.update(clone_strategy=lambda _weights, _count: np.array([0, 1, 0]), positions=positions)
                plan = make_clone_plan(assets, ((0,), (1, 1, 2)), 3, **options)
                with mock.patch.object(
                    newton.ModelBuilder, "add_cloth_mesh", autospec=True, side_effect=newton.ModelBuilder.add_cloth_mesh
                ) as add_cloth:
                    options = dict(plan=plan, asset_prototype_ids=range(4))
                    options.update(positions=positions, quaternions=quaternions)
                    builder, _, _, offsets = replicate_module._replicate_newton(stage, env_ids, self.sim, **options)
                self.assertEqual(add_cloth.call_count, 3)  # Three prototypes, not five destination meshes.
                np.testing.assert_array_equal(np.bincount(np.asarray(builder.particle_world) + 1), [3, 3, 16, 3])
                expected = {"/Shared/sim": vertices[:3]}
                origins = np.zeros((3, 3)) if positions is None else positions
                for world, env_id in enumerate(env_ids):
                    xform = wp.transform(origins[world], quaternions[world])
                    meshes = {"Volume/vis": visual, "Volume_1/vis": visual, "Tet/sim": vertices}
                    if world != 1:
                        meshes = {"Parent/Cloth/sim": vertices[:3] + [15, 0, 0] - origins[0]}
                    for suffix, points in meshes.items():
                        expected[f"/Scene/copy_{env_id}/{suffix}"] = np.asarray(
                            [wp.transform_point(xform, wp.vec3(point)) for point in points]
                        )
                self.assertEqual(set(offsets), set(expected))
                for path, points in expected.items():
                    start = offsets[path]
                    np.testing.assert_allclose(builder.particle_q[start : start + len(points)], points, atol=1e-5)
                self.assertFalse(stage.GetPrimAtPath("/Scene/copy_12"))


class TestReplicationNamesItsCopies(unittest.TestCase):
    _SRC = "/World/envs/env_0/Robot"
    _ENV = "/World/envs/env_{}"

    def test_batched_prefixes_name_each_world_and_preserve_the_prototype(self):
        source = newton.ModelBuilder()
        attributes = source.custom_attributes
        body = source.add_body(xform=wp.transform(), label=self._SRC)
        source.add_shape_box(body=body, label=f"{self._SRC}/shape")
        source.add_shape_box(body=body, label=f"{self._SRC}/sibling_material_shape")
        source.add_shape_box(body=body, label=f"{self._SRC}/shared_material_shape")
        child = source.add_link(xform=wp.transform(), label=f"{self._SRC}/link")
        source.add_joint_revolute(parent=body, child=child, axis=(0.0, 0.0, 1.0), label=f"{self._SRC}/hinge")

        def resolve_motor_owners(builder):
            labels = builder.custom_attributes["syn:motor_label"].values
            targets = builder.custom_attributes["syn:motor_target"].values
            return [0 if label == target else -1 for label, target in zip(labels, targets, strict=True)]

        source.add_custom_frequency(
            newton.ModelBuilder.CustomFrequency(
                name="motor",
                namespace="syn",
                label_attribute="syn:motor_label",
                articulation_owner_attribute="syn:motor_articulation",
                articulation_owner_resolver=resolve_motor_owners,
            )
        )
        for name, dtype, default, references in (
            ("motor_label", str, "", None),
            ("motor_target", str, "", None),
            ("motor_world", int, -1, "world"),
            ("motor_articulation", int, -1, "articulation"),
        ):
            attr = newton.ModelBuilder.CustomAttribute(
                name, dtype, "syn:motor", default=default, namespace="syn", references=references
            )
            source.add_custom_attribute(attr)
        for name in ("syn:motor_label", "syn:motor_target"):
            attributes[name].values = [f"{self._SRC}/motor"]
        attributes["syn:motor_world"].values = [-1]
        source._custom_frequency_counts["syn:motor"] = 1
        for name, namespace in (("visual_material_path", "isaaclab"), ("shape_note", "syn")):
            frequency = newton.Model.AttributeFrequency.SHAPE
            attr = newton.ModelBuilder.CustomAttribute(name, str, frequency, default="", namespace=namespace)
            source.add_custom_attribute(attr)
            attributes[f"{namespace}:{name}"].values[0] = self._SRC + "/Looks/material"
        sibling_material = "/World/envs/env_0/Material"
        attributes["isaaclab:visual_material_path"].values.update({1: sibling_material, 2: "/World/SharedMaterial"})
        label_names = "body_label", "joint_label", "shape_label", "articulation_label"
        original = {name: list(getattr(source, name)) for name in label_names}
        builder = newton.ModelBuilder()
        env_ids = np.array([10, 20], dtype=np.int64)
        plan = make_clone_plan(
            tuple(AssetBaseCfg(prim_path=path) for path in (self._SRC, sibling_material)), ((0, 1),), len(env_ids)
        )
        positions = np.zeros((len(env_ids), 3), dtype=np.float32)
        quaternions = np.tile([0, 0, 0, 1], (len(env_ids), 1)).astype(np.float32)
        assets = {self._SRC: source, sibling_material: newton.ModelBuilder()}
        add_builder = newton.ModelBuilder.add_builder
        source_paths = {name: attr.values.copy() for name, attr in attributes.items() if attr.dtype is str}

        def append_copy(destination, asset, **kwargs):
            if asset is source:
                for name, labels in original.items():
                    self.assertEqual(getattr(asset, name), labels)
                for name, values in source_paths.items():
                    self.assertEqual(asset.custom_attributes[name].values, values)
            return add_builder(destination, asset, **kwargs)

        with mock.patch.object(newton.ModelBuilder, "add_builder", append_copy):
            replicate_builder_mapping(builder, plan, positions, quaternions, assets, env_ids=env_ids)
        for name, source_labels in original.items():
            expected = [
                label.replace(self._SRC, f"{self._ENV.format(i)}/Robot", 1) for i in env_ids for label in source_labels
            ]
            self.assertEqual(getattr(builder, name), expected)
            self.assertEqual(getattr(source, name), source_labels)

        expected_labels = [f"{self._ENV.format(i)}/Robot/motor" for i in env_ids]
        for name in ("syn:motor_label", "syn:motor_target"):
            self.assertEqual(builder.custom_attributes[name].values, expected_labels)
            self.assertEqual(attributes[name].values, [f"{self._SRC}/motor"])
        self.assertEqual(builder.custom_attributes["syn:motor_articulation"].values, [0, source.articulation_count])
        materials = self._ENV + "/Robot/Looks/material", self._ENV + "/Material", "/World/SharedMaterial"
        expected = [path.format(env_id) for env_id in env_ids for path in materials]
        self.assertEqual(builder.custom_attributes["isaaclab:visual_material_path"].values, dict(enumerate(expected)))
        self.assertEqual(
            builder.custom_attributes["syn:shape_note"].values,
            {index * 3: self._SRC + "/Looks/material" for index in range(len(env_ids))},
        )

        # Declaring the environment root must produce the same names as declaring its assets.
        root = self._ENV.format(0)
        plan = make_clone_plan((AssetBaseCfg(prim_path=root),), ((0,),), len(env_ids))
        rooted = newton.ModelBuilder()
        replicate_builder_mapping(rooted, plan, positions, quaternions, {root: source}, env_ids=env_ids)
        for name in label_names:
            self.assertEqual(getattr(rooted, name), getattr(builder, name))
        for name in source_paths:
            self.assertEqual(rooted.custom_attributes[name].values, builder.custom_attributes[name].values)

    def test_hook_labels_are_rewritten_after_the_slow_path(self):
        source = newton.ModelBuilder()
        source.add_body(label=f"{self._SRC}/base")
        builder = newton.ModelBuilder()
        env_ids = np.array([10, 20], dtype=np.int64)
        plan = make_clone_plan((AssetBaseCfg(prim_path=self._SRC),), ((0,),), 2)

        def hook(builder, *_):
            builder.add_body(label=f"{self._SRC}/hook")

        positions = np.zeros((2, 3), dtype=np.float32)
        rotations = np.array([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=np.float32)
        replicate_builder_mapping(
            builder, plan, positions, rotations, {self._SRC: source}, env_ids=env_ids, per_world_builder_hooks=(hook,)
        )
        self.assertEqual(
            builder.body_label,
            [f"/World/envs/env_{env_id}/Robot/{label}" for env_id in env_ids for label in ("base", "hook")],
        )


class TestRootJointNaming(unittest.TestCase):
    """The importer leaves a floating base's root joint unnamed; every other entity is named."""

    def test_only_generated_free_root_names_change(self):
        builder = newton.ModelBuilder()
        root = "/World/envs/env_0/Robot"
        bodies = [builder.add_link(label=f"{root}/link_{index}") for index in range(4)]
        builder.add_joint_free(child=bodies[0])
        builder.add_joint_free(child=bodies[1], label="authored")
        builder.add_joint_revolute(parent=bodies[0], child=bodies[2], axis=(0.0, 0.0, 1.0))
        builder.add_joint_free(parent=bodies[0], child=bodies[3])
        self.assertFalse(builder.joint_label[0].startswith("/"))
        expected = [f"{root}/link_0_free_joint", *builder.joint_label[1:]]
        newton_clone_utils_module._name_root_joints_after_their_body(builder)
        self.assertEqual(builder.joint_label, expected)


if __name__ == "__main__":
    unittest.main()
