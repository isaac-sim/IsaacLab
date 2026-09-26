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
from isaaclab_newton.physics import visualization_deformables as visualization_deformables_module

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import ClonePlan, PrototypeWorldTopology, make_clone_plan
from isaaclab.cloner.path import get_instance_paths
from isaaclab.scene_data.deformable_discovery import (
    DeformableStageEntry,
    deformable_prototypes,
    expand_deformable_entries,
)
from isaaclab.sim import SpawnerCfg
from isaaclab.sim.schemas import define_deformable_curve_properties


class TestReplicateBuilderMapping(unittest.TestCase):
    def test_world_prototypes_preserve_repeated_instances_and_default_poses(self):
        """Two reusable assets compose four world prototypes and 36 distinct native instances."""
        cfgs = tuple(AssetBaseCfg(prim_path="/World/envs/env_[^/]+/" + name) for name in ("Banana", "Franka"))
        plan = make_clone_plan(
            cfgs, ((0, 1), (0, 1, 1), (0, 0, 1), (1,)), 16, positions=np.zeros((16, 3), dtype=np.float32)
        )
        instances = get_instance_paths(plan)
        assets = {}
        for _, source, _, world_ids in instances:
            if not len(world_ids) or source in assets:
                continue
            asset = assets[source] = newton.ModelBuilder()
            asset.add_body(label=source, xform=wp.transform((1, 2, 3), wp.quat_identity()))
        builder = newton.ModelBuilder()
        with mock.patch.object(builder, "replicate", wraps=builder.replicate) as replicate:
            replicate_builder_mapping(
                builder,
                plan,
                instances,
                plan.positions,
                np.tile([0, 0, 0, 1], (16, 1)),
                assets,
                env_ids=np.arange(16),
            )
        self.assertEqual(replicate.call_count, 4)
        self.assertEqual(len(assets), 2)
        self.assertEqual(builder.body_count, 36)
        self.assertEqual(len(set(builder.body_label)), 36)
        np.testing.assert_array_equal(np.bincount(builder.body_world), [2] * 4 + [3] * 8 + [1] * 4)
        np.testing.assert_allclose(np.asarray(builder.body_q)[:, :3], np.tile([1, 2, 3], (36, 1)))
        self.assertIn("/World/envs/env_4/Franka_1", builder.body_label)
        self.assertIn("/World/envs/env_8/Banana_1", builder.body_label)

    def test_local_and_env_root_sites_keep_indices_labels_and_world_positions(self):
        source_path, destination = "/World/envs/env_0/Robot", "/World/envs/env_{}/Robot"
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

        plan = make_clone_plan((source_path, other_path), ((0, 1),), 3)
        instances = (
            (0, source_path, destination, np.arange(3)),
            (1, other_path, "/World/envs/env_{}/Box", np.arange(3)),
        )
        with mock.patch.object(builder, "replicate", wraps=builder.replicate) as replicate:
            local_site_map, _, _ = replicate_builder_mapping(
                builder,
                plan,
                instances,
                positions,
                np.array([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=np.float32),
                {source_path: source, other_path: other},
                env_ids=np.arange(3, dtype=np.int64),
                source_site_indices={id(source): {"ee": [site_idx]}},
                env_root_sites={"origin": wp.transform((0.1, 0.0, 0.0), wp.quat_identity())},
            )

        replicate.assert_called_once()
        for name, index in (("ee", site_idx), ("origin", root_site_idx)):
            self.assertEqual(local_site_map[name], [[base_shape + world * shape_stride + index] for world in range(3)])
            for world, (index,) in enumerate(local_site_map[name]):
                self.assertEqual(builder.shape_label[index], f"/World/envs/env_{world}/{name}")
        np.testing.assert_allclose(
            [tuple(builder.shape_transform[index].p) for (index,) in local_site_map["origin"]],
            positions + [0.1, 0.0, 0.0],
            atol=1e-6,
        )
        self.assertEqual(source.shape_count + other.shape_count, root_site_idx)
        self.assertEqual(
            builder.body_label[1:],
            [f"/World/envs/env_{world}/{name}" for world in range(3) for name in ("Robot", "Box")],
        )

    def test_unselected_asset_prototypes_are_not_copied(self):
        sources = ("/World/envs/env_0/inactive", "/World/envs/env_0/active")
        source_builders = {source: newton.ModelBuilder() for source in sources}
        for path, source in source_builders.items():
            source.add_body(label=path)
        source_builders[sources[0]].add_body(label="/outside/the/plan")
        builder = newton.ModelBuilder()

        plan = make_clone_plan(sources, ((1,), ()), 2)
        instances = ((1, sources[1], "/World/envs/env_{}/active", np.array([0])),)
        replicate_builder_mapping(
            builder,
            plan,
            instances,
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            source_builders,
            env_ids=np.arange(2, dtype=np.int64),
        )

        self.assertEqual(builder.body_label, ["/World/envs/env_0/active"])
        self.assertEqual(builder.body_world, [0])
        self.assertEqual(builder.world_count, 2)


class TestVisualizationClonePlan(unittest.TestCase):
    def setUp(self):
        self.sim = SimpleNamespace(
            cfg=SimpleNamespace(physics=object()),
            device="cpu",
            stage=None,
            physics_manager=SimpleNamespace(register_callback=mock.Mock()),
        )

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

        for positions in (np.array([[3, 0, 0], [5, 0, 0]], dtype=np.float32), None):
            with self.subTest(positions=positions):
                plan = make_clone_plan(
                    (
                        AssetBaseCfg(
                            prim_path="/Copies/env_[^/]+/Body", spawn=SpawnerCfg(spawn_path="/World/Declared")
                        ),
                        AssetBaseCfg(prim_path="/World"),
                    ),
                    ((0,),),
                    2,
                    shared_assets=(1,),
                    positions=positions,
                    env_template="/Copies/env_{}",
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

        # The same definition can also be shared and appear twice in each replicated world.
        plan = make_clone_plan((AssetBaseCfg(prim_path="/World/Declared"),), ((0, 0),), 2, shared_assets=(0,))
        with mock.patch.object(
            newton.ModelBuilder, "add_usd", autospec=True, side_effect=newton.ModelBuilder.add_usd
        ) as add_usd:
            builder, _, _ = NewtonReplicateContext(self.sim).replicate(plan, (0,))
        self.assertEqual(add_usd.call_count, 1)
        self.assertEqual(builder.body_world, [-1, 0, 0, 1, 1])
        self.assertEqual(len(set(builder.body_label)), 5)

    def test_cable_import_binds_only_supported_native_instances_without_destination_prims(self):
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
        plan = make_clone_plan(
            (
                AssetBaseCfg(prim_path="/Scene/copy_[^/]+/Rope", spawn=SpawnerCfg(spawn_path=source)),
                AssetBaseCfg(prim_path="/Scene/copy_[^/]+/OtherRope"),
            )
            + tuple(AssetBaseCfg(prim_path=path) for path in shared),
            ((0, 1),),
            2,
            shared_assets=range(2, len(shared) + 2),
            env_template="/Scene/copy_{}",
        )
        builder, _, _ = NewtonReplicateContext(self.sim).replicate(plan, (0, *range(2, len(shared) + 2)))
        model = builder.finalize(device="cpu")
        with mock.patch(
            "isaaclab_newton.physics.newton_manager.get_current_stage",
            side_effect=AssertionError("Stage discovery is not a binding input."),
        ):
            bindings = replicate_module.NewtonManager.collect_cable_segment_shape_ids()
        self.assertEqual(set(bindings), {"/Scene/SharedRope", "/Scene/copy_0/Rope", "/Scene/copy_1/Rope"})
        for path, shape_ids in bindings.items():
            self.assertEqual(
                [model.shape_label[index] for index in shape_ids], [f"{path}_edge_capsule_{i}" for i in range(2)]
            )
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

        plan = make_clone_plan(
            (AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Robot", spawn=SpawnerCfg(spawn_path=robot_path)),),
            ((0,),),
            2,
            positions=np.asarray(((0, 0, 0), (2, 0, 0)), dtype=np.float32),
        )
        builder, _, _ = NewtonReplicateContext(self.sim).replicate(plan, (0,))
        model = builder.finalize(device="cpu")

        self.assertEqual(model.shape_count, 4)
        self.assertEqual(len(model.shape_collision_filter_pairs), 0)
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

    def test_shadow_deformables_use_plan_placement_without_destination_prims(self):
        stage = Usd.Stage.CreateInMemory()
        self._define_xform(stage, "/Scene/copy_7", (10.0, 0.0, 0.0))
        self._define_xform(stage, "/Scene/copy_7/Parent", (2.0, 0.0, 0.0))
        path = "/Scene/copy_7/Parent/Cloth"
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        cloth = UsdGeom.Mesh.Define(stage, path)
        cloth.AddTranslateOp().Set((3.0, 0.0, 0.0))
        cloth.GetPrim().SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(["OmniPhysicsDeformableBodyAPI"]))
        cloth.CreatePointsAttr(vertices)
        cloth.CreateFaceVertexCountsAttr([3])
        cloth.CreateFaceVertexIndicesAttr([0, 1, 2])
        instances = ((0, path, "/Scene/copy_{}/Parent/Cloth", np.array([0, 2])),)
        env_ids = np.array([7, 9, 12])
        positions = np.array([[10, 0, 0], [20, 0, 0], [30, 0, 0]], dtype=np.float32)
        prototypes = deformable_prototypes(stage, instances)
        for positions in (positions, None):
            with self.subTest(positions=positions):
                builder = newton.ModelBuilder()
                offsets = visualization_deformables_module.add_shadow_deformables_to_builder(
                    builder,
                    expand_deformable_entries(prototypes, instances, env_ids, positions),
                )
                self.assertEqual(offsets, {path: 0, "/Scene/copy_12/Parent/Cloth": 3})
                self.assertFalse(stage.GetPrimAtPath("/Scene/copy_12"))
                offset = np.zeros(3) if positions is None else positions[2] - positions[0]
                # Root translation is baked into vertices, not applied a second time at placement.
                source_points = vertices + [15.0, 0.0, 0.0]
                np.testing.assert_allclose(builder.particle_q, np.concatenate((source_points, source_points + offset)))

    def test_shadow_visual_topologies_keep_heterogeneous_offsets(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]])
        entries = tuple(
            DeformableStageEntry(
                root_path=f"/Source/{name}",
                sim_mesh_path=f"/Source/{name}/Simulation",
                vis_mesh_path=f"/Source/{name}/Visual",
                deformable_type="volume",
                vertex_count=4,
                vis_vertex_count=count,
                vis_vertices=vertices[:count],
                vis_indices=np.arange(count),
            )
            for name, count in (("A", 3), ("B", 6))
        )
        instances = (
            (0, entries[0].root_path, "/Copies/{}/Body", np.array([0, 2])),
            (1, entries[1].root_path, "/Copies/{}/Body", np.array([1])),
        )
        env_ids = np.array([2, 10, 30])
        builder = newton.ModelBuilder()
        builder.add_particle(pos=wp.vec3(), vel=wp.vec3(), mass=1.0)
        offsets = visualization_deformables_module.add_shadow_deformables_to_builder(
            builder, expand_deformable_entries(entries, instances, env_ids)
        )
        self.assertEqual(
            offsets, {"/Copies/2/Body/Visual": 1, "/Copies/30/Body/Visual": 4, "/Copies/10/Body/Visual": 7}
        )
        self.assertEqual(builder.particle_count, 13)
        self.assertEqual(len(builder.tri_indices), 4)


class TestReplicationNamesItsCopies(unittest.TestCase):
    _SRC = "/World/envs/env_0/Robot"
    _ENV = "/World/envs/env_{}"

    def test_batched_prefixes_name_each_world_and_preserve_the_prototype(self):
        source = newton.ModelBuilder()
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
            source.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    namespace="syn",
                    frequency="syn:motor",
                    dtype=dtype,
                    default=default,
                    references=references,
                )
            )
        source.custom_attributes["syn:motor_label"].values = [f"{self._SRC}/motor"]
        source.custom_attributes["syn:motor_target"].values = [f"{self._SRC}/motor"]
        source.custom_attributes["syn:motor_world"].values = [-1]
        source._custom_frequency_counts["syn:motor"] = 1
        for name, namespace in (("visual_material_path", "isaaclab"), ("shape_note", "syn")):
            source.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    namespace=namespace,
                    dtype=str,
                    frequency=newton.Model.AttributeFrequency.SHAPE,
                    default="",
                )
            )
            source.custom_attributes[f"{namespace}:{name}"].values[0] = self._SRC + "/Looks/material"
        sibling_material = "/World/envs/env_0/Material"
        source.custom_attributes["isaaclab:visual_material_path"].values.update(
            {1: sibling_material, 2: "/World/SharedMaterial"}
        )
        original = {
            name: list(getattr(source, name))
            for name in ("body_label", "joint_label", "shape_label", "articulation_label")
        }
        builder = newton.ModelBuilder()
        env_ids = np.array([10, 20], dtype=np.int64)
        plan = make_clone_plan((self._SRC, sibling_material), ((0, 1),), len(env_ids))
        instances = (
            (0, self._SRC, "/World/envs/env_{}/Robot", np.arange(len(env_ids))),
            (1, sibling_material, "/World/envs/env_{}/Material", np.arange(len(env_ids))),
        )
        positions = np.zeros((len(env_ids), 3), dtype=np.float32)
        quaternions = np.zeros((len(env_ids), 4), dtype=np.float32)
        quaternions[:, 3] = 1.0
        replicate_builder_mapping(
            builder,
            plan,
            instances,
            positions,
            quaternions,
            {self._SRC: source, sibling_material: newton.ModelBuilder()},
            env_ids=env_ids,
        )
        for name, source_labels in original.items():
            expected = [
                label.replace(self._SRC, f"{self._ENV.format(i)}/Robot", 1) for i in env_ids for label in source_labels
            ]
            self.assertEqual(getattr(builder, name), expected)
            self.assertEqual(getattr(source, name), source_labels)

        expected_labels = [f"{self._ENV.format(i)}/Robot/motor" for i in env_ids]
        self.assertEqual(builder.custom_attributes["syn:motor_label"].values, expected_labels)
        self.assertEqual(builder.custom_attributes["syn:motor_articulation"].values, [0, source.articulation_count])
        self.assertEqual(builder.custom_attributes["syn:motor_target"].values, expected_labels)
        self.assertEqual(source.custom_attributes["syn:motor_label"].values, [f"{self._SRC}/motor"])
        materials = self._ENV + "/Robot/Looks/material", self._ENV + "/Material", "/World/SharedMaterial"
        expected = [path.format(env_id) for env_id in env_ids for path in materials]
        self.assertEqual(builder.custom_attributes["isaaclab:visual_material_path"].values, dict(enumerate(expected)))
        self.assertEqual(
            builder.custom_attributes["syn:shape_note"].values,
            {index * 3: self._SRC + "/Looks/material" for index in range(len(env_ids))},
        )

    def test_hook_labels_are_rewritten_after_the_slow_path(self):
        source = newton.ModelBuilder()
        source.add_body(label=f"{self._SRC}/base")
        builder = newton.ModelBuilder()
        env_ids = np.array([10, 20], dtype=np.int64)
        plan = make_clone_plan((self._SRC,), ((0,),), 2)
        instances = ((0, self._SRC, "/World/envs/env_{}/Robot", np.arange(2)),)

        def hook(builder, *_):
            builder.add_body(label=f"{self._SRC}/hook")

        replicate_builder_mapping(
            builder,
            plan,
            instances,
            np.zeros((2, 3), dtype=np.float32),
            np.array([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=np.float32),
            {self._SRC: source},
            env_ids=env_ids,
            per_world_builder_hooks=(hook,),
        )
        self.assertEqual(
            builder.body_label,
            [f"/World/envs/env_{env_id}/Robot/{label}" for env_id in env_ids for label in ("base", "hook")],
        )


class TestRootJointNaming(unittest.TestCase):
    """The importer leaves a floating base's root joint unnamed; every other entity is named."""

    _SOURCE = "/World/envs/env_0/Robot"
    _BODY = "/World/envs/env_0/Robot/pelvis"

    @staticmethod
    def _builder_with_free_root(body_label: str) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        body = builder.add_link(xform=wp.transform(), label=body_label)
        builder.add_joint_free(child=body)
        return builder

    def test_a_generated_root_joint_name_becomes_its_body_path(self):
        builder = self._builder_with_free_root(self._BODY)
        self.assertFalse(builder.joint_label[0].startswith("/"))

        newton_clone_utils_module._name_root_joints_after_their_body(builder)

        self.assertEqual(builder.joint_label[0], f"{self._BODY}_free_joint")

    def test_other_joint_labels_are_left_alone(self):
        named = self._builder_with_free_root(self._BODY)
        named.joint_label[0] = "authored"

        non_free = newton.ModelBuilder()
        parent = non_free.add_link(xform=wp.transform(), label=self._BODY)
        child = non_free.add_link(xform=wp.transform(), label=f"{self._BODY}/link")
        non_free.add_joint_revolute(parent=parent, child=child, axis=(0.0, 0.0, 1.0))

        non_root = newton.ModelBuilder()
        parent = non_root.add_link(xform=wp.transform(), label=self._BODY)
        child = non_root.add_link(xform=wp.transform(), label=f"{self._BODY}/link")
        non_root.add_joint_free(parent=parent, child=child)

        for builder in (named, non_free, non_root):
            original = list(builder.joint_label)
            newton_clone_utils_module._name_root_joints_after_their_body(builder)
            self.assertEqual(builder.joint_label, original)


if __name__ == "__main__":
    unittest.main()
