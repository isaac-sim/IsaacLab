# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Newton clone label rewriting and visualization clone-plan sources."""

import unittest
from unittest import mock

import newton
import numpy as np
import warp as wp
from isaaclab_newton.cloner import newton_clone_utils as newton_clone_utils_module
from isaaclab_newton.cloner.newton_clone_utils import rename_builder_labels, replicate_builder_mapping
from isaaclab_newton.physics import visualization_builder as visualization_builder_module
from isaaclab_newton.physics import visualization_deformables as visualization_deformables_module

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.cloner import ClonePlan
from isaaclab.scene_data.deformable_discovery import DeformableStageEntry

_VIS_LABEL_SUFFIXES = {
    "body_label": "Body",
    "joint_label": "Joint",
    "shape_label": "Shape",
    "articulation_label": "Articulation",
    "constraint_mimic_label": "ConstraintMimic",
    "equality_constraint_label": "EqualityConstraint",
}
# Equality constraints live in custom attributes (like real newton), not plain builder lists.
_VIS_BUILTIN_LABEL_ATTRS = tuple(attr for attr in _VIS_LABEL_SUFFIXES if attr != "equality_constraint_label")
_VIS_EQ_FREQ = "mujoco:equality_constraint"

_SRC = "/World/envs/env_0/protoA"
_DST = "/World/envs/env_{}/protoA"


class _FakeVisualizationModelBuilder:
    def __init__(self, up_axis=None):
        self.up_axis = up_axis
        self.shape_collision_filter_pairs = []
        self.shape_collision_group = []
        for attr in _VIS_BUILTIN_LABEL_ATTRS:
            setattr(self, attr, [])
            setattr(self, attr.replace("_label", "_world"), [])
        self.custom_attributes = {
            "mujoco:equality_constraint_label": newton.ModelBuilder.CustomAttribute(
                name="equality_constraint_label", frequency=_VIS_EQ_FREQ, dtype=str, default="", namespace="mujoco"
            ),
            "mujoco:equality_constraint_world": newton.ModelBuilder.CustomAttribute(
                name="equality_constraint_world",
                frequency=_VIS_EQ_FREQ,
                dtype=int,
                default=0,
                namespace="mujoco",
                references="world",
            ),
        }
        self.geometry_sources = []
        self.world_slices = []
        self._current_world = None

    @property
    def shape_count(self):
        return len(self.shape_label)

    def begin_world(self):
        self._current_world = len(self.world_slices)
        self.world_slices.append([])

    def end_world(self):
        self._current_world = None

    def add_usd(self, stage, root_path=None, ignore_paths=None, schema_resolvers=None, **kwargs):
        del stage, ignore_paths, schema_resolvers, kwargs
        if root_path is None:
            return {"path_shape_map": {}}
        label_start = len(self.body_label)
        geometry_start = len(self.geometry_sources)
        for attr in _VIS_BUILTIN_LABEL_ATTRS:
            getattr(self, attr).append(f"{root_path}/{_VIS_LABEL_SUFFIXES[attr]}")
            getattr(self, attr.replace("_label", "_world")).append(self._current_world or 0)
        self.shape_collision_group.append(1)
        self.custom_attributes["mujoco:equality_constraint_label"].values.append(
            f"{root_path}/{_VIS_LABEL_SUFFIXES['equality_constraint_label']}"
        )
        self.custom_attributes["mujoco:equality_constraint_world"].values.append(self._current_world or 0)
        self.geometry_sources.append(root_path)
        self._record_world_slice(label_start, len(self.body_label), geometry_start, len(self.geometry_sources))
        return {"path_shape_map": {}}

    def add_builder(self, builder, xform=None):
        del xform
        label_start = len(self.body_label)
        geometry_start = len(self.geometry_sources)
        for attr in _VIS_BUILTIN_LABEL_ATTRS:
            labels = getattr(builder, attr)
            getattr(self, attr).extend(labels)
            getattr(self, attr.replace("_label", "_world")).extend([self._current_world] * len(labels))
        self.shape_collision_group.extend(builder.shape_collision_group)
        eq_labels = builder.custom_attributes["mujoco:equality_constraint_label"].values
        self.custom_attributes["mujoco:equality_constraint_label"].values.extend(eq_labels)
        self.custom_attributes["mujoco:equality_constraint_world"].values.extend([self._current_world] * len(eq_labels))
        self.geometry_sources.extend(builder.geometry_sources)
        self._record_world_slice(label_start, len(self.body_label), geometry_start, len(self.geometry_sources))

    def labels_for_world(self, world_id, attr):
        if attr == "equality_constraint_label":
            labels = self.custom_attributes["mujoco:equality_constraint_label"].values
        else:
            labels = getattr(self, attr)
        return [label for start, end, _, _ in self.world_slices[world_id] for label in labels[start:end]]

    def geometry_sources_for_world(self, world_id):
        return [
            source for _, _, start, end in self.world_slices[world_id] for source in self.geometry_sources[start:end]
        ]

    def _record_world_slice(self, label_start, label_end, geometry_start, geometry_end):
        if self._current_world is not None:
            self.world_slices[self._current_world].append((label_start, label_end, geometry_start, geometry_end))


def _make_builder(worlds: list[int]) -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    builder.shape_label.extend(f"{_SRC}/shape_{world}" for world in worlds)
    builder.shape_world.extend(worlds)
    return builder


def _add_custom_frequency(builder, freq_name, string_columns):
    freq = f"syn:{freq_name}"
    builder.add_custom_frequency(newton.ModelBuilder.CustomFrequency(name=freq_name, namespace="syn"))
    builder.add_custom_attribute(
        newton.ModelBuilder.CustomAttribute(
            name=f"{freq_name}_world", frequency=freq, dtype=int, default=0, namespace="syn", references="world"
        )
    )
    for column in string_columns:
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(name=column, frequency=freq, dtype=str, default="", namespace="syn")
        )


def _populate_custom_frequency(builder, freq_name, string_columns, worlds):
    builder.custom_attributes[f"syn:{freq_name}_world"].values = list(worlds)
    for column in string_columns:
        builder.custom_attributes[f"syn:{column}"].values = [f"{_SRC}/{column}_{world}" for world in worlds]
    builder._custom_frequency_counts[f"syn:{freq_name}"] = len(worlds)


class TestRenameCustomAttributes(unittest.TestCase):
    def setUp(self):
        self.worlds = [0, 1]
        self.env_ids = np.array([10, 20], dtype=np.int64)
        self.mapping = np.ones((1, len(self.worlds)), dtype=np.bool_)

    def test_custom_string_columns_follow_frequency_worlds(self):
        builder = newton.ModelBuilder()
        _add_custom_frequency(builder, "freqA", ["freqA_label", "freqA_alt"])
        _add_custom_frequency(builder, "freqB", ["freqB_label"])
        _populate_custom_frequency(builder, "freqA", ["freqA_label", "freqA_alt"], self.worlds)
        _populate_custom_frequency(builder, "freqB", ["freqB_label"], self.worlds)
        rename_builder_labels(builder, [_SRC], [_DST], self.env_ids, self.mapping)

        for freq, columns in {"freqA": ("freqA_label", "freqA_alt"), "freqB": ("freqB_label",)}.items():
            worlds = builder.custom_attributes[f"syn:{freq}_world"].values
            for column in columns:
                self.assertEqual(
                    builder.custom_attributes[f"syn:{column}"].values,
                    [f"{_DST.format(int(self.env_ids[w]))}/{column}_{int(w)}" for w in worlds],
                )

    def test_custom_string_columns_ignore_unset_world_rows(self):
        builder = newton.ModelBuilder()
        _add_custom_frequency(builder, "freqA", ["freqA_label"])
        builder.custom_attributes["syn:freqA_world"].values = [None, self.worlds[0]]
        builder.custom_attributes["syn:freqA_label"].values = ["unassigned", f"{_SRC}/freqA_label_{self.worlds[0]}"]
        builder._custom_frequency_counts["syn:freqA"] = 2

        rename_builder_labels(builder, [_SRC], [_DST], self.env_ids, self.mapping)

        self.assertEqual(
            builder.custom_attributes["syn:freqA_label"].values,
            ["unassigned", f"{_DST.format(int(self.env_ids[0]))}/freqA_label_{self.worlds[0]}"],
        )

    def test_shape_material_paths_follow_shape_worlds(self):
        builder = _make_builder(self.worlds)
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="visual_material_path",
                namespace="isaaclab",
                dtype=str,
                frequency=newton.Model.AttributeFrequency.SHAPE,
                default="",
            )
        )
        paths = builder.custom_attributes["isaaclab:visual_material_path"].values
        paths.update({index: f"{_SRC}/Looks/material" for index in range(len(self.worlds))})

        rename_builder_labels(builder, [_SRC], [_DST], self.env_ids, self.mapping)

        self.assertEqual(
            paths, {index: f"{_DST.format(int(self.env_ids[index]))}/Looks/material" for index in self.worlds}
        )

    def test_other_shape_attributes_without_world_references_pass_through(self):
        builder = _make_builder(self.worlds)
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name="shape_note",
                namespace="syn",
                dtype=str,
                frequency=newton.Model.AttributeFrequency.SHAPE,
                default="",
            )
        )
        notes = builder.custom_attributes["syn:shape_note"].values
        notes.update({index: f"{_SRC}/note" for index in range(len(self.worlds))})

        rename_builder_labels(builder, [_SRC], [_DST], self.env_ids, self.mapping)

        self.assertEqual(notes, {index: f"{_SRC}/note" for index in range(len(self.worlds))})


class TestReplicateBuilderMapping(unittest.TestCase):
    @staticmethod
    def _source_builder(root_path: str):
        builder = _FakeVisualizationModelBuilder()
        builder.add_usd(None, root_path=root_path)
        return builder

    def test_source_local_sites_batched_with_correct_indices(self):
        source_path, destination = "/World/envs/env_0", "/World/envs/env_{}"
        source = newton.ModelBuilder()
        source.add_body(xform=wp.transform((2.0, 0.0, 0.0), wp.quat_identity()))
        site_idx = source.add_site(body=0, xform=wp.transform(), label="ee")

        # Non-zero base so site indices are not trivially zero-based.
        builder = newton.ModelBuilder()
        builder.add_body(xform=wp.transform())
        builder.add_shape(body=0, type=newton.GeoType.SPHERE)
        base_shape = builder.shape_count
        stride = source.shape_count

        positions = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=np.float32)
        quaternions = np.array([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=np.float32)

        with mock.patch.object(builder, "replicate", wraps=builder.replicate) as replicate:
            local_site_map, _, _ = replicate_builder_mapping(
                builder,
                (source_path,),
                np.ones((1, 3), dtype=np.bool_),
                positions,
                quaternions,
                {source_path: source},
                destinations=(destination,),
                env_ids=np.arange(3, dtype=np.int64),
                source_site_indices={id(source): {"ee": [site_idx]}},
            )

        replicate.assert_called_once()
        self.assertEqual(
            local_site_map["ee"],
            [[base_shape + world * stride + site_idx] for world in range(3)],
        )
        for world, world_indices in enumerate(local_site_map["ee"]):
            self.assertEqual(builder.shape_label[world_indices[0]], f"/World/envs/env_{world}/ee")

    def test_env_root_sites_batched_at_correct_world_positions(self):
        source_path, destination = "/World/envs/env_0", "/World/envs/env_{}"
        source = newton.ModelBuilder()
        source.add_body(xform=wp.transform((2.0, 0.0, 0.0), wp.quat_identity()))

        builder = newton.ModelBuilder()
        base_shape = builder.shape_count
        positions = np.array([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=np.float32)
        quaternions = np.array([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=np.float32)
        env_root_offset = wp.transform((0.1, 0.0, 0.0), wp.quat_identity())

        with mock.patch.object(builder, "replicate", wraps=builder.replicate) as replicate:
            local_site_map, _, _ = replicate_builder_mapping(
                builder,
                (source_path,),
                np.ones((1, 3), dtype=np.bool_),
                positions,
                quaternions,
                {source_path: source},
                destinations=(destination,),
                env_ids=np.arange(3, dtype=np.int64),
                env_root_sites={"origin": env_root_offset},
            )

        replicate.assert_called_once()
        stride = source.shape_count
        self.assertEqual(source.shape_count, 1)
        self.assertEqual(
            local_site_map["origin"],
            [[base_shape + world * stride] for world in range(3)],
        )
        for world, world_indices in enumerate(local_site_map["origin"]):
            site_pos = builder.shape_transform[world_indices[0]].p
            self.assertAlmostEqual(float(site_pos[0]), float(positions[world][0]) + 0.1, places=5)
            self.assertAlmostEqual(float(site_pos[1]), 0.0, places=5)
            self.assertEqual(builder.shape_label[world_indices[0]], f"/World/envs/env_{world}/origin")

    def test_inactive_source_rows_are_ignored(self):
        sources = ("/World/envs/env_0/inactive", "/World/envs/env_0/active")
        source_builders = {source: self._source_builder(source) for source in sources}
        source_builders[sources[0]].body_label.append("/outside/the/plan")
        builder = _FakeVisualizationModelBuilder()

        replicate_builder_mapping(
            builder,
            sources,
            np.array([[False, False], [True, False]], dtype=np.bool_),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            source_builders,
            destinations=("/World/envs/env_{}/inactive", "/World/envs/env_{}/active"),
            env_ids=np.arange(2, dtype=np.int64),
        )

        self.assertEqual(builder.geometry_sources_for_world(0), ["/World/envs/env_0/active"])
        self.assertEqual(builder.geometry_sources_for_world(1), [])


class TestVisualizationClonePlan(unittest.TestCase):
    def test_clone_plan_expands_prototype_deformables_to_selected_environments(self):
        entry = DeformableStageEntry(
            root_path="/World/envs/env_0/Deformable",
            sim_mesh_path="/World/envs/env_0/Deformable/simulation_mesh",
            vis_mesh_path="/World/envs/env_0/Deformable/visual_mesh",
            deformable_type="surface",
            vertex_count=3,
            vis_vertex_count=3,
        )
        clone_plan = ClonePlan(
            sources=("/World/envs/env_0",),
            destinations=("/World/envs/env_{}",),
            clone_mask=np.ones((1, 4), dtype=np.bool_),
            env_ids=np.arange(4, dtype=np.int64),
        )

        entries = visualization_deformables_module._expand_clone_plan_deformable_entries([entry], clone_plan)

        self.assertEqual(
            [entry.root_path for entry in entries],
            [f"/World/envs/env_{env_id}/Deformable" for env_id in range(4)],
        )
        self.assertEqual(
            [entry.vis_mesh_path for entry in entries],
            [f"/World/envs/env_{env_id}/Deformable/visual_mesh" for env_id in range(4)],
        )

    @staticmethod
    def _define_xform(stage, path, translation=None):
        xform = UsdGeom.Xform.Define(stage, path)
        if translation is not None:
            xform.AddTranslateOp().Set(translation)

    def test_visualization_builder_imports_standalone_stage_as_one_world(self):
        stage = Usd.Stage.CreateInMemory()
        self._define_xform(stage, "/World")
        self._define_xform(stage, "/World/Robot")
        builder = mock.Mock()
        builder.shape_collision_filter_pairs = []
        builder.shape_collision_group = []
        builder.shape_count = 0
        builder.add_usd.return_value = {"path_shape_map": {}}

        with (
            mock.patch.object(visualization_builder_module, "ModelBuilder", return_value=builder),
            mock.patch.object(visualization_builder_module, "SchemaResolverNewton", lambda: "newton"),
            mock.patch.object(visualization_builder_module, "SchemaResolverPhysx", lambda: "physx"),
            mock.patch.object(visualization_builder_module, "import_builder_visual_material_paths"),
        ):
            result, (shadow_entities, registry_groups) = (
                visualization_builder_module.build_visualization_builder_from_stage_envs(stage, [], None)
            )

        self.assertIs(result, builder)
        self.assertEqual(shadow_entities, [])
        self.assertEqual(registry_groups, [])
        builder.add_usd.assert_called_once_with(stage, schema_resolvers=["newton", "physx"], ignore_paths=None)

    def test_visualization_builder_disables_collision_pairs(self):
        stage = Usd.Stage.CreateInMemory()
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

        clone_plan = ClonePlan(
            sources=(robot_path,),
            destinations=("/World/envs/env_{}/Robot",),
            clone_mask=torch.ones((1, 2), dtype=torch.bool),
            env_ids=torch.arange(2),
        )
        for env_paths, plan, expected_shape_count in (
            ([], None, 2),
            ([(0, "/World/envs/env_0"), (1, "/World/envs/env_1")], clone_plan, 4),
        ):
            builder, _shadow_metadata = visualization_builder_module.build_visualization_builder_from_stage_envs(
                stage, env_paths, plan
            )
            model = builder.finalize(device="cpu")

            self.assertEqual(model.shape_count, expected_shape_count)
            self.assertEqual(len(model.shape_collision_filter_pairs), 0)
            self.assertEqual(model.shape_contact_pair_count, 0)

    def test_visualization_builder_rejects_clone_plan_without_environment_paths(self):
        """A cloned scene must not be cached as an incomplete single-world model."""
        stage = Usd.Stage.CreateInMemory()
        self._define_xform(stage, "/World")
        clone_plan = ClonePlan(
            sources=(),
            destinations=(),
            clone_mask=np.empty((0, 0), dtype=np.bool_),
            env_ids=np.empty(0, dtype=np.int64),
        )

        with (
            mock.patch.object(visualization_builder_module, "SchemaResolverNewton", lambda: object()),
            mock.patch.object(visualization_builder_module, "SchemaResolverPhysx", lambda: object()),
            self.assertRaisesRegex(ValueError, "requires at least one environment path"),
        ):
            visualization_builder_module.build_visualization_builder_from_stage_envs(stage, [], clone_plan)

    def test_visualization_builder_uses_clone_plan_sources_and_rewrites_labels(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        self._define_xform(stage, "/World")
        self._define_xform(stage, "/World/envs")
        env_paths = [(env_id, f"/World/envs/env_{env_id}") for env_id in (0, 1, 2)]
        for env_id, env_path in env_paths:
            self._define_xform(stage, env_path, (float(env_id) * 3.0, 0.0, 0.0))
            self._define_xform(stage, f"{env_path}/Object")
        self._define_xform(stage, "/World/envs/env_0/Object/source_0_visual")
        self._define_xform(stage, "/World/envs/env_1/Object/source_1_visual")

        clone_plan = ClonePlan(
            sources=("/World/envs/env_0/Object", "/World/envs/env_1/Object"),
            destinations=("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
            clone_mask=np.array([[True, False, True], [False, True, False]], dtype=np.bool_),
            env_ids=np.array([0, 1, 2], dtype=np.int64),
        )

        with (
            mock.patch.object(visualization_builder_module, "ModelBuilder", _FakeVisualizationModelBuilder),
            mock.patch.object(newton_clone_utils_module, "ModelBuilder", _FakeVisualizationModelBuilder),
            mock.patch.object(visualization_builder_module, "SchemaResolverNewton", lambda: object()),
            mock.patch.object(visualization_builder_module, "SchemaResolverPhysx", lambda: object()),
            mock.patch.object(visualization_builder_module, "import_builder_visual_material_paths"),
            mock.patch.object(newton_clone_utils_module, "import_builder_visual_material_paths"),
            mock.patch.object(newton_clone_utils_module, "replace_newton_builder_shape_colors"),
        ):
            builder, _shadow_metadata = visualization_builder_module.build_visualization_builder_from_stage_envs(
                stage, env_paths, clone_plan
            )

        self.assertEqual(
            [builder.geometry_sources_for_world(i) for i in range(3)],
            [["/World/envs/env_0/Object"], ["/World/envs/env_1/Object"], ["/World/envs/env_0/Object"]],
        )
        for attr, suffix in _VIS_LABEL_SUFFIXES.items():
            self.assertEqual(
                [builder.labels_for_world(i, attr) for i in range(3)],
                [
                    [f"/World/envs/env_0/Object/{suffix}"],
                    [f"/World/envs/env_1/Object/{suffix}"],
                    [f"/World/envs/env_2/Object/{suffix}"],
                ],
            )


class TestReplicationNamesItsCopies(unittest.TestCase):
    _SRC = "/World/envs/env_0/Robot"
    _ENV = "/World/envs/env_{}"

    def test_batched_prefixes_name_each_world_and_preserve_the_prototype(self):
        source = newton.ModelBuilder()
        body = source.add_body(xform=wp.transform(), label=self._SRC)
        source.add_shape_box(body=body, label=f"{self._SRC}/shape")
        child = source.add_link(xform=wp.transform(), label=f"{self._SRC}/link")
        source.add_joint_revolute(parent=body, child=child, axis=(0.0, 0.0, 1.0), label=f"{self._SRC}/hinge")
        original = {
            name: list(getattr(source, name))
            for name in ("body_label", "joint_label", "shape_label", "articulation_label")
        }
        builder = newton.ModelBuilder()
        env_ids = np.array([10, 20], dtype=np.int64)
        mapping = np.ones((1, len(env_ids)), dtype=np.bool_)
        positions = np.zeros((len(env_ids), 3), dtype=np.float32)
        quaternions = np.zeros((len(env_ids), 4), dtype=np.float32)
        quaternions[:, 3] = 1.0
        replicate_builder_mapping(
            builder,
            [self._SRC],
            mapping,
            positions,
            quaternions,
            {self._SRC: source},
            destinations=["/World/envs/env_{}/Robot"],
            env_ids=env_ids,
        )
        for name, source_labels in original.items():
            expected = [
                label.replace(self._SRC, f"{self._ENV.format(i)}/Robot", 1) for i in env_ids for label in source_labels
            ]
            self.assertEqual(getattr(builder, name), expected)
            self.assertEqual(getattr(source, name), source_labels)

    def test_hook_labels_are_rewritten_after_the_slow_path(self):
        source = newton.ModelBuilder()
        source.add_body(label=f"{self._SRC}/base")
        builder = newton.ModelBuilder()
        env_ids = np.array([10, 20], dtype=np.int64)
        mapping = np.ones((1, 2), dtype=np.bool_)

        def hook(builder, *_):
            builder.add_body(label=f"{self._SRC}/hook")

        replicate_builder_mapping(
            builder,
            (self._SRC,),
            mapping,
            np.zeros((2, 3), dtype=np.float32),
            np.array([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=np.float32),
            {self._SRC: source},
            destinations=("/World/envs/env_{}/Robot",),
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
