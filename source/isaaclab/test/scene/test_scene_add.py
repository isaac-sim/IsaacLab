# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the narrow scene composition contract."""

import pytest

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.cloner import CloneCfg, InclusionSet, make_valid_clone_combinations, sequential
from isaaclab.scene import InteractiveSceneCfg, scene_add
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.terrains import TerrainImporterCfg


def _asset(prim_path: str, *, size: float = 0.1) -> AssetBaseCfg:
    """Return a simple environment asset."""
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(size=(size, size, size)),
    )


def _variant_asset(prim_path: str) -> AssetBaseCfg:
    """Return an environment asset with two native spawn variants."""
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
                sim_utils.SphereCfg(radius=0.2),
            ]
        ),
    )


def _scene(
    *,
    clone_cfg: CloneCfg | None = None,
    num_envs: int = 8,
    env_spacing: float = 2.0,
    **entities: object,
) -> InteractiveSceneCfg:
    """Return a scene config with entities inserted in keyword order."""
    scene = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=env_spacing,
        clone_cfg=clone_cfg if clone_cfg is not None else CloneCfg(),
    )
    for name, cfg in entities.items():
        setattr(scene, name, cfg)
    return scene


def _combination_records(scene: InteractiveSceneCfg) -> list[tuple[list[str], int]]:
    """Return clone-combination assets and weights in declaration order."""
    return [(combination.assets, combination.weight) for combination in scene.clone_cfg.clone_combinations]


def _entity_names(scene: InteractiveSceneCfg) -> list[str]:
    """Return entity field names in declaration order."""
    base_fields = InteractiveSceneCfg.__dataclass_fields__
    return [name for name in vars(scene) if name not in base_fields]


def _light_fields(scene: InteractiveSceneCfg) -> dict[str, AssetBaseCfg]:
    """Return composed light fields."""
    return {
        name: value
        for name, value in vars(scene).items()
        if isinstance(value, AssetBaseCfg) and isinstance(value.spawn, sim_utils.LightCfg)
    }


def test_scene_add_keeps_left_world_and_adds_environment_alternatives():
    """Left-side globals remain active while each scene contributes one row."""
    left = _scene(
        ground=_asset("/World/Ground", size=10.0),
        light=AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DistantLightCfg()),
        robot=_asset("{ENV_REGEX_NS}/Robot"),
    )
    right = _scene(object=_asset("{ENV_REGEX_NS}/Object", size=0.2))

    combined = scene_add(left, right)

    assert _entity_names(combined) == ["ground", "light", "robot", "object"]
    assert _combination_records(combined) == [(["robot"], 1), (["object"], 1)]


def test_scene_add_shares_equal_definitions_in_the_same_logical_slot():
    """Root bindings do not affect equality within one logical slot."""
    left = _scene(robot=_asset("{ENV_REGEX_NS}/LeftRobot"))
    right = _scene(robot=_asset("{ENV_REGEX_NS}/RightRobot"))

    combined = scene_add(left, right)

    assert _entity_names(combined) == ["robot"]
    assert combined.robot.prim_path == "{ENV_REGEX_NS}/LeftRobot"
    assert _combination_records(combined) == [(["robot"], 1), (["robot"], 1)]


def test_scene_add_suffixes_different_definitions_in_the_same_slot():
    """Different definitions receive deterministic field and prim-path suffixes."""
    left = _scene(robot=_asset("{ENV_REGEX_NS}/Robot", size=0.1))
    right = _scene(robot=_asset("{ENV_REGEX_NS}/Robot", size=0.2))

    combined = scene_add(left, right)

    assert _entity_names(combined) == ["robot", "robot_1"]
    assert combined.robot_1.prim_path == "{ENV_REGEX_NS}/Robot_1"
    assert _combination_records(combined) == [(["robot"], 1), (["robot_1"], 1)]


def test_scene_add_does_not_share_different_logical_slots():
    """Different source field names stay distinct even when definitions match."""
    left = _scene(robot=_asset("{ENV_REGEX_NS}/Asset"))
    right = _scene(manipulator=_asset("{ENV_REGEX_NS}/Asset"))

    combined = scene_add(left, right)

    assert _entity_names(combined) == ["robot", "manipulator"]
    assert combined.manipulator.prim_path == "{ENV_REGEX_NS}/Asset_1"
    assert _combination_records(combined) == [(["robot"], 1), (["manipulator"], 1)]


def test_scene_add_reuses_an_earlier_slot_variant_when_chained():
    """A later operand can reuse any earlier definition from its logical slot."""
    first = _scene(robot=_asset("{ENV_REGEX_NS}/Robot", size=0.1))
    second = _scene(robot=_asset("{ENV_REGEX_NS}/Robot", size=0.2))
    third = _scene(robot=_asset("{ENV_REGEX_NS}/OtherRobot", size=0.1))

    combined = scene_add(scene_add(first, second), third)

    assert _entity_names(combined) == ["robot", "robot_1"]
    assert _combination_records(combined) == [
        (["robot"], 1),
        (["robot_1"], 1),
        (["robot"], 1),
    ]


def test_scene_add_normalizes_existing_inclusion_sets():
    """Always-active fields join every local row while weights and order survive."""
    left = _scene(
        clone_cfg=CloneCfg(
            clone_strategy=sequential,
            clone_combinations=[
                InclusionSet(assets=["choice"], weight=2),
                InclusionSet(assets=["alternative"], weight=3),
            ],
        ),
        common=_asset("{ENV_REGEX_NS}/Common", size=0.1),
        choice=_asset("{ENV_REGEX_NS}/Choice", size=0.2),
        alternative=_asset("{ENV_REGEX_NS}/Alternative", size=0.3),
    )
    right = _scene(
        clone_cfg=CloneCfg(clone_strategy=sequential),
        robot=_asset("{ENV_REGEX_NS}/Robot", size=0.4),
    )

    combined = scene_add(left, right)

    assert [(set(assets), weight) for assets, weight in _combination_records(combined)] == [
        ({"common", "choice"}, 2),
        ({"common", "alternative"}, 3),
        ({"robot"}, 1),
    ]
    assert combined.clone_cfg.clone_strategy is sequential


def test_scene_add_preserves_multi_asset_spawner_variants():
    """Composition remains compatible with the cloner's native variant expansion."""
    combined = scene_add(
        _scene(choice=_variant_asset("{ENV_REGEX_NS}/Choice")),
        _scene(robot=_asset("{ENV_REGEX_NS}/Robot")),
    )

    valid_set = make_valid_clone_combinations(
        ["choice", "robot"],
        [len(combined.choice.spawn.assets_cfg), 1],
        combined.clone_cfg.clone_combinations,
    )

    assert valid_set.tolist() == [[0, -1], [1, -1], [-1, 0]]


def test_scene_add_deep_copies_inputs():
    """Mutating an input or the result cannot affect the other configs."""
    left = _scene(robot=_asset("{ENV_REGEX_NS}/Robot", size=0.1))
    right = _scene(object=_asset("{ENV_REGEX_NS}/Object", size=0.2))

    combined = scene_add(left, right)
    left.robot.spawn.size = (0.3, 0.3, 0.3)
    right.object.prim_path = "{ENV_REGEX_NS}/Changed"
    combined.robot.prim_path = "{ENV_REGEX_NS}/Combined"

    assert combined.robot.spawn.size == (0.1, 0.1, 0.1)
    assert combined.object.prim_path == "{ENV_REGEX_NS}/Object"
    assert left.robot.prim_path == "{ENV_REGEX_NS}/Robot"


def test_scene_add_keeps_all_execution_settings_from_the_left_operand():
    """Right-side execution settings are ignored instead of becoming merge policy."""
    left = _scene(
        clone_cfg=CloneCfg(
            clone_strategy=sequential,
            device="cpu",
            clone_regex="/World/left/left_.*",
        ),
        num_envs=4,
        env_spacing=1.25,
        robot=_asset("{ENV_REGEX_NS}/Robot"),
    )
    left.lazy_sensor_update = False
    left.replicate_physics = False
    left.filter_collisions = False
    left.clone_in_fabric = True
    right = _scene(
        clone_cfg=CloneCfg(
            device="cuda:0",
            clone_regex="/World/right/right_.*",
        ),
        num_envs=9,
        env_spacing=4.5,
        object=_asset("{ENV_REGEX_NS}/Object"),
    )
    right.lazy_sensor_update = True
    right.replicate_physics = True
    right.filter_collisions = True
    right.clone_in_fabric = False

    combined = scene_add(left, right)

    for name in (
        "num_envs",
        "env_spacing",
        "lazy_sensor_update",
        "replicate_physics",
        "filter_collisions",
        "clone_in_fabric",
    ):
        assert getattr(combined, name) == getattr(left, name)
    for name in ("clone_strategy", "device", "clone_regex"):
        assert getattr(combined.clone_cfg, name) == getattr(left.clone_cfg, name)
    assert _combination_records(combined) == [(["robot"], 1), (["object"], 1)]


@pytest.mark.parametrize(
    "prim_path",
    [
        "{ENV_REGEX_NS}/Robot.*",
        "{ENV_REGEX_NS}/Robot[0-9]",
        "{ENV_REGEX_NS}/Robot?",
        "{ENV_REGEX_NS}/(Robot)",
    ],
)
def test_scene_add_rejects_regex_or_meta_environment_leaves(prim_path: str):
    """Environment asset roots must use one literal path segment."""
    left = _scene(robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(object=_asset(prim_path))

    with pytest.raises(ValueError, match="literal|metacharacter|regex"):
        scene_add(left, right)


@pytest.mark.parametrize("prim_path", ["/World/envs/env_.*/Robot", "/World/envs/env_0/Robot"])
def test_scene_add_rejects_expanded_environment_roots(prim_path: str):
    """Resolved environment paths cannot be mistaken for global assets."""
    left = _scene(robot=_asset(prim_path))
    right = _scene(object=_asset("{ENV_REGEX_NS}/Object"))

    with pytest.raises(ValueError, match="literal|global"):
        scene_add(left, right)
    with pytest.raises(ValueError, match="literal|global"):
        scene_add(right, left)


def test_scene_add_rejects_different_global_asset_sets():
    """Two declared global worlds must contain exactly equal assets."""
    left = _scene(
        ground=_asset("/World/Ground", size=10.0),
        robot=_asset("{ENV_REGEX_NS}/Robot"),
    )
    right = _scene(
        wall=_asset("/World/Wall", size=5.0),
        object=_asset("{ENV_REGEX_NS}/Object"),
    )

    with pytest.raises(ValueError, match="Global scene assets must match exactly"):
        scene_add(left, right)


def test_scene_add_skips_assets_selected_by_spawn_predicate():
    """The optional predicate can remove mismatched lights without adding policy to scene_add."""
    left = _scene(
        sky=AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                color=(1.0, 1.0, 1.0),
                intensity=750.0,
                texture_file="sky.hdr",
            ),
        ),
        robot=_asset("{ENV_REGEX_NS}/Robot"),
    )
    right = _scene(
        key=AssetBaseCfg(
            prim_path="/World/KeyLight",
            spawn=sim_utils.DistantLightCfg(color=(0.9, 0.8, 0.7), intensity=1000.0),
        ),
        fill=AssetBaseCfg(
            prim_path="/World/FillLight",
            spawn=sim_utils.DomeLightCfg(color=(0.1, 0.2, 0.3), intensity=500.0),
        ),
        object=_asset("{ENV_REGEX_NS}/Object"),
    )
    seen = []

    def skip_lights(asset: object) -> bool:
        seen.append(asset)
        return isinstance(asset, sim_utils.LightCfg)

    with pytest.raises(ValueError, match="Global scene assets must match exactly"):
        scene_add(left, right)

    combined = scene_add(left, right, asset_skip=skip_lights)

    assert [type(asset) for asset in seen] == [
        sim_utils.DomeLightCfg,
        sim_utils.CuboidCfg,
        sim_utils.DistantLightCfg,
        sim_utils.DomeLightCfg,
        sim_utils.CuboidCfg,
    ]
    assert _light_fields(combined) == {}
    assert _entity_names(combined) == ["robot", "object"]
    assert _combination_records(combined) == [(["robot"], 1), (["object"], 1)]
    assert left.sky.spawn.texture_file == "sky.hdr"
    assert isinstance(right.key.spawn, sim_utils.DistantLightCfg)


def test_scene_add_applies_asset_skip_to_each_chained_call():
    """A caller can use the same exclusion policy throughout a composition fold."""

    def skip_lights(asset: object) -> bool:
        return isinstance(asset, sim_utils.LightCfg)

    first = scene_add(
        _scene(
            sky=AssetBaseCfg(prim_path="/World/Sky", spawn=sim_utils.DomeLightCfg()),
            robot=_asset("{ENV_REGEX_NS}/Robot"),
        ),
        _scene(object=_asset("{ENV_REGEX_NS}/Object")),
        asset_skip=skip_lights,
    )
    combined = scene_add(
        first,
        _scene(
            fill=AssetBaseCfg(prim_path="/World/Fill", spawn=sim_utils.DistantLightCfg()),
            tool=_asset("{ENV_REGEX_NS}/Tool"),
        ),
        asset_skip=skip_lights,
    )

    assert _light_fields(combined) == {}
    assert _combination_records(combined) == [(["robot"], 1), (["object"], 1), (["tool"], 1)]


def test_scene_add_does_not_reserve_a_canonical_light_root():
    """The core operation should treat /World/light like every other global root."""
    left = _scene(marker=_asset("/World/light"), robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(object=_asset("{ENV_REGEX_NS}/Object"))

    combined = scene_add(left, right)

    assert combined.marker.prim_path == "/World/light"
    assert _entity_names(combined) == ["marker", "robot", "object"]


@pytest.mark.parametrize(
    ("right_path", "right_size"),
    [
        ("/World/Ground", 11.0),
        ("/World/OtherGround", 10.0),
    ],
)
def test_scene_add_rejects_nonidentical_global_slots(right_path: str, right_size: float):
    """A global slot must match in both its root binding and its definition."""
    left = _scene(ground=_asset("/World/Ground", size=10.0), robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(ground=_asset(right_path, size=right_size), object=_asset("{ENV_REGEX_NS}/Object"))

    with pytest.raises(ValueError, match="Global scene assets must match exactly"):
        scene_add(left, right)


def test_scene_add_matches_exact_global_assets_across_field_names():
    """Field names do not prevent two exactly equal global assets from being shared."""
    left = _scene(ground=_asset("/World/Ground", size=10.0), robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(floor=_asset("/World/Ground", size=10.0), object=_asset("{ENV_REGEX_NS}/Object"))

    combined = scene_add(left, right)

    assert _entity_names(combined) == ["ground", "robot", "object"]


@pytest.mark.parametrize(
    ("left_path", "right_path"),
    [
        ("/World/Asset", "{ENV_REGEX_NS}/Asset"),
        ("{ENV_REGEX_NS}/Asset", "/World/Asset"),
    ],
)
def test_scene_add_rejects_scope_changes_in_a_logical_slot(left_path: str, right_path: str):
    """One logical slot cannot alternate between global and environment scope."""
    left = _scene(asset=_asset(left_path), left_anchor=_asset("{ENV_REGEX_NS}/LeftAnchor"))
    right = _scene(asset=_asset(right_path), right_anchor=_asset("{ENV_REGEX_NS}/RightAnchor"))

    with pytest.raises(ValueError, match="cannot mix global and environment-scoped"):
        scene_add(left, right)


def test_scene_add_discards_sensor_fields():
    """Sensors do not participate in clone-only scene composition."""
    left = _scene(robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(
        ee_frame=FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/tool")],
        ),
        contacts=ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*"),
        object=_asset("{ENV_REGEX_NS}/Object"),
    )

    combined = scene_add(left, right)

    assert _entity_names(combined) == ["robot", "object"]
    assert _combination_records(combined) == [(["robot"], 1), (["object"], 1)]


def test_scene_add_lowers_flat_terrain_to_an_environment_ground():
    """A flat terrain importer should retain its physical and visible plane settings."""
    physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.6)
    terrain = TerrainImporterCfg(
        prim_path="/World/Terrain",
        terrain_type="plane",
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.3)),
        physics_material=physics_material,
    )
    left = _scene(env_spacing=1.5, robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(
        clone_cfg=CloneCfg(clone_combinations=[InclusionSet(assets=["terrain", "object"], weight=3)]),
        terrain=terrain,
        object=_asset("{ENV_REGEX_NS}/Object"),
    )

    combined = scene_add(left, right)

    assert _entity_names(combined) == ["robot", "terrain", "object"]
    assert combined.terrain.prim_path == "{ENV_REGEX_NS}/GroundPlane"
    assert combined.terrain.collision_group == 0
    assert combined.terrain.spawn.size == (1.5, 1.5)
    assert combined.terrain.spawn.color == (0.1, 0.2, 0.3)
    assert combined.terrain.spawn.physics_material == physics_material
    assert combined.terrain.spawn.physics_material is not terrain.physics_material
    assert _combination_records(combined) == [(["robot"], 1), (["terrain", "object"], 3)]


def test_scene_add_calls_asset_skip_after_flat_terrain_lowering():
    """The predicate should receive a concrete ground spawner for a plane terrain importer."""
    seen = []

    def skip_ground(asset: object) -> bool:
        seen.append(asset)
        return isinstance(asset, sim_utils.GroundPlaneCfg)

    combined = scene_add(
        _scene(robot=_asset("{ENV_REGEX_NS}/Robot")),
        _scene(
            terrain=TerrainImporterCfg(prim_path="/World/Terrain", terrain_type="plane"),
            object=_asset("{ENV_REGEX_NS}/Object"),
        ),
        asset_skip=skip_ground,
    )

    assert any(isinstance(asset, sim_utils.GroundPlaneCfg) for asset in seen)
    assert _entity_names(combined) == ["robot", "object"]


def test_scene_add_uses_native_flat_terrain_color_fallback():
    """A plane importer without a visual material should retain its black fallback."""
    combined = scene_add(
        _scene(robot=_asset("{ENV_REGEX_NS}/Robot")),
        _scene(terrain=TerrainImporterCfg(prim_path="/World/Terrain", terrain_type="plane", visual_material=None)),
    )

    assert combined.terrain.spawn.color == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("terrain_type", ["generator", "usd"])
def test_scene_add_rejects_nonflat_terrain(terrain_type: str):
    """Generated and USD terrain require composition semantics beyond flat scenes."""
    left = _scene(robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(terrain=TerrainImporterCfg(prim_path="/World/Terrain", terrain_type=terrain_type))

    with pytest.raises(ValueError, match=f"terrain.*terrain_type='{terrain_type}'"):
        scene_add(left, right)


def test_scene_add_rejects_rigid_object_collections():
    """Collections must not disappear while member-level composition is unsupported."""
    left = _scene(robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(
        objects=RigidObjectCollectionCfg(
            rigid_objects={
                "cube": RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/Cube",
                    spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
                )
            }
        )
    )

    with pytest.raises(ValueError, match="rigid-object collection.*objects"):
        scene_add(left, right)


def test_scene_add_discards_noncloneable_fields_from_combination_references():
    """Non-cloneable declared fields are removed when local rows become self-contained."""
    left = _scene(robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(
        clone_cfg=CloneCfg(
            clone_combinations=[
                InclusionSet(assets=["object", "existing", "contacts", "light"], weight=3),
            ]
        ),
        object=_asset("{ENV_REGEX_NS}/Object"),
        existing=AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Existing", spawn=None),
        contacts=ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Object/.*"),
        light=AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DistantLightCfg()),
    )

    combined = scene_add(left, right, asset_skip=lambda asset: isinstance(asset, sim_utils.LightCfg))

    assert _entity_names(combined) == ["robot", "object"]
    assert _combination_records(combined) == [(["robot"], 1), (["object"], 3)]


def test_scene_add_rejects_nested_spawned_asset_roots():
    """Spawned assets still require a literal one-segment environment root."""
    left = _scene(object=_asset("{ENV_REGEX_NS}/Object"))
    right = _scene(
        robot=_asset("{ENV_REGEX_NS}/Robot"),
        tool=_asset("{ENV_REGEX_NS}/Robot/Tool"),
    )

    with pytest.raises(ValueError, match="literal|nested|child"):
        scene_add(left, right)


def test_scene_add_rejects_global_collision_groups_on_environment_ground():
    """A per-environment ground must not collide with assets in other environments."""
    left = _scene(robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(
        ground=AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/GroundPlane",
            spawn=sim_utils.GroundPlaneCfg(size=(2.0, 2.0)),
            collision_group=-1,
        )
    )

    with pytest.raises(ValueError, match="ground.*collision_group=0"):
        scene_add(left, right)


def test_scene_add_requires_collision_filtering_for_environment_ground():
    """Infinite per-environment plane colliders require environment isolation."""
    left = _scene(robot=_asset("{ENV_REGEX_NS}/Robot"))
    left.filter_collisions = False
    right = _scene(
        ground=AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/GroundPlane",
            spawn=sim_utils.GroundPlaneCfg(size=(2.0, 2.0)),
            collision_group=0,
        )
    )

    with pytest.raises(ValueError, match="ground planes require filter_collisions=True"):
        scene_add(left, right)


def test_scene_add_rejects_unequal_global_asset_counts():
    """A partial non-light global-world match must not add another global asset."""
    shared = _asset("/World/Shared", size=1.0)
    left = _scene(shared=shared, robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(
        shared=shared,
        wall=_asset("/World/Wall", size=2.0),
        object=_asset("{ENV_REGEX_NS}/Object"),
    )

    with pytest.raises(ValueError, match="Global scene assets must match exactly"):
        scene_add(left, right)


def test_scene_add_rejects_unknown_non_asset_fields():
    """Unknown scene fields must not disappear from the composed result."""
    left = _scene(robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(metadata={"owner": "task"}, object=_asset("{ENV_REGEX_NS}/Object"))

    with pytest.raises(TypeError, match="scene field 'metadata'.*dict"):
        scene_add(left, right)


def test_scene_add_rejects_rows_that_become_empty():
    """A row containing only ignored fields must not become an empty world."""
    left = _scene(robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(
        clone_cfg=CloneCfg(
            clone_combinations=[
                InclusionSet(assets=["object"]),
                InclusionSet(assets=["contacts"]),
            ]
        ),
        object=_asset("{ENV_REGEX_NS}/Object"),
        contacts=ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Object/.*"),
    )

    with pytest.raises(ValueError, match="cannot become empty"):
        scene_add(left, right)


@pytest.mark.parametrize("empty_first", [True, False])
def test_scene_add_rejects_operands_without_environment_assets(empty_first: bool):
    """Every operand must contribute a concrete environment world."""
    empty = _scene()
    populated = _scene(robot=_asset("{ENV_REGEX_NS}/Robot"))
    operands = (empty, populated) if empty_first else (populated, empty)

    with pytest.raises(ValueError, match="spawned environment asset"):
        scene_add(*operands)


def test_scene_add_allows_zero_weight_rows_to_become_empty():
    """Ignored fields may leave an empty row when the native cloner skips it."""
    left = _scene(robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(
        clone_cfg=CloneCfg(
            clone_combinations=[
                InclusionSet(assets=["object"]),
                InclusionSet(assets=["contacts"], weight=0),
            ]
        ),
        object=_asset("{ENV_REGEX_NS}/Object"),
        contacts=ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Object/.*"),
    )

    combined = scene_add(left, right)
    valid_set = make_valid_clone_combinations(["robot", "object"], [1, 1], combined.clone_cfg.clone_combinations)

    assert _combination_records(combined) == [(["robot"], 1), (["object"], 1), ([], 0)]
    assert valid_set.tolist() == [[0, -1], [-1, 0]]


def test_scene_add_rejects_ground_visuals_larger_than_output_spacing():
    """Per-environment ground visuals must fit within the output grid spacing."""
    left = _scene(env_spacing=1.0, robot=_asset("{ENV_REGEX_NS}/Robot"))
    right = _scene(
        ground=AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/GroundPlane",
            spawn=sim_utils.GroundPlaneCfg(size=(2.0, 2.0)),
        )
    )

    with pytest.raises(ValueError, match="ground.*exceeds.*env_spacing"):
        scene_add(left, right)
