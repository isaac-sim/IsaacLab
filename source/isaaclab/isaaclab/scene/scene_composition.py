# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.cloner import InclusionSet
from isaaclab.sensors import SensorBaseCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import find_unique_string_name
from isaaclab.utils.configclass import configclass

from .interactive_scene_cfg import InteractiveSceneCfg

_ENV_ROOT = "{ENV_REGEX_NS}/"
_SLOT_METADATA = "_scene_add_slots"


def scene_add(
    scene_cfg1: InteractiveSceneCfg,
    scene_cfg2: InteractiveSceneCfg,
    *,
    asset_skip: Callable[[SpawnerCfg], bool] | None = None,
) -> InteractiveSceneCfg:
    """Add a flat scene fragment as another clone combination.

    The first operand owns the scene execution settings. Both operands
    contribute spawned :class:`~isaaclab.assets.AssetBaseCfg` fields at
    literal, one-segment global or environment roots. When both operands
    declare global assets, their unordered asset sets must match exactly,
    including prim paths; field names do not affect that match. A global world
    declared by only one operand is carried into the result. Environment field
    names are logical slots: equal definitions in one slot reuse an existing
    binding, while different definitions receive a unique field name and, when
    needed, a unique prim path. Flat
    :class:`~isaaclab.terrains.TerrainImporterCfg` fields become bounded,
    environment-local ground assets. Sensors and spawnless assets are ignored.
    Each operand must contribute at least one spawned environment asset.

    Args:
        scene_cfg1: Scene containing the shared world and existing combinations.
        scene_cfg2: Scene whose supported environment assets are added.
        asset_skip: Optional predicate called with each spawned asset
            configuration. An asset is omitted when the predicate returns
            :data:`True`. Flat terrain importers are lowered before the
            predicate is called.

    Returns:
        A new scene configuration containing both operands' clone combinations.

    Raises:
        TypeError: If an operand or scene field type is unsupported, or a
            participating asset cannot be compared.
        ValueError: If global worlds differ, an operand has no environment
            asset or contains unsupported terrain or collections, a clone row
            becomes empty, or an asset has an invalid configuration.
    """
    for name, scene_cfg in (("scene_cfg1", scene_cfg1), ("scene_cfg2", scene_cfg2)):
        if not isinstance(scene_cfg, InteractiveSceneCfg):
            raise TypeError(f"{name} must be an InteractiveSceneCfg, got {type(scene_cfg).__name__}.")
        scene_cfg.validate()

    ground_size = min(2.0, scene_cfg1.env_spacing)
    left = _scene_assets(scene_cfg1, ground_size, asset_skip)
    right = _scene_assets(scene_cfg2, ground_size, asset_skip)
    for name, assets in (("scene_cfg1", left), ("scene_cfg2", right)):
        if not any(env_scoped for _, _, _, env_scoped in assets):
            raise ValueError(f"{name} must contain at least one spawned environment asset.")

    entities = {name: cfg for name, _, cfg, _ in left}
    definitions = {name: _asset_definition(cfg) for name, _, cfg, env_scoped in left if env_scoped}
    if not scene_cfg1.filter_collisions and any(
        env_scoped and isinstance(cfg.spawn, sim_utils.GroundPlaneCfg) for _, _, cfg, env_scoped in left + right
    ):
        raise ValueError("Environment-local ground planes require filter_collisions=True.")
    for name, _, cfg, env_scoped in left + right:
        if (
            env_scoped
            and isinstance(cfg.spawn, sim_utils.GroundPlaneCfg)
            and any(size > scene_cfg1.env_spacing for size in cfg.spawn.size)
        ):
            raise ValueError(f"Environment-local ground field {name!r} exceeds the output env_spacing.")
    slots = {name: slot for name, slot, _, _ in left}
    slot_scopes = {slot: env_scoped for _, slot, _, env_scoped in left}
    global_names = {slot: name for name, slot, _, env_scoped in left if not env_scoped}
    global_matches = _global_matches(left, right)
    paths = {cfg.prim_path for cfg in entities.values()}
    if len(paths) != len(entities):
        raise ValueError("scene_cfg1 contains duplicate asset roots.")

    used_names = set(dir(InteractiveSceneCfg)) | set(entities)
    right_name_map: dict[str, str] = {}
    for source_name, slot, cfg, env_scoped in right:
        if not env_scoped and source_name in global_matches:
            right_name_map[source_name] = global_matches[source_name]
            continue

        existing_scope = slot_scopes.get(slot)
        if existing_scope is not None and existing_scope != env_scoped:
            raise ValueError(f"Scene slot {slot!r} cannot mix global and environment-scoped assets.")

        if not env_scoped:
            target_name = global_names.get(slot)
            if target_name is not None:
                if not _configs_equal(entities[target_name], cfg):
                    raise ValueError(f"Global scene slot {slot!r} must match exactly across operands.")
                right_name_map[source_name] = target_name
                continue
            if cfg.prim_path in paths:
                raise ValueError(f"Global asset root {cfg.prim_path!r} is already bound to another scene slot.")

            target_name = find_unique_string_name(source_name, lambda name: name not in used_names)
            entities[target_name] = cfg
            slots[target_name] = slot
            slot_scopes[slot] = False
            global_names[slot] = target_name
            used_names.add(target_name)
            paths.add(cfg.prim_path)
            right_name_map[source_name] = target_name
            continue

        definition = _asset_definition(cfg)
        target_name = next(
            (
                name
                for name, existing in definitions.items()
                if slots[name] == slot and _configs_equal(existing, definition)
            ),
            None,
        )
        if target_name is None:
            target_name = find_unique_string_name(source_name, lambda name: name not in used_names)
            cfg.prim_path = find_unique_string_name(cfg.prim_path, lambda path: path not in paths)
            entities[target_name] = cfg
            definitions[target_name] = definition
            slots[target_name] = slot
            slot_scopes[slot] = True
            used_names.add(target_name)
            paths.add(cfg.prim_path)
        right_name_map[source_name] = target_name

    rows = _scene_rows(scene_cfg1, left)
    rows += _remap_rows(_scene_rows(scene_cfg2, right), right_name_map)
    return _make_scene(scene_cfg1, entities, slots, rows)


def _scene_assets(
    scene_cfg: InteractiveSceneCfg,
    ground_size: float,
    asset_skip: Callable[[SpawnerCfg], bool] | None,
) -> list[tuple[str, str, AssetBaseCfg, bool]]:
    """Copy supported assets and lower flat terrain importers."""
    base_fields = InteractiveSceneCfg.__dataclass_fields__
    slots = getattr(type(scene_cfg), _SLOT_METADATA, {})
    assets = []
    for name, value in vars(scene_cfg).items():
        if name in base_fields or value is None:
            continue
        if isinstance(value, TerrainImporterCfg):
            if value.terrain_type != "plane":
                raise ValueError(
                    f"scene_add does not support terrain field {name!r} with terrain_type={value.terrain_type!r}."
                )
            material = value.visual_material.to_dict() if value.visual_material is not None else {}
            value = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/GroundPlane",
                spawn=sim_utils.GroundPlaneCfg(
                    size=(ground_size, ground_size),
                    color=material.get("diffuse_color", (0.0, 0.0, 0.0)),
                    physics_material=copy.deepcopy(value.physics_material),
                ),
                collision_group=0,
            )
        elif isinstance(value, RigidObjectCollectionCfg):
            raise ValueError(f"scene_add does not support rigid-object collection field {name!r}.")
        elif isinstance(value, SensorBaseCfg):
            continue
        elif not isinstance(value, AssetBaseCfg):
            raise TypeError(f"scene_add does not support scene field {name!r} of type {type(value).__name__}.")
        if not isinstance(value, AssetBaseCfg) or value.spawn is None:
            continue
        if asset_skip is not None and asset_skip(value.spawn):
            continue
        if not isinstance(value.prim_path, str):
            raise TypeError(f"scene asset {name!r} must have a string prim_path.")

        env_scoped = value.prim_path.startswith(_ENV_ROOT)
        if env_scoped and not value.prim_path[len(_ENV_ROOT) :].isidentifier():
            raise ValueError(
                f"scene_add requires a literal, non-nested environment root; {name!r} uses {value.prim_path!r}."
            )
        if env_scoped and isinstance(value.spawn, sim_utils.GroundPlaneCfg) and value.collision_group != 0:
            raise ValueError(f"Environment-local ground field {name!r} must use collision_group=0.")
        if not env_scoped and "{ENV_REGEX_NS}" in value.prim_path:
            raise ValueError(f"scene_add requires an asset below the environment root, got {value.prim_path!r}.")
        if not env_scoped and (
            not value.prim_path.startswith("/World/") or not value.prim_path[len("/World/") :].isidentifier()
        ):
            raise ValueError(f"scene_add requires a literal one-segment global root, got {value.prim_path!r}.")
        cfg = copy.deepcopy(value)
        assets.append((name, slots.get(name, name), cfg, env_scoped))
    return assets


def _asset_definition(cfg: AssetBaseCfg) -> AssetBaseCfg:
    """Remove the output binding before comparing two asset configs."""
    definition = copy.deepcopy(cfg)
    definition.prim_path = "{SCENE_SLOT}"
    return definition


def _global_matches(
    left: list[tuple[str, str, AssetBaseCfg, bool]],
    right: list[tuple[str, str, AssetBaseCfg, bool]],
) -> dict[str, str]:
    """Match exact global asset sets when both operands declare globals."""
    left_globals = {name: cfg for name, _, cfg, env_scoped in left if not env_scoped}
    right_globals = {name: cfg for name, _, cfg, env_scoped in right if not env_scoped}
    if not left_globals or not right_globals:
        return {}
    if len(left_globals) != len(right_globals):
        raise ValueError("Global scene assets must match exactly across operands.")

    available = dict(left_globals)
    matches = {}
    for source_name, source_cfg in right_globals.items():
        target_name = next(
            (name for name, cfg in available.items() if _configs_equal(cfg, source_cfg)),
            None,
        )
        if target_name is None:
            raise ValueError("Global scene assets must match exactly across operands.")
        matches[source_name] = target_name
        del available[target_name]
    return matches


def _configs_equal(left: AssetBaseCfg, right: AssetBaseCfg) -> bool:
    """Compare two configs through their native equality implementation."""
    if type(left) is not type(right):
        return False
    try:
        result = left == right
    except Exception as exc:
        raise TypeError("scene_add requires asset configs with scalar native equality.") from exc
    if not isinstance(result, bool):
        raise TypeError("scene_add requires asset configs with scalar native equality.")
    return result


def _scene_rows(
    scene_cfg: InteractiveSceneCfg, assets: list[tuple[str, str, AssetBaseCfg, bool]]
) -> list[InclusionSet]:
    """Make one operand's clone rows self-contained."""
    cloneable_names = [name for name, _, _, env_scoped in assets if env_scoped]
    combinations = scene_cfg.clone_cfg.clone_combinations
    if not combinations:
        return [InclusionSet(assets=cloneable_names)] if cloneable_names else []

    base_fields = InteractiveSceneCfg.__dataclass_fields__
    declared_names = {name for name in vars(scene_cfg) if name not in base_fields}
    referenced_names = {name for row in combinations for name in row.assets}
    unknown = sorted(referenced_names - declared_names)
    if unknown:
        raise ValueError(f"Clone combinations reference unknown scene fields: {unknown}.")
    claimed = referenced_names.intersection(cloneable_names)
    rows = [
        InclusionSet(
            assets=[name for name in cloneable_names if name in row.assets or name not in claimed],
            weight=row.weight,
        )
        for row in combinations
    ]
    if any(row.weight > 0 and not row.assets for row in rows):
        raise ValueError("Clone combinations cannot become empty after unsupported scene fields are removed.")
    return rows


def _remap_rows(rows: list[InclusionSet], names: dict[str, str]) -> list[InclusionSet]:
    """Replace source field names with composed field names."""
    return [
        InclusionSet(assets=list(dict.fromkeys(names[name] for name in row.assets)), weight=row.weight) for row in rows
    ]


def _make_scene(
    source: InteractiveSceneCfg,
    entities: dict[str, AssetBaseCfg],
    slots: dict[str, str],
    rows: list[InclusionSet],
) -> InteractiveSceneCfg:
    """Build a configclass that InteractiveScene can parse normally."""
    clone_cfg = copy.deepcopy(source.clone_cfg)
    clone_cfg.clone_combinations = rows
    values = {name: copy.deepcopy(getattr(source, name)) for name in InteractiveSceneCfg.__dataclass_fields__}
    values["clone_cfg"] = clone_cfg
    namespace = {"__module__": __name__, "__doc__": "Scene configuration produced by scene_add.", **entities}
    scene_type = configclass(type("_AddedSceneCfg", (InteractiveSceneCfg,), namespace))
    scene_cfg = scene_type(**values)
    setattr(scene_type, _SLOT_METADATA, slots)
    return scene_cfg
