# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.cloner import CloneCfg, InclusionSet
from isaaclab.cloner import add as clone_add
from isaaclab.utils import find_unique_string_name
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.assets import AssetBaseCfg

    from .interactive_scene import InteractiveScene


@configclass
class InteractiveSceneCfg:
    """Configuration for the interactive scene.

    The users can inherit from this class to add entities to their scene. This is then parsed by the
    :class:`InteractiveScene` class to create the scene.

    .. note::
        The adding of entities to the scene is sensitive to the order of the attributes in the configuration.
        Please make sure to add the entities in the order you want them to be added to the scene.
        The recommended order of specification is terrain, physics-related assets (articulations and rigid bodies),
        sensors and non-physics-related assets (lights).

    For example, to add a robot to the scene, the user can create a configuration class as follows:

    .. code-block:: python

        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sensors.ray_caster import GridPatternCfg, RayCasterCfg
        from isaaclab.utils.configclass import configclass

        from isaaclab_assets.robots.anymal import ANYMAL_C_CFG


        @configclass
        class MySceneCfg(InteractiveSceneCfg):
            # terrain - flat terrain plane
            terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="plane",
            )

            # articulation - robot 1
            robot_1 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_1")
            # articulation - robot 2
            robot_2 = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot_2")
            robot_2.init_state.pos = (0.0, 1.0, 0.6)

            # sensor - ray caster attached to the base of robot 1 that scans the ground
            height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot_1/base",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
                debug_vis=True,
                mesh_prim_paths=["/World/ground"],
            )

            # extras - light
            light = AssetBaseCfg(
                prim_path="/World/light",
                spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
            )

    """

    class_type: type[InteractiveScene] | str = "{DIR}.interactive_scene:InteractiveScene"
    """The class to use for the interactive scene.

    Defaults to :class:`isaaclab.scene.InteractiveScene`.
    """

    num_envs: int = MISSING
    """Number of environment instances handled by the scene."""

    env_spacing: float = MISSING
    """Spacing between environments.

    This is the default distance between environment origins in the scene. Used only when the
    number of environments is greater than one.
    """

    lazy_sensor_update: bool = True
    """Whether to update sensors only when they are accessed. Default is True.

    If true, the sensor data is only updated when their attribute ``data`` is accessed. Otherwise, the sensor
    data is updated every time sensors are updated.
    """

    replicate_physics: bool = True
    """Enable/disable replication of physics schemas when using the Cloner APIs. Default is True.

    If True, the simulation will have the same asset instances (USD prims) in all the cloned environments.
    Internally, this ensures optimization in setting up the scene and parsing it via the physics stage parser.

    If False, the simulation allows having separate asset instances (USD prims) in each environment.
    This flexibility comes at a cost of slowdowns in setting up and parsing the scene.

    .. note::
        Optimized parsing of certain prim types (such as deformable objects) is not currently supported
        by the physics engine. In these cases, this flag needs to be set to False.

    .. attention::
        Setting this flag to False is currently not supported on the Newton physics backend:
        Newton discovers the scene through its replication path, which stage parsing cannot
        replace for cloned environments.

    .. note::
        The scene pipes this flag into :attr:`~isaaclab.cloner.CloneCfg.replicate_physics`;
        the policy is applied by :func:`~isaaclab.cloner.replicate`. Direct workflows that
        call :func:`~isaaclab.cloner.replicate` themselves pass ``replicate_physics``
        explicitly.
    """

    filter_collisions: bool = True
    """Enable/disable collision filtering between cloned environments. Default is True.

    If True, collisions will not occur between cloned environments.

    If False, the simulation will generate collisions between environments.

    .. note::
        Collisions can only be filtered automatically in direct workflows when physics replication is enabled.
        If :attr:`replicated_physics` is ``False`` and collision filtering is desired, make sure to call
        ``scene.filter_collisions()``.
    """

    clone_in_fabric: bool = False
    """Deprecated legacy Fabric cloning flag. Default is False.

    Queued replication no longer forwards this flag to the PhysX replicator;
    ``useFabricForReplication`` is always ``False``.
    """

    clone_cfg: CloneCfg = CloneCfg()
    """Clone execution and legal scene-combination configuration."""


def add(
    target: InteractiveSceneCfg,
    source: InteractiveSceneCfg,
    *,
    asset_skip: Callable[[AssetBaseCfg], bool] | None = None,
) -> InteractiveSceneCfg:
    """Fold the source scene's environment assets into ``target`` as a new clone combination.

    ``target`` accumulates the fold: it keeps its execution settings and clone
    combinations, gains the source scene's assets, and is returned. Both scenes
    contribute spawned :class:`~isaaclab.assets.AssetBaseCfg` fields at literal
    ``{ENV_REGEX_NS}/Leaf`` roots. An asset equal to an existing binding reuses
    it (each binding at most once per call); other assets receive a unique field
    name and prim path. Sensors, terrain importers, rigid-object collections,
    and spawnless assets do not participate and are cleared from ``target``.
    Global assets are not composed: skip them with :paramref:`asset_skip` and
    attach shared world assets after the fold. Both operands are consumed.

    Args:
        target: Scene that accumulates the fold.
        source: Scene whose environment assets are added. It must not declare
            clone combinations of its own.
        asset_skip: Optional predicate called with each spawned asset
            configuration. An asset is omitted when the predicate returns
            :data:`True`. The predicate must not mutate the configuration.

    Returns:
        ``target``, extended with the source scene's assets and one new clone combination.

    Raises:
        ValueError: If a scene has no environment asset or an asset root is
            global or malformed, or if the added scene declares clone combinations.
    """
    if source.clone_cfg.clone_combinations:
        raise ValueError("the added scene must not declare clone combinations; fold it as the base scene instead.")

    target_assets = _scene_assets(target, asset_skip)
    source_assets = _scene_assets(source, asset_skip)
    if not target_assets or not source_assets:
        raise ValueError("both scenes must contain at least one spawned environment asset.")

    # ``target`` accumulates only the participating assets: clear everything else
    base_fields = InteractiveSceneCfg.__dataclass_fields__
    target_names = {name for name, _ in target_assets}
    for name in [
        n for n, v in vars(target).items() if n not in base_fields and v is not None and n not in target_names
    ]:
        setattr(target, name, None)

    # an empty clone configuration is homogeneous: make the base combination explicit
    if not target.clone_cfg.clone_combinations:
        clone_add(target.clone_cfg, InclusionSet(assets=[name for name, _ in target_assets]))

    # a duplicate is an asset equal to an existing binding; each binding is
    # reused at most once per call so distinct twins survive
    available = dict(target_assets)
    added_names: list[str] = []
    used_names = set(dir(target))
    paths = {cfg.prim_path for _, cfg in target_assets}
    for source_name, cfg in source_assets:
        target_name = next((name for name, existing in available.items() if existing == cfg), None)
        if target_name is not None:
            del available[target_name]
        else:
            target_name = find_unique_string_name(source_name, lambda name: name not in used_names)
            if cfg.prim_path in paths:
                cfg = cfg.replace(prim_path=find_unique_string_name(cfg.prim_path, lambda path: path not in paths))
            setattr(target, target_name, cfg)
            used_names.add(target_name)
            paths.add(cfg.prim_path)
        added_names.append(target_name)

    clone_add(target.clone_cfg, InclusionSet(assets=list(dict.fromkeys(added_names))))
    return target


def _scene_assets(
    scene_cfg: InteractiveSceneCfg,
    asset_skip: Callable[[AssetBaseCfg], bool] | None,
) -> list[tuple[str, AssetBaseCfg]]:
    """List the environment-scoped spawned assets of one scene."""
    # Deferred: config construction must stay importable without isaaclab.assets
    # (kitless factories); only scene composition pays for it.
    from isaaclab.assets import AssetBaseCfg  # noqa: PLC0415

    env_root = "{ENV_REGEX_NS}/"
    base_fields, assets = InteractiveSceneCfg.__dataclass_fields__, []
    for name, value in vars(scene_cfg).items():
        # only spawned assets compose; sensors, terrains, and collections are not AssetBaseCfg
        if name in base_fields or not isinstance(value, AssetBaseCfg) or value.spawn is None:
            continue
        if asset_skip is not None and asset_skip(value):
            continue
        path = value.prim_path
        if not (isinstance(path, str) and path.startswith(env_root) and path[len(env_root) :].isidentifier()):
            raise ValueError(
                f"scene composition requires assets at literal '{{ENV_REGEX_NS}}/Leaf' roots; {name!r} uses"
                f" {path!r}. Skip global assets with asset_skip and attach shared world assets to the composed scene."
            )
        assets.append((name, value))
    return assets
