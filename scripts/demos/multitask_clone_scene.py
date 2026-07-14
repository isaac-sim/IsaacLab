# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose a multi-robot scene from task configs and step physics only.

This demo resolves registered task configurations and combines their scenes
into heterogeneous clone combinations without constructing task environments.
No action, command, observation, reward, termination, event, or curriculum
managers are created.

Each task configuration supplies its scene directly. The
:func:`~isaaclab.scene.scene_add` operation deduplicates equivalent scene
entities and assigns unique bindings to non-equivalent collisions.
The demo owns generic PhysX simulation settings and discards task MDP and
simulator configuration. It skips every input light through ``scene_add``'s
asset predicate, then adds one global Dome light. By default, every supported
registered flat PhysX task scene is loaded; Newton scenes, unsupported scenes,
and known embedded-camera USD assets are reported and excluded.
The selected environment count must cover every spawn variant and weighted clone
combination. Omit ``--num_envs`` to size it automatically. The default launch is
large; use ``--num_task`` for a smaller smoke test.

.. note::
    This demo targets the PhysX backend. PhysX allows each environment to hold a
    different set of articulations, which is exactly what the heterogeneous clone
    composition produces. The ``--physics newton_mjwarp`` backend is wired up (and
    fixed-base articulations spawn correctly on it), but Newton's ``ArticulationView``
    requires every environment to contain an identical articulation count and topology
    and raises ``ValueError: Varying articulation counts per world are not supported``
    otherwise. Newton therefore only works here for a homogeneous composition (every
    environment gets the same task assets); the heterogeneous multitask case must use
    PhysX.

Usage:

.. code-block:: bash

    # Default PhysX physics with the Kit visualizer.
    ./isaaclab.sh -p scripts/demos/multitask_clone_scene.py --visualizer kit

    ./isaaclab.sh -p scripts/demos/multitask_clone_scene.py --num_task 3 \
        --visualizer kit --num_envs 3

    ./isaaclab.sh -p scripts/demos/multitask_clone_scene.py --task TASK_ID \
        --task ANOTHER_TASK_ID --visualizer kit --num_envs 8

    # Kitless Newton (MJWarp) physics with the Newton visualizer (no Isaac Sim).
    # Note: only supported for homogeneous compositions (identical articulations per env).
    ./isaaclab.sh -p scripts/demos/multitask_clone_scene.py \
        --physics newton_mjwarp --visualizer newton

"""

from __future__ import annotations

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Demo: clone-only multi-robot multi-task scene.")
parser.add_argument("--physics", default="physx", choices=["physx", "newton_mjwarp"], help="Physics backend.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of environments. Omit to use one environment per expanded clone combination.",
)
parser.add_argument("--env_spacing", type=float, default=2.5, help="Distance between environment origins [m].")
parser.add_argument("--sim_dt", type=float, default=1.0 / 60.0, help="Physics timestep [s].")
parser.add_argument(
    "--task",
    dest="task_ids",
    action="append",
    metavar="TASK_ID",
    help="Registered task whose scene to include. Repeat to select at least two tasks.",
)
parser.add_argument(
    "--num_task",
    type=int,
    default=None,
    help="Number of tasks to use from the selected/default order. Omit to use all tasks.",
)
parser.add_argument(
    "--disable_replicate_physics",
    action="store_true",
    help="Set replicate_physics=False on the composed scene.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=-1,
    help="Maximum physics steps before exit. Negative values run until the app closes.",
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]
if args_cli.num_task is not None and args_cli.num_task < 2:
    parser.error("--num_task must be at least 2")
if args_cli.task_ids is not None:
    if len(args_cli.task_ids) < 2:
        parser.error("select at least two tasks with repeated --task arguments")
    if len(set(args_cli.task_ids)) != len(args_cli.task_ids):
        parser.error("--task values must be unique")
    if args_cli.num_task is not None and args_cli.num_task > len(args_cli.task_ids):
        parser.error(f"--num_task cannot exceed the {len(args_cli.task_ids)} explicitly selected tasks")

"""Everything else follows."""

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import make_valid_clone_combinations, sequential
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg, scene_add
from isaaclab.sim import SimulationContext
from isaaclab.terrains import TerrainImporterCfg

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # isort:skip

from isaaclab_tasks.utils import resolve_task_config

_BANNED_USD_PATH_SUFFIXES = {
    "/Robots/Agility/Digit/digit_v4.usd": "contains embedded cameras with unsupported RTX projection metadata",
}


def _spawn_usd_paths(spawn_cfg: object) -> list[str]:
    """Return every USD path referenced by one spawned asset configuration."""
    if isinstance(spawn_cfg, sim_utils.MultiAssetSpawnerCfg):
        return [path for asset_cfg in spawn_cfg.assets_cfg for path in _spawn_usd_paths(asset_cfg)]
    if isinstance(spawn_cfg, sim_utils.MultiUsdFileCfg):
        return [spawn_cfg.usd_path] if isinstance(spawn_cfg.usd_path, str) else list(spawn_cfg.usd_path)
    if isinstance(spawn_cfg, sim_utils.UsdFileCfg):
        return [spawn_cfg.usd_path]
    return []


def _skip_light(asset: object) -> bool:
    """Return whether a spawned asset is a light source omitted by this demo."""
    return isinstance(asset, sim_utils.LightCfg)


def _registered_task_ids() -> list[str]:
    """Return registered task IDs owned by isaaclab_tasks."""
    task_ids = []
    for task_spec in gym.registry.values():
        entry_point = task_spec.kwargs.get("env_cfg_entry_point")
        if isinstance(entry_point, str):
            module_name = entry_point.split(":", maxsplit=1)[0]
        else:
            module_name = getattr(entry_point, "__module__", type(entry_point).__module__)
        if module_name.startswith("isaaclab_tasks."):
            task_ids.append(task_spec.id)
    return sorted(task_ids)


def _flat_physx_rejection(env_cfg: object) -> str | None:
    """Return why an environment config is outside this demo's flat PhysX scope."""
    physics_cfg = env_cfg.sim.physics
    if physics_cfg is not None and any(
        cls.__module__.startswith("isaaclab_newton.") for cls in type(physics_cfg).__mro__
    ):
        return f"Newton physics config {type(physics_cfg).__name__}"

    scene_cfg = env_cfg.scene
    fields = {
        name: value
        for name, value in vars(scene_cfg).items()
        if name not in InteractiveSceneCfg.__dataclass_fields__ and value is not None
    }
    for asset_name, asset_cfg in fields.items():
        if not isinstance(asset_cfg, AssetBaseCfg):
            continue
        for usd_path in _spawn_usd_paths(asset_cfg.spawn):
            for banned_suffix, reason in _BANNED_USD_PATH_SUFFIXES.items():
                if usd_path.endswith(banned_suffix):
                    return f"asset {asset_name!r} uses banned USD {banned_suffix!r}: {reason}"

    terrains = [value for value in fields.values() if isinstance(value, TerrainImporterCfg)]
    if any(terrain.terrain_type != "plane" for terrain in terrains):
        return "non-flat terrain"
    has_flat_ground = any(
        isinstance(value, AssetBaseCfg) and isinstance(value.spawn, sim_utils.GroundPlaneCfg)
        for value in fields.values()
    ) or any(terrain.terrain_type == "plane" for terrain in terrains)
    if not has_flat_ground:
        return "no declarative flat ground"

    try:
        scene_add(scene_cfg, scene_cfg, asset_skip=_skip_light)
    except (TypeError, ValueError) as exc:
        return str(exc)
    return None


def _load_task_scenes() -> tuple[tuple[str, ...], list[InteractiveSceneCfg]]:
    """Resolve explicit tasks or discover every supported registered flat PhysX task."""
    explicit = args_cli.task_ids is not None
    requested_ids = tuple(args_cli.task_ids) if explicit else tuple(_registered_task_ids())
    if explicit and args_cli.num_task is not None:
        requested_ids = requested_ids[: args_cli.num_task]

    accepted_ids = []
    accepted_scenes = []
    rejected = []
    for task_id in requested_ids:
        env_cfg, _ = resolve_task_config(task_id, "")
        rejection = _flat_physx_rejection(env_cfg)
        if rejection is not None:
            if explicit:
                raise ValueError(f"Task {task_id!r} is not supported by this demo: {rejection}.")
            rejected.append((task_id, rejection))
            continue

        accepted_ids.append(task_id)
        accepted_scenes.append(env_cfg.scene)
        if not explicit and args_cli.num_task is not None and len(accepted_ids) == args_cli.num_task:
            break

    if args_cli.num_task is not None and len(accepted_ids) < args_cli.num_task:
        raise ValueError(f"Only {len(accepted_ids)} supported tasks are available; requested {args_cli.num_task}.")
    if len(accepted_ids) < 2:
        raise ValueError("Select at least two supported task scenes.")

    if rejected:
        print("\n[INFO] Skipped registered task scenes outside the flat PhysX composition scope:")
        for task_id, reason in rejected:
            print(f"  {task_id}: {reason}")
    return tuple(accepted_ids), accepted_scenes


def _num_expanded_clone_combinations(scene_cfg: InteractiveSceneCfg) -> int:
    """Return the number of clone combinations after spawn-variant expansion."""
    asset_names = []
    variant_counts = []
    base_fields = InteractiveSceneCfg.__dataclass_fields__
    for name, asset_cfg in vars(scene_cfg).items():
        if (
            name in base_fields
            or not isinstance(asset_cfg, AssetBaseCfg)
            or asset_cfg.spawn is None
            or not asset_cfg.prim_path.startswith("{ENV_REGEX_NS}/")
        ):
            continue

        if isinstance(asset_cfg.spawn, sim_utils.MultiAssetSpawnerCfg):
            count = len(asset_cfg.spawn.assets_cfg)
        elif isinstance(asset_cfg.spawn, sim_utils.MultiUsdFileCfg):
            count = 1 if isinstance(asset_cfg.spawn.usd_path, str) else len(asset_cfg.spawn.usd_path)
        else:
            count = 1
        asset_names.append(name)
        variant_counts.append(count)

    combinations = make_valid_clone_combinations(asset_names, variant_counts, scene_cfg.clone_cfg.clone_combinations)
    return len(combinations)


def print_composition_summary(scene_cfg: InteractiveSceneCfg, selected_task_ids: tuple[str, ...]) -> None:
    """Print the entity and inclusion-row result produced by scene composition."""
    base_fields = InteractiveSceneCfg.__dataclass_fields__
    print("\n" + "=" * 72)
    print("scene_add composition")
    print("=" * 72)
    print("\ntask scenes")
    for task_id in selected_task_ids:
        print(f"  {task_id}")

    print("\nscene entities")
    for name, entity_cfg in vars(scene_cfg).items():
        if name in base_fields:
            continue
        prim_path = getattr(entity_cfg, "prim_path", None)
        print(f"  {name:12s} -> {prim_path or type(entity_cfg).__name__}")

    print("\nclone-combination rows")
    for row, combination in enumerate(scene_cfg.clone_cfg.clone_combinations):
        print(f"  row {row:02d}: {combination.assets}")


def print_scene_summary(scene: InteractiveScene) -> None:
    """Print clone-plan details for the constructed scene."""
    print("\n" + "=" * 72)
    print("Multitask clone-only scene")
    print("=" * 72)
    print(f"num_envs             : {scene.num_envs}")
    print(f"articulations        : {sorted(scene.articulations)}")
    print(f"rigid_objects        : {sorted(scene.rigid_objects)}")
    print(f"extras               : {sorted(scene.extras)}")

    plan = scene.clone_plan
    if plan is None:
        return

    print("\nclone plan")
    for row, (source, destination) in enumerate(zip(plan.sources, plan.destinations)):
        env_ids = plan.clone_mask[row].nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        print(f"  row {row:02d}: {source} -> {destination}; envs={env_ids}")


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """Step physics for the scene without applying MDP actions."""
    sim_dt = sim.get_physics_dt()
    step = 0
    # Step while a visualizer window is still open (or none exist, e.g. headless); works for kit and newton.
    while sim.is_headless_or_exist_active_visualizer():
        if args_cli.max_steps >= 0 and step >= args_cli.max_steps:
            break
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step()
            continue

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step += 1


def main() -> None:
    """Run the scene-only clone demo."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # The default newton mjwarp solver configuration needs to be tuned for this demo.
        if isinstance(physics_cfg, NewtonCfg) and isinstance(physics_cfg.solver_cfg, MJWarpSolverCfg):
            physics_cfg.solver_cfg.nconmax = 128
            physics_cfg.solver_cfg.naconmax = 2048
            physics_cfg.solver_cfg.njmax = 512

        task_ids, task_scene_cfgs = _load_task_scenes()
        for task_scene_cfg in task_scene_cfgs:
            task_scene_cfg.env_spacing = args_cli.env_spacing

        replicate_physics = not args_cli.disable_replicate_physics and all(
            scene_cfg.replicate_physics for scene_cfg in task_scene_cfgs
        )
        scene_cfg = task_scene_cfgs[0]
        for task_scene_cfg in task_scene_cfgs[1:]:
            scene_cfg = scene_add(scene_cfg, task_scene_cfg, asset_skip=_skip_light)

        scene_fields = {
            name: value
            for name, value in vars(scene_cfg).items()
            if name not in InteractiveSceneCfg.__dataclass_fields__
        }
        if "light" in scene_fields:
            raise ValueError("Cannot add the demo light because a non-light scene field already uses 'light'.")
        occupied_light_root = next(
            (
                name
                for name, value in scene_fields.items()
                if isinstance(value, AssetBaseCfg) and value.prim_path == "/World/light"
            ),
            None,
        )
        if occupied_light_root is not None:
            raise ValueError(f"Cannot add the demo light because {occupied_light_root!r} already uses /World/light.")
        scene_cfg.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )

        num_combinations = _num_expanded_clone_combinations(scene_cfg)
        num_envs = num_combinations if args_cli.num_envs is None else args_cli.num_envs
        if num_envs < num_combinations:
            raise ValueError(f"--num_envs must be at least the {num_combinations} expanded clone combinations.")

        scene_cfg.num_envs = num_envs
        scene_cfg.replicate_physics = replicate_physics
        scene_cfg.clone_cfg.clone_strategy = sequential
        print_composition_summary(scene_cfg, task_ids)

        sim_cfg = sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device, physics=physics_cfg)
        sim = SimulationContext(sim_cfg)
        sim.set_camera_view(eye=[6.0, 6.0, 4.0], target=[0.0, 0.0, 0.5])
        scene = InteractiveScene(scene_cfg)

        sim.reset()
        scene.reset()
        scene.write_data_to_sim()
        print_scene_summary(scene)
        print("\n[INFO] Setup complete. Stepping physics without MDP managers.")
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
