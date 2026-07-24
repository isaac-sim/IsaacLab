# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose a curated selection of task scenes into one heterogeneous simulation.

The pipeline is: resolve the scene config of every task in :data:`DEFAULT_TASKS`,
fold the scenes together with :func:`~isaaclab.scene.add` while skipping every
task's own light and floor, add one Dome light and one shared ground plane, and
clone the composition so each environment hosts one task's assets. No task
environments or MDP managers are constructed; the demo owns generic PhysX
simulation settings.

.. code-block:: bash

    # Usage with the full default task selection.
    ./isaaclab.sh -p scripts/demos/heterogeneous_scene.py

    # Usage with a smaller composition.
    ./isaaclab.sh -p scripts/demos/heterogeneous_scene.py --num_task 3 --num_envs 3

"""

from __future__ import annotations

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="Demo: clone-only multi-robot multi-task scene.",
    conflict_handler="resolve",
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--env_spacing", type=float, default=2.5, help="Distance between environment origins [m].")
parser.add_argument("--sim_dt", type=float, default=1.0 / 60.0, help="Physics timestep [s].")
parser.add_argument(
    "--num_task",
    type=int,
    default=None,
    help="Number of tasks to use from the default order. Omit to use all tasks.",
)
parser.add_argument("--physics", default="physx", choices=["physx"], help="Physics backend.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = parser.parse_known_args()
# strip consumed args so hydra-based task-config resolution does not re-parse them
sys.argv = [sys.argv[0], *hydra_args]

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.scene import add as scene_add

from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# Tasks composed by default. The selection criterion is simple: every listed
# scene is a PhysX task whose floor is a single flat plane at height zero, so
# one shared ground plane can serve the whole composition. Registered tasks
# not listed here either place their floor elsewhere (e.g. tabletop scenes
# with the ground at -1.05 m), use procedural terrain (the -Rough velocity
# tasks), require camera rendering flags or optional packages, target the
# Newton backend, or duplicate a listed task under another alias.
DEFAULT_TASKS = (
    # classic control
    "Isaac-Cartpole",
    "Isaac-Fourbar-Pole-Swingup",
    "Isaac-Ant",
    "Isaac-Humanoid",
    # legged locomotion
    "Isaac-Velocity-Flat-AnymalD",
    "IsaacContrib-Velocity-Flat-AnymalB",
    "IsaacContrib-Velocity-Flat-AnymalC",
    "IsaacContrib-Velocity-Flat-UnitreeA1",
    "IsaacContrib-Velocity-Flat-UnitreeGo1",
    "Isaac-Velocity-Flat-UnitreeGo2",
    "Isaac-Velocity-Flat-Cassie",
    "IsaacContrib-Velocity-Flat-Digit",
    "Isaac-Velocity-Flat-G1",
    "Isaac-Velocity-Flat-H1",
    "IsaacContrib-Navigation-Flat-AnymalC",
    # arm and hand manipulation
    "Isaac-Lift-Franka",
    "Isaac-Reorient-Franka",
    "Isaac-Lift-KukaAllegro",
    "Isaac-Reorient-KukaAllegro",
    "Isaac-Open-Drawer-Franka",
    "IsaacContrib-Open-Drawer-Franka-IK-Abs",
    "IsaacContrib-Open-Drawer-Franka-IK-Rel",
)


def _load_task_scenes() -> tuple[list[str], list[InteractiveSceneCfg]]:
    """Resolve the scene config of every selected task."""
    task_ids = list(DEFAULT_TASKS if args_cli.num_task is None else DEFAULT_TASKS[: args_cli.num_task])
    if len(task_ids) < 2:
        raise ValueError("Select at least two task scenes.")
    scene_cfgs = []
    for task_id in task_ids:
        # resolve preset placeholders (e.g. object-set choices) to their defaults
        env_cfg = resolve_presets(load_cfg_from_registry(task_id, "env_cfg_entry_point"))
        scene_cfgs.append(env_cfg.scene)
    return task_ids, scene_cfgs


def main() -> None:
    """Resolve the selected task scenes, compose, add light and floor, simulate."""
    # Resolve and compose every task scene before Kit launches: config resolution is
    # simulator-free, and the launch swaps module state that must not interleave with it.
    task_ids, task_scene_cfgs = _load_task_scenes()
    print(f"\n[INFO] Composing task scenes: {task_ids}")
    for task_scene_cfg in task_scene_cfgs:
        task_scene_cfg.env_spacing = args_cli.env_spacing

    scene_cfg = task_scene_cfgs[0]

    def is_global_asset(a: AssetBaseCfg) -> bool:
        return isinstance(a.spawn, (sim_utils.LightCfg, sim_utils.GroundPlaneCfg))

    for task_scene_cfg in task_scene_cfgs[1:]:
        scene_cfg = scene_add(scene_cfg, task_scene_cfg, asset_skip=is_global_asset)
    scene_cfg.light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    scene_cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    scene_cfg.num_envs = args_cli.num_envs
    scene_cfg.replicate_physics = True

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=args_cli.sim_dt, device=args_cli.device, physics=physics_cfg)
        )
        sim.set_camera_view(eye=[6.0, 6.0, 4.0], target=[0.0, 0.0, 0.5])
        scene = scene_cfg.class_type(scene_cfg)
        sim.reset()
        scene.reset()
        scene.write_data_to_sim()
        print(f"[INFO] Composed {len(task_ids)} task scenes into {args_cli.num_envs} environments. Stepping physics.")

        sim_dt = sim.get_physics_dt()
        # Step while a visualizer window is still open (or none exist, e.g. headless).
        while True:
            if not sim.is_playing():
                sim.step()
                continue
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)


if __name__ == "__main__":
    main()
