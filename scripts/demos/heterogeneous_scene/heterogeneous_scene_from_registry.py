# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Unified heterogeneous-scene demo built from the whole ``core`` task registry.

One script that harvests **any mix** of core tasks and clones them with **either**
API -- selectable by flags, because the task source and the cloning API are
orthogonal once tasks are harvested into the shared :class:`~registry_harvest.Prototype`
form:

* ``--workflow {manager, direct, both}`` -- which tasks to combine. ``both`` merges
  manager-based and Direct-workflow tasks into one scene; same-model assets shared
  across the two workflows collapse to a single cloned prototype.
* ``--clone-api {implicit, explicit}`` --
    * ``implicit`` : high-level :class:`~isaaclab.scene.InteractiveScene` + a
      heterogeneous :class:`~isaaclab.cloner.CloneCfg` (cloning/selector/physics
      managed for you).
    * ``explicit`` : low-level ``grid_transforms`` + ``usd_replicate`` +
      :class:`~isaaclab.cloner.ReplicateSession` (you spawn each prototype in
      ``env_0`` and clone it yourself, then drive the bare asset objects).
* ``--clone_strategy {sequential, interleaved}`` -- prototype-combination -> env
  assignment (both round-robin). ``random`` is intentionally not offered; see
  :data:`~registry_harvest.STRATEGIES` for why.

Harvest/identity/report live in :mod:`registry_harvest`; the two clone engines
(``InteractiveSceneEngine`` / ``ManualCloneEngine``) live in :mod:`clone_engines`.

.. code-block:: bash

    DEMO=scripts/demos/heterogeneous_scene/heterogeneous_scene_from_registry.py
    ./isaaclab.sh -p $DEMO --workflow both --clone_api implicit     # all core tasks, high-level
    ./isaaclab.sh -p $DEMO --workflow direct --clone_api explicit   # Direct tasks, low-level
    ./isaaclab.sh -p $DEMO --include_contrib --list_only            # add contrib tasks; inspect only

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Demo: unified heterogeneous scene from the core task registry.")
parser.add_argument("--workflow", choices=["manager", "direct", "both"], default="both", help="Which tasks to combine.")
parser.add_argument(
    "--include_contrib",
    action="store_true",
    help="Also harvest isaaclab_tasks.contrib tasks, not just isaaclab_tasks.core.",
)
parser.add_argument(
    "--clone_api",
    choices=["implicit", "explicit"],
    default="implicit",
    help="implicit = InteractiveScene + CloneCfg; explicit = manual ClonePlan / ReplicateSession.",
)
parser.add_argument("--envs_per_task", type=int, default=4, help="Envs per discovered task (ignored if --num_envs).")
parser.add_argument("--num_envs", type=int, default=None, help="Total envs. Defaults to num_tasks * envs_per_task.")
parser.add_argument("--env_spacing", type=float, default=3.0, help="Grid spacing between environments [m].")
parser.add_argument("--include", type=str, default=None, help="Regex; keep only tasks whose id matches.")
parser.add_argument("--exclude", type=str, default=None, help="Regex; drop tasks whose id matches.")
parser.add_argument(
    "--max_tasks", type=int, default=None, help="Cap total tasks (after filtering) for a lighter scene."
)
parser.add_argument(
    "--clone_strategy",
    choices=["sequential", "interleaved"],
    default="sequential",
    help="Prototype-combination -> env assignment (both round-robin); see registry_harvest.STRATEGIES.",
)
parser.add_argument("--list_only", action="store_true", help="Print the pre-processing report and exit (no sim).")
parser.add_argument(
    "--randomize_object_variants",
    action="store_true",
    help="Keep multi-asset spawners (random object per env) instead of collapsing them to one variant.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import traceback

import clone_engines as engines
import registry_harvest as common

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

# Importing isaaclab_tasks registers every core and contrib gym task as a side effect.
import isaaclab_tasks  # noqa: F401


def main() -> None:
    explicit = args_cli.clone_api == "explicit"
    engine_cls = engines.ManualCloneEngine if explicit else engines.InteractiveSceneEngine
    print(f"[INFO] Discovering core tasks (workflow={args_cli.workflow}, clone_api={args_cli.clone_api}) ...")

    # 1) Discovery of task ids for the selected workflow (manager, direct, or both).
    # Harvesting more task families is a one-liner: widen module_prefixes -- here,
    # --include_contrib adds isaaclab_tasks.contrib (already registered by the import above).
    prefixes = (common.CORE_TASK_PREFIX,)
    if args_cli.include_contrib:
        prefixes += (common.CONTRIB_TASK_PREFIX,)
    workflows = {"manager": [True], "direct": [False], "both": [True, False]}[args_cli.workflow]
    task_ids = sorted(
        tid
        for manager_based in workflows
        for tid in common.discover_task_ids(
            args_cli.include, args_cli.exclude, manager_based=manager_based, module_prefixes=prefixes
        )
    )
    print(f"[INFO] {len(task_ids)} candidate task(s) match the filters.")

    # 2) Harvest is workflow-agnostic: asset_fields merges cfg.scene (managers) and the
    # cfg top level (Direct), so a single pass handles a mixed task set. The prim
    # prefix is the one thing that depends on the chosen clone engine.
    tasks, prototypes = common.harvest_tasks(
        task_ids,
        fields_of=common.asset_fields,
        prim_prefix=engine_cls.PRIM_PREFIX,
        device=args_cli.device,
        max_tasks=args_cli.max_tasks,
        randomize_variants=args_cli.randomize_object_variants,
    )
    if not tasks:
        print("[ERROR] No tasks yielded env-scoped assets; nothing to build.")
        return

    num_envs = args_cli.num_envs if args_cli.num_envs is not None else len(tasks) * args_cli.envs_per_task
    num_envs = max(num_envs, len(tasks))
    common.assign_env_ids(tasks, prototypes, num_envs)
    common.print_preprocess_report(
        tasks,
        prototypes,
        num_envs,
        title=f"PRE-PROCESSING REPORT  (workflow={args_cli.workflow}, {args_cli.clone_api})",
    )

    if args_cli.list_only:
        print("[INFO] --list_only set; skipping SimulationContext and cloning.")
        return

    # 3) Build the scene
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device))
    sim.set_camera_view(eye=[6.0, 6.0, 4.0], target=[0.0, 0.0, 0.5])
    print(f"[INFO] Building scene via the {args_cli.clone_api} clone API; locomotion / manipulation groups alternate.")

    # 4) Run the simulation
    strategy = common.STRATEGIES[args_cli.clone_strategy]
    engine_cls(sim, simulation_app, tasks, prototypes, num_envs, args_cli.env_spacing, args_cli.device, strategy).run()


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - surface the traceback before closing the app
        traceback.print_exc()
    finally:
        simulation_app.close()
