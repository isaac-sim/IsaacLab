# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark the legacy vs batched Newton model builder replication paths.

For each task, the script creates the environment once (to obtain the USD stage and
the published clone plan), parses the per-source builders once, and then times, for
each requested world count and for both builder paths:

* build time: replication of all worlds into a fresh main builder (for the legacy
  path this includes the label rename pass, which the batched path does not need);
* finalize time: ``ModelBuilder.finalize`` on the requested device;
* total time: build + finalize.

Each task runs in its own subprocess because Isaac Sim can only be launched once per
process. Results are printed as a table and written to JSON (and optionally CSV).

Example:

.. code-block:: bash

    uv run python benchmark_model_builder.py \\
        --tasks Isaac-Lift-KukaAllegro-Camera Isaac-Cartpole-Direct \\
        --world_counts 1 128 1024 4096 --repeats 3 --verify
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

DEFAULT_TASKS = [
    "Isaac-Lift-KukaAllegro-Camera",
    "Isaac-Lift-KukaAllegro",
    "Isaac-Cartpole-Direct",
]
DEFAULT_WORLD_COUNTS = [1, 128, 1024, 4096, 8192]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="Task names to benchmark.")
    parser.add_argument(
        "--world_counts", nargs="+", type=int, default=DEFAULT_WORLD_COUNTS, help="World counts to benchmark."
    )
    parser.add_argument("--physics", default="newton_mjwarp", help="Physics preset applied to each task.")
    parser.add_argument("--renderer", default=None, help="Optional renderer preset applied to each task.")
    parser.add_argument("--device", default="cuda", help="Device passed to ModelBuilder.finalize().")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per (task, world count, builder).")
    parser.add_argument("--repeats", type=int, default=3, help="Timed runs per (task, world count, builder).")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare the two builders' outputs at the smallest world count and report mismatches.",
    )
    parser.add_argument("--output", default=None, help="JSON output path (default: benchmark_model_builder_<ts>.json).")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--_single_task", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_result_file", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _benchmark_task(args: argparse.Namespace) -> dict:
    """Benchmark one task inside an Isaac Sim process. Returns a result dict."""
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    _simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from isaaclab_newton.cloner.batched_model_builder import replicate_builder_mapping_batched
    from isaaclab_newton.cloner.builder_diff import compare_builder_states
    from isaaclab_newton.cloner.newton_clone_utils import (
        build_source_builders,
        rename_builder_labels,
        replicate_builder_mapping,
    )
    from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

    import isaaclab.sim as sim_utils
    from isaaclab.cloner.cloner_utils import grid_transforms
    from isaaclab.physics import PhysicsManager
    from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    task = args._single_task
    result: dict = {"task": task, "rows": [], "verify": None}

    # -- Create the environment once to obtain the stage and clone plan.
    # resolve_task_config reads preset tokens (physics=..., renderer=...) from sys.argv.
    sim_utils.create_new_stage()
    tokens = [f"physics={args.physics}"] + ([f"renderer={args.renderer}"] if args.renderer else [])
    argv = sys.argv
    sys.argv = [argv[0], *tokens]
    try:
        env_cfg, _ = resolve_task_config(task, None)
    finally:
        sys.argv = argv
    env_cfg.sim.device = "cuda"
    env_cfg.scene.num_envs = 2
    env = gym.make(task, cfg=env_cfg)
    env.unwrapped.sim._app_control_on_stop_handle = None

    plan = env.unwrapped.scene.clone_plan
    if plan is None or not plan.sources:
        raise RuntimeError(f"Task '{task}' did not publish a clone plan; cannot benchmark replication.")
    stage = sim_utils.get_current_stage()
    env_spacing = float(env_cfg.scene.env_spacing)
    sources = tuple(plan.sources)
    destinations = tuple(plan.destinations)
    base_mask = plan.clone_mask.detach().cpu()

    # -- Parse source builders once (shared setup for both paths).
    manager_cls = PhysicsManager._sim.physics_manager
    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    setup_start = time.perf_counter()
    source_builders = build_source_builders(
        stage, sources, lambda: manager_cls.create_builder(up_axis="Z"), schema_resolvers
    )
    result["source_parse_time_s"] = time.perf_counter() - setup_start

    def fresh_main_builder():
        builder = manager_cls.create_builder(up_axis="Z")
        builder.add_usd(stage, ignore_paths=["/World/envs", *sources], schema_resolvers=schema_resolvers)
        replace_newton_builder_shape_colors(builder, stage)
        return builder

    def run_once(num_worlds: int, batched: bool) -> tuple[float, float, object]:
        # Cycle the plan's world columns to synthesize the requested world count.
        mapping = base_mask[:, torch.arange(num_worlds) % base_mask.size(1)]
        env_ids = torch.arange(num_worlds)
        positions, _ = grid_transforms(num_worlds, env_spacing, device="cpu")
        quaternions = torch.zeros((num_worlds, 4))
        quaternions[:, 3] = 1.0

        builder = fresh_main_builder()
        build_start = time.perf_counter()
        if batched:
            replicate_builder_mapping_batched(
                builder, sources, destinations, env_ids, mapping, positions, quaternions, source_builders
            )
        else:
            replicate_builder_mapping(builder, sources, mapping, positions, quaternions, source_builders)
            rename_builder_labels(builder, sources, destinations, env_ids, mapping)
        build_time = time.perf_counter() - build_start

        finalize_start = time.perf_counter()
        model = builder.finalize(device=args.device)
        finalize_time = time.perf_counter() - finalize_start
        del model
        return build_time, finalize_time, builder

    if args.verify:
        num_worlds = min(args.world_counts)
        _, _, legacy_builder = run_once(num_worlds, batched=False)
        _, _, batched_builder = run_once(num_worlds, batched=True)
        mismatches = compare_builder_states(legacy_builder, batched_builder)
        result["verify"] = {"num_worlds": num_worlds, "mismatches": mismatches}
        del legacy_builder, batched_builder

    for num_worlds in args.world_counts:
        for batched in (False, True):
            builder_name = "batched" if batched else "legacy"
            try:
                for _ in range(args.warmup):
                    run_once(num_worlds, batched)
                build_times, finalize_times = [], []
                for _ in range(args.repeats):
                    build_time, finalize_time, _b = run_once(num_worlds, batched)
                    build_times.append(build_time)
                    finalize_times.append(finalize_time)
                row = {
                    "task": task,
                    "num_worlds": num_worlds,
                    "builder": builder_name,
                    "build_time_s": min(build_times),
                    "finalize_time_s": min(finalize_times),
                    "total_time_s": min(b + f for b, f in zip(build_times, finalize_times)),
                }
            except Exception as exc:  # keep going: report which combination failed and why
                row = {
                    "task": task,
                    "num_worlds": num_worlds,
                    "builder": builder_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            result["rows"].append(row)
            print(f"[done] {task} worlds={num_worlds} builder={builder_name}: {row}", flush=True)

    env.close()
    return result


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'task':<36} {'worlds':>7} {'builder':>8} "
        f"{'build [s]':>10} {'finalize [s]':>13} {'total [s]':>10} {'speedup':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    legacy_totals = {
        (row["task"], row["num_worlds"]): row["total_time_s"]
        for row in rows
        if row["builder"] == "legacy" and "error" not in row
    }
    for row in rows:
        if "error" in row:
            print(f"{row['task']:<36} {row['num_worlds']:>7} {row['builder']:>8}  ERROR: {row['error']}")
            continue
        speedup = ""
        if row["builder"] == "batched":
            legacy_total = legacy_totals.get((row["task"], row["num_worlds"]))
            if legacy_total:
                speedup = f"{legacy_total / row['total_time_s']:.2f}x"
        print(
            f"{row['task']:<36} {row['num_worlds']:>7} {row['builder']:>8} "
            f"{row['build_time_s']:>10.3f} {row['finalize_time_s']:>13.3f} {row['total_time_s']:>10.3f} {speedup:>8}"
        )


def _write_csv(path: str, rows: list[dict]) -> None:
    import csv

    fields = ["task", "num_worlds", "builder", "build_time_s", "finalize_time_s", "total_time_s", "error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    args = _parse_args()

    if args._single_task is not None:
        result = _benchmark_task(args)
        with open(args._result_file, "w") as f:
            json.dump(result, f)
        return 0

    # Coordinator: run each task in its own subprocess (Isaac Sim launches once per process).
    all_rows: list[dict] = []
    task_results: list[dict] = []
    for task in args.tasks:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            result_file = tmp.name
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--_single_task",
            task,
            "--_result_file",
            result_file,
            "--world_counts",
            *map(str, args.world_counts),
            "--physics",
            args.physics,
            "--device",
            args.device,
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
        ]
        if args.renderer:
            cmd += ["--renderer", args.renderer]
        if args.verify:
            cmd.append("--verify")
        print(f"\n=== Benchmarking {task} ===", flush=True)
        proc = subprocess.run(cmd)
        try:
            with open(result_file) as f:
                result = json.load(f)
        except (OSError, json.JSONDecodeError):
            result = {"task": task, "rows": [], "error": f"subprocess exited with code {proc.returncode}"}
            print(f"[ERROR] {task}: benchmark subprocess failed (exit code {proc.returncode}).")
        finally:
            if os.path.exists(result_file):
                os.unlink(result_file)
        task_results.append(result)
        all_rows.extend(result.get("rows", []))
        verify = result.get("verify")
        if verify is not None:
            status = "OK" if not verify["mismatches"] else f"{len(verify['mismatches'])} MISMATCHES"
            print(f"[verify] {task} at {verify['num_worlds']} worlds: {status}")
            for mismatch in verify["mismatches"][:10]:
                print(f"    - {mismatch}")

    _print_table(all_rows)

    output = args.output or f"benchmark_model_builder_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    with open(output, "w") as f:
        json.dump(
            {"args": {k: v for k, v in vars(args).items() if not k.startswith("_")}, "results": task_results},
            f,
            indent=2,
        )
    print(f"\nResults written to {output}")
    if args.csv:
        _write_csv(args.csv, all_rows)
        print(f"CSV written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
