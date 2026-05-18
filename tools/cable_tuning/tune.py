# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Iterative-bisection driver for the FrankaCable env tuning.

Each round: pick the next parameter from a fixed schedule, sweep its trial values
(each in a fresh subprocess of :mod:`tools.cable_tuning.eval_run`), score, keep
the best, append to ``tuning_log.jsonl``, move on.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass

from tools.cable_tuning.scoring import score_summary


@dataclass
class ParamSpec:
    path: str
    values: list[float | int | str]


PARAM_SCHEDULE: list[ParamSpec] = [
    # Stage A — solver stability
    ParamSpec("sim.physics.num_substeps", [10, 20, 40, 80]),
    ParamSpec("sim.physics.solver_cfg.vbd_cfg.iterations", [5, 10, 20, 40]),
    ParamSpec("sim.physics.solver_cfg.vbd_cfg.rigid_avbd_beta", [1.0e3, 1.0e4, 1.0e5, 1.0e6]),
    ParamSpec("sim.physics.solver_cfg.mjwarp_cfg.ls_iterations", [10, 20, 40]),
    ParamSpec("sim.physics.solver_cfg.proxy_collide_interval", [1, 2, 5, 10]),
    ParamSpec("sim.physics.solver_cfg.proxy_mass_scale", [0.1, 1.0, 10.0]),
    # Stage B — cable material
    ParamSpec("scene.cable.spawn.physics_material.stretch_stiffness", [1.0e6, 1.0e7, 1.0e8, 1.0e9]),
    ParamSpec("scene.cable.spawn.physics_material.stretch_damping", [0.1, 1.0, 10.0, 100.0]),
    ParamSpec("scene.cable.spawn.physics_material.bend_stiffness", [1.0e-3, 1.0e-2, 1.0e-1, 1.0]),
    ParamSpec("scene.cable.spawn.physics_material.bend_damping", [1.0e-3, 1.0e-2, 1.0e-1, 1.0]),
    ParamSpec("scene.cable.spawn.physics_material.density", [100.0, 1000.0, 10000.0]),
]


def _new_run_dir(root: str) -> str:
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run_dir = os.path.join(root, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _failure_summary(error: str, run_dir: str) -> dict:
    return {
        "nan_flag": 1, "max_state_reached": 0, "exploded_flag": 0,
        "settle_time_s": 1e6, "mean_goal_pos_error_lift": 1e6,
        "cable_oscillation_rms": 1e6, "steps_executed": 0, "error": error,
        "run_dir": run_dir,
    }


def _run_one(overrides: dict, root: str, max_steps: int, timeout_s: int) -> dict:
    """Spawn one eval_run subprocess. Return summary dict."""
    run_dir = _new_run_dir(root)
    overrides_path = os.path.join(run_dir, "overrides.json")
    with open(overrides_path, "w") as f:
        json.dump(overrides, f, indent=2)

    cmd = [
        "./isaaclab.sh", "-p", "tools/cable_tuning/eval_run.py",
        "--overrides", overrides_path,
        "--out", run_dir,
        "--max-steps", str(max_steps),
        "--headless",
    ]
    log_path = os.path.join(run_dir, "stdout.log")
    with open(log_path, "w") as log:
        try:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired:
            log.write(f"\n[tune.py] subprocess timeout after {timeout_s}s\n")

    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(summary_path):
        return _failure_summary("no_summary_written", run_dir)
    with open(summary_path) as f:
        s = json.load(f)
    s["run_dir"] = run_dir
    return s


def _sweep_one_param(
    spec: ParamSpec,
    baseline: dict,
    root: str,
    max_steps: int,
    timeout_s: int,
    log_path: str,
    round_idx: int,
) -> tuple[float | int | str, float, dict]:
    """Evaluate ``spec.values`` against ``baseline``. Returns (best_value, best_cost, best_summary)."""
    best_value: float | int | str = baseline.get(spec.path, spec.values[0])
    best_cost = float("inf")
    best_summary: dict | None = None
    for v in spec.values:
        overrides = {**baseline, spec.path: v}
        summary = _run_one(overrides, root, max_steps, timeout_s)
        cost = score_summary(summary)
        entry = {
            "round": round_idx,
            "param": spec.path,
            "value": v,
            "overrides": overrides,
            "summary": summary,
            "cost": cost,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"  [{spec.path}={v}] cost={cost:.3f}  (run={summary['run_dir']})", flush=True)
        if cost < best_cost:
            best_cost, best_value, best_summary = cost, v, summary
    return best_value, best_cost, best_summary or _failure_summary("no_runs", "")


def _resume_baseline(log_path: str) -> tuple[dict, set[str]]:
    """Rebuild baseline + done_params from prior log lines."""
    baseline: dict[str, float | int | str] = {}
    done_params: set[str] = set()
    if not os.path.exists(log_path):
        return baseline, done_params

    per_param_best: dict[str, tuple[float, float | int | str]] = {}
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            done_params.add(entry["param"])
            p, v, c = entry["param"], entry["value"], entry["cost"]
            if p not in per_param_best or c < per_param_best[p][0]:
                per_param_best[p] = (c, v)
    for p, (_, v) in per_param_best.items():
        baseline[p] = v
    return baseline, done_params


def main() -> int:
    parser = argparse.ArgumentParser(description="Iterative-bisection cable tuner.")
    parser.add_argument("--root", type=str, default="tuning_results/franka_cable")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--timeout", type=int, default=300, help="Per-run wall-clock timeout (s).")
    parser.add_argument("--resume", action="store_true", help="Reload prior best from tuning_log.jsonl.")
    args = parser.parse_args()

    os.makedirs(args.root, exist_ok=True)
    log_path = os.path.join(args.root, "tuning_log.jsonl")

    baseline: dict[str, float | int | str] = {}
    done_params: set[str] = set()
    if args.resume:
        baseline, done_params = _resume_baseline(log_path)
        print(f"[tune.py] resumed; baseline = {baseline}", flush=True)

    for i, spec in enumerate(PARAM_SCHEDULE):
        if spec.path in done_params and args.resume:
            print(f"[round {i}] skip {spec.path} (already done)", flush=True)
            continue
        print(f"[round {i}] sweeping {spec.path} over {spec.values}", flush=True)
        best_value, best_cost, _ = _sweep_one_param(
            spec, baseline, args.root, args.max_steps, args.timeout, log_path, i,
        )
        baseline[spec.path] = best_value
        print(f"[round {i}] best {spec.path}={best_value!r} cost={best_cost:.3f}", flush=True)

    final_path = os.path.join(args.root, "final_best.json")
    with open(final_path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"[tune.py] final best saved to {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
