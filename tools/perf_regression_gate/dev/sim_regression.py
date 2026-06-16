# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression simulation helper that injects degraded FPS artifacts into a scratch artifacts directory.

Used to demonstrate that the gate produces BLOCK verdicts when a real regression
is introduced, WITHOUT re-running the full benchmark suite.

    python3 tools/perf_regression_gate/dev/sim_regression.py --fps_scale 0.53

Then run aggregate.py against the output to see BLOCK verdicts:

    python3 tools/perf_regression_gate/aggregate.py \\
        --artifacts_dir /tmp/sim_artifacts \\
        --gpu_model L40S \\
        --baselines_dir tools/perf_regression_gate/local_baselines \\
        --allow_baseline_update false
"""

import argparse
import json
import random
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
_GATE_DIR = _MODULE_DIR.parent  # tools/perf_regression_gate/

sys.path.insert(0, str(_GATE_DIR))
sys.path.insert(0, str(_GATE_DIR.parent))

from baseline_manager import load_baseline  # noqa: E402
from task_config import load_tasks  # noqa: E402


def _make_perf_info(task_id: str, fps_mean: float, num_frames: int) -> list:
    rng = random.Random(42)
    fps_series = [max(1.0, fps_mean + rng.gauss(0, fps_mean * 0.01)) for _ in range(num_frames)]
    return [
        {
            "phase_name": "runtime",
            "measurements": [
                {
                    "name": f"{task_id} Step Frametimes",
                    "value": {"Environment step effective FPS": fps_series},
                }
            ],
            "metadata": [],
        }
    ]


def _make_bench_result(task, fps_mean: float) -> dict:
    return {
        "task_id": task.task_id,
        "backend": task.backend_key,
        "backend_key": task.backend_key,
        "physics_backend": task.physics_backend,
        "render_backend": task.render_backend,
        "preset": task.preset,
        "attempt": 1,
        "was_retried": False,
        "exit_code": 0,
        "failure_phase": None,
        "stdout_tail": "",
        "wall_time_s": 60.0,
        "startup_time_s": 10.0,
        "perf_regression_gate_info_present": True,
        "raw_fps_mean": fps_mean,
        "raw_fps_std": fps_mean * 0.01,
        "raw_fps_min": fps_mean * 0.95,
        "raw_fps_max": fps_mean * 1.05,
        "raw_fps_median": fps_mean,
        "raw_fps_p5": fps_mean * 0.96,
        "raw_fps_p95": fps_mean * 1.04,
        "outlier_count": 0,
        "gpu_diag": None,
        "task_config_snapshot": {
            "task_id": task.task_id,
            "backend": task.backend_key,
            "backend_key": task.backend_key,
            "physics_backend": task.physics_backend,
            "render_backend": task.render_backend,
            "preset": task.preset,
            "num_envs": task.num_envs,
            "num_frames": task.num_frames,
            "excluded_frames_raw": task.excluded_frames_raw,
            "timeout_minutes": task.timeout_minutes,
            "camera_resolution": task.camera_resolution,
            "tags": task.tags,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Inject simulated regression artifacts.")
    p.add_argument(
        "--fps_scale",
        type=float,
        default=0.53,
        help="Multiply baseline FPS by this factor (0.53 = 47%% regression, default)",
    )
    p.add_argument("--tags", nargs="+", default=["always"], help="Task tags to include (default: always)")
    p.add_argument("--gpu_model", default="L40S")
    p.add_argument("--baselines_dir", type=Path, default=_GATE_DIR / "local_baselines")
    p.add_argument("--out_dir", type=Path, default=Path("/tmp/sim_artifacts"))
    args = p.parse_args()

    all_tasks = load_tasks()
    tag_set = frozenset(args.tags)
    tasks = [t for t in all_tasks if tag_set.intersection(frozenset(t.tags))]

    print(f"\n[sim_regression] fps_scale={args.fps_scale} ({(1 - args.fps_scale) * 100:.0f}% regression)")
    print(f"[sim_regression] writing artifacts to {args.out_dir}\n")

    generated = 0
    for task in tasks:
        baseline = load_baseline(args.baselines_dir, args.gpu_model, task.task_id, task.backend_key)
        if baseline is None:
            print(f"  SKIP (no baseline): {task.task_id}/{task.backend_key}")
            continue

        baseline_fps = baseline.median_fps
        regressed_fps = baseline_fps * args.fps_scale

        art_dir = args.out_dir / task.task_id / task.backend_key
        art_dir.mkdir(parents=True, exist_ok=True)

        perf_info = _make_perf_info(task.task_id, regressed_fps, task.num_frames)
        (art_dir / "perf_regression_gate_info.json").write_text(json.dumps(perf_info))

        bench_result = _make_bench_result(task, regressed_fps)
        (art_dir / "perf_regression_gate_result.json").write_text(json.dumps(bench_result))

        print(f"  {task.task_id}/{task.backend_key}: baseline={baseline_fps:.1f}  regressed={regressed_fps:.1f}")
        generated += 1

    print(f"\n[sim_regression] wrote {generated} artifact sets")
    print("\nNow run aggregate.py:")
    print(f"  python3 {_GATE_DIR}/aggregate.py \\")
    print(f"      --artifacts_dir {args.out_dir} \\")
    print(f"      --gpu_model {args.gpu_model} \\")
    print(f"      --baselines_dir {args.baselines_dir} \\")
    print("      --allow_baseline_update false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
