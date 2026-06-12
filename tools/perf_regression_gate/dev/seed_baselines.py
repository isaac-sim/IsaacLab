# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seed the flat-file baseline store from each task's calibrated ``ref_fps``.

The gate only enforces thresholds once a rolling window has at least
``oracle.MIN_WINDOW`` samples (below that it seed-PASSes). Collecting those from
live GPU runs is slow, so for local development / demos this script bootstraps a
deterministic window centered on the per-GPU ``ref_fps`` in ``tasks.json``.

This is a dev/test convenience only -- production baselines accumulate from real
PASS/WARN runs on the protected branch via ``aggregate.py --allow_baseline_update``.

Usage::

    python3 tools/perf_regression_gate/dev/seed_baselines.py \\
        --baselines_dir tools/perf_regression_gate/local_baselines \\
        --gpu_model "NVIDIA L40S"
"""

import argparse
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).parent.parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from baseline_manager import seed_baseline_with_spread  # noqa: E402
from task_config import load_tasks  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed flat-file baselines from tasks.json ref_fps.")
    p.add_argument(
        "--baselines_dir",
        type=Path,
        default=_MODULE_DIR / "local_baselines",
        help="Flat-file baseline directory to populate (default: tools/perf_regression_gate/local_baselines/)",
    )
    p.add_argument("--gpu_model", default="NVIDIA L40S", help="GPU model key matching tasks.json ref_fps")
    p.add_argument("--n_samples", type=int, default=8, help="Samples to seed per task/backend (>= oracle.MIN_WINDOW)")
    p.add_argument("--noise_pct", type=float, default=1.5, help="Gaussian spread as %% of ref_fps")
    p.add_argument("--fingerprint", default=None, help="Optional fingerprint bucket to seed into")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    seeded = 0
    for task in load_tasks():
        # Resolve the per-GPU reference FPS (substring match tolerates "L40S" vs "NVIDIA L40S").
        ref = None
        for key, val in task.ref_fps.items():
            if key == args.gpu_model or key in args.gpu_model or args.gpu_model in key:
                ref = val
                break
        if ref is None:
            print(f"[seed_baselines] skip {task.task_id}/{task.backend_key}: no ref_fps for {args.gpu_model!r}")
            continue
        seed_baseline_with_spread(
            args.baselines_dir,
            args.gpu_model,
            task.task_id,
            task.backend_key,
            center_fps=ref,
            noise_fps=max(1.0, args.noise_pct / 100.0 * ref),
            n_samples=args.n_samples,
            seed=42,
            fingerprint=args.fingerprint,
        )
        seeded += 1
        print(f"[seed_baselines] seeded {task.task_id}/{task.backend_key} center={ref:.1f} n={args.n_samples}")
    print(f"[seed_baselines] done: {seeded} task/backend buckets under {args.baselines_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
