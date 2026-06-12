#!/usr/bin/env python3
import argparse
import json
import math
import random
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id")
    parser.add_argument("--backend")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fps_mean", type=float, default=200.0)
    parser.add_argument("--failure_phase", default="none")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simulate failures
    if args.failure_phase == "import":
        # Emit a fake traceback and exit non-zero without writing perf file
        print("Traceback (most recent call last):")
        print("  File \"<string>\", line 1, in <module>")
        print("ImportError: simulated import failure")
        sys.exit(1)

    if args.failure_phase == "init":
        # Emit AppLauncher init message then exit non-zero (no perf file)
        print("AppLauncher initialization complete")
        sys.exit(2)

    # Prepare FPS series
    n = int(args.num_frames)
    rng = random.Random(0)
    noise = 5.0
    fps_series = [max(0.0, rng.gauss(args.fps_mean, noise)) for _ in range(n)]

    # Write perf_regression_gate_info.json
    runtime_phase = {
        "phase_name": "runtime",
        "measurements": [
            {
                "name": "Step Frametimes",
                "value": {"Environment step effective FPS": fps_series},
            }
        ],
    }
    info_path = out_dir / "perf_regression_gate_info.json"
    with info_path.open("w") as fh:
        json.dump([runtime_phase], fh)

    # Print Step Frametimes marker so classify_failure_phase can see it
    print("Step Frametimes")

    # For runtime failure emulate crash after printing frame times
    if args.failure_phase == "runtime":
        print("RuntimeError: simulated crash during runtime")
        sys.exit(3)

    # Successful exit
    sys.exit(0)


if __name__ == "__main__":
    main()
