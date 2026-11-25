# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record the Rizon-Sharpa coffee demo's arm motion for the coffee kitchen camera benchmark.

Command to run, from the repository root:

    uv run python scripts/benchmarks/record_coffee_arm_motion.py \
        source/isaaclab_tasks/isaaclab_tasks/benchmark/coffee_kitchen_camera/coffee_arm_motion.json

Runs ``examples/demos/rizon_sharpa_coffee.py --visualizer none --headless`` with one environment, as its ``main()``
does, until the scripted sequence finishes, and records:

* ``positions`` and ``velocities``: the measured arm joint state after the reset and after each control step;
* ``targets``: the arm actuator targets that the demo's IK action set in each physics substep of each step.

The demo's configuration comes from its private ``_parse_args`` and ``_make_env_cfg`` helpers, so changing them
changes the recording. The demo, its task package and this script must be committed, so the recorded commit
reproduces the file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_PATH = REPO_ROOT / "examples" / "demos" / "rizon_sharpa_coffee.py"
DEMO_ARGUMENTS = ("--visualizer", "none", "--headless")
MAX_POSE_ERROR = 1e-3
"""Largest accepted start-pose error and hand drift [rad]."""
SOURCES = (
    "examples/demos/rizon_sharpa_coffee.py",
    "source/isaaclab_tasks/isaaclab_tasks/contrib/rizon_sharpa_coffee",
    "scripts/benchmarks/record_coffee_arm_motion.py",
)


def _committed_head() -> str:
    """Return ``HEAD``, after checking that the sources of the recording are committed."""
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *SOURCES], cwd=REPO_ROOT, capture_output=True)
    if dirty.returncode or dirty.stdout.strip():
        raise RuntimeError(f"Commit the recording's sources first: {', '.join(SOURCES)}")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _load_demo():
    spec = importlib.util.spec_from_file_location("rizon_sharpa_coffee_demo", DEMO_PATH)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return demo


def _rounded(rows):
    if isinstance(rows, list):
        return [_rounded(row) for row in rows]
    if not math.isfinite(rows):
        raise RuntimeError("The recording contains a non-finite value.")
    return round(rows, 6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="JSON file to write.")
    output_path = parser.parse_args().output
    command = shlex.join(["scripts/benchmarks/record_coffee_arm_motion.py", output_path.as_posix()])
    commit = _committed_head()

    demo = _load_demo()
    sys.argv = [str(DEMO_PATH), *DEMO_ARGUMENTS]
    args, _ = demo._parse_args()
    cfg = demo._make_env_cfg(args, args.control_hz // args.video_fps)
    os.environ["HEADLESS"] = "1"

    with demo.launch_simulation(cfg, args):
        import torch

        from isaaclab_tasks.contrib.rizon_sharpa_coffee.demo_env import CoffeeDemoEnv
        from isaaclab_tasks.contrib.rizon_sharpa_coffee.robot import (
            ARM_JOINT_NAMES,
            COFFEE_READY_POSITIONS,
            HAND_HOLD_POSITIONS,
            HAND_JOINT_NAMES,
        )
        from isaaclab_tasks.contrib.rizon_sharpa_coffee.scripted import CoffeeScoopDemo

        # As the demo's main() does without a visualizer.
        cfg.sim.physics.load_visual_shapes = bool(cfg.sim.visualizer_cfgs)
        cfg.grains_per_particle = 0
        env = CoffeeDemoEnv(cfg)
        try:
            env.reset()
            controller = CoffeeScoopDemo(env)
            robot = env.scene["robot"]
            arm_ids = [robot.joint_names.index(name) for name in ARM_JOINT_NAMES]
            hand_ids = [robot.joint_names.index(name) for name in HAND_JOINT_NAMES]
            hand_hold = torch.tensor(HAND_HOLD_POSITIONS, device=env.device)

            substep_targets: list[list[float]] = []
            apply_action = env.action_manager.apply_action

            def recording_apply_action() -> None:
                apply_action()
                substep_targets.append(robot.actuators.target_command.position.torch[0, arm_ids].tolist())

            env.action_manager.apply_action = recording_apply_action

            positions, velocities, targets, hand_drift = [], [], [], 0.0
            with torch.inference_mode():
                while True:
                    joint_pos = robot.data.joint_pos.torch[0]
                    positions.append(joint_pos[arm_ids].tolist())
                    velocities.append(robot.data.joint_vel.torch[0, arm_ids].tolist())
                    hand_drift = max(hand_drift, float((joint_pos[hand_ids] - hand_hold).abs().max()))
                    if controller.finished:
                        break
                    if len(targets) >= args.max_steps:
                        raise RuntimeError(f"The scripted sequence did not finish within {args.max_steps} steps.")
                    env.step(controller.compute())
                    # Physics that ran the decimation loop itself would call apply_action once and hide the substeps.
                    if len(substep_targets) != cfg.decimation:
                        raise RuntimeError(f"Expected {cfg.decimation} substep targets, got {len(substep_targets)}.")
                    targets.append(list(substep_targets))
                    substep_targets.clear()
                    if not env.is_state_finite().all():
                        raise RuntimeError(
                            f"Non-finite state in phase {controller.phase!r} after {len(targets)} steps."
                        )
        finally:
            env.close()

    start_error = max(abs(a - b) for a, b in zip(positions[0], COFFEE_READY_POSITIONS))
    # The benchmark starts from the ready pose and holds the hand at its default grip, so both must match the demo.
    if start_error > MAX_POSE_ERROR or hand_drift > MAX_POSE_ERROR:
        raise RuntimeError(
            f"Start error {start_error:.2e} or hand drift {hand_drift:.2e} exceeds {MAX_POSE_ERROR} rad."
        )
    recording = {
        "isaac_lab_commit": commit,
        "command": command,
        "demo_command": shlex.join(["examples/demos/rizon_sharpa_coffee.py", *DEMO_ARGUMENTS]),
        "rate_hz": args.control_hz,
        "physics_hz": args.physics_hz,
        "decimation": cfg.decimation,
        "joint_names": list(ARM_JOINT_NAMES),
        "positions": _rounded(positions),
        "velocities": _rounded(velocities),
        "targets": _rounded(targets),
    }
    output_path.write_text(json.dumps(recording) + "\n", encoding="utf-8")
    print(
        f"RECORDED frames={len(positions)} rate_hz={args.control_hz} physics_hz={args.physics_hz} "
        f"decimation={cfg.decimation} start_error={start_error:.2e} hand_drift={hand_drift:.2e} commit={commit}",
        flush=True,
    )


if __name__ == "__main__":
    main()
