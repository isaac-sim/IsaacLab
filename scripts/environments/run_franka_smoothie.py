# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the repository-owned fruit, tap, lid and blender task with its scripted controller."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path


def main() -> None:
    """Step the scripted smoothie sequence, optionally rendering it live."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--seed", type=int, default=70044)
    parser.add_argument("--robot_usd_path", type=Path, help="Optional local Franka asset instead of the bundled asset.")
    playback = parser.add_mutually_exclusive_group()
    playback.add_argument("--replay", type=Path, help="Replay joint actions from a previously saved .npz recording.")
    playback.add_argument(
        "--record",
        type=Path,
        help=("Run the live, fully controlled sequence and write its actions/stages to this .npz path after success."),
    )
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max_steps must be positive.")
    if not args.headless and not os.environ.get("DISPLAY"):
        parser.error("Run from a graphical desktop terminal, or pass --headless.")
    if args.robot_usd_path is not None:
        if not args.robot_usd_path.is_file():
            parser.error("--robot_usd_path must be an existing USD file.")
        os.environ["ISAACLAB_FRANKA_POUR_ROBOT_USD_PATH"] = str(args.robot_usd_path.resolve())
    os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    import numpy as np
    import torch

    from isaaclab.app import launch_simulation

    from isaaclab_tasks.contrib.franka_smoothie.recorded_controller import RecordedSequenceController
    from isaaclab_tasks.contrib.franka_smoothie.smoothie_controller import SmoothieSequenceController
    from isaaclab_tasks.contrib.franka_smoothie.smoothie_env import SmoothieBlenderEnv
    from isaaclab_tasks.contrib.franka_smoothie.smoothie_env_cfg import FrankaSmoothieEnvCfg

    cfg = FrankaSmoothieEnvCfg()
    cfg.seed = args.seed

    with contextlib.ExitStack() as resources:
        resources.enter_context(launch_simulation(cfg))
        with torch.inference_mode():
            env = SmoothieBlenderEnv(cfg)
            resources.callback(env.close)
            env.reset(seed=args.seed)

            viewer = visuals = None
            if not args.headless:
                from isaaclab_visualizers.newton import NewtonGLVisualizer, NewtonGLVisualizerCfg

                from isaaclab_tasks.contrib.franka_smoothie.smoothie_visuals import SmoothieVisuals

                viewer = NewtonGLVisualizer(
                    NewtonGLVisualizerCfg(
                        enable_picking=False,
                        show_particles=False,
                        streaming_view=False,
                        window_width=1280,
                        window_height=800,
                        eye=(1.4, -1.2, 1.05),
                        lookat=(0.45, 0.0, 0.20),
                    )
                )
                resources.callback(viewer.close)
                viewer.initialize(env.sim.get_scene_data_provider())
                visuals = SmoothieVisuals()
                resources.callback(visuals.close)

            recording = args.record is not None
            controller = (
                RecordedSequenceController(env, recording_path=args.replay)
                if args.replay is not None
                else SmoothieSequenceController(env)
            )
            resources.callback(controller.close_trace)
            recorded_actions: list[np.ndarray] = []
            recorded_stages: list[str] = []
            mode = "replaying recorded sequence" if args.replay is not None else "live sequence"
            print(f"Running franka smoothie task ({mode}); seed {args.seed}", flush=True)
            for step in range(args.max_steps or env.max_episode_length):
                if viewer is not None and not viewer.is_running():
                    print("Window closed.", flush=True)
                    break
                actions = controller.compute(step)
                if recording:
                    recorded_actions.append(actions[0].detach().cpu().numpy().copy())
                    recorded_stages.append(controller.stage)
                _, _, terminated, truncated, _ = env.step(actions)
                done = bool(terminated[0] | truncated[0])
                if viewer is not None and not done:
                    visuals.update(
                        viewer,
                        env.pose("cup")[0].cpu().numpy(),
                        float(env.task.fill_fraction[0]),
                        bool(env.task.tap_on[0]),
                    )
                    viewer.step(env.step_dt)
                if step % 30 == 0 or done:
                    print(f"{(step + 1) * env.step_dt:.1f}s {controller.stage}", flush=True)
                if done:
                    succeeded = bool(env.termination_manager.get_term("success")[0])
                    print(f"Terminated at step {step + 1}: {'success' if succeeded else 'failure'}.", flush=True)
                    if recording:
                        if not succeeded:
                            parser.error("Refusing to save a recording from a run that did not succeed.")
                        np.savez(
                            args.record,
                            actions=np.stack(recorded_actions).astype(np.float32),
                            stages=np.asarray(recorded_stages),
                        )
                        print(f"Saved recording to {args.record}.", flush=True)
                    break
            else:
                print("Reached step limit without terminating.", flush=True)


if __name__ == "__main__":
    main()
