# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Insert, release, regrasp and extract an Ethernet cable in one continuous simulation.

The non-RL manager-based environment owns physics and reset. This example returns ordinary TCP
and finger targets. Measured poses, finger opening and latch clearance gate the
sequence. The managed dependencies include the Newton hydroelastic contact correction.

.. code-block:: bash

    uv run isaaclab example sharpa-rj45
    uv run --extra isaacsim --extra video isaaclab example sharpa-rj45 --viz kit --video
    uv run isaaclab example sharpa-rj45 --viz none

"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from isaaclab.examples.demos.rj45.cycle import CableCycleStateMachine

logger = logging.getLogger(__name__)


def main() -> int:
    """Run the bounded physical cycle; return success only after extraction."""
    from isaaclab.app import add_launcher_args

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max_steps", "--num_steps", dest="num_steps", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--settle_duration_s", type=float, default=0.0, help="Initial physical grasp hold [s].")
    parser.add_argument("--video", action="store_true", help="Record with the standard task video recorder.")
    parser.add_argument(
        "--failure_tail_s", type=float, default=0.0, help="Simulate with held targets after a fault [s]."
    )
    parser.add_argument("--output_dir", type=Path, default=Path("logs/demos/sharpa_rj45"))
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["newton_gl"])
    args = parser.parse_args()
    if args.num_steps <= 0:
        parser.error("--num_steps must be positive")
    for name in ("settle_duration_s", "failure_tail_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name} must be finite and nonnegative")
    return _run(args)


def _run(args: argparse.Namespace) -> int:
    import torch

    from isaaclab.app import launch_simulation
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.utils import instantiate
    from isaaclab.utils.math import subtract_frame_transforms

    from isaaclab_tasks.contrib.gb300_sharpa_insertion import mdp
    from isaaclab_tasks.contrib.gb300_sharpa_insertion.env_cfg import CLOSE_HAND_RAD, GB300SharpaInsertionEnvCfg
    from isaaclab_tasks.contrib.gb300_sharpa_insertion.visual_scene import add_gl_visuals, configure_visual_scene
    from isaaclab_tasks.contrib.gb300_sharpa_insertion.workcell import head_goal_for_socket

    from isaaclab_assets.robots._rizon_sharpa_bundle import resolve_robot_usd
    from isaaclab_assets.robots.rizon_sharpa import HAND_JOINT_NAMES

    cfg = GB300SharpaInsertionEnvCfg()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    if args.video:
        cfg.video_recorders = [
            VideoRecorderCfg(
                output_dir=str(args.output_dir),
                output_filename_prefix=f"cycle_{datetime.now():%Y%m%d_%H%M%S_%f}",
                video_length=args.num_steps,
            )
        ]
    with launch_simulation(cfg, args):
        cfg.scene.robot.spawn.usd_path = resolve_robot_usd()
        rack_path = configure_visual_scene(cfg)
        env = instantiate(cfg)
        try:
            observations, _ = env.reset(seed=args.seed)
            add_gl_visuals(env, rack_path)
            if env.action_manager.total_action_dim != 29:
                raise RuntimeError("The example must provide absolute TCP and 22 finger targets.")
            robot = env.scene["robot"]
            hand_ids, _ = robot.find_joints(HAND_JOINT_NAMES, preserve_order=True)

            def controller_state(observations: dict[str, torch.Tensor]) -> dict:
                # Transfer the controller measurements and validity flag together once per control step.
                captured, margins, release, exit_margin = mdp.latch_metrics(env)
                fields = (
                    mdp.tcp_pose(env)[0],
                    mdp.head_pose(env)[0],
                    mdp.latch_state(env)[0, :1],
                    captured[0, None],
                    margins[0],
                    release[0, None],
                    exit_margin[0, None],
                    robot.data.joint_pos.torch[0, hand_ids],
                    torch.isfinite(observations["state"]).all()[None],
                )
                packed = torch.cat(fields).cpu()
                tcp, head, angle, engaged, margins, release, exit_margin, hand, finite = packed.split(
                    [field.numel() for field in fields]
                )
                if not finite.item():
                    raise RuntimeError("Cable cycle encountered a non-finite simulation state.")
                return {
                    "tcp_pose_w": tcp.numpy(),
                    "head_pose_w": head.numpy(),
                    "latch_engaged": bool(engaged.item()),
                    "latch_angle_rad": angle.item(),
                    "capture_margins_m": margins.numpy(),
                    "keeper_release_margin_m": release.item(),
                    "keeper_exit_margin_m": exit_margin.item(),
                    "hand_joint_pos": hand.numpy(),
                }

            state = controller_state(observations)
            goal_position = np.asarray(head_goal_for_socket(cfg.target_socket_position_e))
            goal_position += env.scene.env_origins[0].cpu().numpy()
            machine = CableCycleStateMachine(
                state["tcp_pose_w"],
                np.r_[goal_position, 2**-0.5, 0.0, 0.0, 2**-0.5],
                env.step_dt,
                CLOSE_HAND_RAD,
                settle_duration_s=args.settle_duration_s,
            )
            initial_hand = robot.data.joint_pos.torch[:, hand_ids].clone()
            closed_hand = initial_hand.new_tensor(CLOSE_HAND_RAD)[None]
            step = 0
            failure_step: int | None = None
            failure_tail_steps = math.ceil(min(args.failure_tail_s / env.step_dt, args.num_steps))
            reason = "Simulation closed or step budget exhausted before the complete cycle."
            prior_phase = machine.phase
            with torch.inference_mode():
                while env.sim.is_running() and step < args.num_steps:
                    if not env.sim.is_playing():
                        env.sim.render()
                        continue
                    if failure_step is None:
                        command, fingers = machine.update(**state)
                    if machine.phase != prior_phase:
                        logger.info("[GB300 cycle] step=%d phase=%s", step, machine.phase)
                        prior_phase = machine.phase
                    if machine.phase == "DONE":
                        reason = None
                        break
                    if machine.phase == "FAILED":
                        if failure_step is None:
                            failure_step = step
                            reason = machine.failure
                            logger.warning(
                                "[GB300 cycle] fault=%s; holding targets for %g s", reason, args.failure_tail_s
                            )
                        if step - failure_step >= failure_tail_steps:
                            break
                    targets = torch.as_tensor(np.r_[command, fingers], dtype=torch.float32, device=env.device)[None]
                    root = robot.data.root_pose_w.torch
                    position, rotation = subtract_frame_transforms(
                        root[:, :3], root[:, 3:], targets[:, :3], targets[:, 3:7]
                    )
                    hand_targets = targets[:, 7:]
                    if failure_step is None and step * env.step_dt < 0.2:
                        fraction = min((step + 1) * env.step_dt / 0.2, 1.0)
                        hand_targets = torch.lerp(initial_hand, closed_hand, fraction)
                    observations, _ = env.step(torch.cat((position, rotation, hand_targets), dim=-1))
                    step += 1
                    state = controller_state(observations)
            success = machine.phase == "DONE" and machine.insertion_completed
            print(
                json.dumps(
                    {
                        "success": success,
                        "phase": machine.phase,
                        "reason": reason,
                        "insertion_completed": machine.insertion_completed,
                        "steps": step,
                        "failure_step": failure_step,
                        "post_failure_simulation_s": 0.0
                        if failure_step is None
                        else (step - failure_step) * env.step_dt,
                        "measured_withdrawal_m": machine.measured_withdrawal_m,
                    },
                    indent=2,
                ),
                flush=True,
            )
            return 0 if success else 1
        finally:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
