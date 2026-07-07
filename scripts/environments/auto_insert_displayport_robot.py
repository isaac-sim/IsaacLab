# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Automated DisplayPort plug insertion: robot IK-grasps plug then descends into socket.

The environment setup (task, config overrides, grasp) is identical to
``step_displayport_robot.py``. Instead of manual control, the script runs an
autonomous descent trajectory with XY locking until the plug seats.

Phases per trial:
  1. Reset  — IK grasp positions arm, plug at z_clearance above socket.
  2. Settle — N hold steps to let the IK-solved arm stabilise.
  3. Descent — repeated z-action steps (negative = down) with XY locking.
               Stops on success (Δz reaches mated position) or stall/timeout.
  4. Hold   — a few steps at the final position for visual inspection.

Usage:
    ./isaaclab.sh -p scripts/environments/auto_insert_displayport_robot.py --num_envs 1 --viz kit

Typical one-liner (faster descent, 3 trials):
    ./isaaclab.sh -p scripts/environments/auto_insert_displayport_robot.py \\
        --num_envs 1 --viz kit --num_trials 3 --descent_action 1.0 --z_clearance 0.025
"""

import argparse
import sys

import gymnasium as gym
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    compute_plug_pose,
    compute_socket_root,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.joint_pos_env_cfg import (
    _GEOMETRY_POS,
    _SOCKET_ROT,
)

TASK = "Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-v0"

# 1 / _ACTION_SCALE from task_space_env_cfg: converts metres of EEF error → action units.
_GAIN = 1.0 / 0.025  # hold-step gain (proportional only, no coupling issue)

parser = argparse.ArgumentParser(description="Automated DisplayPort robot insertion.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--z_clearance",
    type=float,
    default=0.030,
    help="Initial plug clearance above socket [m] (default 30 mm).",
)
parser.add_argument(
    "--descent_action",
    type=float,
    default=0.5,
    help="Z-action magnitude per step applied downward (default 0.5). Increase for faster descent.",
)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=30,
    help="Hold steps after reset before descending (default 30).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=300,
    help="Max descent steps before declaring timeout (default 300).",
)
parser.add_argument(
    "--num_trials",
    type=int,
    default=1,
    help="Number of insertion trials (each preceded by a reset).",
)
parser.add_argument(
    "--stall_window",
    type=int,
    default=15,
    help="Steps over which to measure progress for stall detection (default 15).",
)
parser.add_argument(
    "--stall_threshold_mm",
    type=float,
    default=0.2,
    help="Minimum Δz progress over stall_window to avoid stall detection [mm] (default 0.2).",
)
parser.add_argument(
    "--seated_tolerance_mm",
    type=float,
    default=4.0,
    help="Δz within this of mated position → success [mm] (default 4.0).",
)
parser.add_argument(
    "--xy_kp",
    type=float,
    default=3.0,
    help="Proportional gain multiplier for XY locking during descent (default 3.0 × base gain).",
)
parser.add_argument(
    "--xy_ki",
    type=float,
    default=1.5,
    help="Integral gain for XY locking during descent [action_units / (m·step)] (default 1.5). "
         "Eliminates accumulated X drift from Jacobian coupling. Set 0 to disable.",
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _get_eef_pos(env) -> torch.Tensor | None:
    robot = env.unwrapped.scene["robot"]
    try:
        eef_idx = robot.find_bodies("flange")[0][0]
        return robot.data.body_pos_w[0, eef_idx].cpu()
    except Exception:
        return None


def _get_plug_pos(env) -> torch.Tensor:
    return wp.to_torch(env.unwrapped.scene["dp_plug"].data.root_pos_w)[0].cpu()


def _get_socket_pos(env) -> torch.Tensor:
    return wp.to_torch(env.unwrapped.scene["dp_socket"].data.root_pos_w)[0].cpu()


def _dz_m(env) -> float:
    """Plug root Z minus socket root Z in metres. Decreases as plug inserts."""
    return float(_get_plug_pos(env)[2] - _get_socket_pos(env)[2])


def _print_state(env, *, step: int | None = None, tag: str = "") -> None:
    eef = _get_eef_pos(env)
    plug = _get_plug_pos(env)
    sock = _get_socket_pos(env)
    dz = (plug[2] - sock[2]) * 1e3
    prefix = f"[{step:3d}] " if step is not None else "      "
    eef_str = (
        f"EEF ({eef[0]*1e3:.1f}, {eef[1]*1e3:.1f}, {eef[2]*1e3:.1f})  " if eef is not None else ""
    )
    print(
        f"  {prefix}{eef_str}"
        f"Plug ({plug[0]*1e3:.1f}, {plug[1]*1e3:.1f}, {plug[2]*1e3:.1f})  "
        f"Δz={dz:.2f} mm  {tag}"
    )


# ---------------------------------------------------------------------------
# Step primitives
# ---------------------------------------------------------------------------


def _hold_step(env, actions: torch.Tensor, hold_pos: torch.Tensor | None) -> None:
    """Step with a proportional action that drives EEF back to hold_pos."""
    step_action = torch.zeros(actions.shape[-1], device=actions.device)
    eef = _get_eef_pos(env)
    if hold_pos is not None and eef is not None:
        err = hold_pos.to(actions.device) - eef.to(actions.device)
        step_action[:3] += err * _GAIN
    actions[:] = step_action
    with torch.inference_mode():
        env.step(actions)


def _descent_step(
    env,
    actions: torch.Tensor,
    xy_lock_ref: torch.Tensor | None,
    z_action: float,
    xy_integral: torch.Tensor | None,
    kp: float,
    ki: float,
) -> None:
    """Step with z_action plus XY PI correction to cancel kinematic X-Z coupling.

    Proportional-only control (gain 40) can't fully cancel Jacobian cross-terms at each
    step — error accumulates across steps, causing visible X drift. The integral term
    drives the accumulated XY error to zero regardless of descent speed.
    """
    step_action = torch.zeros(actions.shape[-1], device=actions.device)
    step_action[2] = z_action
    eef = _get_eef_pos(env)
    if xy_lock_ref is not None and eef is not None:
        xy_err = xy_lock_ref[:2].to(actions.device) - eef[:2].to(actions.device)
        step_action[:2] += xy_err * kp
        if xy_integral is not None and ki > 0.0:
            xy_integral += xy_err
            step_action[:2] += xy_integral * ki
    actions[:] = step_action
    with torch.inference_mode():
        env.step(actions)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    env_cfg, _ = resolve_task_config(TASK, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        # Override plug starting height.
        new_plug_root, new_plug_rot = compute_plug_pose(
            _GEOMETRY_POS, _SOCKET_ROT, z_clearance=args_cli.z_clearance
        )
        env_cfg.scene.dp_plug.init_state.pos = new_plug_root
        env_cfg.scene.dp_plug.init_state.rot = new_plug_rot

        # Compute mated Δz for insertion detection.
        mated_plug_root, _ = compute_plug_pose(_GEOMETRY_POS, _SOCKET_ROT, z_clearance=0.0)
        socket_root = compute_socket_root(_GEOMETRY_POS, _SOCKET_ROT)
        mated_dz = mated_plug_root[2] - socket_root[2]  # metres
        seated_dz = mated_dz + args_cli.seated_tolerance_mm * 1e-3

        print(f"  Plug start:      {args_cli.z_clearance*1e3:.0f} mm above socket")
        print(f"  Mated Δz target: {mated_dz*1e3:.2f} mm  (success when Δz < {seated_dz*1e3:.2f} mm)")

        # Disable pose randomization — fixed geometry for reproducible tests.
        zero_range = {
            "x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0],
            "roll": [0.0, 0.0], "pitch": [0.0, 0.0], "yaw": [0.0, 0.0],
        }
        env_cfg.events.randomize_socket_pose.params["pose_range"] = zero_range
        if hasattr(env_cfg.events, "randomize_plug_pose"):
            env_cfg.events.randomize_plug_pose.params["pose_range"] = zero_range
        if hasattr(env_cfg.events, "reset_plug_curriculum"):
            env_cfg.events.reset_plug_curriculum.params["at_goal_prob"] = 0.0
            env_cfg.events.reset_plug_curriculum.params["normal_pose_range"] = {
                "x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]
            }

        # Inertial decoupling reduces (but does not eliminate) kinematic coupling.
        # gravity_compensation stays False: robot links have disable_gravity=True in PhysX,
        # so enabling compensation would push the arm upward against non-existent gravity.
        if hasattr(env_cfg.actions, "arm_action") and hasattr(
            env_cfg.actions.arm_action, "controller_cfg"
        ):
            env_cfg.actions.arm_action.controller_cfg.inertial_dynamics_decoupling = True

        env = gym.make(TASK, cfg=env_cfg)
        device = env.unwrapped.device
        actions = torch.zeros(env.action_space.shape, device=device)
        z_action = -abs(args_cli.descent_action)
        xy_kp = _GAIN * args_cli.xy_kp
        xy_ki = args_cli.xy_ki
        results: list[tuple[str, float]] = []

        for trial in range(args_cli.num_trials):
            print(f"\n{'='*50}")
            print(f"Trial {trial + 1} / {args_cli.num_trials}")
            print(f"{'='*50}")

            # --- Reset ---
            env.reset()
            actions[:] = 0.0
            _print_state(env, tag="(post-reset)")

            # --- Phase 1: Settle ---
            print(f"  Settling ({args_cli.settle_steps} steps)...")
            hold_pos = _get_eef_pos(env)
            for _ in range(args_cli.settle_steps):
                _hold_step(env, actions, hold_pos)
            hold_pos = _get_eef_pos(env)
            _print_state(env, tag="(settled)")

            # --- Phase 2: Descent ---
            print(f"  Descending (z_action={z_action:.2f}, xy_kp={xy_kp:.1f}, xy_ki={xy_ki:.2f}, max {args_cli.max_steps} steps)...")
            xy_ref = _get_eef_pos(env)  # XY target held constant throughout descent
            xy_integral = torch.zeros(2, device=device)  # reset integrator each trial
            dz_history: list[float] = []
            success = stall = False
            final_step = 0

            for step in range(args_cli.max_steps):
                _descent_step(env, actions, xy_ref, z_action, xy_integral, xy_kp, xy_ki)
                cur_dz = _dz_m(env)
                dz_history.append(cur_dz)
                final_step = step + 1

                if (step + 1) % 10 == 0:
                    _print_state(env, step=step + 1)

                # Success: plug reached the mated position within tolerance.
                if cur_dz <= seated_dz:
                    success = True
                    break

                # Stall: not enough downward progress over the last stall_window steps.
                w = args_cli.stall_window
                if len(dz_history) >= w:
                    progress = (dz_history[-w] - dz_history[-1]) * 1e3  # mm
                    if progress < args_cli.stall_threshold_mm:
                        stall = True
                        break

            # --- Phase 3: Hold for inspection ---
            final_eef = _get_eef_pos(env)
            for _ in range(30):
                _hold_step(env, actions, final_eef)

            cur_dz = _dz_m(env)
            gap_mm = (cur_dz - mated_dz) * 1e3
            outcome = "SUCCESS" if success else ("STALL" if stall else "TIMEOUT")
            print(f"\n  {outcome} at step {final_step}")
            _print_state(env, tag=f"({outcome})")
            print(f"  Final Δz: {cur_dz*1e3:.2f} mm  |  gap to mated: {gap_mm:.2f} mm")
            results.append((outcome, cur_dz))

        # --- Summary ---
        print(f"\n{'='*50}")
        print(f"Summary: {args_cli.num_trials} trial(s)")
        print(f"{'='*50}")
        successes = sum(1 for r, _ in results if r == "SUCCESS")
        print(f"  {successes}/{args_cli.num_trials} successful")
        for i, (r, dz) in enumerate(results):
            gap = (dz - mated_dz) * 1e3
            print(f"  Trial {i+1:2d}: {r:8s}  Δz={dz*1e3:.2f} mm  gap={gap:+.2f} mm")

        env.close()


if __name__ == "__main__":
    main()
