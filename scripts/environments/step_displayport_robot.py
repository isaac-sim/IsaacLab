# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive insertion tester: Flexiv Rizon 4S holding a DisplayPort plug, driven by user into socket.

Usage:
    ./isaaclab.sh -p scripts/environments/step_displayport_robot.py --num_envs 1 --viz kit

How it works:
    At reset, the existing IK-grasp event positions the robot arm so its gripper holds the
    DisplayPort plug, which starts ~68 mm above the socket opening with perfect X/Y alignment.
    The user then sends OSC (task-space) actions to drive the EEF downward. The OSC controller
    converts those into joint torques that apply a real continuous force — the robot pushes the
    plug in, it does not teleport.

    The gripper stays closed automatically via PD actuator stiffness (no gripper action needed).

    Socket pose randomization is disabled so the geometry is fixed and reproducible.

Action convention (insertion axis = world -Z):
    z -1.0 50  →  push down for 50 steps (burst mode — runs immediately, no per-step delay)
    z -1.0     →  set persistent z-action (keeps applying every loop iteration)
    0          →  stop / zero all actions

Why burst mode matters:
    The OSC impedance controller (stiffness=300 N/m) only achieves ~10% of the target offset per
    8-physics-sub-step call. Each env.step() moves the EEF ~2-3 mm at action magnitude 1.0.
    Burst mode runs N steps back-to-back so a single command produces noticeable travel.
    Suggested starting point: "z -1.0 30" (≈60–90 mm of EEF travel, enough to reach the socket).

Commands (type at the prompt, then press Enter):
    z <val> [n]  set EEF z-action (no n=persistent, n given=burst n steps)  e.g. "z -1.0 30"
    x <val> [n]  set EEF x-action
    y <val> [n]  set EEF y-action
    rx/ry/rz <val> [n]  rotation actions
    0            zero ALL actions (stop)
    a            print current action vector
    p            print EEF, plug, socket positions and insertion depth
    r            reset scene (re-runs IK grasp)
    f            fast mode (full sim speed, no per-step delay)
    s            slow mode (default, 0.1 s/step between persistent steps)
    q            quit

Action values are not clamped — use magnitudes > 1 for faster motion (e.g. "z -3.0 20").
"""

import argparse
import select
import sys
import time

import gymnasium as gym
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    compute_plug_pose,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.joint_pos_env_cfg import (
    _GEOMETRY_POS,
    _SOCKET_ROT,
)
from isaaclab_tasks.contrib.deploy.cable_insertion.config.displayport_rizon_4s.task_space_env_cfg import (
    _INSERTION_LENGTH,
)

TASK = "Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-TaskSpace-v0"

parser = argparse.ArgumentParser(description="Interactive DisplayPort robot insertion test.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--step_delay", type=float, default=0.1, help="Seconds between steps in slow mode.")
parser.add_argument("--z_clearance", type=float, default=0.015, help="Initial plug-to-socket clearance in metres (default 15 mm). Negative = start inside socket.")
parser.add_argument("--start_seated", action="store_true", help="Spawn plug inside socket. Use --insert_depth to control how far in (default 3 mm).")
parser.add_argument("--insert_depth", type=float, default=0.003, help="Insertion depth in metres when using --start_seated (default 3 mm).")
parser.add_argument("--show_colliders", action="store_true", help="Overlay collision meshes in the viewport.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

_AXIS_MAP = {"x": 0, "y": 1, "z": 2, "rx": 3, "ry": 4, "rz": 5}

HELP = """
=== DisplayPort Robot Insertion Test ===
Robot holds plug via IK grasp. Drive it into the socket with OSC actions.

  z <val> [n]  → EEF z-action, negative=down  (e.g. "z -1.0 30" = 30-step burst)
  x <val> [n]  → EEF x-action (lateral)
  y <val> [n]  → EEF y-action (lateral)
  rx/ry/rz <v> [n]  → EEF rotation actions
  0            → zero all actions (stop)
  a            → print current action vector
  p            → print EEF, plug, socket positions + insertion depth
  r            → reset (re-runs IK grasp, plug back to start)
  f            → fast mode (full speed)
  s            → slow mode (default, 0.1 s/step between persistent steps)
  q            → quit

With [n]: burst — runs n steps immediately (no per-step delay), then stops.
Without [n]: persistent — action keeps applying every loop iteration until '0'.
Suggested start: "z -1.0 30"  (~60-90 mm of EEF travel downward).
Use --start_seated to spawn plug already inserted for in-socket movement testing.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_eef_pos(env):
    robot = env.unwrapped.scene["robot"]
    try:
        eef_idx = robot.find_bodies("flange")[0][0]
        return robot.data.body_pos_w[0, eef_idx].cpu()
    except Exception:
        return None


def _get_plug_pos(env):
    return wp.to_torch(env.unwrapped.scene["dp_plug"].data.root_pos_w)[0].cpu()


def _get_socket_pos(env):
    return wp.to_torch(env.unwrapped.scene["dp_socket"].data.root_pos_w)[0].cpu()


def _print_poses(env):
    eef = _get_eef_pos(env)
    plug = _get_plug_pos(env)
    sock = _get_socket_pos(env)

    if eef is not None:
        print(f"  EEF (flange): ({eef[0]*1e3:.1f}, {eef[1]*1e3:.1f}, {eef[2]*1e3:.1f}) mm")
    print(f"  Plug  root:   ({plug[0]*1e3:.2f}, {plug[1]*1e3:.2f}, {plug[2]*1e3:.2f}) mm")
    print(f"  Socket root:  ({sock[0]*1e3:.2f}, {sock[1]*1e3:.2f}, {sock[2]*1e3:.2f}) mm")

    # Insertion depth proxy: plug z - socket z.
    # Insertion axis = world -Z, so this delta decreases as plug enters socket.
    # SOCKET_INSERTION_OFFSET = [0.0375, 0, 0] in socket-local X, which maps to world -Z
    # with the default socket rotation. So the socket slot opening is ~37.5 mm above sock root in -Z.
    # This is an approximate metric; it captures the trend clearly.
    dz_mm = (plug[2] - sock[2]) * 1e3
    print(f"  Δz (plug - socket root): {dz_mm:.2f} mm  (decreasing → inserting)")


def _check_stdin():
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.readline().strip()
    return None


def _handle_command(raw, current_action):
    """Parse a command and update current_action in place.

    Returns (quit: bool, reset: bool, mode: str | None, burst_steps: int).
    burst_steps > 0 means caller should run that many steps immediately.
    """
    parts = raw.split()
    if not parts:
        return False, False, None, 0
    cmd = parts[0].lower()

    if cmd == "q":
        return True, False, None, 0
    elif cmd == "r":
        current_action[:] = 0.0
        return False, True, None, 0
    elif cmd == "0":
        current_action[:] = 0.0
        print("  All actions zeroed.")
    elif cmd == "a":
        print(f"  Current action: {[f'{v:.3f}' for v in current_action.tolist()]}")
    elif cmd in _AXIS_MAP:
        if len(parts) < 2:
            print(f"  Usage: {cmd} <val> [n_steps]  (e.g. '{cmd} -1.0 30')")
        else:
            try:
                val = float(parts[1])
                burst = int(parts[2]) if len(parts) >= 3 else 0
                current_action[_AXIS_MAP[cmd]] = val
                mode_str = f"burst {burst} steps" if burst > 0 else "persistent"
                print(f"  action[{cmd}] = {val:.3f}  ({mode_str})  full: {[f'{v:.3f}' for v in current_action.tolist()]}")
                return False, False, None, burst
            except ValueError:
                print(f"  Invalid value(s): {parts[1:]}")
    elif cmd == "f":
        return False, False, "fast", 0
    elif cmd == "s":
        return False, False, "slow", 0
    else:
        print("  Unknown command. Try 'z -1.0 30' to push down, '0' to stop, 'q' to quit.")
    return False, False, None, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    env_cfg, _ = resolve_task_config(TASK, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device

        # Disable pose randomization — fixed geometry for reproducible insertion tests.
        zero_range = {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0],
                      "roll": [0.0, 0.0], "pitch": [0.0, 0.0], "yaw": [0.0, 0.0]}
        env_cfg.events.randomize_socket_pose.params["pose_range"] = zero_range

        # --start_seated: spawn plug at the fully-mated position (blade fully inside socket).
        # Uses geometry: z_clearance = -_INSERTION_LENGTH (negative = inside socket).
        z_clearance = -args_cli.insert_depth if args_cli.start_seated else args_cli.z_clearance
        new_plug_root, new_plug_rot = compute_plug_pose(_GEOMETRY_POS, _SOCKET_ROT, z_clearance=z_clearance)
        env_cfg.scene.dp_plug.init_state.pos = new_plug_root
        env_cfg.scene.dp_plug.init_state.rot = new_plug_rot
        if args_cli.start_seated:
            print(f"  Plug start: {args_cli.insert_depth*1e3:.0f} mm inside socket  root={tuple(f'{v:.4f}' for v in new_plug_root)}")
            # Spawning inside the socket causes collision overlap → huge depenetration forces.
            # Cap the depenetration velocity so PhysX resolves the overlap gradually.
            env_cfg.scene.dp_plug.spawn.rigid_props.max_depenetration_velocity = 0.05
        else:
            print(f"  Plug start: {z_clearance*1e3:.0f} mm above socket  root={tuple(f'{v:.4f}' for v in new_plug_root)}")

        if hasattr(env_cfg.events, "randomize_plug_pose"):
            env_cfg.events.randomize_plug_pose.params["pose_range"] = zero_range
        if hasattr(env_cfg.events, "reset_plug_curriculum"):
            env_cfg.events.reset_plug_curriculum.params["at_goal_prob"] = 0.0
            env_cfg.events.reset_plug_curriculum.params["normal_pose_range"] = zero_range

        # Enable inertial dynamics decoupling for cleaner manual control.
        # gravity_compensation must remain False: the robot links have disable_gravity=True in PhysX,
        # so OSC gravity torques would push the arm upward against non-existent gravity.
        if hasattr(env_cfg.actions, "arm_action") and hasattr(env_cfg.actions.arm_action, "controller_cfg"):
            env_cfg.actions.arm_action.controller_cfg.inertial_dynamics_decoupling = True

        env = gym.make(TASK, cfg=env_cfg)
        env.reset()

        if args_cli.start_seated:
            # Run settle steps with zero action so PhysX can gently resolve any spawn overlap
            # before handing control to the user.
            print("  Settling physics (50 steps) ...")
            settle_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            for _ in range(50):
                with torch.inference_mode():
                    env.step(settle_actions)
            print("  Settled.")

        if args_cli.show_colliders:
            import carb
            s = carb.settings.get_settings()
            s.set("/physics/visualizationDisplayColliders", True)
            print("  Collision meshes enabled.")

        action_dim = env.action_space.shape[-1]
        current_action = torch.zeros(action_dim)  # 6D OSC action, persists across steps
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

        # pose_rel with zero action sets target = current pos each step → no restoring force.
        # We hold position explicitly by tracking a reference EEF position and issuing a
        # corrective action to drive back to it during the idle loop.
        hold_eef_pos = _get_eef_pos(env)  # world-frame (3,) tensor, updated after reset/burst

        slow_mode = True
        step_delay = args_cli.step_delay

        print(HELP)
        _print_poses(env)
        print(f"Sim running — slow mode ({step_delay}s/step). Type commands anytime.\n")

        while True:
            line = _check_stdin()
            if line is not None:
                cmd0 = line.lower().split()[0] if line.split() else ""

                if cmd0 == "p":
                    _print_poses(env)
                    continue

                quit_flag, reset_flag, mode, burst = _handle_command(line, current_action)
                if quit_flag:
                    break
                if reset_flag:
                    env.reset()
                    current_action[:] = 0.0
                    hold_eef_pos = _get_eef_pos(env)
                    print("  Scene reset — robot re-grasping plug.")
                    _print_poses(env)
                    continue
                if mode == "fast":
                    slow_mode = False
                    print("  Fast mode.")
                    continue
                elif mode == "slow":
                    slow_mode = True
                    print(f"  Slow mode ({step_delay}s/step).")
                    continue

                if burst > 0:
                    # Run burst steps with XY locking: record starting EEF XY and add
                    # proportional corrections each step to cancel kinematic X-Z coupling.
                    # The coupling is geometric (arm Jacobian cross-terms), not inertial,
                    # so OSC tuning alone cannot eliminate it.
                    start_eef = _get_eef_pos(env)
                    # Gain converts metres of XY error → action units (1 / position_scale).
                    # 0.025 is _ACTION_SCALE from task_space_env_cfg.
                    _XY_LOCK_GAIN = 1.0 / 0.025
                    for _ in range(burst):
                        step_action = current_action.clone()
                        if start_eef is not None:
                            cur_eef = _get_eef_pos(env)
                            if cur_eef is not None:
                                xy_err = start_eef[:2] - cur_eef[:2]
                                step_action[0] += float(xy_err[0]) * _XY_LOCK_GAIN
                                step_action[1] += float(xy_err[1]) * _XY_LOCK_GAIN
                        actions[:] = torch.tensor(step_action, device=env.unwrapped.device)
                        with torch.inference_mode():
                            env.step(actions)
                    # After burst: update hold reference to the new EEF position.
                    current_action[:] = 0.0
                    actions[:] = 0.0
                    hold_eef_pos = _get_eef_pos(env)
                    print(f"  Burst done ({burst} steps).")
                    _print_poses(env)
                    continue
                else:
                    # Persistent action set: snapshot current EEF XY as lock reference so
                    # the idle loop can cancel X-Z Jacobian coupling during the push.
                    new_eef = _get_eef_pos(env)
                    if new_eef is not None:
                        hold_eef_pos = new_eef

            # Idle loop: apply current_action + XY proportional correction to cancel
            # kinematic coupling. XY correction always runs; Z hold only when fully idle.
            _HOLD_GAIN = 1.0 / 0.025  # metres → action units
            idle_action = current_action.clone()
            if hold_eef_pos is not None:
                cur_eef = _get_eef_pos(env)
                if cur_eef is not None:
                    xy_err = hold_eef_pos[:2] - cur_eef[:2]
                    idle_action[0] += float(xy_err[0]) * _HOLD_GAIN
                    idle_action[1] += float(xy_err[1]) * _HOLD_GAIN
                    if current_action.abs().max() < 1e-6:
                        # Full XYZ hold when no action is active.
                        idle_action[2] += float(hold_eef_pos[2] - cur_eef[2]) * _HOLD_GAIN
            actions[:] = torch.tensor(idle_action, device=env.unwrapped.device)
            with torch.inference_mode():
                env.step(actions)

            if slow_mode:
                time.sleep(step_delay)

        env.close()


if __name__ == "__main__":
    main()
