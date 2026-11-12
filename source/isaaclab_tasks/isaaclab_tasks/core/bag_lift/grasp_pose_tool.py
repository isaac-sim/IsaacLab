# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Define the grasp by hand, then test it open-loop (no learning).

You describe the grasp as joint angles [deg] in ``grasp_poses.json`` (next to this file). Two modes:

  live  (default)  Hold the waypoint named by ``"show"`` and RELOAD the file whenever you save it, so you can edit
                   the numbers in your editor and watch the arm move in the Kit window. The terminal prints where the
                   rigid-jaw tip is in the BAG frame, jaw-to-strap gaps, and whether the geometric grasp is true.
                   When it looks right, that tip pose is the grasp pose.
  --record         Play the sequence up to the "show" waypoint (or --show), then WRITE the pose to ``grasp_goal.json``
                   as the training target (rigid-jaw tip position and orientation in the BAG frame, gripper angle,
                   joints); then exit.
  run              Replay the file's ``sequence`` (pregrasp -> grasp -> grasp_closed -> lift -> ee_up ->
                   return_home), print the result of every stage, and
                   report whether the bag was lifted 5 cm while pinched. This is the "does the physics work" test.
                   --stop_after grasp_closed   stop after that waypoint (e.g. test only the gripper closing)
                   --save_traj [FILE]          write every control step (joints, tip in the bag frame, jaw-to-strap
                                               gaps, grasped flag) to a JSON file: an example trajectory

Trajectory timing (top-level keys in the pose file):
  "sequence"          optional list of waypoint names to visit in order, e.g. ["pregrasp", "approach", "grasp", ...].
                      Add your own waypoint blocks (any name) and list them here to shape the path.
  "speed_deg_per_s"   optional {waypoint: {joint: speed}}. Speed of EACH JOINT during the move into that waypoint,
                      e.g. "grasp": {"revolute_1": 5, "revolute_2": 12, "default": 8}. A joint takes
                      |change| / speed seconds (smooth start and stop, peak 1.5x the number) and then waits for the
                      others; the move lasts as long as the slowest joint. "default" covers joints you don't list.
                      A plain number instead of a dict means that speed for every joint. Joints without a speed
                      finish together with the move (--move_s when nothing is given). The motors are capped at
                      28.6 deg/s, so a peak above that is reported and the arm will lag the command.

Waypoints (each lists only the joints it changes; the rest are inherited from the previous waypoint):
  pregrasp      jaws open, a few cm away from the strap
  grasp         jaws open, positioned around the strap
  grasp_closed  same as grasp, gripper closed on the strap
  lift          arm raised with the gripper still closed, so the bag hangs from the jaws
  ee_up         wrist pitched so the end-effector points up (before the carry home)

The bag is NOT randomised and the actions are not noisy here, so the same file gives the same result every time.
The training env increments joint targets by ``action_scale * action`` each step (zero = hold). This tool widens
``action_scale`` to 4 rad so a single command can jump to any pose inside the USD joint limits.

Usage (from the Isaac Lab repo root):
    uv run python source/isaaclab_tasks/isaaclab_tasks/core/bag_lift/grasp_pose_tool.py --viz kit
    uv run python source/isaaclab_tasks/isaaclab_tasks/core/bag_lift/grasp_pose_tool.py --viz kit --mode run
    uv run python source/isaaclab_tasks/isaaclab_tasks/core/bag_lift/grasp_pose_tool.py --viz kit --mode run \\
        --stop_after grasp_closed --save_traj
"""

# Warp captures ``enable_backward`` when a module is created, so set it before importing Warp kernels.
import warp as wp

wp.config.enable_backward = False

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul  # noqa: E402

from isaaclab_rl.entrypoints.common import apply_env_overrides, close_env  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli  # noqa: E402

WAYPOINTS = ("pregrasp", "grasp", "grasp_closed", "lift")
VEL_LIMIT_DEG = 28.6  # [deg/s] the arm motors are capped at 0.5 rad/s (bag_sand_scene.py joint_velocity_limit)
MIN_MOVE_S = 0.3


def sequence_of(poses: dict) -> list[str]:
    """Order in which --mode run visits the waypoints (default: the four standard ones)."""
    return list(poses.get("sequence", WAYPOINTS))


DEFAULT_POSES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_poses.json")
DEFAULT_GOAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_goal.json")
DEFAULT_TRAJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_trajectory.json")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand-posed grasp: live editing and open-loop test.")
    parser.add_argument("--task", type=str, default="Isaac-BagLift-Direct")
    parser.add_argument("--mode", choices=("live", "run"), default="live")
    parser.add_argument("--poses", type=str, default=DEFAULT_POSES, help="Pose file (created if missing).")
    parser.add_argument("--record", action="store_true", help="Write the 'show' waypoint to the goal file and exit.")
    parser.add_argument("--goal", type=str, default=DEFAULT_GOAL, help="Goal file written by --record.")
    parser.add_argument("--stop_after", type=str, default=None, help="[run] stop after this waypoint.")
    parser.add_argument(
        "--save_traj",
        nargs="?",
        const=DEFAULT_TRAJ,
        default=None,
        help="[run] write the per-step trajectory to this JSON file (default grasp_trajectory.json next to the tool).",
    )
    parser.add_argument("--move_s", type=float, default=2.0, help="[run] seconds per arm move.")
    parser.add_argument("--close_s", type=float, default=1.5, help="[run] seconds to close the gripper.")
    parser.add_argument(
        "--hold_s", type=float, default=None, help="[run] seconds to hold after each stage (default: cfg.demo_hold_s)."
    )
    parser.add_argument(
        "--show", type=str, default=None, help="Waypoint to hold / record (default: the file's 'show')."
    )
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    add_launcher_args(parser)
    parser.set_defaults(device=None)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    sys.argv = [sys.argv[0]] + hydra_args
    args_cli.num_envs = 1
    return args_cli


class GraspRig:
    """One env, joint targets in degrees, and a status line about the tip and the contacts."""

    def __init__(self, env: gym.Env):
        self.env = env
        self.u = env.unwrapped
        _, self.names = self.u.arm.find_joints(".*")
        self.home = self.u.arm.data.default_joint_pos.torch[0].clone()
        self.target = self.home.clone()
        self.ever_success = False  # the env's success (pinched + at the final pose) at any step
        self.max_lift = 0.0  # [m] highest the bag got above its resting height, over every control step
        self.stage = ""  # label stored with each logged step
        self.log: list[dict] | None = None  # set to [] to record every control step

    # --- poses ------------------------------------------------------------------------------------------
    def apply_pose(self, base: torch.Tensor, pose: dict) -> torch.Tensor:
        """Return ``base`` with the joints named in ``pose`` (degrees) replaced."""
        out = base.clone()
        for name, deg in pose.items():
            if name not in self.names:
                raise KeyError(f"unknown joint '{name}'. Joints are: {self.names}")
            out[self.names.index(name)] = math.radians(float(deg))
        return out

    def waypoint_targets(self, poses: dict, upto: str) -> dict[str, torch.Tensor]:
        """Joint targets [rad] of every waypoint up to ``upto`` (inclusive), with inheritance."""
        targets, current = {}, self.home.clone()
        for name in sequence_of(poses):
            current = self.apply_pose(current, poses.get(name, {}))
            targets[name] = current.clone()
            if name == upto:
                break
        return targets

    # --- stepping ---------------------------------------------------------------------------------------
    def step(self, target: torch.Tensor) -> bool:
        """Command ``target`` [rad] for one policy step. Returns True when the env ended the episode."""
        self.target = target
        # Incremental actions: one step of action_scale (the tool sets this to 4 rad) reaches any in-limit pose.
        action = (target - self.u._joint_targets[0]) / self.u.cfg.action_scale
        action = action.clamp(-1.0, 1.0).unsqueeze(0).to(self.u.device)
        with torch.inference_mode():
            _, _, terminated, truncated, _ = self.env.step(action)
        self.max_lift = max(self.max_lift, float(self.u._lift_amt[0]))
        self.ever_success |= bool(self.u._success[0])
        if self.log is not None:
            self._record()
        return bool((terminated | truncated).any())

    def _record(self) -> None:
        """Append the state after this control step to the trajectory log."""
        u = self.u
        bag_p, bag_q = u.bag.data.root_pos_w.torch[0:1], u.bag.data.root_quat_w.torch[0:1]
        tip = quat_apply_inverse(bag_q, u._ee_pos_w()[0:1] - bag_p)[0]
        self.log.append(
            {
                "t": round(len(self.log) * u.step_dt, 4),
                "stage": self.stage,
                "target_deg": [round(math.degrees(v), 2) for v in self.target.tolist()],
                "joint_deg": [round(math.degrees(v), 2) for v in u.arm.data.joint_pos.torch[0].tolist()],
                "tip_bag": [round(v, 5) for v in tip.tolist()],
                "rigid_gap_mm": round(float(u._rigid_gap[0]) * 1000.0, 2),
                "moving_gap_mm": round(float(u._moving_gap[0]) * 1000.0, 2),
                "grasped": bool(u._grasped[0]),
                "grasped_raw": bool(u._grasp_raw[0]),
                "strap_pinched": [int(u._strap_count[0]), round(float(u._strap_depth[0]), 3)],
                "strap_resting": [int(u._rest_count[0]), round(float(u._rest_depth[0]), 3)],
                "success": bool(u._success[0]),
                "bag_lift_cm": round(100.0 * float(u._lift_amt[0]), 3),
            }
        )

    def move(self, start: torch.Tensor, goal: torch.Tensor, seconds: float | torch.Tensor) -> bool:
        """Smooth joint-space move. ``seconds`` is one duration, or one per joint (each joint finishes on its own
        schedule and then waits). Returns True if the episode ended on the way."""
        durations = torch.as_tensor(seconds, dtype=start.dtype, device=start.device).expand_as(start)
        steps = max(1, int(round(float(durations.max()) / self.u.step_dt)))
        for k in range(1, steps + 1):
            s = (k * self.u.step_dt / durations.clamp(min=1e-6)).clamp(0.0, 1.0)
            s = s * s * (3.0 - 2.0 * s)  # smoothstep per joint
            if self.step(start + s * (goal - start)):
                return True
        return False

    def hold(self, seconds: float) -> bool:
        return any(self.step(self.target) for _ in range(max(1, int(round(seconds / self.u.step_dt)))))

    # --- status -----------------------------------------------------------------------------------------
    def status(self, label: str) -> dict:
        u = self.u
        tip = u._ee_pos_w()[0]
        bag_pos = u.bag.data.root_pos_w.torch[0]
        rel = quat_apply_inverse(u.bag.data.root_quat_w.torch[0:1], (tip - bag_pos).unsqueeze(0))[0]
        info = {
            "lift_cm": 100.0 * float(u._lift_amt[0]),
            "grasped": bool(u._grasped[0]),
            "rest_count": int(u._rest_count[0]),
            "strap_count": int(u._strap_count[0]),
        }

        def jaw_text(name: str, on: bool, gap: float) -> str:
            if u._deformable_handle:  # strap contacts are invisible to PhysX: report the distance to the strap
                return f"{name} jaw {gap * 1000:5.1f} mm from strap {'TOUCH' if on else '     '}"
            return f"{name} jaw {'strap' if on else 'none/body'}"

        grip = math.degrees(float(u.arm.data.joint_pos.torch[0, u._grip_idx]))
        strap_text = ""
        if u._deformable_handle:  # strap between the jaws and how deep into the mouth (0 = housing end, 1 = tips)
            strap_text = (
                f"strap pinched {int(u._strap_count[0]):2d} nodes @ depth {float(u._strap_depth[0]):4.2f}, "
                f"on rigid jaw {int(u._rest_count[0]):2d} nodes @ depth {float(u._rest_depth[0]):4.2f} "
                f"(want pinched {u.cfg.mouth_mid_frac:.2f}+-{u.cfg.mouth_tol:.2f} OR resting <"
                f"{u.cfg.mouth_mid_frac + u.cfg.mouth_tol:.2f}: {'OK' if bool(u._strap_midway[0]) else 'no'})"
            )
        print(
            f"[grasp_pose] {label:13s} tip in bag frame (x,y,z)=({rel[0]:+.3f},{rel[1]:+.3f},{rel[2]:+.3f}) m"
            f" | to pinch target {float(u._dist[0]) * 100:5.1f} cm"
            f" | {jaw_text('rigid', bool(u._rigid_on[0]), float(u._rigid_gap[0]))}"
            f" | {jaw_text('moving', bool(u._moving_on[0]), float(u._moving_gap[0]))}"
            f" | {strap_text}"
            f" | gripper {grip:+5.1f} deg | grasped {info['grasped']} | bag lift {info['lift_cm']:.1f} cm"
            f" | final-pose err {math.degrees(float(u._pose_max[0])):5.1f} deg",
            flush=True,
        )
        # commanded vs actual joint angle; '!' = more than 3 deg short of its target (blocked or still moving)
        actual = u.arm.data.joint_pos.torch[0]
        cells = []
        for i, name in enumerate(self.names):
            want, got = math.degrees(float(self.target[i])), math.degrees(float(actual[i]))
            cells.append(f"{name[-1]}: {want:+6.1f}/{got:+6.1f}{'!' if abs(want - got) > 3.0 else ' '}")
        print("[grasp_pose] joint target/actual deg   " + " | ".join(cells), flush=True)
        return info


def _load_poses(path: str, rig: GraspRig) -> dict:
    if not os.path.exists(path):
        home_deg = {n: round(math.degrees(float(v)), 1) for n, v in zip(rig.names, rig.home)}
        template = {
            "_comment": "Joint angles in DEGREES. A waypoint lists only the joints it changes; the rest are "
            "inherited from the previous waypoint (the first inherits the home pose). "
            "'show' picks the waypoint held in live mode.",
            "_home": home_deg,
            "show": "grasp",
            **{name: dict(home_deg) for name in WAYPOINTS},
        }
        with open(path, "w") as f:
            json.dump(template, f, indent=2)
        print(f"[grasp_pose] wrote a template with the home pose to {path}. Edit it and save.", flush=True)
    with open(path) as f:
        return json.load(f)


def _record_goal(rig: GraspRig, poses: dict, args: argparse.Namespace) -> None:
    """Play the sequence up to the 'show' waypoint, then save the rigid jaw's pose in the bag frame as the goal.

    Playing the real sequence (instead of jumping straight to the waypoint) records the tip where the demo
    actually puts it, after the strap has been met on the way in.
    """
    u = rig.u
    shown = args.show or poses.get("show", "grasp")
    args.stop_after = shown
    _run_sequence(rig, poses, args)
    rig.hold(1.0)  # let the arm settle
    info = rig.status(f"record:{shown}")
    bag_p = u.bag.data.root_pos_w.torch[0:1]
    bag_q = u.bag.data.root_quat_w.torch[0:1]
    tip = quat_apply_inverse(bag_q, u._ee_pos_w()[0:1] - bag_p)[0]
    jaw_q = quat_mul(quat_conjugate(bag_q), u.arm.data.body_quat_w.torch[0:1, u._ee_idx])[0]
    moving_idx = u.arm.find_bodies(u.cfg.moving_jaw_body_name)[0][0]
    moving = quat_apply_inverse(bag_q, u.arm.data.body_pos_w.torch[0:1, moving_idx] - bag_p)[0]
    grip_deg = math.degrees(float(u.arm.data.joint_pos.torch[0, u._grip_idx]))
    goal = {
        "_comment": "Written by grasp_pose_tool.py --record. Everything is in the BAG frame (x, y, z) / (x, y, z, w).",
        "waypoint": shown,
        "tip_pos_bag": [round(float(v), 5) for v in tip],
        "jaw_quat_bag": [round(float(v), 5) for v in jaw_q],
        "moving_jaw_pos_bag": [round(float(v), 5) for v in moving],
        "gripper_deg": round(grip_deg, 1),
        "joints_deg": {n: round(math.degrees(float(v)), 1) for n, v in zip(rig.names, u.arm.data.joint_pos.torch[0])},
        "contact_at_record": {
            "grasped": info["grasped"],
            "rest_count": info["rest_count"],
            "strap_count": info["strap_count"],
        },
    }
    with open(args.goal, "w") as f:
        json.dump(goal, f, indent=2)
    print(f"[grasp_pose] RECORDED goal -> {args.goal}\n{json.dumps(goal, indent=2)}", flush=True)
    if not info["grasped"]:
        print(
            "[grasp_pose] note: grasped is False at this pose (expected for an open-jaw waypoint such as 'grasp'). "
            "Check the jaw-to-strap gaps above if this waypoint should already hold the strap.",
            flush=True,
        )


def _run_sequence(rig: GraspRig, poses: dict, args: argparse.Namespace) -> None:
    sequence = sequence_of(poses)
    if args.stop_after is not None:
        if args.stop_after not in sequence:
            raise SystemExit(f"[grasp_pose] --stop_after '{args.stop_after}' is not in the sequence {sequence}")
        sequence = sequence[: sequence.index(args.stop_after) + 1]
    targets = rig.waypoint_targets(poses, sequence[-1])
    if args.save_traj:
        rig.log = []
    speeds = poses.get("speed_deg_per_s", {})
    lift_goal = rig.u.cfg.lift_height
    best_lift, ended, prev = 0.0, False, rig.home
    for name in sequence:
        delta_deg = [math.degrees(float(d)) for d in (targets[name] - prev)]
        spec = speeds.get(name, {})
        if not isinstance(spec, dict):  # a plain number = the same speed for every joint
            spec = {"default": spec}
        fallback = args.close_s if name == "grasp_closed" else args.move_s
        per_joint = [float(spec.get(j, spec.get("default", 0.0))) for j in rig.names]
        secs = [abs(d) / v if v > 0.0 else 0.0 for d, v in zip(delta_deg, per_joint)]
        stage_s = max([MIN_MOVE_S, *secs]) if any(v > 0.0 for v in per_joint) else fallback
        # a joint with no speed finishes together with the move
        secs = [t if v > 0.0 else stage_s for t, v in zip(secs, per_joint)]
        cells = []
        for j, d, v, t in zip(rig.names, delta_deg, per_joint, secs):
            if abs(d) < 0.05:
                continue
            peak = 1.5 * abs(d) / max(t, 1e-6)
            cells.append(
                f"{j[-1]}: {d:+6.1f} deg in {t:4.1f} s (peak {peak:4.1f}{'!' if peak > VEL_LIMIT_DEG else ''})"
            )
        rig.stage = name
        first_row = len(rig.log) if rig.log is not None else 0
        print(f"[grasp_pose] -> {name} ({stage_s:.1f} s) " + " | ".join(cells), flush=True)
        if any("!" in c for c in cells):
            print("[grasp_pose]    '!' = above the 28.6 deg/s motor cap, that joint will lag the command.", flush=True)
        ended = rig.move(prev, targets[name], torch.tensor(secs)) or rig.hold(args.hold_s)
        info = rig.status(name)
        if rig.log is not None and len(rig.log) > first_row:
            rows = rig.log[first_row:]
            held = sum(r["grasped"] for r in rows) / len(rows)
            raw = sum(r["grasped_raw"] for r in rows) / len(rows)
            print(
                f"[grasp_pose]    {name}: grasped {held * 100:3.0f}% (raw test {raw * 100:3.0f}%) of {len(rows)} "
                f"steps | bag lift at the end {rows[-1]['bag_lift_cm']:.1f} cm | strap gap at the end: "
                f"rigid {rows[-1]['rigid_gap_mm']:.1f} mm, moving {rows[-1]['moving_gap_mm']:.1f} mm",
                flush=True,
            )
        best_lift = max(best_lift, rig.max_lift)
        prev = targets[name]
        if ended:
            break
    print("[grasp_pose] ----------------------------------------------------------------", flush=True)
    print(
        f"[grasp_pose] env SUCCESS (closed on the strap, bag {rig.u.cfg.lift_height * 100:.0f} cm up, joints "
        f"within {math.degrees(rig.u.cfg.final_pose_tol):.1f} deg of '{rig.u.cfg.final_pose_waypoint}'): "
        f"{rig.ever_success}",
        flush=True,
    )
    if "lift" not in sequence:
        print(
            f"[grasp_pose] stopped after '{sequence[-1]}': grasped at the end = {info['grasped']} "
            f"(lift not tested). Strap gaps and the grasped flag per step are in the trajectory file, if saved.",
            flush=True,
        )
    elif best_lift >= lift_goal:
        print(f"[grasp_pose] SUCCESS: bag lifted {best_lift * 100:.1f} cm while pinched.", flush=True)
    else:
        print(
            f"[grasp_pose] FAILED: best bag lift {best_lift * 100:.1f} cm (needs {lift_goal * 100:.0f} cm)"
            f"{'; episode ended early (bag knocked away?)' if ended else ''}. See the per-stage lines above:"
            " jaw-to-strap gaps and whether grasped was true.",
            flush=True,
        )
    if rig.log is not None:
        names = [n[-1] for n in rig.names]
        out = {
            "_comment": f"Written by grasp_pose_tool.py --save_traj. One row per control step "
            f"({1.0 / rig.u.step_dt:.0f} Hz). Angles in degrees, tip_bag in the BAG frame [m]. joint order: "
            + ", ".join(rig.names),
            "pose_file": args.poses,
            "stages": sequence,
            "step_dt": rig.u.step_dt,
            "joint_order": rig.names,
            "steps": rig.log,
        }
        with open(args.save_traj, "w") as f:
            json.dump(out, f)
        print(f"[grasp_pose] saved {len(rig.log)} steps ({names}) -> {args.save_traj}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    env_cfg, _ = resolve_task_config(args.task, "")
    apply_env_overrides(args, env_cfg)
    # deterministic scene: bag at its default pose, no start-pose jitter, no action / observation noise
    env_cfg.bag_xy_range = (0.0, 0.0)
    env_cfg.bag_yaw_range = (0.0, 0.0)
    env_cfg.bag_tilt_range = 0.0
    env_cfg.arm_joint_noise = 0.0
    env_cfg.action_noise_model = None
    env_cfg.observation_noise_model = None
    env_cfg.episode_length_s = 3600.0
    env_cfg.terminate_on_success = False  # keep going after the bag is lifted (e.g. on to return_home)
    # Tool only: drive the joints directly. Training uses residual-on-demo; that would fight hand-posed targets.
    env_cfg.use_demo_residual = False
    env_cfg.action_scale = 4.0
    if args.hold_s is None:
        args.hold_s = env_cfg.demo_hold_s

    with launch_simulation(env_cfg, args), contextlib.ExitStack() as cleanup:
        env = gym.make(args.task, cfg=env_cfg)
        cleanup.callback(lambda: close_env(env))
        env.reset()
        rig = GraspRig(env)
        print(f"[grasp_pose] joints (degrees, in this order): {rig.names}", flush=True)
        poses = _load_poses(args.poses, rig)
        sim = rig.u.sim

        if args.record:
            rig.step(rig.home)
            rig.hold(1.0)
            _record_goal(rig, poses, args)
            return 0

        if args.mode == "run":
            rig.step(rig.home)  # settle one step so the contact state is valid
            rig.hold(1.0)
            _run_sequence(rig, poses, args)
            if not sim.has_gui:
                return 0
            print("[grasp_pose] holding the last pose; close the window or press Ctrl+C to exit.", flush=True)
            while sim.is_running():
                rig.step(rig.target)
            return 0

        # live mode: hold the 'show' waypoint, reload the file when it changes
        mtime, shown, steps = -1.0, None, 0
        while sim.is_running():
            now = os.path.getmtime(args.poses)
            if now != mtime:
                mtime = now
                try:
                    with open(args.poses) as f:
                        poses = json.load(f)
                    shown = args.show or poses.get("show", "grasp")
                    rig.target = rig.waypoint_targets(poses, shown)[shown]
                    print(f"[grasp_pose] loaded '{shown}' from {args.poses}", flush=True)
                except (ValueError, KeyError) as exc:  # half-saved file or a typo: keep the old target
                    print(f"[grasp_pose] could not use the pose file ({exc}); keeping the previous pose.", flush=True)
            rig.step(rig.target)
            steps += 1
            if steps % 60 == 0:  # about once a second
                rig.status(str(shown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
