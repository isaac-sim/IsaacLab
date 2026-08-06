# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose + tune the scripted pick-and-lift on the PhysX deformable backend.

The bundled state machine is tuned for Newton and stalls in the approach phase on
PhysX. This script runs the same state machine on a chosen backend but exposes the
knobs that matter for grasping a PhysX FEM softbody, and logs per-step diagnostics
(state, end-effector vs object distance) so the failure mode is visible.

Tunable knobs:
  --position_threshold   distance [m] to advance approach -> grasp (kernel default 0.03)
  --grasp_z_offset       vertical offset [m] added to the grasp target (negative = press down into object)
  --wait_mult            multiplier on all state wait times (give the soft, low-PD arm time to settle)
  --grip_close           gripper close command magnitude (default -1.0; more negative = squeeze harder)

.. code-block:: bash

    isaaclab.bat -p scripts/environments/state_machine/lift_franka_soft_physx_tune.py --backend physx --viz newton
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose/tune scripted pick-lift on PhysX deformable backend.")
parser.add_argument("--backend", type=str, default="physx", choices=["newton", "physx"], help="Physics backend.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--max_steps", type=int, default=400, help="Steps before exiting.")
parser.add_argument("--position_threshold", type=float, default=0.03, help="Approach->grasp distance threshold [m].")
parser.add_argument("--grasp_z_offset", type=float, default=0.0, help="Vertical offset on the grasp target [m].")
parser.add_argument("--wait_mult", type=float, default=1.0, help="Multiplier on all state wait times.")
parser.add_argument("--grip_close", type=float, default=-1.0, help="Gripper close command (more negative=harder).")
parser.add_argument("--episode_s", type=float, default=0.0, help="Override episode length [s] (0 = keep default 5s).")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the rollout.")
parser.add_argument("--video_length", type=int, default=400, help="Video length in steps.")
parser.add_argument("--video_folder", type=str, default="", help="Override output folder for the video.")
parser.add_argument("--video_tag", type=str, default="", help="Suffix added to the video filename (for batch runs).")
parser.add_argument("--seed", type=int, default=-1, help="Environment seed (>=0 to set; varies the target per video).")
parser.add_argument("--gravity", type=float, default=0.0, help="Enable gravity: magnitude [m/s^2] applied as -z (e.g. 9.81). 0 = keep env default (off).")
parser.add_argument("--arm_stiffness", type=float, default=0.0, help="Override Franka arm joint stiffness (0 = keep default ~80).")
parser.add_argument("--arm_damping", type=float, default=0.0, help="Override Franka arm joint damping (0 = keep default ~4).")
parser.add_argument("--grip_effort", type=float, default=0.0, help="Cap finger joint effort [N] = grasp force (0 = keep default 500). With high stiffness the closing force saturates at this value.")
parser.add_argument("--grip_stiffness", type=float, default=0.0, help="Override finger joint stiffness (0 = keep default 1000). Raise it so high grasp forces are actually delivered.")
parser.add_argument("--no_lift", action="store_true", default=False, help="Static-squeeze mode: grasp the object and hold in place (never lift) to measure compression vs force cleanly.")
parser.add_argument("--log_every", type=int, default=15, help="Print a per-step diagnostic line every N steps (set 1 to inspect jitter).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaacsim.core.experimental.utils.app import enable_extension

enable_extension("omni.usd.metrics.assembler.ui", enabled=False)

import os
from collections.abc import Sequence

import gymnasium as gym
import torch
import warp as wp

from isaaclab.assets.deformable_object.deformable_object_data import DeformableObjectData

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

wp.init()

TASK = "Isaac-Lift-Soft-Franka-v0"

# state ids
REST, APPROACH_ABOVE, APPROACH, GRASP, LIFT, OPEN = 0, 1, 2, 3, 4, 5
STATE_NAMES = ["REST", "APPROACH_ABOVE", "APPROACH", "GRASP", "LIFT", "OPEN"]


@wp.func
def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
    return wp.length(current_pos - desired_pos) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    wait_thresh: wp.array(dtype=float),  # per-state wait threshold [s], indexed by state
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    grasp_offset: wp.array(dtype=wp.transform),
    position_threshold: float,
    grip_open: float,
    grip_close: float,
):
    tid = wp.tid()
    state = sm_state[tid]
    if state == 0:  # REST
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = grip_open
        if sm_wait_time[tid] >= wait_thresh[0]:
            sm_state[tid] = 1
            sm_wait_time[tid] = 0.0
    elif state == 1:  # APPROACH_ABOVE
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = grip_open
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= wait_thresh[1]:
                sm_state[tid] = 2
                sm_wait_time[tid] = 0.0
    elif state == 2:  # APPROACH (grasp target = object + grasp_offset)
        des_ee_pose[tid] = wp.transform_multiply(grasp_offset[tid], object_pose[tid])
        gripper_state[tid] = grip_open
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= wait_thresh[2]:
                sm_state[tid] = 3
                sm_wait_time[tid] = 0.0
    elif state == 3:  # GRASP
        des_ee_pose[tid] = wp.transform_multiply(grasp_offset[tid], object_pose[tid])
        gripper_state[tid] = grip_close
        if sm_wait_time[tid] >= wait_thresh[3]:
            sm_state[tid] = 4
            sm_wait_time[tid] = 0.0
    elif state == 4:  # LIFT
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = grip_close
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= wait_thresh[4]:
                sm_state[tid] = 5
                sm_wait_time[tid] = 0.0
    elif state == 5:  # OPEN/hold
        gripper_state[tid] = grip_close
        if sm_wait_time[tid] >= wait_thresh[5]:
            sm_state[tid] = 5
            sm_wait_time[tid] = 0.0
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    def __init__(self, dt, num_envs, device, position_threshold, grasp_z_offset, wait_mult, grip_close):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        self.grip_open = 1.0
        self.grip_close = grip_close

        self.sm_dt = torch.full((num_envs,), self.dt, device=device)
        self.sm_state = torch.zeros((num_envs,), dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros((num_envs,), device=device)

        base_wait = torch.tensor([0.2, 1.0, 1.0, 1.0, 1.5, 0.0], device=device) * wait_mult
        self.wait_thresh = base_wait

        self.des_ee_pose = torch.zeros((num_envs, 7), device=device)
        self.des_gripper_state = torch.zeros((num_envs,), device=device)

        # approach-above offset (0.1 m up)
        self.offset = torch.zeros((num_envs, 7), device=device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0
        # grasp offset (apply grasp_z_offset on z; identity quat)
        self.grasp_offset = torch.zeros((num_envs, 7), device=device)
        self.grasp_offset[:, 2] = grasp_z_offset
        self.grasp_offset[:, -1] = 1.0

        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.wait_thresh_wp = wp.from_torch(self.wait_thresh, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)
        self.grasp_offset_wp = wp.from_torch(self.grasp_offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose, object_pose, des_object_pose):
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                self.wait_thresh_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.grasp_offset_wp,
                self.position_threshold,
                self.grip_open,
                self.grip_close,
            ],
            device=self.device,
        )
        return torch.cat([self.des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def main():
    print(
        f"[TUNE] backend={args_cli.backend} thr={args_cli.position_threshold} "
        f"grasp_z={args_cli.grasp_z_offset} wait_mult={args_cli.wait_mult} grip_close={args_cli.grip_close}",
        flush=True,
    )
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg = resolve_presets(cfg, selected=(["physx"] if args_cli.backend == "physx" else []))
    cfg.sim.device = args_cli.device
    cfg.scene.num_envs = args_cli.num_envs
    cfg.viewer.eye = (2.1, 1.0, 1.3)
    if args_cli.episode_s > 0.0:
        cfg.episode_length_s = args_cli.episode_s
        print(f"[TUNE] episode_length_s set to {args_cli.episode_s}", flush=True)
    if args_cli.seed >= 0:
        cfg.seed = args_cli.seed
        print(f"[TUNE] seed set to {args_cli.seed}", flush=True)
    if args_cli.gravity > 0.0:
        cfg.sim.gravity = (0.0, 0.0, -args_cli.gravity)
        print(f"[TUNE] gravity ENABLED: {cfg.sim.gravity} (note: Newton ignores per-body disable_gravity)", flush=True)
    if args_cli.arm_stiffness > 0.0 or args_cli.arm_damping > 0.0:
        for name, act in cfg.scene.robot.actuators.items():
            if name == "panda_hand":
                continue
            if args_cli.arm_stiffness > 0.0:
                act.stiffness = args_cli.arm_stiffness
            if args_cli.arm_damping > 0.0:
                act.damping = args_cli.arm_damping
            print(f"[TUNE] arm actuator '{name}': stiffness={act.stiffness} damping={act.damping}", flush=True)
    if args_cli.grip_stiffness > 0.0:
        cfg.scene.robot.actuators["panda_hand"].stiffness = args_cli.grip_stiffness
    if args_cli.grip_effort > 0.0:
        hand = cfg.scene.robot.actuators["panda_hand"]
        hand.effort_limit_sim = args_cli.grip_effort
        print(f"[TUNE] gripper effort capped at {args_cli.grip_effort} N (stiffness={hand.stiffness}) -> grasp force ~= {args_cli.grip_effort} N", flush=True)

    if args_cli.backend == "newton" and args_cli.video:
        import newton.viewer as _nv

        _Orig = _nv.ViewerGL

        def _Windowed(*a, **k):
            import pyglet

            pyglet.options["headless"] = False
            k["headless"] = False
            return _Orig(*a, **k)

        _nv.ViewerGL = _Windowed

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(TASK, cfg=cfg, render_mode=render_mode)

    if args_cli.video:
        folder = os.path.abspath(args_cli.video_folder or f"videos/tune/{args_cli.backend}")
        os.makedirs(folder, exist_ok=True)
        tag = f"_{args_cli.video_tag}" if args_cli.video_tag else ""
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=folder,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            name_prefix=f"lift_{args_cli.backend}{tag}",
            disable_logger=True,
        )
        print(f"[TUNE] recording to {folder}", flush=True)

    env.reset()

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 0] = 1.0
    object_grasp_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    object_grasp_orientation[:, 0] = 1.0

    pick_sm = PickAndLiftSm(
        cfg.sim.dt * cfg.decimation,
        env.unwrapped.num_envs,
        env.unwrapped.device,
        args_cli.position_threshold,
        args_cli.grasp_z_offset,
        args_cli.wait_mult,
        args_cli.grip_close,
    )

    robot = env.unwrapped.scene["robot"]
    finger_ids, _ = robot.find_joints("panda_finger.*")

    step = 0
    peak_z = -1e9
    base_z = None
    max_state = 0
    min_dist_in_approach = 1e9  # closest the EE got to the grasp target while in APPROACH
    hold_start = max(0, args_cli.max_steps - 120)  # measurement window at the end of the run
    hold_gaps: list[float] = []
    hold_zs: list[float] = []
    while simulation_app.is_running():
        with torch.inference_mode():
            dones = env.step(actions)[-2]

            ee = env.unwrapped.scene["ee_frame"]
            tcp_pos = ee.data.target_pos_w.torch[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_quat = ee.data.target_quat_w.torch[..., 0, :].clone()

            obj: DeformableObjectData = env.unwrapped.scene["deformable"].data
            obj_pos = obj.root_pos_w.torch - env.unwrapped.scene.env_origins

            des_pos = env.unwrapped.command_manager.get_command("deformable_pose")[..., :3]

            actions = pick_sm.compute(
                torch.cat([tcp_pos, tcp_quat], dim=-1),
                torch.cat([obj_pos, object_grasp_orientation], dim=-1),
                torch.cat([des_pos, desired_orientation], dim=-1),
            )
            # static-squeeze mode: never advance past GRASP, so the gripper holds the
            # object in place and only the squeeze force acts (clean force-vs-width).
            if args_cli.no_lift:
                pick_sm.sm_state.clamp_(max=GRASP)

            st = int(pick_sm.sm_state[0].item())
            max_state = max(max_state, st)
            ez = float(tcp_pos[0, 2].item())
            oz = float(obj_pos[0, 2].item())
            dist = float(torch.norm(tcp_pos[0] - obj_pos[0]).item())
            if st == APPROACH:
                min_dist_in_approach = min(min_dist_in_approach, dist)
            if base_z is None:
                base_z = oz
            peak_z = max(peak_z, oz)

            # gripper opening (sum of both prismatic finger joints) + hold-window sampling
            finger_gap = float(robot.data.joint_pos[0, finger_ids].sum().item())
            if step >= hold_start:
                hold_gaps.append(finger_gap)
                hold_zs.append(oz)

            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

            step += 1
            if step % args_cli.log_every == 0:
                print(
                    f"[TUNE] step={step:3d} state={STATE_NAMES[st]:<14} "
                    f"ee=({tcp_pos[0,0]:.4f},{tcp_pos[0,1]:.4f},{ez:.4f}) "
                    f"obj=({obj_pos[0,0]:.4f},{obj_pos[0,1]:.4f},{oz:.4f}) "
                    f"finger_gap={finger_gap:.5f} dist_ee_obj={dist:.4f}",
                    flush=True,
                )
            if step >= args_cli.max_steps:
                break

    lift = (peak_z - base_z) if base_z is not None else 0.0
    grasped = max_state >= GRASP
    lifted = lift > 0.10
    mean_gap = (sum(hold_gaps) / len(hold_gaps)) if hold_gaps else float("nan")
    final_z = (sum(hold_zs[-20:]) / len(hold_zs[-20:])) if hold_zs else peak_z
    held = bool(final_z > 0.15)  # still lifted at end of hold window (vs slipped back toward table ~0.05)
    print(
        f"[TUNE] RESULT backend={args_cli.backend} thr={args_cli.position_threshold} grasp_z={args_cli.grasp_z_offset} "
        f"wait_mult={args_cli.wait_mult} grip_close={args_cli.grip_close} | max_state={STATE_NAMES[max_state]} "
        f"min_approach_dist={min_dist_in_approach:.4f} base_z={base_z:.4f} peak_z={peak_z:.4f} lift={lift:+.4f}m "
        f"grasped={grasped} lifted={lifted}",
        flush=True,
    )
    print(
        f"[TUNE] GRIP grip_effort={args_cli.grip_effort} gravity={args_cli.gravity} "
        f"hold_gap={mean_gap:.5f} final_z={final_z:.4f} held={held}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
