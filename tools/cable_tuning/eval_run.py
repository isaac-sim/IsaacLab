# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single headless evaluation of the FrankaCable env under the lift state machine.

Mirrors :mod:`scripts.environments.state_machine.lift_franka_soft` but:
- locked to ``Isaac-Lift-Cable-Franka-v0`` and one env,
- applies caller-supplied dotted-path overrides to the env cfg,
- captures per-step metrics to ``<out>/metrics.parquet`` and
  aggregated scalars to ``<out>/summary.json``,
- exits cleanly on NaN/exception with ``nan_flag=1`` in the summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single cable-env tuning evaluation.")
    parser.add_argument("--overrides", type=str, default=None, help="Path to overrides JSON.")
    parser.add_argument("--out", type=str, required=True, help="Output run directory.")
    parser.add_argument("--max-steps", type=int, default=600, help="Max env steps (~dt*N seconds).")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs (kept at 1 for tuning).")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _build_pick_and_lift_sm(wp, torch):
    """Construct a PickAndLiftSm class inline.

    Inlined (rather than imported from
    ``scripts.environments.state_machine.lift_franka_soft``) because that
    module parses CLI args at import time, conflicting with this script's own
    argparse. Logic and timings are kept identical.
    """

    class GripperState:
        OPEN = wp.constant(1.0)
        CLOSE = wp.constant(-1.0)

    class PickSmState:
        REST = wp.constant(0)
        APPROACH_ABOVE_OBJECT = wp.constant(1)
        APPROACH_OBJECT = wp.constant(2)
        GRASP_OBJECT = wp.constant(3)
        LIFT_OBJECT = wp.constant(4)

    class PickSmWaitTime:
        REST = wp.constant(0.2)
        APPROACH_ABOVE_OBJECT = wp.constant(1.0)
        APPROACH_OBJECT = wp.constant(1.0)
        GRASP_OBJECT = wp.constant(1.0)
        LIFT_OBJECT = wp.constant(1.5)

    @wp.func
    def distance_below_threshold(current_pos: wp.vec3, desired_pos: wp.vec3, threshold: float) -> bool:
        return wp.length(current_pos - desired_pos) < threshold

    @wp.kernel
    def infer_state_machine(
        dt: wp.array(dtype=float),
        sm_state: wp.array(dtype=int),
        sm_wait_time: wp.array(dtype=float),
        ee_pose: wp.array(dtype=wp.transform),
        object_pose: wp.array(dtype=wp.transform),
        des_object_pose: wp.array(dtype=wp.transform),
        des_ee_pose: wp.array(dtype=wp.transform),
        gripper_state: wp.array(dtype=float),
        offset: wp.array(dtype=wp.transform),
        position_threshold: float,
    ):
        tid = wp.tid()
        state = sm_state[tid]
        if state == PickSmState.REST:
            des_ee_pose[tid] = ee_pose[tid]
            gripper_state[tid] = GripperState.OPEN
            if sm_wait_time[tid] >= PickSmWaitTime.REST:
                sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
                sm_wait_time[tid] = 0.0
        elif state == PickSmState.APPROACH_ABOVE_OBJECT:
            des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
            gripper_state[tid] = GripperState.OPEN
            if distance_below_threshold(
                wp.transform_get_translation(ee_pose[tid]),
                wp.transform_get_translation(des_ee_pose[tid]),
                position_threshold,
            ):
                if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                    sm_state[tid] = PickSmState.APPROACH_OBJECT
                    sm_wait_time[tid] = 0.0
        elif state == PickSmState.APPROACH_OBJECT:
            des_ee_pose[tid] = object_pose[tid]
            gripper_state[tid] = GripperState.OPEN
            if distance_below_threshold(
                wp.transform_get_translation(ee_pose[tid]),
                wp.transform_get_translation(des_ee_pose[tid]),
                position_threshold,
            ):
                if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                    sm_state[tid] = PickSmState.GRASP_OBJECT
                    sm_wait_time[tid] = 0.0
        elif state == PickSmState.GRASP_OBJECT:
            des_ee_pose[tid] = object_pose[tid]
            gripper_state[tid] = GripperState.CLOSE
            if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
                sm_state[tid] = PickSmState.LIFT_OBJECT
                sm_wait_time[tid] = 0.0
        elif state == PickSmState.LIFT_OBJECT:
            des_ee_pose[tid] = des_object_pose[tid]
            gripper_state[tid] = GripperState.CLOSE
        sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]

    class PickAndLiftSm:
        def __init__(self, dt, num_envs, device, position_threshold=0.03):
            self.dt = float(dt)
            self.num_envs = num_envs
            self.device = device
            self.position_threshold = position_threshold
            self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
            self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
            self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)
            self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
            self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)
            self.offset = torch.zeros((self.num_envs, 7), device=self.device)
            self.offset[:, 2] = 0.1
            self.offset[:, -1] = 1.0
            self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
            self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
            self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
            self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
            self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
            self.offset_wp = wp.from_torch(self.offset, wp.transform)

        def compute(self, ee_pose, object_pose, des_object_pose):
            ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
            object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
            des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)
            wp.launch(
                kernel=infer_state_machine,
                dim=self.num_envs,
                inputs=[
                    self.sm_dt_wp, self.sm_state_wp, self.sm_wait_time_wp,
                    ee_pose_wp, object_pose_wp, des_object_pose_wp,
                    self.des_ee_pose_wp, self.des_gripper_state_wp,
                    self.offset_wp, self.position_threshold,
                ],
                device=self.device,
            )
            return torch.cat([self.des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)

    return PickAndLiftSm


_FAILURE_SUMMARY = {
    "nan_flag": 1,
    "max_state_reached": 0,
    "exploded_flag": 0,
    "settle_time_s": 1e6,
    "mean_goal_pos_error_lift": 1e6,
    "cable_oscillation_rms": 1e6,
    "steps_executed": 0,
    "error": "not_started",
}


def main() -> int:
    args = _parse_args()
    os.makedirs(args.out, exist_ok=True)

    summary = dict(_FAILURE_SUMMARY)
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f)

    simulation_app = None
    try:
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app

        from isaacsim.core.experimental.utils.app import enable_extension
        enable_extension("omni.usd.metrics.assembler.ui", enabled=False)

        import gymnasium as gym
        import pandas as pd
        import torch
        import warp as wp

        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.manager_based.manipulation.lift_franka_soft.franka_cable_env_cfg import FrankaCableEnvCfg
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        wp.init()

        PickAndLiftSm = _build_pick_and_lift_sm(wp, torch)

        env_cfg: FrankaCableEnvCfg = parse_env_cfg(
            "Isaac-Lift-Cable-Franka-v0",
            device=args.device,
            num_envs=args.num_envs,
        )

        if args.overrides:
            from tools.cable_tuning.overrides import apply_overrides
            with open(args.overrides) as f:
                overrides = json.load(f)
            apply_overrides(env_cfg, overrides)

        env = gym.make("Isaac-Lift-Cable-Franka-v0", cfg=env_cfg)
        env.reset()

        actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        actions[:, 3] = 1.0

        desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
        desired_orientation[:, 0] = 1.0
        object_grasp_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
        object_grasp_orientation[:, 0] = 1.0
        object_local_grasp_position = torch.tensor([0.0, 0.0, -0.01], device=env.unwrapped.device)

        pick_sm = PickAndLiftSm(env_cfg.sim.dt * env_cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

        metrics_rows: list[dict] = []
        nan_flag = 0
        exploded_flag = 0
        max_state = 0
        steps = 0
        summary["error"] = ""

        with torch.inference_mode():
            while steps < args.max_steps and simulation_app.is_running():
                try:
                    env.step(actions)
                except Exception as exc:  # noqa: BLE001
                    nan_flag = 1
                    summary["error"] = f"env.step exception: {exc!r}"
                    break

                ee_frame = env.unwrapped.scene["ee_frame"]
                tcp_pos = ee_frame.data.target_pos_w.torch[..., 0, :].clone() - env.unwrapped.scene.env_origins
                tcp_quat = ee_frame.data.target_quat_w.torch[..., 0, :].clone()
                cable_data = env.unwrapped.scene["cable"].data
                cable_com = cable_data.body_com_pos_w.torch[:, 3] - env.unwrapped.scene.env_origins
                object_pos = cable_com + object_local_grasp_position
                desired_pos = env.unwrapped.command_manager.get_command("object_pose")[..., :3]
                robot = env.unwrapped.scene["robot"]
                joint_pos = robot.data.joint_pos.torch
                joint_vel = robot.data.joint_vel.torch

                row_finite = bool(
                    torch.isfinite(tcp_pos).all()
                    and torch.isfinite(cable_com).all()
                    and torch.isfinite(joint_pos).all()
                    and torch.isfinite(joint_vel).all()
                )
                if not row_finite:
                    nan_flag = 1
                    summary["error"] = "non-finite tensor detected"
                    break

                if (
                    joint_pos.abs().max().item() > 50.0
                    or joint_vel.abs().max().item() > 100.0
                    or cable_com.abs().max().item() > 5.0
                ):
                    exploded_flag = 1

                actions = pick_sm.compute(
                    torch.cat([tcp_pos, tcp_quat], dim=-1),
                    torch.cat([object_pos, object_grasp_orientation], dim=-1),
                    torch.cat([desired_pos, desired_orientation], dim=-1),
                )

                state_val = int(pick_sm.sm_state[0].item())
                if state_val > max_state:
                    max_state = state_val

                metrics_rows.append({
                    "step": steps,
                    "state": state_val,
                    "ee_x": float(tcp_pos[0, 0]),
                    "ee_y": float(tcp_pos[0, 1]),
                    "ee_z": float(tcp_pos[0, 2]),
                    "goal_x": float(desired_pos[0, 0]),
                    "goal_y": float(desired_pos[0, 1]),
                    "goal_z": float(desired_pos[0, 2]),
                    "cable_com_x": float(cable_com[0, 0]),
                    "cable_com_y": float(cable_com[0, 1]),
                    "cable_com_z": float(cable_com[0, 2]),
                    "joint_pos_absmax": float(joint_pos.abs().max()),
                    "joint_vel_absmax": float(joint_vel.abs().max()),
                })
                steps += 1

        env.close()

        df = pd.DataFrame(metrics_rows)
        metrics_path = os.path.join(args.out, "metrics.parquet")
        if not df.empty:
            df.to_parquet(metrics_path, index=False)

        dt = env_cfg.sim.dt * env_cfg.decimation
        lift_state = 4  # PickSmState.LIFT_OBJECT
        full_state = lift_state + 1
        window = max(1, int(0.3 / dt))
        held_lift = (
            not df.empty
            and int(df.iloc[-1]["state"]) == lift_state
            and (df["state"] == lift_state).sum() >= window
        )
        max_state_reached = full_state if held_lift else max_state

        settle_time_s = float(args.max_steps * dt)
        mean_goal_pos_error_lift = 1e6
        cable_oscillation_rms = 1e6

        if not df.empty:
            lift_df = df[df["state"] == lift_state].reset_index(drop=True)
            if len(lift_df) > 0:
                ee = lift_df[["ee_x", "ee_y", "ee_z"]].to_numpy()
                goal = lift_df[["goal_x", "goal_y", "goal_z"]].to_numpy()
                err = ((ee - goal) ** 2).sum(axis=1) ** 0.5
                mean_goal_pos_error_lift = float(err.mean())

                com = lift_df[["cable_com_x", "cable_com_y", "cable_com_z"]].to_numpy()
                if len(com) >= 3:
                    vel = (com[1:] - com[:-1]) / dt
                    acc = (vel[1:] - vel[:-1]) / dt
                    cable_oscillation_rms = float(((acc**2).sum(axis=1).mean()) ** 0.5)
                else:
                    cable_oscillation_rms = 0.0

                if len(com) >= 2:
                    vel = (com[1:] - com[:-1]) / dt
                    speed = (vel ** 2).sum(axis=1) ** 0.5
                    err_aligned = err[1:]
                    n = min(len(speed), len(err_aligned))
                    if n >= window:
                        settled_run = 0
                        for i in range(n):
                            ok = speed[i] < 0.05 and err_aligned[i] < 0.05
                            settled_run = settled_run + 1 if ok else 0
                            if settled_run >= window:
                                settle_time_s = float((i - window + 1) * dt)
                                break

        summary.update({
            "nan_flag": int(nan_flag),
            "max_state_reached": int(max_state_reached),
            "exploded_flag": int(exploded_flag),
            "settle_time_s": float(settle_time_s),
            "mean_goal_pos_error_lift": float(mean_goal_pos_error_lift),
            "cable_oscillation_rms": float(cable_oscillation_rms),
            "steps_executed": int(steps),
            "error": summary.get("error", ""),
        })
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return 0

    except Exception:  # noqa: BLE001
        summary["error"] = traceback.format_exc()
        summary["nan_flag"] = 1
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return 1

    finally:
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    sys.exit(main())
