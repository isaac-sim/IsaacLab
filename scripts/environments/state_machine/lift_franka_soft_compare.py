# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend comparison harness for Isaac-Lift-Soft-Franka-v0 (Newton vs PhysX).

Runs the same Franka deformable-lift task and warp state machine on a selectable
physics backend, optionally records a video, and reports timing + lift height so
the two backends can be compared.

.. code-block:: bash

    # Newton (default backend), record video
    isaaclab.bat -p scripts/environments/state_machine/lift_franka_soft_compare.py --backend newton --video
    # PhysX, record video
    isaaclab.bat -p scripts/environments/state_machine/lift_franka_soft_compare.py --backend physx --video
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare Newton vs PhysX backends on the soft-lift task.")
parser.add_argument("--backend", type=str, default="newton", choices=["newton", "physx"], help="Physics backend.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--max_steps", type=int, default=400, help="Number of environment steps before exiting.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the rollout.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in env steps).")
parser.add_argument("--video_folder", type=str, default="", help="Directory to write the recorded video into.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# RecordVideo needs an rgb_array render mode, which in turn requires cameras to be enabled.
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# disable metrics assembler due to scene graph instancing
from isaacsim.core.experimental.utils.app import enable_extension

enable_extension("omni.usd.metrics.assembler.ui", enabled=False)

"""Rest everything else."""

import os
import time
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


class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    OPEN_GRIPPER = wp.constant(5)


class PickSmWaitTime:
    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(1.0)
    APPROACH_OBJECT = wp.constant(1.0)
    GRASP_OBJECT = wp.constant(1.0)
    LIFT_OBJECT = wp.constant(1.5)
    OPEN_GRIPPER = wp.constant(0.0)


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
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            position_threshold,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                sm_state[tid] = PickSmState.OPEN_GRIPPER
                sm_wait_time[tid] = 0.0
    elif state == PickSmState.OPEN_GRIPPER:
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.OPEN_GRIPPER:
            sm_state[tid] = PickSmState.OPEN_GRIPPER
            sm_wait_time[tid] = 0.0
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.03):
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

    def reset_idx(self, env_ids: Sequence[int] = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor):
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
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
                self.position_threshold,
            ],
            device=self.device,
        )
        return torch.cat([self.des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def _describe(label, obj):
    try:
        print(f"[COMPARE] {label}: {type(obj).__name__}", flush=True)
    except Exception:
        pass


def main():
    backend = args_cli.backend
    print(f"[COMPARE] ===== backend = {backend} =====", flush=True)

    # load the raw cfg (presets still intact) and resolve to the requested backend
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    selected = ["physx"] if backend == "physx" else []
    cfg = resolve_presets(cfg, selected=selected)
    cfg.sim.device = args_cli.device
    cfg.scene.num_envs = args_cli.num_envs
    cfg.viewer.eye = (2.1, 1.0, 1.3)

    # report the resolved backend-defining config bits
    _describe("sim.physics", cfg.sim.physics)
    print(f"[COMPARE] scene.replicate_physics = {getattr(cfg.scene, 'replicate_physics', 'n/a')}", flush=True)
    try:
        solver = getattr(cfg.sim.physics, "solver_cfg", None)
        if solver is not None:
            _describe("solver_cfg", solver)
    except Exception:
        pass

    # Newton records video through its own pyglet/OpenGL viewer. Its default headless
    # path uses EGL, which does not exist on Windows ("Library 'EGL' not found"). Force a
    # windowed WGL context so frame readback works on Windows (a viewer window will appear).
    if args_cli.backend == "newton" and args_cli.video:
        import newton.viewer as _nv

        _OrigViewerGL = _nv.ViewerGL

        def _WindowedViewerGL(*a, **k):
            import pyglet

            pyglet.options["headless"] = False
            k["headless"] = False
            return _OrigViewerGL(*a, **k)

        _nv.ViewerGL = _WindowedViewerGL
        print("[COMPARE] Patched Newton ViewerGL -> windowed (WGL) for video on Windows.", flush=True)

    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(TASK, cfg=cfg, render_mode=render_mode)

    if args_cli.video:
        folder = args_cli.video_folder or os.path.abspath(f"videos/compare/{backend}")
        os.makedirs(folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=folder,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            name_prefix=f"lift_soft_{backend}",
            disable_logger=True,
        )
        print(f"[COMPARE] recording video to {folder}", flush=True)

    print(
        f"[COMPARE] env created: num_envs={env.unwrapped.num_envs}, device={env.unwrapped.device}, "
        f"obs/action={env.unwrapped.action_space.shape}",
        flush=True,
    )

    env.reset()
    print("[COMPARE] env.reset() done.", flush=True)

    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 0] = 1.0
    object_grasp_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    object_grasp_orientation[:, 0] = 1.0
    object_local_grasp_position = torch.tensor([0.0, 0.0, 0.0], device=env.unwrapped.device)

    pick_sm = PickAndLiftSm(cfg.sim.dt * cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

    step_count = 0
    max_state_reached = 0
    peak_obj_z = -1.0e9
    base_obj_z = None
    warmup = 20
    t_start = None
    timed_steps = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            dones = env.step(actions)[-2]

            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = (
                ee_frame_sensor.data.target_pos_w.torch[..., 0, :].clone() - env.unwrapped.scene.env_origins
            )
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w.torch[..., 0, :].clone()

            object_data: DeformableObjectData = env.unwrapped.scene["deformable"].data
            object_position = object_data.root_pos_w.torch - env.unwrapped.scene.env_origins
            object_position += object_local_grasp_position

            desired_position = env.unwrapped.command_manager.get_command("deformable_pose")[..., :3]

            actions = pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([object_position, object_grasp_orientation], dim=-1),
                torch.cat([desired_position, desired_orientation], dim=-1),
            )

            cur_state = int(pick_sm.sm_state.max().item())
            max_state_reached = max(max_state_reached, cur_state)
            obj_z = float(object_data.root_pos_w.torch[0, 2].item())
            if base_obj_z is None:
                base_obj_z = obj_z
            peak_obj_z = max(peak_obj_z, obj_z)

            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

            step_count += 1
            if step_count == warmup:
                torch.cuda.synchronize()
                t_start = time.perf_counter()
            elif step_count > warmup:
                timed_steps += 1
            if step_count % 50 == 0:
                print(f"[COMPARE] step={step_count}/{args_cli.max_steps} sm_state={cur_state} obj_z={obj_z:.4f}", flush=True)
            if step_count >= args_cli.max_steps:
                break

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t_start) if t_start else 0.0
    fps = (timed_steps / elapsed) if elapsed > 0 else 0.0
    lift = (peak_obj_z - base_obj_z) if base_obj_z is not None else 0.0
    mode = "with rendering(video)" if args_cli.video else "physics-only(headless)"
    print(
        f"[COMPARE] RESULT backend={backend} steps={step_count} max_sm_state={max_state_reached} "
        f"base_z={base_obj_z:.4f} peak_z={peak_obj_z:.4f} lift={lift:+.4f}m "
        f"| {mode}: {fps:.1f} steps/s ({timed_steps} timed steps in {elapsed:.2f}s)",
        flush=True,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
