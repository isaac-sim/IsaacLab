# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Script to run an environment with a pick‑and‑lift **state machine that drives the
**relative IK** controller**.  The state machine logic is still implemented as a
Warp kernel, but at the very end we convert the *absolute* desired pose coming
from the kernel into a **delta position + delta rotation‑vector** that the
relative‑IK policy expects.

Launch, for example, with 32 environments:

.. code‑block:: bash

    ./isaaclab.sh \
        -p scripts/environments/state_machine/lift_cube_sm_rel_ik.py \
        --num_envs 32
"""

# ────────────────────────────────────────────────────────────────────────────────
# Basics: Omniverse app setup
# ────────────────────────────────────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Pick and lift state machine for lift environments (relative IK)."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")

# Let the AppLauncher inject its own CLI flags (e.g. --headless, --renderer‑gpu‑id …)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse Kit in the requested mode (headless / GUI)
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

# ────────────────────────────────────────────────────────────────────────────────
# Imports that require Kit to be running
# ────────────────────────────────────────────────────────────────────────────────
import gymnasium as gym
from collections.abc import Sequence

import torch
import warp as wp

from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData

import isaaclab_tasks  # noqa: F401 – registers tasks with gym
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Initialise Warp once at startup
wp.init()
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
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)


@wp.func
def distance_below_threshold(a: wp.vec3, b: wp.vec3, thresh: float) -> bool:
    return wp.length(a - b) < thresh


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    target_obj_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_cmd: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    pos_thresh: float,
):
    tid = wp.tid()
    state = sm_state[tid]

    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_cmd[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_cmd[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            pos_thresh,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_cmd[tid] = GripperState.OPEN
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            pos_thresh,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PickSmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0

    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_cmd[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0

    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = target_obj_pose[tid]
        gripper_cmd[tid] = GripperState.CLOSE
        if distance_below_threshold(
            wp.transform_get_translation(ee_pose[tid]),
            wp.transform_get_translation(des_ee_pose[tid]),
            pos_thresh,
        ):
            if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
                sm_state[tid] = PickSmState.REST
                sm_wait_time[tid] = 0.0

    sm_wait_time[tid] += dt[tid]
# ────────────────────────────────────────────────────────────────────────────────
# Utility: quaternion helpers (PyTorch‑side, for delta orientation)
# ────────────────────────────────────────────────────────────────────────────────

def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = torch.unbind(q1, -1)
    w2, x2, y2, z2 = torch.unbind(q2, -1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), -1)


def quat_to_rotvec(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    q = q / torch.norm(q, dim=-1, keepdim=True)
    qw = torch.clamp(q[..., 0], -1.0, 1.0)
    angle = 2.0 * torch.acos(qw)
    s = torch.sqrt(1.0 - qw * qw)
    axis = torch.where((s > eps).unsqueeze(-1), q[..., 1:] / s.unsqueeze(-1), torch.zeros_like(q[..., 1:]))
    return axis * angle.unsqueeze(-1)
# ────────────────────────────────────────────────────────────────────────────────
# State‑machine wrapper that *outputs RELATIVE actions*
# ────────────────────────────────────────────────────────────────────────────────
class PickAndLiftSm:
    """Wrap the Warp kernel and convert its *absolute* pose output to *relative* deltas.

    The final action tensor is `(N, 7)`
        ┌─────────┬──────────────┬─────────┐
        │ index   │ meaning      │ units   │
        ├─────────┼──────────────┼─────────┤
        │ 0‑2     │ Δx, Δy, Δz   │ metres  │
        │ 3‑5     │ ωx, ωy, ωz   │ rad     │
        │ 6       │ gripper      │ ∈[-1,1] │
        └─────────┴──────────────┴─────────┘
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", pos_threshold=0.01):
        self.dt = float(dt)
        self.n = num_envs
        self.device = device
        self.pos_threshold = pos_threshold

        self._dt = torch.full((self.n,), self.dt, device=device)
        self._state = torch.zeros((self.n,), dtype=torch.int32, device=device)
        self._wait = torch.zeros((self.n,), device=device)
        self._des_pose = torch.zeros((self.n, 7), device=device)
        self._grip = torch.zeros((self.n,), device=device)

        self._offset = torch.zeros((self.n, 7), device=device)
        self._offset[:, 2] = 0.10
        self._offset[:, 6] = 1.0  # quat (x,y,z,w)
        # ── Warp views ───────────────────────────────────────────────────────────
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    # ---------------------------------------------------------------------
    # Main compute step
    # ---------------------------------------------------------------------
    def compute(
        self,
        ee_pose: torch.Tensor,  # (N, 7) (pos + quat(w,x,y,z))
        object_pose: torch.Tensor,  # (N, 7)
        des_object_pose: torch.Tensor,  # (N, 7)
    ) -> torch.Tensor:
        """Run the state machine **and return RELATIVE actions**.
        
        Returns
        -------
        actions: tensor (N, 7) – Δposition (3), rotation‑vector (3), gripper (1).
        """
        # --- Re‑order (w,x,y,z) → (x,y,z,w) for Warp --------------------------
        ee_pose_warp = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose_warp = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose_warp = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # --- Convert to Warp arrays and run kernel ---------------------------
        ee_pose_wp = wp.from_torch(ee_pose_warp.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose_warp.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose_warp.contiguous(), wp.transform)

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

        # --- Warp → torch (and back to (w,x,y,z)) ----------------------------
        des_ee_pose_abs = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]  # (N,7) pos + quat(w,x,y,z)

        # --- Δposition -------------------------------------------------------
        delta_pos = des_ee_pose_abs[:, :3] - ee_pose[:, :3]

        # --- Δorientation as rot‑vector --------------------------------------
        q_cur = ee_pose[:, 3:]  # (wxyz)
        q_des = des_ee_pose_abs[:, 3:]
        q_diff = quat_mul(q_des, quat_conj(q_cur))  # (wxyz)
        delta_rotvec = quat_to_rotvec(q_diff)  # (3,)

        # --- Assemble action --------------------------------------------------
        actions = torch.cat(
            [delta_pos, delta_rotvec, self.des_gripper_state.unsqueeze(-1)], dim=-1
        )
        return actions(q_des, quat_conj(q_cur))  # still (wxyz)

        # --- Assemble action --------------------------------------------------
        actions = torch.cat(
            [delta_pos, delta_quat, self.des_gripper_state.unsqueeze(-1)], dim=-1
        )
        return actions

# ────────────────────────────────────────────────────────────────────────────────
# Main entry‑point
# ────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Parse environment configuration (RELATIVE IK flavour)
    # ------------------------------------------------------------------
    env_cfg: LiftEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-IK-Rel-v0",  # << changed from "…IK-Abs…"
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    env = gym.make("Isaac-Lift-Cube-Franka-IK-Rel-v0", cfg=env_cfg)
    env.reset()

    # ------------------------------------------------------------------
    # 2. Create buffers
    # ------------------------------------------------------------------
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    # no initial orientation / gripper command → zeros is neutral for Δ‑actions

    desired_orientation = torch.zeros((env.unwrapped.num_envs, 4), device=env.unwrapped.device)
    desired_orientation[:, 1] = 1.0  # (w,x,y,z) = (0,1,0,0) 180° about X

    pick_sm = PickAndLiftSm(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=env.unwrapped.num_envs,
        device=env.unwrapped.device,
        position_threshold=0.01,
    )

    # ------------------------------------------------------------------
    # 3. Main simulation loop
    # ------------------------------------------------------------------
    while simulation_app.is_running():
        with torch.inference_mode():
            dones = env.step(actions)[-2]

            # --- Query observations --------------------------------------
            ee_sensor = env.unwrapped.scene["ee_frame"]
            ee_pos = ee_sensor.data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
            ee_quat = ee_sensor.data.target_quat_w[..., 0, :]

            obj_data: RigidObjectData = env.unwrapped.scene["object"].data
            obj_pos = obj_data.root_pos_w - env.unwrapped.scene.env_origins

            desired_pos = env.unwrapped.command_manager.get_command("object_pose")[..., :3]

            # --- Run SM and get RELATIVE action --------------------------
            actions = pick_sm.compute(
                torch.cat([ee_pos, ee_quat], dim=-1),
                torch.cat([obj_pos, desired_orientation], dim=-1),
                torch.cat([desired_pos, desired_orientation], dim=-1),
            )

            # --- Reset per‑env state machine when env terminates ----------
            if dones.any():
                pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

    # ------------------------------------------------------------------
    # 4. Clean‑up
    # ------------------------------------------------------------------
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
