import logging
import queue
import sys
import threading
from collections.abc import Callable
from typing import TextIO

import numpy as np
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, quat_inv, quat_mul, subtract_frame_transforms

from .observation import get_observation
from openpi_client import websocket_client_policy


def simple_swing_control(scene, sim_time: float):
    """Simple debug swing for first 6 joints only."""
    action = scene["robot"].data.default_joint_pos.clone()
    action[:, 0:6] = 0.4 * np.sin(2 * np.pi * 0.5 * sim_time)
    if action.shape[-1] >= 7:
        action[:, 6] = 0.0

    scene["robot"].set_joint_position_target(action)


def _extract_action_chunk_7d(
    infer_result: dict,
    chunk_size: int = 10,
    *,
    zero_gripper_dim: bool = True,
) -> np.ndarray:
    if "actions" not in infer_result:
        raise KeyError(f"Model response missing 'actions' key. Got keys: {list(infer_result.keys())}")

    actions = np.asarray(infer_result["actions"], dtype=np.float32)

    # Normalize to (T, D) with explicit shape checks
    if actions.ndim == 1:
        actions = actions[None, :]
    elif actions.ndim == 3:
        if actions.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for inference, got shape {actions.shape}")
        actions = actions[0]
    elif actions.ndim != 2:
        raise ValueError(f"Unsupported action shape: {actions.shape}")

    # Strictly require 7D action for this robot
    if actions.shape[-1] != 7:
        raise ValueError(f"Expected action dim 7, got {actions.shape[-1]} with shape {actions.shape}")

    # Normalize chunk length
    if actions.shape[0] > chunk_size:
        actions = actions[:chunk_size]
    elif actions.shape[0] < chunk_size:
        pad = np.zeros((chunk_size - actions.shape[0], 7), dtype=np.float32)
        actions = np.concatenate([actions, pad], axis=0)

    # Ensure writable memory before in-place edits (server may return read-only buffers/views).
    actions = np.array(actions, dtype=np.float32, copy=True)

    if zero_gripper_dim:
        # Optional: ignore model gripper when not using policy gripper targets.
        actions[:, 6] = 0.0

    return actions.astype(np.float32, copy=False)


def pi_control(
    scene,
    sim_time,
    client: websocket_client_policy.WebsocketClientPolicy,
    prompt: str = "Reach the yellow lemon on the table.",
    *,
    chunk_size: int = 10,
    zero_gripper_dim: bool = True,
):
    # 1) Build observation (wrist + top camera + 7-DoF state)
    obs = get_observation(scene, prompt=prompt)

    # 2) Run policy inference and normalize to one 7D chunk (T=chunk_size)
    infer_result = client.infer(obs)
    action_chunk_7d = _extract_action_chunk_7d(
        infer_result, chunk_size=chunk_size, zero_gripper_dim=zero_gripper_dim
    )
    return action_chunk_7d


class JointAbsExecutor:
    """Apply policy outputs as *absolute* joint angles (rad) for joints 1–6.

    Each step: ``target ≈ current + clip(wrap(desired_abs - current), ±step_limit)`` so the
    commanded target moves toward the model's absolute pose without jumping. Joint 7 is
    gripper: either fixed open/close or taken from the model when ``use_policy_gripper`` is set.
    """

    def __init__(
        self,
        max_abs_step: float = 0.4,
        fixed_gripper_target: float = 0.0,
        freeze_joint6: bool = False,
        invert_joint4: bool = False,
        use_policy_gripper: bool = False,
        gripper_clip: float = 0.08,
    ):
        self.max_abs_step = max_abs_step
        self.fixed_gripper_target = fixed_gripper_target
        self.freeze_joint6 = freeze_joint6
        self.invert_joint4 = invert_joint4
        self.use_policy_gripper = use_policy_gripper
        self.gripper_clip = gripper_clip

    @staticmethod
    def _wrap_to_pi_torch(x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(x), torch.cos(x))

    def apply(
        self,
        scene,
        pred_q6: torch.Tensor,
        step_count: int | None = None,
        logger: logging.Logger | None = None,
        source: str = "policy",
        raw_action6: np.ndarray | None = None,
        pred_gripper: float | None = None,
    ) -> None:
        robot = scene["robot"]
        cur_q = robot.data.joint_pos[0].clone()
        prev_q = cur_q.clone()

        pred_q = pred_q6.to(device=cur_q.device, dtype=cur_q.dtype).clone()
        if self.invert_joint4:
            pred_q[3] = -pred_q[3]

        # Absolute-angle error on a circle for all revolute arm joints (avoids ±π wrap bugs).
        q_err = self._wrap_to_pi_torch(pred_q - cur_q[:6])

        step_limit = torch.tensor(
            [self.max_abs_step, self.max_abs_step, self.max_abs_step, 0.03, self.max_abs_step, 0.02],
            device=cur_q.device,
            dtype=cur_q.dtype,
        )
        q_step = torch.clamp(q_err, min=-step_limit, max=step_limit)
        target_q6 = cur_q[:6] + q_step

        if self.freeze_joint6:
            target_q6[5] = cur_q[5]

        cur_q[0:6] = target_q6

        if cur_q.shape[0] >= 8:
            if self.use_policy_gripper and pred_gripper is not None:
                g = float(pred_gripper)
                g = max(-self.gripper_clip, min(self.gripper_clip, g))
                cur_q[6] = g
                cur_q[7] = g
            else:
                cur_q[6] = self.fixed_gripper_target
                cur_q[7] = self.fixed_gripper_target

        if logger is not None and step_count is not None and step_count % 20 == 0:
            logger.debug(
                "step=%d source=%s raw_abs_action6=%s pred_gripper=%s prev_q=%s q_err=%s target_q=%s",
                step_count,
                source,
                np.array(raw_action6) if raw_action6 is not None else pred_q.detach().cpu().numpy(),
                pred_gripper,
                prev_q[:6].detach().cpu().numpy(),
                q_err.detach().cpu().numpy(),
                cur_q[:8].detach().cpu().numpy(),
            )

        robot.set_joint_position_target(cur_q.unsqueeze(0))


class LemonReachController:
    """Classical human-like lemon reaching controller based on Differential IK."""

    def __init__(
        self,
        scene,
        device: str,
        approach_height: float = 0.14,
        descend_height: float = 0.06,
        approach_threshold: float = 0.03,
        descend_threshold: float = 0.02,
        xy_jitter: float = 0.008,
        z_jitter: float = 0.005,
        angle_jitter_rad: float = 0.03,
        max_joint_step_rad: float = 0.06,
    ):
        self.scene = scene
        self.device = device

        self.approach_height = approach_height
        self.descend_height = descend_height
        self.approach_threshold = approach_threshold
        self.descend_threshold = descend_threshold
        self.xy_jitter = xy_jitter
        self.z_jitter = z_jitter
        self.angle_jitter_rad = angle_jitter_rad
        self.max_joint_step_rad = max_joint_step_rad

        self.active = False
        self.phase = "idle"  # idle -> approach -> descend -> hold
        self.target_approach_pos_w: torch.Tensor | None = None
        self.target_descend_pos_w: torch.Tensor | None = None
        self.target_quat_w: torch.Tensor | None = None

        self.controller: DifferentialIKController | None = None
        self.robot_entity: SceneEntityCfg | None = None
        self.ee_jacobi_idx: int | None = None
        self.joint_ids: list[int] = []

        self._initialize_controller()

    @property
    def is_available(self) -> bool:
        return self.controller is not None and self.robot_entity is not None and self.ee_jacobi_idx is not None

    def _initialize_controller(self) -> None:
        robot_entity = SceneEntityCfg(
            "robot",
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
            body_names=["link6"],
        )
        robot_entity.resolve(self.scene)
        self.robot_entity = robot_entity

        robot = self.scene["robot"]
        if robot.is_fixed_base:
            self.ee_jacobi_idx = robot_entity.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = robot_entity.body_ids[0]
        self.joint_ids = list(robot_entity.joint_ids)

        # Position-only IK: avoids position/orientation fighting that causes jitter.
        ik_cfg = DifferentialIKControllerCfg(
            command_type="position",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.12},
        )
        self.controller = DifferentialIKController(ik_cfg, num_envs=self.scene.num_envs, device=self.device)

    def start(self) -> bool:
        if not self.is_available or self.controller is None:
            return False
        self.controller.reset()
        self.active = True
        self.phase = "approach"
        self.target_approach_pos_w = None
        self.target_descend_pos_w = None
        self.target_quat_w = None
        return True

    def stop(self) -> None:
        self.active = False
        self.phase = "idle"
        # Freeze at current pose so stale IK targets do not keep pulling the arm.
        if self.scene is not None and "robot" in self.scene.keys():
            robot = self.scene["robot"]
            q = robot.data.joint_pos.clone()
            robot.set_joint_position_target(q)

    def reset(self) -> None:
        self.stop()
        self.target_approach_pos_w = None
        self.target_descend_pos_w = None
        self.target_quat_w = None

    def _get_lemon_pos_w(self) -> torch.Tensor | None:
        if "lemon" not in self.scene.keys():
            return None
        lemon = self.scene["lemon"]
        if hasattr(lemon, "get_world_poses"):
            lemon_pos_w, _ = lemon.get_world_poses()
            return lemon_pos_w
        if hasattr(lemon, "data") and hasattr(lemon.data, "root_state_w"):
            return lemon.data.root_state_w[:, :3]
        return None

    def _sample_targets(
        self, lemon_pos_w: torch.Tensor, ee_pose_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_envs = lemon_pos_w.shape[0]
        device = lemon_pos_w.device
        dtype = lemon_pos_w.dtype

        xy_noise = (torch.rand((num_envs, 2), device=device, dtype=dtype) * 2.0 - 1.0) * self.xy_jitter
        z_noise = (torch.rand((num_envs, 1), device=device, dtype=dtype) * 2.0 - 1.0) * self.z_jitter

        approach_pos = lemon_pos_w.clone()
        approach_pos[:, :2] += xy_noise
        approach_pos[:, 2:3] += self.approach_height + z_noise

        descend_pos = lemon_pos_w.clone()
        descend_pos[:, :2] += xy_noise
        descend_pos[:, 2:3] += self.descend_height + z_noise

        # Unused for position-only IK; kept for API compatibility if we switch back to pose mode.
        roll = (torch.rand((num_envs,), device=device, dtype=dtype) * 2.0 - 1.0) * self.angle_jitter_rad
        pitch = (torch.rand((num_envs,), device=device, dtype=dtype) * 2.0 - 1.0) * self.angle_jitter_rad
        yaw = (torch.rand((num_envs,), device=device, dtype=dtype) * 2.0 - 1.0) * self.angle_jitter_rad
        delta_quat = quat_from_euler_xyz(roll, pitch, yaw)
        target_quat = quat_mul(delta_quat, ee_pose_w[:, 3:7])
        return approach_pos, descend_pos, target_quat

    def apply_step(self) -> bool:
        if not self.active or not self.is_available:
            return False
        if self.controller is None or self.robot_entity is None or self.ee_jacobi_idx is None or len(self.joint_ids) == 0:
            return False
        if "robot" not in self.scene.keys():
            return False

        lemon_pos_w = self._get_lemon_pos_w()
        if lemon_pos_w is None:
            return False

        robot = self.scene["robot"]
        ee_body_id = self.robot_entity.body_ids[0]

        jacobian = robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.joint_ids].clone()
        ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
        root_pose_w = robot.data.root_pose_w
        joint_pos = robot.data.joint_pos[:, self.joint_ids]

        # PhysX Jacobian is expressed in the world frame; IK uses EE pose in the robot *base* frame.
        # Rotate Jacobian rows into the base frame (same as test_differential_ik.py).
        base_rot = root_pose_w[:, 3:7]
        base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        if self.target_approach_pos_w is None or self.target_descend_pos_w is None or self.target_quat_w is None:
            (
                self.target_approach_pos_w,
                self.target_descend_pos_w,
                self.target_quat_w,
            ) = self._sample_targets(lemon_pos_w, ee_pose_w)

        if self.phase == "approach":
            target_pos_w = self.target_approach_pos_w
            dist = torch.norm(ee_pose_w[:, 0:3] - target_pos_w, dim=-1).mean()
            if float(dist.item()) < self.approach_threshold:
                self.phase = "descend"
        elif self.phase in {"descend", "hold"}:
            target_pos_w = self.target_descend_pos_w
            dist = torch.norm(ee_pose_w[:, 0:3] - target_pos_w, dim=-1).mean()
            if self.phase == "descend" and float(dist.item()) < self.descend_threshold:
                self.phase = "hold"
        else:
            target_pos_w = self.target_approach_pos_w
            self.phase = "approach"

        target_pos_b, _ = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_pos_w, self.target_quat_w
        )

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # Position-only: pass desired EE orientation as current (keeps wrist stable in null space sense).
        self.controller.set_command(target_pos_b, ee_quat=ee_quat_b)

        joint_pos_des = self.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        delta = joint_pos_des - joint_pos
        lim = float(self.max_joint_step_rad)
        delta = torch.clamp(delta, min=-lim, max=lim)
        joint_pos_des = joint_pos + delta

        robot.set_joint_position_target(joint_pos_des, joint_ids=self.joint_ids)

        # Keep gripper fixed: only overwrite gripper joint targets. A full-vector
        # set_joint_position_target(cur_q) would wipe the arm targets set above.
        n = joint_pos_des.shape[0]
        if robot.data.joint_pos.shape[-1] >= 8:
            grip = torch.zeros((n, 2), device=joint_pos_des.device, dtype=joint_pos_des.dtype)
            robot.set_joint_position_target(grip, joint_ids=[6, 7])
        return True


def _runtime_command_input_stream() -> TextIO | None:
    """Line input for runtime commands.

    Isaac/Cursor often run Python with stdin redirected (non-TTY). ``input()`` then hits EOF
    immediately and the listener thread exits. Opening ``/dev/tty`` uses the controlling terminal
    instead so commands still work when launched from an IDE.
    """
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            return sys.stdin
    except (AttributeError, ValueError):
        pass
    try:
        return open("/dev/tty", "r", encoding="utf-8", errors="replace")
    except OSError:
        return None


class RuntimeCommandListener:
    """Local terminal command listener: start / stop / reset."""

    def __init__(self, enabled: bool, is_app_running: Callable[[], bool]):
        self.enabled = enabled
        self._is_app_running = is_app_running
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return

        def _reader() -> None:
            stream = _runtime_command_input_stream()
            if stream is None:
                print(
                    "[runtime-cmd] No TTY: stdin is not interactive and /dev/tty is unavailable. "
                    "start|stop|reset will not work; run from a real terminal (e.g. bash ssh/console)."
                )
                return
            print("[runtime-cmd] Type command and press Enter: start | stop | reset")
            while self._is_app_running():
                try:
                    raw = stream.readline()
                except Exception:
                    break
                if raw == "":
                    break
                cmd = raw.strip().lower()
                if cmd in {"start", "stop", "reset"}:
                    self._queue.put(cmd)
                elif cmd:
                    print(f"[runtime-cmd] Unknown command: {cmd}")

        self._thread = threading.Thread(target=_reader, daemon=True)
        self._thread.start()

    def pop_all(self) -> list[str]:
        if not self.enabled:
            return []
        cmds: list[str] = []
        while True:
            try:
                cmds.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return cmds


class RuntimeCommandHandler:
    """Runtime command state machine for start/stop/reset."""

    def __init__(self, lemon_reach_ctrl: LemonReachController | None, logger: logging.Logger):
        self.running_control = True
        self.lemon_reach_ctrl = lemon_reach_ctrl
        self.logger = logger

    def process(self, commands: list[str], on_reset: Callable[[], None]) -> bool:
        handled_reset = False
        for cmd in commands:
            if cmd == "reset":
                self.logger.info("Received runtime command: reset")
                on_reset()
                if self.lemon_reach_ctrl is not None:
                    self.lemon_reach_ctrl.reset()
                handled_reset = True
            elif cmd == "start":
                self.logger.info("Received runtime command: start")
                self.running_control = True
                if self.lemon_reach_ctrl is None or not self.lemon_reach_ctrl.start():
                    self.logger.warning("Received start, but classical lemon controller is unavailable.")
                else:
                    self.logger.info("Lemon reach mode enabled (human-like approach, classical IK).")
            elif cmd == "stop":
                self.logger.info("Received runtime command: stop")
                self.running_control = False
                if self.lemon_reach_ctrl is not None:
                    self.lemon_reach_ctrl.stop()
        return handled_reset

