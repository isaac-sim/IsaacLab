"""Shared A10 single-arm simulation step loop (policy, lemon reach, optional recording)."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from control.controller import JointAbsExecutor, LemonReachController, RuntimeCommandHandler, RuntimeCommandListener, pi_control
from control.robot_reset import reset_fruits, reset_robot
from control.a10_tcp_server import A10TcpServer
from openpi_client import websocket_client_policy


def run(
    simulation_app,
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    client: websocket_client_policy.WebsocketClientPolicy | None,
    logger: logging.Logger,
    prompt: str,
    no_policy: bool,
    *,
    frame_recording: bool = False,
    record_dir: str = "A10_Single/recordings",
    record_interval: int = 1,
    tcp_recording: bool = False,
    tcp_port: int = 8000,
    use_policy_gripper: bool = False,
    policy_chunk_size: int = 10,
    policy_exec_horizon: int = 7,
    enable_runtime_commands: bool = False,
) -> tuple[Path | None, int]:
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    frame_idx = 0
    output_dir: Path | None = None
    tcp_server: A10TcpServer | None = None
    lemon_reach_ctrl: LemonReachController | None = None
    runtime_cmd_listener: RuntimeCommandListener | None = None
    runtime_cmd_handler: RuntimeCommandHandler | None = None

    chunk_size = max(1, int(policy_chunk_size))
    exec_horizon = max(1, min(int(policy_exec_horizon), chunk_size))
    joint_executor = JointAbsExecutor(
        max_abs_step=0.4,
        fixed_gripper_target=0.0,
        freeze_joint6=False,
        invert_joint4=False,
        use_policy_gripper=use_policy_gripper,
    )
    logger.info(
        "Policy execution: absolute joint targets + per-step rate limit. "
        "policy_chunk_size=%d policy_exec_horizon=%d (rows applied per infer). use_policy_gripper=%s.",
        chunk_size,
        exec_horizon,
        use_policy_gripper,
    )

    if frame_recording:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(record_dir) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Recording frames to: %s", output_dir)

    if tcp_recording:
        server = A10TcpServer()
        if server.start(tcp_port):
            tcp_server = server
            logger.info("TCP data recorder enabled at 0.0.0.0:%d", tcp_port)
        else:
            logger.error("Failed to start TCP data recorder at port %d; disabling tcp_recording.", tcp_port)
            tcp_server = None

    runtime_cmd_listener = RuntimeCommandListener(enable_runtime_commands, lambda: simulation_app.is_running())
    runtime_cmd_listener.start()
    if enable_runtime_commands:
        logger.info(
            "Runtime commands on: type start | stop | reset in the terminal that has the TTY "
            "(often the same shell you used to launch Isaac). stop pauses policy/lemon IK; "
            "start enables classical lemon-reach IK; reset reloads scene."
        )

    try:
        lemon_reach_ctrl = LemonReachController(scene, sim.device)
        if lemon_reach_ctrl.is_available:
            logger.info("Classical lemon reach controller initialized.")
        else:
            logger.warning("Classical lemon reach controller is unavailable.")
    except Exception as e:
        logger.exception("Failed to initialize classical lemon reach controller: %s", e)
        lemon_reach_ctrl = None
    runtime_cmd_handler = RuntimeCommandHandler(lemon_reach_ctrl, logger)

    def _save_frame(step_count: int) -> None:
        nonlocal frame_idx
        if not frame_recording or output_dir is None:
            return
        if record_interval > 1 and (step_count % record_interval) != 0:
            return
        if "wrist_camera" not in scene.keys():
            return
        import imageio.v2 as imageio

        try:
            rgb = scene["wrist_camera"].data.output["rgb"]
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.detach().cpu().numpy()
            rgb = np.asarray(rgb)
            if rgb.ndim == 4 and rgb.shape[0] == 1:
                rgb = rgb[0]
            if rgb.ndim != 3 or rgb.shape[-1] < 3 or rgb.size == 0:
                return
            frame = rgb[..., :3].astype(np.uint8, copy=False)
            imageio.imwrite(output_dir / f"frame_{frame_idx:06d}.png", frame)
            frame_idx += 1
        except Exception:
            logger.exception("Failed to save frame at step %d", step_count)

    def _get_external_target_q() -> np.ndarray | None:
        if tcp_server is None:
            return None
        q = tcp_server.get_target_q()
        if not q:
            return None
        return np.asarray(q, dtype=np.float32)

    def _publish_robot_state_to_tcp() -> None:
        if tcp_server is None:
            return
        try:
            q_now = scene["robot"].data.joint_pos[0]
            if isinstance(q_now, torch.Tensor):
                q_now = q_now.detach().cpu().numpy()
            tcp_server.send_set_joints(np.asarray(q_now).tolist())
        except Exception:
            logger.exception("Failed to publish robot state to TCP server.")

    def _reset_scene_and_counters() -> None:
        nonlocal count, sim_time
        if lemon_reach_ctrl is not None:
            lemon_reach_ctrl.reset()
        reset_robot(scene)
        reset_fruits(scene)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        _publish_robot_state_to_tcp()
        count = 0
        sim_time = 0.0

    logger.info("Running simulator ...")

    try:
        while simulation_app.is_running():
            try:
                if count > 0 and count % 800 == 0:
                    logger.info("Reset scene at step %d", count)
                    _reset_scene_and_counters()
                    continue

                if runtime_cmd_handler is not None and runtime_cmd_listener is not None:
                    if runtime_cmd_handler.process(runtime_cmd_listener.pop_all(), _reset_scene_and_counters):
                        continue

                if runtime_cmd_handler is not None and not runtime_cmd_handler.running_control:
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(sim_dt)
                    _publish_robot_state_to_tcp()
                    _save_frame(count)
                    count += 1
                    sim_time += sim_dt
                    continue

                if lemon_reach_ctrl is not None and lemon_reach_ctrl.active:
                    applied = lemon_reach_ctrl.apply_step()
                    if not applied:
                        logger.warning("Lemon reach mode active but controller step failed. Falling back to default control.")
                        lemon_reach_ctrl.stop()
                    else:
                        scene.write_data_to_sim()
                        sim.step()
                        scene.update(sim_dt)
                        _publish_robot_state_to_tcp()
                        _save_frame(count)
                        count += 1
                        sim_time += sim_dt
                        continue

                if no_policy:
                    ext_q = _get_external_target_q()
                    if ext_q is not None and ext_q.size > 0:
                        robot = scene["robot"]
                        cur_q = robot.data.joint_pos[0].clone()
                        copy_n = min(cur_q.shape[0], int(ext_q.shape[0]))
                        cur_q[:copy_n] = torch.as_tensor(ext_q[:copy_n], device=cur_q.device, dtype=cur_q.dtype)
                        robot.set_joint_position_target(cur_q.unsqueeze(0))

                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(sim_dt)
                    _publish_robot_state_to_tcp()
                    _save_frame(count)
                    count += 1
                    sim_time += sim_dt
                    continue

                if client is None:
                    raise RuntimeError("Policy client is required when --no_policy is not set.")

                try:
                    action_chunk = pi_control(
                        scene,
                        sim_time,
                        client,
                        prompt=prompt,
                        chunk_size=chunk_size,
                        zero_gripper_dim=not use_policy_gripper,
                    )
                except Exception as e:
                    logger.exception("Policy inference failed at step %d: %s", count, e)
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(sim_dt)
                    _publish_robot_state_to_tcp()
                    _save_frame(count)
                    count += 1
                    sim_time += sim_dt
                    continue

                if action_chunk.ndim != 2 or action_chunk.shape != (chunk_size, 7):
                    raise ValueError(f"Expected action chunk shape ({chunk_size}, 7), got {action_chunk.shape}")

                for i in range(min(exec_horizon, action_chunk.shape[0])):
                    ext_q = _get_external_target_q()
                    a = action_chunk[i]

                    use_external_target = ext_q is not None and ext_q.size >= 6
                    if use_external_target:
                        ext_target = ext_q if ext_q is not None else np.zeros(6, dtype=np.float32)
                        pred_q6 = torch.as_tensor(
                            ext_target[:6],
                            device=scene["robot"].data.joint_pos.device,
                            dtype=scene["robot"].data.joint_pos.dtype,
                        ).clone()
                    else:
                        pred_q6 = torch.as_tensor(
                            a[:6],
                            device=scene["robot"].data.joint_pos.device,
                            dtype=scene["robot"].data.joint_pos.dtype,
                        ).clone()

                    joint_executor.apply(
                        scene=scene,
                        pred_q6=pred_q6,
                        step_count=count,
                        logger=logger,
                        source="tcp" if use_external_target else "policy",
                        raw_action6=np.array(a[:6]),
                        pred_gripper=float(a[6]) if use_policy_gripper and not use_external_target else None,
                    )

                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(sim_dt)
                    _publish_robot_state_to_tcp()
                    _save_frame(count)
                    count += 1
                    sim_time += sim_dt

            except Exception as e:
                logger.exception("Fatal loop error at step %d: %s", count, e)
                try:
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(sim_dt)
                    _publish_robot_state_to_tcp()
                    _save_frame(count)
                    count += 1
                    sim_time += sim_dt
                except Exception:
                    logger.exception("Safe-step fallback also failed at step %d", count)
    finally:
        if tcp_server is not None:
            tcp_server.stop()
            logger.info("TCP data recorder stopped.")

    return output_dir, frame_idx


def stitch_frames_to_mp4(output_dir: Path, fps: int, logger: logging.Logger) -> Path | None:
    import imageio.v2 as imageio

    frame_paths = sorted(output_dir.glob("frame_*.png"))
    if len(frame_paths) == 0:
        logger.warning("No recorded frames found. Skip mp4 stitching.")
        return None

    video_path = output_dir / "recording.mp4"
    writer = imageio.get_writer(str(video_path), fps=fps)
    try:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    finally:
        writer.close()

    logger.info("Saved mp4 recording: %s", video_path)
    return video_path
