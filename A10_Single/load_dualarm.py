#!/usr/bin/env python3
# Policy-only entry (no frame / TCP data recording). For recording, use load_dualarm_record.py.
#
# ./isaaclab.sh -p A10_Single/load_dualarm.py --num_envs 1 --enable_cameras --enable_runtime_commands --policy_chunk_size 10 --policy_exec_horizon 7
# ./isaaclab.sh -p A10_Single/load_dualarm.py --num_envs 1 --enable_cameras --no_policy
# ./isaaclab.sh -p A10_Single/load_dualarm.py --num_envs 1 --enable_cameras --no_policy --enable_runtime_commands
# start / stop / reset
# OpenPI websocket (本机默认): --policy_host 127.0.0.1 --policy_port 8000
# 或用环境变量: OPENPI_POLICY_HOST / OPENPI_POLICY_PORT
# 排查 observation: 加 --debug_observation，DEBUG 会写入本次 run_*.log（量很大）

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path

from isaaclab.app import AppLauncher

_DEFAULT_POLICY_HOST = os.environ.get("OPENPI_POLICY_HOST", "127.0.0.1")
_DEFAULT_POLICY_PORT = int(os.environ.get("OPENPI_POLICY_PORT", "8000"))

parser = argparse.ArgumentParser(description="Run A10 single-arm pi policy evaluation scene (policy response only).")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--policy_host",
    type=str,
    default=_DEFAULT_POLICY_HOST,
    help="OpenPI server host (default: 127.0.0.1 or OPENPI_POLICY_HOST).",
)
parser.add_argument(
    "--policy_port",
    type=int,
    default=_DEFAULT_POLICY_PORT,
    help="OpenPI server websocket port (default: 8000 or OPENPI_POLICY_PORT).",
)
parser.add_argument("--prompt", type=str, default="Reach the yellow lemon.")
parser.add_argument("--no_policy", action="store_true")
parser.add_argument(
    "--enable_runtime_commands",
    action="store_true",
    help="Enable local runtime commands from terminal input: start / stop / reset.",
)
parser.add_argument(
    "--use_policy_gripper",
    action="store_true",
    help="Use 7th action dim from the model for both gripper joints (absolute); default fixes gripper to 0.",
)
parser.add_argument(
    "--policy_chunk_size",
    type=int,
    default=10,
    help="Pad/trim model actions to this many timesteps per inference (chunk shape is this x 7).",
)
parser.add_argument(
    "--policy_exec_horizon",
    type=int,
    default=7,
    help="How many leading rows of each chunk to execute as sim steps before the next infer. Capped by --policy_chunk_size.",
)
parser.add_argument(
    "--debug_observation",
    action="store_true",
    help="Write control.observation DEBUG lines (state7, image shape/mean) into the same run_*.log file.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

_hub_cache_dir = Path.home() / ".cache" / "ov" / "hub_isaaclab_a10_single"
_hub_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OMNICLIENT_HUB_CACHE_DIR", str(_hub_cache_dir))

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from a10_sim_loop import run as run_sim_loop
from scene.a10_scene_cfg import A10SceneCfg
from control.robot_reset import reset_fruits, reset_robot
from openpi_client import websocket_client_policy


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("a10_single.run")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)

    log_dir = Path("A10_Single/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.info("Debug log file: %s", log_path)
    return logger


def _configure_observation_logging(root_logger: logging.Logger, *, enabled: bool) -> None:
    """Route control.observation DEBUG to the run log file (root uses propagate=False, so this is explicit)."""
    obs = logging.getLogger("control.observation")
    if not enabled:
        obs.setLevel(logging.WARNING)
        return
    obs.setLevel(logging.DEBUG)
    if obs.handlers:
        return
    for h in root_logger.handlers:
        if isinstance(h, logging.FileHandler):
            obs.addHandler(h)
            break
    obs.propagate = False


LOGGER = _build_logger()
_configure_observation_logging(LOGGER, enabled=args.debug_observation)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 1.8, 1.8], [0.5, 0.0, 0.3])

    scene_cfg = A10SceneCfg(num_envs=args.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    reset_robot(scene)
    reset_fruits(scene)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    LOGGER.info("Setup complete.")

    client = None
    if not args.no_policy:
        client = websocket_client_policy.WebsocketClientPolicy(
            host=args.policy_host,
            port=args.policy_port,
        )
    else:
        LOGGER.info("--no_policy enabled: running static scene only.")

    run_sim_loop(
        simulation_app,
        sim,
        scene,
        client,
        LOGGER,
        prompt=args.prompt,
        no_policy=args.no_policy,
        frame_recording=False,
        tcp_recording=False,
        enable_runtime_commands=args.enable_runtime_commands,
        use_policy_gripper=args.use_policy_gripper,
        policy_chunk_size=args.policy_chunk_size,
        policy_exec_horizon=args.policy_exec_horizon,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
