"""
Evaluate a trained SmolVLA policy inside Isaac Sim.

This script connects to smolvla_server.py (running in the lerobot conda env)
over a local TCP socket, steps the Isaac Lab environment, and reports success rate.

Run smolvla_server.py FIRST, then run this script:

  Terminal 1 (lerobot env):
    cd ~/CSL/lerobot
    conda run -n lerobot python \\
        ~/Stanley_ws/IsaacLab/scripts/imitation_learning/lerobot/smolvla_server.py \\
        --checkpoint outputs/train/openarm_visuomotor/checkpoints/001000/pretrained_model \\
        --task "Pick up the red cube."

  Terminal 2 (Isaac Sim):
    cd ~/Stanley_ws/IsaacLab
    ./isaaclab.sh -p scripts/imitation_learning/lerobot/eval_smolvla.py \\
        --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \\
        --num_rollouts 5 \\
        --horizon 300

Action mapping (policy 8D → env 14D):
  env[ 0: 6] = policy[0:6]   left arm IK delta (dx,dy,dz,drx,dry,drz)
  env[ 6   ] = policy[6  ]   left gripper (+1=open / -1=close)
  env[ 7:13] = 0             right arm stays at reset pose
  env[13   ] = 1.0           right gripper stays open
  (If the policy outputs extra dims beyond index 7, they are discarded.)

State mapping (env → policy 6D):
  Uses left-arm joint positions [0:6] from the robot articulation.
  If your convert_hdf5_to_lerobot.py stored a different state representation,
  adjust _get_state() below accordingly.
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate a SmolVLA policy in Isaac Lab via smolvla_server.py"
)
parser.add_argument("--task",        type=str, required=True, help="Gym env id")
parser.add_argument("--num_rollouts",type=int, default=5,   help="Number of evaluation episodes")
parser.add_argument("--horizon",     type=int, default=300,  help="Max steps per episode")
parser.add_argument("--port",        type=int, default=5556, help="Policy server TCP port")
parser.add_argument("--seed",        type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything else after Isaac Sim is up."""

import pickle
import random
import socket
import struct

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.stack  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# ── TCP client helpers (must match server) ─────────────────────────────────────

def _recv_exactly(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Policy server closed connection")
        buf.extend(chunk)
    return bytes(buf)

def _send(sock, payload: bytes):
    sock.sendall(struct.pack(">I", len(payload)) + payload)

def _recv(sock) -> dict:
    raw_len = _recv_exactly(sock, 4)
    (n,) = struct.unpack(">I", raw_len)
    return pickle.loads(_recv_exactly(sock, n))

def policy_reset(sock):
    _send(sock, pickle.dumps({"cmd": "reset"}))
    _recv(sock)

def policy_step(sock, state: np.ndarray, cameras: dict) -> np.ndarray:
    _send(sock, pickle.dumps({"cmd": "step", "state": state, "cameras": cameras}))
    resp = _recv(sock)
    if "error" in resp:
        raise RuntimeError(f"Policy server error: {resp['error']}")
    # Server sends action as a plain Python list to avoid numpy version mismatch
    # (lerobot env uses numpy 2.x, isaaclab env uses numpy 1.x).
    return np.asarray(resp["action"], dtype=np.float32)

def policy_close(sock):
    _send(sock, pickle.dumps({"cmd": "close"}))
    _recv(sock)


# ── observation extraction ─────────────────────────────────────────────────────

def _get_state(env, _cache=[]) -> np.ndarray:
    """Extract 6D state: left arm joint positions (first 6 of openarm_left_joint[1-7]).

    The training dataset stored states/articulation/robot/joint_position, which
    is the full joint state of the robot articulation in Isaac Sim joint order.
    We look up the indices of openarm_left_joint1..6 once and cache them.
    """
    if not _cache:
        robot = env.scene["robot"]
        # Find joint indices for left arm joints 1-6
        ids, _ = robot.find_joints(["openarm_left_joint[1-6]"])
        _cache.append(ids)

    joint_pos = env.scene["robot"].data.joint_pos[0]   # (num_joints,)
    left_ids = _cache[0]
    return joint_pos[left_ids].cpu().numpy().astype(np.float32)


def _get_cameras(obs_dict: dict) -> dict:
    """Extract camera images as uint8 HWC numpy arrays.

    Training rename_map:
      front_cam   → camera1
      body_cam    → camera2
      wrist_cam   → camera3
    """
    policy_obs = obs_dict["policy"]
    cameras = {}
    mapping = {
        "front_cam":  "camera1",
        "body_cam":   "camera2",
        "wrist_cam":  "camera3",
    }
    for env_key, policy_key in mapping.items():
        if env_key in policy_obs:
            img = policy_obs[env_key]        # (1, H, W, 3) uint8
            cameras[policy_key] = img.squeeze(0).cpu().numpy().astype(np.uint8)
    return cameras


# ── action mapping ─────────────────────────────────────────────────────────────

def _map_action_to_env(policy_action: np.ndarray, env_action_dim: int = 14) -> torch.Tensor:
    """Map policy action (any dim) → 14D environment action.

    Mapping for left-arm pick-up (policy outputs 8D):
      env[ 0: 6] = policy[0:6]   left arm IK delta (dx,dy,dz,drx,dry,drz)
      env[ 6   ] = policy[6  ]   left gripper (+1=open / -1=close)
      env[ 7:13] = 0             right arm stays at reset pose
      env[13   ] = 1.0           right gripper stays open

    If policy action is shorter than 7D (unexpected), we fill what we can.
    """
    env_action = np.zeros(env_action_dim, dtype=np.float32)
    # Fill left arm + gripper (at most indices 0..6)
    n = min(7, len(policy_action))
    env_action[:n] = policy_action[:n]
    env_action[env_action_dim - 1] = 1.0   # right gripper open
    return torch.from_numpy(env_action).unsqueeze(0)  # (1, env_action_dim)


# ── rollout ────────────────────────────────────────────────────────────────────

def rollout(env, sock, success_term, horizon: int) -> bool:
    obs_dict, _ = env.reset()
    policy_reset(sock)

    for step in range(horizon):
        state   = _get_state(env)
        cameras = _get_cameras(obs_dict)

        if not cameras:
            print("[warn] No camera images found in obs — check image_obs_list config")
            break

        policy_action = policy_step(sock, state, cameras)         # (8,)
        env_action    = _map_action_to_env(policy_action).to(env.device)   # (1,14)

        obs_dict, _, terminated, truncated, _ = env.step(env_action)

        # Check success
        if bool(success_term.func(env, **success_term.params)[0]):
            print(f"  [step {step+1:3d}] SUCCESS")
            return True

        if terminated or truncated:
            break

    print(f"  [step {step+1:3d}] failed")
    return False


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    # ── environment ──────────────────────────────────────────────────────────
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None

    # Extract success term before disabling terminations
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    print(f"\nEnv action dim : {env.action_space.shape}")
    print(f"Env obs keys   : {list(env.observation_space.spaces.keys())}")

    # ── connect to policy server ──────────────────────────────────────────────
    print(f"\nConnecting to policy server on port {args_cli.port} …")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", args_cli.port))
    print("Connected.\n")

    # ── rollouts ──────────────────────────────────────────────────────────────
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"── Trial {trial + 1}/{args_cli.num_rollouts} ──────────────────")
        success = rollout(env, sock, success_term, args_cli.horizon)
        results.append(success)

    # ── summary ───────────────────────────────────────────────────────────────
    n_ok = results.count(True)
    print(f"\n{'='*50}")
    print(f"Success: {n_ok} / {len(results)}")
    print(f"Rate   : {n_ok / len(results):.1%}")
    print(f"Results: {results}")

    policy_close(sock)
    sock.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
