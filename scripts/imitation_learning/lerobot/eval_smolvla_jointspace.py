"""
Evaluate a JOINT-SPACE-trained SmolVLA policy inside Isaac Sim.

Sibling of eval_smolvla.py, for checkpoints trained on joint-space actions/observations (both
LJ1.pos..LJ8.pos -- e.g. ethanCSL/openarm_visuomotor_no_domain_randomization_1000_joints, see
convert_hdf5_to_lerobot.py). Unlike eval_smolvla.py, this does NOT go through the env's IK-based
action term at all: the model's 8D output IS the target joint configuration directly, so this
bypasses the action manager entirely and drives the robot's joints directly every physics
substep, replicating everything else ManagerBasedRLEnv.step() normally does (scene.write_data_
to_sim(), sim.step(), sim.render() at the configured interval, scene.update() for fresh sensor/
camera data) -- see _step_direct() below, confirmed line-by-line against
source/isaaclab/isaaclab/envs/manager_based_rl_env.py's own step() (lines ~153-197), not guessed.
The two calls that ARE skipped are action_manager.process_action() and .apply_action() -- the IK
solve and its resulting set_joint_position_target() call, replaced by our own direct one.

Run smolvla_server.py FIRST (same server, unmodified -- it's checkpoint-agnostic, does plain
state-normalize / policy.select_action / action-unnormalize with no assumption about what the
action represents), then this:

  Terminal 1 (lerobot env):
    cd ~/CSL/lerobot
    conda run -n lerobot python \\
        ~/Stanley_ws/IsaacLab/scripts/imitation_learning/lerobot/smolvla_server.py \\
        --checkpoint ethanCSL/openarm_visuomotor_no_domain_randomization_1000_joints \\
        --task "Pick up the red cube."

  Terminal 2 (Isaac Sim):
    cd ~/Stanley_ws/IsaacLab
    ./isaaclab.sh -p scripts/imitation_learning/lerobot/eval_smolvla_jointspace.py \\
        --task Isaac-PickUp-RedCube-OpenArm-IK-Abs-v0 \\
        --num_rollouts 5 \\
        --horizon 300 \\
        --cameras front_cam,body_cam,wrist_cam

State (8D): openarm_left_joint1..7 + openarm_left_finger_joint1, extracted via LJ_IDX below --
the physics Articulation's joint order is INTERLEAVED dual-arm (even indices 0,2,4,...,12 = left
arm 1-7, then 14 = left gripper's actuated finger), confirmed against a live Isaac Sim
robot.data.joint_names printout and matching convert_hdf5_to_lerobot.py's own LJ_IDX exactly.
This deliberately does NOT reuse eval_smolvla.py's _get_state() (joint_pos[:7]) -- that slice
mixes left- and right-arm values under this interleaved ordering and would feed the model a
wrong, mismatched state even for the old EE-delta checkpoint's 7D state.

Action (8D): applied DIRECTLY as target joint positions for those same 8 joints -- no IK, no
delta interpretation, no gripper open/close threshold. openarm_left_finger_joint2 is a native
PhysX mimic of finger_joint1 in this USD (confirmed: no PhysicsDriveAPI on joint2), so only the
one actuated joint needs a target -- its mimic partner follows automatically. The right arm is
left undriven (holds wherever env.reset() put it), matching every real-hardware deploy/replay
script in this project's convention for this single-arm task.

--cameras: comma-separated env camera names in policy slot order, same contract as
eval_smolvla.py's own --cameras -- MUST match this exact checkpoint's --rename_map at train
time. Verify before trusting a rollout; a mismatch is silent (no error), it just degrades to
generic/wrong behavior.
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Evaluate a joint-space-action SmolVLA policy in Isaac Lab via smolvla_server.py"
)
parser.add_argument("--task",        type=str, required=True, help="Gym env id")
parser.add_argument("--num_rollouts",type=int, default=5,   help="Number of evaluation episodes")
parser.add_argument("--horizon",     type=int, default=300,  help="Max steps per episode")
parser.add_argument("--port",        type=int, default=5556, help="Policy server TCP port")
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument(
    "--cameras", type=str, default="front_cam,body_cam,wrist_cam",
    help=(
        "Comma-separated env camera names, IN POLICY SLOT ORDER (1st -> camera1, 2nd -> camera2, "
        "...). MUST match the --rename_map used for lerobot-train on this exact checkpoint. "
        "Sending the wrong env camera into a policy slot is silent (no error, no crash)."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything else after Isaac Sim is up."""

import pickle
import random
import socket
import struct

import carb
import carb.input
import omni.appwindow
import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.stack  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# ── TCP client helpers (identical to eval_smolvla.py, must match server) ───────

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
    return np.asarray(resp["action"], dtype=np.float32)

def policy_close(sock):
    _send(sock, pickle.dumps({"cmd": "close"}))
    _recv(sock)


# ── joint-space state/action ────────────────────────────────────────────────────

# Matches convert_hdf5_to_lerobot.py's LJ_IDX exactly: left arm 1-7 (interleaved dual-arm even
# indices) + left gripper's actuated finger joint.
LJ_IDX = list(range(0, 14, 2)) + [14]


def _get_state(env) -> np.ndarray:
    """Return the 8D joint-space state (openarm_left_joint1..7 + left gripper), matching the
    exact index order the training dataset used -- see module docstring for why this is NOT
    joint_pos[:7]."""
    joint_pos = env.scene["robot"].data.joint_pos[0]   # (num_joints,)
    return joint_pos[LJ_IDX].cpu().numpy().astype(np.float32)


def _get_cameras(obs_dict: dict, camera_names: list[str]) -> dict:
    """Extract camera images as uint8 HWC numpy arrays (identical to eval_smolvla.py)."""
    policy_obs = obs_dict["policy"]
    cameras = {}
    for i, env_key in enumerate(camera_names):
        policy_key = f"camera{i + 1}"
        if env_key in policy_obs:
            img = policy_obs[env_key]        # (1, H, W, 3) uint8
            cameras[policy_key] = img.squeeze(0).cpu().numpy().astype(np.uint8)
        else:
            print(f"[warn] Camera '{env_key}' (-> {policy_key}) not found in policy observations")
    return cameras


def _step_direct(env, target_left_q: np.ndarray) -> dict:
    """Advance the sim by one env "step" worth of physics (env.cfg.decimation substeps), driving
    the 8 LJ joints directly instead of going through the env's IK action term -- see module
    docstring for the exact ManagerBasedRLEnv.step() sequence this replicates. Returns a fresh
    observation dict in the same {"policy": {...}} shape env.step()/env.reset() return.
    """
    robot = env.scene["robot"]
    target_t = torch.as_tensor(target_left_q, dtype=torch.float32, device=env.device).unsqueeze(0)  # (1, 8)

    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        robot.set_joint_position_target(target_t, joint_ids=LJ_IDX)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if (env._sim_step_counter % env.cfg.sim.render_interval == 0) and is_rendering:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)

    return env.observation_manager.compute()


# ── keyboard reset (identical to eval_smolvla.py) ───────────────────────────────

class KeyboardHandler:
    """Listens for R = manual reset, Q = quit."""

    def __init__(self):
        self.reset_requested = False
        self.quit_requested  = False
        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()
        self._input    = carb.input.acquire_input_interface()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_key
        )
        print("[keyboard] Press  R  to reset episode early,  Q  to quit.")

    def _on_key(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.R:
                self.reset_requested = True
                print("[keyboard] Manual reset requested.")
            elif event.input == carb.input.KeyboardInput.Q:
                self.quit_requested = True
                print("[keyboard] Quit requested.")
        return True

    def consume_reset(self) -> bool:
        if self.reset_requested:
            self.reset_requested = False
            return True
        return False


# ── rollout ────────────────────────────────────────────────────────────────────

def rollout(env, sock, success_term, horizon: int, kb: "KeyboardHandler", camera_names: list[str]) -> bool:
    obs_dict, _ = env.reset()
    policy_reset(sock)

    for step in range(horizon):
        if kb.consume_reset():
            print(f"  [step {step+1:3d}] manually reset")
            return False

        state   = _get_state(env)
        cameras = _get_cameras(obs_dict, camera_names)

        if not cameras:
            print("[warn] No camera images found in obs — check image_obs_list config")
            break

        policy_action = policy_step(sock, state, cameras)   # (8,) absolute joint targets

        obs_dict = _step_direct(env, policy_action)

        if bool(success_term.func(env, **success_term.params)[0]):
            print(f"  [step {step+1:3d}] SUCCESS")
            return True

    print(f"  [step {step+1:3d}] failed")
    return False


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None

    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    print(f"\nEnv obs keys   : {list(env.observation_space.spaces.keys())}")
    print("Action mode    : DIRECT joint-space (bypassing the env's own IK action term)")

    print(f"\nConnecting to policy server on port {args_cli.port} …")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(("127.0.0.1", args_cli.port))
    print("Connected.\n")

    kb = KeyboardHandler()

    camera_names = [name.strip() for name in args_cli.cameras.split(",") if name.strip()]
    print(f"[INFO] Camera slots (in order -> camera1, camera2, ...): {camera_names}")

    results = []
    for trial in range(args_cli.num_rollouts):
        if kb.quit_requested:
            print("[keyboard] Quitting early.")
            break
        print(f"── Trial {trial + 1}/{args_cli.num_rollouts} ──────────────────")
        success = rollout(env, sock, success_term, args_cli.horizon, kb, camera_names)
        results.append(success)

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
