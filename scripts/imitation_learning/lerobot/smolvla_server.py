#!/usr/bin/env python3
"""
SmolVLA policy server — runs in the 'lerobot' conda env.

Loads a trained SmolVLA checkpoint and exposes it over a TCP socket so that
the Isaac Sim eval script (running in a different Python env) can query it.

Usage (in a separate terminal FIRST):
  cd ~/CSL/lerobot
  python ~/Stanley_ws/IsaacLab/scripts/imitation_learning/lerobot/smolvla_server.py \\
      --checkpoint outputs/train/openarm_visuomotor/checkpoints/last/pretrained_model \\
      --task "Pick up the red cube." \\
      --port 5556

Protocol (TCP, framed by 4-byte big-endian length prefix + pickle payload):
  Client → Server:
    {"cmd": "reset"}                   → clear action queue
    {"cmd": "step",
     "state": ndarray(state_dim,),
     "cameras": {"camera1": uint8 HWC, "camera2": uint8 HWC, "camera3": uint8 HWC}}
    {"cmd": "close"}                   → shut down server
  Server → Client:
    {"ok": True}                       → after reset / close
    {"action": ndarray(action_dim,)}   → after step
"""

import argparse
import glob
import os
import pickle
import socket
import struct
import sys

import numpy as np
import torch

LEROBOT_SRC = "/home/csl/CSL/lerobot/src"
if LEROBOT_SRC not in sys.path:
    sys.path.insert(0, LEROBOT_SRC)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


# ── TCP framing helpers ────────────────────────────────────────────────────────

def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return b""
        buf.extend(chunk)
    return bytes(buf)

def send_msg(sock: socket.socket, data: bytes):
    sock.sendall(struct.pack(">I", len(data)) + data)

def recv_msg(sock: socket.socket) -> bytes:
    raw_len = _recv_exactly(sock, 4)
    if not raw_len:
        return b""
    (n,) = struct.unpack(">I", raw_len)
    return _recv_exactly(sock, n)


# ── normalization helpers ──────────────────────────────────────────────────────

_NORM_EPS = 1e-8

def _resolve_norm_file(checkpoint: str, pattern: str) -> str:
    """Return a local path to a safetensors file matching *pattern*, downloading from HF if needed."""
    if os.path.isdir(checkpoint):
        matches = sorted(glob.glob(os.path.join(checkpoint, pattern)))
        if not matches:
            raise FileNotFoundError(f"No file matching '{pattern}' found in {checkpoint}")
        return matches[0]
    else:
        # HuggingFace repo ID — list repo files and download the matching one.
        from huggingface_hub import hf_hub_download, list_repo_files
        import fnmatch
        repo_files = list(list_repo_files(checkpoint))
        matches = [f for f in repo_files if fnmatch.fnmatch(f, pattern)]
        if not matches:
            raise FileNotFoundError(f"No file matching '{pattern}' found in HF repo '{checkpoint}'")
        return hf_hub_download(repo_id=checkpoint, filename=matches[0])


def _load_norm_stats(checkpoint: str, device: str):
    """Load mean/std for observation.state and action from the checkpoint safetensors."""
    import safetensors.torch as st

    pre_path  = _resolve_norm_file(checkpoint, "*preprocessor*normalizer*.safetensors")
    post_path = _resolve_norm_file(checkpoint, "*postprocessor*unnormalizer*.safetensors")

    pre  = st.load_file(pre_path)
    post = st.load_file(post_path)

    def _t(stats, key):
        return torch.tensor(stats[key], dtype=torch.float32).to(device)

    state_mean = _t(pre,  "observation.state.mean")
    state_std  = torch.clamp(_t(pre,  "observation.state.std"), min=_NORM_EPS)
    action_mean = _t(post, "action.mean")
    action_std  = torch.clamp(_t(post, "action.std"), min=_NORM_EPS)

    print(f"[server] Loaded norm stats  state_dim={state_mean.shape[0]}  action_dim={action_mean.shape[0]}")
    return state_mean, state_std, action_mean, action_std


# ── policy wrapper ─────────────────────────────────────────────────────────────

class PolicyServer:
    def __init__(self, checkpoint: str, task: str, device: str):
        print(f"[server] Loading  {checkpoint}")
        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint)
        self.policy.to(device)
        self.policy.eval()

        # Load MEAN_STD normalization statistics from the checkpoint.
        # The policy was trained on normalized inputs and outputs normalized actions.
        # We must normalize state IN and unnormalize action OUT to match training.
        self.state_mean, self.state_std, self.action_mean, self.action_std = \
            _load_norm_stats(checkpoint, device)

        # Pre-tokenize task instruction once — fixed for the whole session.
        tokenizer = self.policy.model.vlm_with_expert.processor.tokenizer
        prompt = task if task.endswith("\n") else task + "\n"
        tok = tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.policy.config.tokenizer_max_length,
        )
        self.lang_tokens = tok["input_ids"].to(device)
        self.lang_mask   = tok["attention_mask"].to(device).bool()

        state_dim  = self.policy.config.input_features[OBS_STATE].shape[0]
        action_dim = self.policy.config.output_features["action"].shape[0]
        print(f"[server] Ready — state_dim={state_dim}  action_dim={action_dim}")
        print(f"[server] Task: {prompt.strip()!r}")

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def step(self, state: np.ndarray, cameras: dict) -> np.ndarray:
        state_t = torch.from_numpy(state).float().to(self.device)

        # Normalize state with the same MEAN_STD stats used during training.
        n_state = min(state_t.shape[0], self.state_mean.shape[0])
        norm_state = state_t.clone()
        norm_state[:n_state] = (state_t[:n_state] - self.state_mean[:n_state]) / self.state_std[:n_state]

        batch = {
            OBS_STATE: norm_state.unsqueeze(0),
            OBS_LANGUAGE_TOKENS:         self.lang_tokens,
            OBS_LANGUAGE_ATTENTION_MASK: self.lang_mask,
        }
        for cam_key, img_hwc in cameras.items():
            key = f"observation.images.{cam_key}"
            # HWC uint8 → CHW float32 in [0, 1], batch dim added
            batch[key] = (
                torch.from_numpy(img_hwc).float()
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(self.device)
                    / 255.0
            )

        # Policy returns action in normalized space — must unnormalize back to real units.
        norm_action = self.policy.select_action(batch)  # (action_dim,) or (1, action_dim)
        norm_action = norm_action.squeeze()

        action = norm_action.clone()
        n_act = min(norm_action.shape[0], self.action_mean.shape[0])
        action[:n_act] = norm_action[:n_act] * self.action_std[:n_act] + self.action_mean[:n_act]

        return action.cpu().numpy().astype(np.float32)


# ── server loop ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to pretrained_model dir (contains config.json + model.safetensors)")
    ap.add_argument("--task", required=True,
                    help="Language instruction sent to SmolVLA at every step")
    ap.add_argument("--port",   type=int, default=5556)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    server = PolicyServer(args.checkpoint, args.task, args.device)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print(f"[server] Listening on 127.0.0.1:{args.port} — waiting for Isaac Sim client …")

    try:
        conn, addr = srv.accept()
        print(f"[server] Client connected from {addr}")
        with conn:
            while True:
                raw = recv_msg(conn)
                if not raw:
                    print("[server] Client disconnected.")
                    break
                msg = pickle.loads(raw)
                cmd = msg.get("cmd", "step")

                if cmd == "close":
                    send_msg(conn, pickle.dumps({"ok": True}))
                    break
                elif cmd == "reset":
                    server.reset()
                    send_msg(conn, pickle.dumps({"ok": True}))
                elif cmd == "step":
                    action = server.step(msg["state"], msg["cameras"])
                    # Send as a plain Python list so the client can unpickle it
                    # regardless of numpy version (lerobot=numpy2, isaaclab=numpy1).
                    send_msg(conn, pickle.dumps({"action": action.tolist()}))
                else:
                    send_msg(conn, pickle.dumps({"error": f"unknown cmd {cmd!r}"}))
    finally:
        srv.close()
        print("[server] Shut down.")


if __name__ == "__main__":
    main()
