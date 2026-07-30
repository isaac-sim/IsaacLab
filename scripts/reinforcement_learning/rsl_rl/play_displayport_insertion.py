# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Production DisplayPort insertion policy inference in simulation.

Run a trained RSL-RL checkpoint **or** a LEAPP-exported ONNX policy with
deterministic initial conditions, optional perception-error injection on the
socket observation, and per-step logging of observations, actions, end-effector
pose, and plug/socket state.

Per-step telemetry is written to ``policy_io.csv`` (plus ``run_config.json``) under
``--log_dir`` or ``<checkpoint_dir>/inference_logs/<timestamp>/``. The CSV column
layout matches ``rollouts/sim_real_compare/policy_io.csv`` so the same plotting
tools can overlay sim and real rollouts.

This is intentionally separate from ``play.py`` so DisplayPort-specific pose and
logging knobs stay out of the generic play path.

Example (recommended shipping task, RSL-RL checkpoint):

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_displayport_insertion.py \\
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0 \\
        --checkpoint logs/rsl_rl/displayport_insertion_rizon4s/<run>/model_1500.pt \\
        --num_envs 1 \\
        --socket_pos 0.476 0.127 0.07 \\
        --observed_socket_pos 0.486 0.127 0.07 \\
        --max_steps 200 \\
        --log_dir logs/dp_inference_runs \\
        --visualizer kit

LEAPP-exported policy (ONNX + deploy YAML). Pass a YAML path or the export directory:

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_displayport_insertion.py \\
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0 \\
        --leapp_model logs/rsl_rl/dp_exps/displayport_default_model900 \\
        --num_envs 1 \\
        --socket_pos 0.476 0.127 0.07 \\
        --max_steps 200 \\
        --log_dir logs/dp_inference_runs \\
        --visualizer kit

Open-loop replay of a real (or sim) ``policy_io`` CSV:

.. code-block:: bash

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_displayport_insertion.py \\
        --task Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0 \\
        --replay_csv rollouts/sim_real_compare/real_policy.csv \\
        --socket_pos 0.475 0.125 0.07 \\
        --log_dir rollouts/sim_real_compare/replay_run \\
        --visualizer kit

Pose conventions
----------------
* ``--socket_pos`` / ``--observed_socket_pos`` are socket **insertion-geometry**
  world positions (mate point), matching ``fixed_asset_init_pos_center``, not the
  USD root. The USD root is derived via ``compute_socket_root``.
* Quaternions are ``(x, y, z, w)``.
* If ``--observed_socket_*`` is omitted, the policy observes the true simulated
  socket pose.
* If ``--robot_joint_pos`` is set, IK grasp reset is replaced with a plug snap
  into the gripper so the requested joints are preserved.
* ``--replay_csv`` open-loop-replays absolute arm joint targets from a
  ``policy_io.csv`` / ``real_policy.csv`` instead of running the policy. The
  same ``policy_io.csv`` logging path is used. Checkpoint is optional in this
  mode. When init overrides are omitted, row 0 of the CSV seeds
  ``--robot_joint_pos`` and ``--observed_socket_*``.
* ``--leapp_model`` runs a LEAPP-exported ONNX policy (``InferenceManager``)
  instead of an RSL-RL ``.pt`` checkpoint. Accepts the deploy ``.yaml`` or the
  export directory containing it. Pose overrides and ``policy_io.csv`` logging
  still apply. Mutually exclusive with ``--checkpoint`` / ``--replay_csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import importlib.metadata as metadata
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import torch
import yaml
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path

# isaaclab.assets / warp must not be imported at module scope before SimulationApp starts.
if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    compute_plug_pose,
    compute_socket_root,
)
from isaaclab_tasks.utils import get_checkpoint_path, setup_preset_cli
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from success_utils import SuccessTracker  # isort: skip

_DEFAULT_TASK = "Isaac-Deploy-DisplayportInsertion-Rizon4s-Grav-NoJointVel-ROS-Inference-v0"

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="DisplayPort insertion policy inference with controllable poses and step logging.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=_DEFAULT_TASK, help="Registered gym task id.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--use_pretrained_checkpoint", action="store_true", help="Use the published Nucleus pretrained checkpoint."
)
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle stepping to wall-clock dt.")

# Initial / observed state
parser.add_argument(
    "--robot_joint_pos",
    type=str,
    nargs="+",
    default=None,
    metavar="VALUE_RAD | JOINT_NAME=VALUE_RAD",
    help=(
        "Starting joint positions in radians. Either 7 bare values in arm-joint order"
        " (e.g. --robot_joint_pos 0.44 -0.25 0.04 2.23 -0.01 0.91 2.08) or 'name=value' pairs"
        " (e.g. joint1=0.44). Disables IK grasp reset."
    ),
)
parser.add_argument(
    "--socket_pos",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Actual simulated socket insertion-geometry world position [m].",
)
parser.add_argument(
    "--socket_rot",
    type=float,
    nargs=4,
    default=None,
    metavar=("X", "Y", "Z", "W"),
    help="Actual simulated socket orientation quaternion (x, y, z, w).",
)
parser.add_argument(
    "--observed_socket_pos",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Socket insertion-geometry world position [m] fed to the policy (perception belief).",
)
parser.add_argument(
    "--observed_socket_rot",
    type=float,
    nargs=4,
    default=None,
    metavar=("X", "Y", "Z", "W"),
    help="Socket orientation quaternion (x, y, z, w) fed to the policy.",
)
parser.add_argument(
    "--z_clearance",
    type=float,
    default=0.068,
    help="Initial plug-above-socket clearance [m] when --socket_pos is set.",
)

# Replay / open-loop debugging
parser.add_argument(
    "--replay_csv",
    type=str,
    default=None,
    metavar="PATH",
    help=(
        "Open-loop replay absolute arm joint targets from a policy_io / real_policy CSV "
        "instead of running the policy. Accepts target_joint_pos_* (sim) or "
        "safety_cmd_* + measured joints (real: absolute = joint + safety_cmd). "
        "Checkpoint is optional. Ends after the CSV is exhausted unless --max_steps is smaller."
    ),
)
parser.add_argument(
    "--leapp_model",
    type=str,
    default=None,
    metavar="PATH",
    help=(
        "Path to a LEAPP-exported deploy YAML, or a directory containing one "
        "(e.g. logs/rsl_rl/dp_exps/displayport_default_model900). Runs the exported "
        "ONNX policy via InferenceManager instead of an RSL-RL .pt checkpoint."
    ),
)

# Run control / logging
parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many policy steps.")
parser.add_argument(
    "--num_episodes",
    type=int,
    default=None,
    help="Stop after this many completed episodes (requires env auto-reset).",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help=(
        "Directory for run logs. Default: <checkpoint_or_leapp_dir>/inference_logs/<timestamp>."
    ),
)
parser.add_argument("--no_print", action="store_true", help="Disable per-step terminal printing.")
parser.add_argument(
    "--no_log_file",
    action="store_true",
    help="Disable policy_io.csv / run_config.json file logging.",
)
parser.add_argument(
    "--print_every",
    type=int,
    default=1,
    help="Print every N steps (file logging still records every step unless --no_log_file).",
)

cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


# -- helpers -----------------------------------------------------------------


def _to_torch(x):
    """Convert a warp/torch-backed data field to a torch tensor (mappings pass through)."""
    from collections.abc import Mapping

    if x is None:
        return None
    if isinstance(x, Mapping):
        return x
    if hasattr(x, "torch"):
        return x.torch
    try:
        import warp as wp

        return wp.to_torch(x)
    except Exception:
        return x


def _fmt(t) -> str:
    """Format a tensor/array/mapping as a compact rounded string."""
    from collections.abc import Mapping

    if t is None:
        return "N/A"
    if isinstance(t, Mapping):
        return "{" + ", ".join(f"{k}: {_fmt(v)}" for k, v in t.items()) + "}"
    if hasattr(t, "detach"):
        arr = t.detach().cpu().numpy().reshape(-1)
        return np.array2string(arr, precision=4, suppress_small=True, max_line_width=240)
    try:
        arr = np.asarray(t).reshape(-1)
        return np.array2string(arr, precision=4, suppress_small=True, max_line_width=240)
    except Exception:
        return str(t)


def _get_attr(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _get_asset(scene, candidate_names):
    for n in candidate_names:
        try:
            return scene[n], n
        except Exception:
            continue
    return None, None


def _as_numpy_1d(x) -> np.ndarray | None:
    t = _to_torch(x)
    if t is None:
        return None
    if hasattr(t, "detach"):
        return t.detach().cpu().numpy().reshape(-1)
    return np.asarray(t).reshape(-1)


def _extract_lstm_hidden_state(policy) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return per-env-0 LSTM state as ``(h, c)`` with shape ``(num_layers, hidden_dim)``.

    Supports:
    * rsl-rl >= 4 ``RNNModel.get_hidden_state()`` → ``(h, c)`` for LSTM / ``h`` for GRU
    * older ``ActorCriticRecurrent`` via ``memory_a.hidden_states`` / ``.hidden_state``

    Pack into deploy ``rnn_*`` columns with :func:`_pack_rnn_state`.
    """
    if policy is None:
        return None, None

    hs = None
    if hasattr(policy, "get_hidden_state"):
        try:
            hs = policy.get_hidden_state()
        except Exception:
            hs = None
    if hs is None:
        for attr in ("memory_a", "rnn", "actor"):
            mod = getattr(policy, attr, None)
            if mod is None:
                continue
            hs = getattr(mod, "hidden_state", None)
            if hs is None:
                hs = getattr(mod, "hidden_states", None)
            if hs is not None:
                break
            if hasattr(mod, "get_hidden_state"):
                try:
                    hs = mod.get_hidden_state()
                except Exception:
                    hs = None
                if hs is not None:
                    break

    if hs is None:
        return None, None

    def _env0_layers(t) -> np.ndarray | None:
        tt = _to_torch(t)
        if tt is None:
            return None
        if hasattr(tt, "detach"):
            tt = tt.detach()
        # Expected shapes: (num_layers, num_envs, hidden_dim) or already (L, H).
        if tt.ndim == 3:
            tt = tt[:, 0, :]
        elif tt.ndim == 1:
            # Already flat; treat as a single layer.
            tt = tt.reshape(1, -1)
        return tt.contiguous().cpu().numpy()

    if isinstance(hs, (tuple, list)):
        if len(hs) == 0:
            return None, None
        h = _env0_layers(hs[0])
        c = _env0_layers(hs[1]) if len(hs) > 1 else None
        return h, c
    return _env0_layers(hs), None


def _pack_rnn_state(h: np.ndarray | None, c: np.ndarray | None) -> np.ndarray | None:
    """Pack LSTM state into the real-robot ``rnn_*`` flatten order.

    Matches ``policy_io_csv_schema.md``: for each layer, concatenate hidden then
    cell (ONNX ``actor_state_L_out`` with shape ``[2, 1, H]``), then stack layers.
    DisplayPort (2 x 256 LSTM) → 1024 floats: ``[h0, c0, h1, c1]``.
    """
    if h is None:
        return None
    h_arr = np.asarray(h, dtype=np.float64)
    if h_arr.ndim == 1:
        h_arr = h_arr.reshape(1, -1)
    parts: list[np.ndarray] = []
    if c is None:
        for layer in range(h_arr.shape[0]):
            parts.append(h_arr[layer].reshape(-1))
    else:
        c_arr = np.asarray(c, dtype=np.float64)
        if c_arr.ndim == 1:
            c_arr = c_arr.reshape(1, -1)
        if c_arr.shape != h_arr.shape:
            # Fall back to all-h then all-c so something useful is still logged.
            return np.concatenate([h_arr.reshape(-1), c_arr.reshape(-1)], axis=0)
        for layer in range(h_arr.shape[0]):
            parts.append(h_arr[layer].reshape(-1))
            parts.append(c_arr[layer].reshape(-1))
    return np.concatenate(parts, axis=0)


def _parse_robot_joint_pos(entries: list[str], ordered_joint_names: list[str]) -> dict[str, float]:
    """Parse ``--robot_joint_pos`` as either bare ordered values or ``name=value`` pairs.

    Args:
        entries: CLI tokens, e.g. ``["0.44", "-0.25", ...]`` or ``["joint1=0.44", ...]``.
        ordered_joint_names: Arm joint names in policy order, used to map bare values.
    """
    named = [e for e in entries if "=" in e]
    if named and len(named) != len(entries):
        raise ValueError(
            "--robot_joint_pos must be either all bare values (positional) or all 'name=value' pairs, not mixed."
        )

    if named:
        return {e.partition("=")[0]: float(e.partition("=")[2]) for e in entries}

    values = [float(e) for e in entries]
    if len(values) > len(ordered_joint_names):
        raise ValueError(
            f"--robot_joint_pos got {len(values)} values but only {len(ordered_joint_names)} arm joints are known"
            f" ({ordered_joint_names}). Use 'name=value' pairs to set gripper joints."
        )
    return dict(zip(ordered_joint_names, values))


def _arm_joint_names(env_cfg) -> list[str]:
    """Ordered arm joint names, taken from the policy ``joint_pos`` observation term."""
    try:
        names = env_cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names
        if names:
            return list(names)
    except Exception:
        pass
    num_arm = int(getattr(env_cfg, "num_arm_joints", 7) or 7)
    return [f"joint{i + 1}" for i in range(num_arm)]


def _trailing_int(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 1_000_000


@dataclass
class ReplayTrajectory:
    """Open-loop arm joint targets loaded from a policy I/O CSV."""

    path: Path
    source: str  # how absolute targets were derived
    targets: np.ndarray  # (T, num_arm) absolute joint positions [rad]
    init_joint_pos: np.ndarray | None = None  # (num_arm,)
    observed_socket_pos: np.ndarray | None = None  # (3,)
    observed_socket_rot: np.ndarray | None = None  # (4,) xyzw
    csv_actions: np.ndarray | None = None  # (T, num_arm) original logged actions, if present

    @property
    def num_steps(self) -> int:
        return int(self.targets.shape[0])

    @property
    def num_arm_joints(self) -> int:
        return int(self.targets.shape[1])


def load_replay_trajectory(path: str | Path, num_arm_joints: int = 7) -> ReplayTrajectory:
    """Load absolute arm joint targets from a sim or real ``policy_io``-style CSV.

    Column priority (see ``policy_io_csv_schema.md``):
    1. ``blend_cmd_*`` — absolute blended/safety-limited command written to hardware (real).
    2. ``target_joint_pos_{i}`` — absolute commands (sim logger extra).
    3. measured ``joint_*_pos`` + ``safety_cmd_*`` — real pre-blend path when
       ``safety_cmd`` is a scaled delta (``joint + safety_cmd``).
    4. ``action_{i}`` — policy output; for this DisplayPort deploy the values are
       absolute joint targets (rejected if they do not look like joint angles).
    """
    csv_path = Path(path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"--replay_csv not found: {csv_path}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"--replay_csv has no header: {csv_path}")
        header = list(reader.fieldnames)
        rows = list(reader)
    if not rows:
        raise ValueError(f"--replay_csv is empty: {csv_path}")

    def _stack(keys: list[str]) -> np.ndarray:
        return np.asarray([[float(r[k]) for k in keys] for r in rows], dtype=np.float64)

    tgt_keys = sorted(
        [k for k in header if re.fullmatch(r"target_joint_pos_\d+", k)],
        key=_trailing_int,
    )
    blend_keys = sorted(
        [k for k in header if re.search(r"blend_cmd_.*joint\d+$", k) or re.fullmatch(r"blend_cmd_\d+", k)],
        key=_trailing_int,
    )
    safety_keys = sorted(
        [k for k in header if re.search(r"safety_cmd_.*joint\d+$", k) or re.fullmatch(r"safety_cmd_\d+", k)],
        key=_trailing_int,
    )
    # Prefer explicit arm joint_pos columns; fall back to obs_0..N-1.
    joint_keys = [k for k in header if re.fullmatch(r"joint_joint\d+_pos", k)]
    if not joint_keys:
        joint_keys = [k for k in header if re.search(r"joint_.*joint\d+_pos$", k)]
    joint_keys = sorted(joint_keys, key=_trailing_int)
    act_keys = sorted([k for k in header if re.fullmatch(r"action_\d+", k)], key=_trailing_int)
    obs_keys = sorted([k for k in header if re.fullmatch(r"obs_\d+", k)], key=_trailing_int)

    source: str
    if len(blend_keys) >= num_arm_joints:
        targets = _stack(blend_keys[:num_arm_joints])
        source = "blend_cmd"
    elif len(tgt_keys) >= num_arm_joints:
        targets = _stack(tgt_keys[:num_arm_joints])
        source = "target_joint_pos"
    elif len(safety_keys) >= num_arm_joints and len(joint_keys) >= num_arm_joints:
        measured = _stack(joint_keys[:num_arm_joints])
        deltas = _stack(safety_keys[:num_arm_joints])
        targets = measured + deltas
        source = "joint_pos+safety_cmd"
        print(
            "[INFO] Replay CSV uses real-style safety_cmd deltas; "
            "absolute targets = measured_joint + safety_cmd."
        )
    elif len(act_keys) >= num_arm_joints:
        candidate = _stack(act_keys[:num_arm_joints])
        # Guard against older sim logs where action_* held raw normalized output.
        reference = (
            _stack(joint_keys[:num_arm_joints])[0]
            if len(joint_keys) >= num_arm_joints
            else (_stack(obs_keys[:num_arm_joints])[0] if len(obs_keys) >= num_arm_joints else None)
        )
        if reference is not None and np.max(np.abs(candidate[0] - reference)) > 0.5:
            raise ValueError(
                f"--replay_csv {csv_path}: action_* does not look like absolute joint angles "
                f"(row 0 differs from measured joints by "
                f"{np.max(np.abs(candidate[0] - reference)):.3f} rad). Prefer blend_cmd_* / "
                "target_joint_pos_* or re-log with the current schema."
            )
        targets = candidate
        source = "action (absolute)"
    else:
        raise ValueError(
            f"--replay_csv {csv_path} has no usable joint targets. Need "
            f"blend_cmd_*, target_joint_pos_*, safety_cmd_* + joint_*_pos, or absolute action_*. "
            f"Found: blend={len(blend_keys)}, target={len(tgt_keys)}, safety_cmd={len(safety_keys)}, "
            f"joint={len(joint_keys)}, action={len(act_keys)}."
        )

    init_joint_pos = None
    if len(joint_keys) >= num_arm_joints:
        init_joint_pos = _stack(joint_keys[:num_arm_joints])[0]
    elif len(obs_keys) >= num_arm_joints:
        init_joint_pos = _stack(obs_keys[:num_arm_joints])[0]

    observed_socket_pos = observed_socket_rot = None
    if all(k in header for k in ("goal_px", "goal_py", "goal_pz")):
        observed_socket_pos = np.asarray(
            [float(rows[0]["goal_px"]), float(rows[0]["goal_py"]), float(rows[0]["goal_pz"])],
            dtype=np.float64,
        )
    elif len(obs_keys) >= num_arm_joints + 3:
        observed_socket_pos = _stack(obs_keys[num_arm_joints : num_arm_joints + 3])[0]
    if all(k in header for k in ("goal_qx", "goal_qy", "goal_qz", "goal_qw")):
        observed_socket_rot = np.asarray(
            [
                float(rows[0]["goal_qx"]),
                float(rows[0]["goal_qy"]),
                float(rows[0]["goal_qz"]),
                float(rows[0]["goal_qw"]),
            ],
            dtype=np.float64,
        )
    elif len(obs_keys) >= num_arm_joints + 7:
        observed_socket_rot = _stack(obs_keys[num_arm_joints + 3 : num_arm_joints + 7])[0]

    csv_actions = _stack(act_keys[:num_arm_joints]) if len(act_keys) >= num_arm_joints else None

    print(
        f"[INFO] Loaded replay trajectory from {csv_path}: "
        f"{targets.shape[0]} steps, {targets.shape[1]} arm joints, source={source}"
    )
    return ReplayTrajectory(
        path=csv_path,
        source=source,
        targets=targets,
        init_joint_pos=init_joint_pos,
        observed_socket_pos=observed_socket_pos,
        observed_socket_rot=observed_socket_rot,
        csv_actions=csv_actions,
    )


def _seed_cli_from_replay(traj: ReplayTrajectory):
    """Fill unset pose CLI args from the first CSV row so replay starts matched."""
    if args_cli.robot_joint_pos is None and traj.init_joint_pos is not None:
        args_cli.robot_joint_pos = [f"{v:.6f}" for v in traj.init_joint_pos.tolist()]
        print(f"[INFO] --robot_joint_pos seeded from CSV row 0: {args_cli.robot_joint_pos}")
    if args_cli.observed_socket_pos is None and traj.observed_socket_pos is not None:
        args_cli.observed_socket_pos = [float(v) for v in traj.observed_socket_pos.tolist()]
        print(f"[INFO] --observed_socket_pos seeded from CSV: {args_cli.observed_socket_pos}")
    if args_cli.observed_socket_rot is None and traj.observed_socket_rot is not None:
        args_cli.observed_socket_rot = [float(v) for v in traj.observed_socket_rot.tolist()]
        print(f"[INFO] --observed_socket_rot seeded from CSV: {args_cli.observed_socket_rot}")


def _action_scale(env) -> float:
    """Relative joint action scale used by the env (default 0.025)."""
    base = env.unwrapped
    cfg_scale = getattr(base.cfg, "joint_action_scale", None)
    if cfg_scale is not None:
        return float(cfg_scale)
    try:
        term = base.action_manager.get_term("arm_action")
        scale = getattr(term, "_scale", None)
        if scale is not None:
            s = _as_numpy_1d(scale)
            if s is not None and s.size:
                return float(s.reshape(-1)[0])
    except Exception:
        pass
    return 0.025


def _absolute_targets_to_relative_actions(
    env,
    targets_1d: np.ndarray,
    clip_actions: float | None,
) -> torch.Tensor:
    """Convert absolute arm joint targets to RelativeJointPositionAction inputs.

    ``apply_actions`` does ``target = action * scale + q``, so
    ``action = (target - q) / scale``.
    """
    base = env.unwrapped
    robot = base.scene["robot"]
    num_arm = int(targets_1d.shape[0])
    q = _as_numpy_1d(_to_torch(robot.data.joint_pos)[0])
    if q is None:
        raise RuntimeError("Cannot read robot joint_pos for replay action conversion.")
    scale = _action_scale(env)
    rel = (targets_1d - q[:num_arm]) / scale
    if clip_actions is not None:
        rel = np.clip(rel, -clip_actions, clip_actions)
    actions = torch.zeros((base.num_envs, env.num_actions), device=base.device, dtype=torch.float32)
    actions[:, :num_arm] = torch.as_tensor(rel, device=base.device, dtype=torch.float32)
    return actions


def resolve_leapp_model_yaml(path: str | Path) -> Path:
    """Resolve ``--leapp_model`` to a LEAPP deploy YAML file.

    Accepts either a ``.yaml``/``.yml`` file or a directory that contains one
    (prefers ``*deploy*.yaml``, then any single YAML in the directory).
    """
    model_path = Path(path).expanduser().resolve()
    if model_path.is_file():
        if model_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"--leapp_model must be a YAML file or directory, got: {model_path}")
        return model_path
    if not model_path.is_dir():
        raise FileNotFoundError(f"--leapp_model not found: {model_path}")

    yaml_files = sorted(model_path.glob("*.yaml")) + sorted(model_path.glob("*.yml"))
    if not yaml_files:
        raise FileNotFoundError(f"No YAML found under --leapp_model directory: {model_path}")
    deploy_candidates = [p for p in yaml_files if "deploy" in p.name.lower()]
    if len(deploy_candidates) == 1:
        return deploy_candidates[0]
    if len(yaml_files) == 1:
        return yaml_files[0]
    if deploy_candidates:
        names = ", ".join(p.name for p in deploy_candidates)
        raise ValueError(
            f"Multiple deploy YAML files under {model_path}: {names}. Pass the YAML path explicitly."
        )
    names = ", ".join(p.name for p in yaml_files)
    raise ValueError(f"Multiple YAML files under {model_path}: {names}. Pass the YAML path explicitly.")


class LeappDisplayportPolicy:
    """Run a LEAPP-exported DisplayPort policy against a ManagerBased gym env.

    LEAPP packages observation preprocessing, LSTM state, and action decoding into
    an ONNX graph that emits **absolute** arm joint targets. This adapter:

    1. Reads ``robot_joint_pos`` / ``socket_pos`` / ``socket_quat`` from the live
       scene and observation terms (so ``--observed_socket_*`` overrides apply).
    2. Runs ``InferenceManager.run_policy``.
    3. Converts absolute targets back to relative action-manager inputs so the
       existing ``env.step`` / logging path stays unchanged.
    """

    def __init__(self, env, leapp_yaml: Path, clip_actions: float | None = None):
        try:
            from leapp import InferenceManager
        except ImportError as exc:
            raise ImportError(
                "LEAPP is required for --leapp_model. Install with: ./isaaclab.sh -p -m pip install leapp"
            ) from exc

        self.env = env
        self.base = env.unwrapped
        self.clip_actions = clip_actions
        self.yaml_path = Path(leapp_yaml)
        self.inference = InferenceManager(str(self.yaml_path))

        with open(self.yaml_path, encoding="utf-8") as f:
            desc = yaml.safe_load(f)
        pipeline = desc["pipeline"]
        models = desc["models"]
        # Prefer the single pipeline input node; fall back to the only model key.
        input_nodes = list(pipeline.get("inputs", {}).keys())
        if not input_nodes:
            raise ValueError(f"LEAPP YAML has no pipeline inputs: {self.yaml_path}")
        self.node_name = input_nodes[0]
        if self.node_name not in models:
            # Some exports use a shortened model key; keep going with pipeline name.
            pass

        self.input_names = list(pipeline["inputs"][self.node_name])
        self._joint_ids = self._resolve_arm_joint_ids()
        self.last_outputs: dict[str, torch.Tensor] = {}
        self.last_absolute_targets: np.ndarray | None = None

        print(f"[INFO] LEAPP policy loaded from: {self.yaml_path}")
        print(f"[INFO] LEAPP node='{self.node_name}', inputs={self.input_names}")

    def _resolve_arm_joint_ids(self) -> list[int] | None:
        robot = self.base.scene["robot"]
        try:
            names = list(self.base.cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names)
        except Exception:
            names = [f"joint{i + 1}" for i in range(int(getattr(self.base.cfg, "num_arm_joints", 7) or 7))]
        if not names:
            return None
        # Exact names if provided; otherwise treat as regex patterns via find_joints.
        if all(n in list(robot.joint_names) for n in names):
            return [list(robot.joint_names).index(n) for n in names]
        joint_ids, _ = robot.find_joints(names, preserve_order=True)
        return list(joint_ids)

    def _read_observation_term(self, term_name: str) -> torch.Tensor:
        """Evaluate a policy observation term (honors live overrides / noise flags)."""
        obs_mgr = self.base.observation_manager
        names = list(obs_mgr._group_obs_term_names["policy"])
        cfgs = list(obs_mgr._group_obs_term_cfgs["policy"])
        if term_name not in names:
            raise KeyError(f"Policy observation term '{term_name}' not found (have {names})")
        cfg = cfgs[names.index(term_name)]
        out = cfg.func(self.base, **cfg.params)
        if not torch.is_tensor(out):
            out = torch.as_tensor(out, device=self.base.device, dtype=torch.float32)
        return out

    def _gather_inputs(self) -> dict[str, torch.Tensor]:
        robot = self.base.scene["robot"]
        joint_pos = _to_torch(robot.data.joint_pos)
        if self._joint_ids is not None:
            joint_pos = joint_pos[:, self._joint_ids]

        values: dict[str, torch.Tensor] = {}
        for name in self.input_names:
            if name in ("robot_joint_pos", "arm_dof_pos", "joint_pos"):
                values[name] = joint_pos
            elif name in ("socket_pos",):
                values[name] = self._read_observation_term("socket_pos")
            elif name in ("socket_quat",):
                values[name] = self._read_observation_term("socket_quat")
            elif name.startswith("actor_state_"):
                # Feedback tensors are owned by InferenceManager; omit from external inputs.
                continue
            else:
                raise KeyError(
                    f"Unsupported LEAPP input '{name}' for DisplayPort play. "
                    "Expected robot_joint_pos / socket_pos / socket_quat (plus LSTM feedback)."
                )

        return {f"{self.node_name}/{name}": tensor for name, tensor in values.items()}

    def __call__(self, obs=None) -> torch.Tensor:
        """Run one LEAPP inference step; ``obs`` is ignored (inputs are re-read)."""
        del obs  # LEAPP graph takes structured I/O, not the flat RSL-RL vector.
        inputs = self._gather_inputs()
        with torch.inference_mode():
            self.last_outputs = self.inference.run_policy(inputs)

        abs_key = f"{self.node_name}/arm_action"
        if abs_key not in self.last_outputs:
            # Fall back to the first tensor that looks like an arm command.
            candidates = [k for k in self.last_outputs if k.endswith("/arm_action") or "action" in k.split("/")[-1]]
            if not candidates:
                raise KeyError(
                    f"LEAPP outputs missing arm_action. Keys: {list(self.last_outputs.keys())}"
                )
            abs_key = candidates[0]

        absolute = self.last_outputs[abs_key]
        abs_np = _as_numpy_1d(absolute[0] if absolute.ndim > 1 else absolute)
        if abs_np is None:
            raise RuntimeError("Failed to read LEAPP arm_action tensor.")
        self.last_absolute_targets = abs_np.copy()
        return _absolute_targets_to_relative_actions(self.env, abs_np, self.clip_actions)

    def reset(self, dones=None):
        """Reset LEAPP recurrent state when any env episode ends."""
        if dones is None:
            self.inference.reset()
            return
        dones_b = torch.as_tensor(dones).bool().view(-1)
        if bool(dones_b.any()):
            self.inference.reset()

    def extract_lstm_state(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return ``(h, c)`` with shape ``(num_layers, hidden)`` from LEAPP outputs."""
        layer_states: list[torch.Tensor] = []
        for i in range(8):
            key = f"{self.node_name}/actor_state_{i}_out"
            if key not in self.last_outputs:
                # Some InferenceManager builds keep feedback only internally.
                alt = [k for k in self.last_outputs if k.endswith(f"/actor_state_{i}_out")]
                if not alt:
                    break
                key = alt[0]
            layer_states.append(self.last_outputs[key])
        if not layer_states:
            return None, None

        # Each layer is typically [2, num_envs, hidden] for LSTM (h/c), or [1, ...] for GRU.
        h_layers = []
        c_layers = []
        has_cell = True
        for state in layer_states:
            t = state
            if t.ndim == 3:
                # [2, B, H] or [1, B, H]
                if t.shape[0] >= 2:
                    h_layers.append(_as_numpy_1d(t[0, 0]))
                    c_layers.append(_as_numpy_1d(t[1, 0]))
                else:
                    h_layers.append(_as_numpy_1d(t[0, 0]))
                    has_cell = False
            elif t.ndim == 2:
                h_layers.append(_as_numpy_1d(t[0]))
                has_cell = False
            else:
                h_layers.append(_as_numpy_1d(t))
                has_cell = False

        if not h_layers or any(h is None for h in h_layers):
            return None, None
        h = np.stack(h_layers, axis=0)
        if not has_cell or any(c is None for c in c_layers):
            return h, None
        c = np.stack(c_layers, axis=0)
        return h, c


def place_plug_at_grasp_pose(
    env,
    env_ids: torch.Tensor,
    robot_asset_cfg: SceneEntityCfg,
    target_object_name: str,
    end_effector_body_name: str,
    grasp_offset: list,
    grasp_rot_offset: list,
    gripper_joint_setter_func,
    hand_hold_width: float,
    hand_close_width: float,
    num_arm_joints: int,
):
    """Snap the plug into the gripper from the robot's already-reset end-effector pose."""
    import warp as wp

    robot: Articulation = env.scene[robot_asset_cfg.name]
    held_object: RigidObject = env.scene[target_object_name]

    eef_indices, _ = robot.find_bodies([end_effector_body_name])
    eef_idx = eef_indices[0]

    num_reset_envs = len(env_ids)
    grasp_offset_tensor = (
        torch.tensor(grasp_offset, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(num_reset_envs, 1)
    )
    grasp_rot_offset_tensor = (
        torch.tensor(grasp_rot_offset, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(num_reset_envs, 1)
    )

    achieved_hand_pos = wp.to_torch(robot.data.body_pos_w)[env_ids, eef_idx].clone()
    achieved_hand_quat = wp.to_torch(robot.data.body_quat_w)[env_ids, eef_idx].clone()

    inv_grasp_rot_offset = math_utils.quat_conjugate(grasp_rot_offset_tensor)
    target_obj_quat = math_utils.quat_mul(achieved_hand_quat, inv_grasp_rot_offset)
    grasp_offset_in_world = math_utils.quat_apply(achieved_hand_quat, grasp_offset_tensor)
    target_obj_pos = achieved_hand_pos - grasp_offset_in_world

    new_root_pose = torch.cat([target_obj_pos, target_obj_quat], dim=-1)
    zero_velocity = torch.zeros((num_reset_envs, 6), device=env.device, dtype=torch.float32)
    held_object.write_root_pose_to_sim_index(root_pose=new_root_pose, env_ids=env_ids)
    held_object.write_root_velocity_to_sim_index(root_velocity=zero_velocity, env_ids=env_ids)

    all_joints, _ = robot.find_joints([".*"])
    finger_joints = all_joints[num_arm_joints:]
    joint_pos = wp.to_torch(robot.data.joint_pos)[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    gripper_joint_setter_func(joint_pos, list(range(num_reset_envs)), finger_joints, hand_hold_width)
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

    gripper_joint_setter_func(joint_pos, list(range(num_reset_envs)), finger_joints, hand_close_width)
    robot.set_joint_position_target_index(target=joint_pos, joint_ids=all_joints, env_ids=env_ids)


class _FixedObservedSocketPos:
    """Override ``socket_pos`` observation with a fixed insertion-geometry world position."""

    def __init__(self, env, geometry_pos_w):
        self.pos_w = torch.tensor(geometry_pos_w, device=env.device, dtype=torch.float32)

    def __call__(self, env, **kwargs):
        pos = self.pos_w.unsqueeze(0).repeat(env.num_envs, 1)
        return pos - env.scene.env_origins

    def reset(self, env_ids=None):
        pass


class _FixedObservedSocketQuat:
    """Override ``socket_quat`` observation with a fixed world orientation."""

    def __init__(self, env, rot_w):
        self.rot_w = torch.tensor(rot_w, device=env.device, dtype=torch.float32)

    def __call__(self, env, **kwargs):
        quat = self.rot_w.unsqueeze(0).repeat(env.num_envs, 1)
        w_negative = quat[:, 3] < 0
        positive_quat = quat.clone()
        positive_quat[w_negative] = -quat[w_negative]
        return positive_quat

    def reset(self, env_ids=None):
        pass


def _flatten_policy_obs(obs) -> np.ndarray | None:
    """Return a flat 1D numpy observation for env 0 (policy vector if dict/TensorDict)."""
    # Prefer the policy group when present (RSL-RL / Isaac Lab convention).
    if isinstance(obs, dict) or hasattr(obs, "items"):
        try:
            items = dict(obs.items())
            if "policy" in items:
                arr = _as_numpy_1d(items["policy"][0] if hasattr(items["policy"], "__getitem__") else items["policy"])
                if arr is not None:
                    return arr
            parts = []
            for _, v in items.items():
                arr = _as_numpy_1d(v[0] if hasattr(v, "__getitem__") else v)
                if arr is not None:
                    parts.append(arr)
            if parts:
                return np.concatenate(parts, axis=0)
        except Exception:
            pass
    return _as_numpy_1d(obs[0] if hasattr(obs, "__getitem__") else obs)


class InferenceLogger:
    """Print and/or save per-step inference telemetry for env 0.

    CSV layout mirrors ``policy_io_csv_schema.md`` / real-robot ``policy_io.csv``:

    ``wall_time, ros_time, step, joint_<name>_pos, obs_*, action_*, rnn_*,
    goal_p*/q*, eef_p*/q*, blend_cmd_<name>`` (+ sim-only extras).

    Column convention (aligned with the real logger for this DisplayPort deploy):

    * ``action_{i}`` — absolute arm joint target [rad] produced by the policy
      (``joint_pos_target`` after the action term). Matches real ``action_*``,
      which for this LEAPP export already lives in joint-angle space.
    * ``action_raw_{i}`` — sim-only: normalized network output before the action
      term scales/decodes it.
    * ``rnn_{i}`` — concatenated LSTM ``_out`` state this step, per-layer
      ``[h, c]`` then layers (1024 floats for 2x256). Same as real ``rnn_*``.
    * ``blend_cmd_<name>`` — absolute command applied this cycle. In sim there is
      no safety blend, so this equals ``action_*`` / ``joint_pos_target``.
    * ``target_joint_pos_{i}`` — sim-only alias of the absolute target (plot tools).

    Extra sim-only columns (plug/socket GT, success metrics, episode) are appended.
    """

    def __init__(self, env, log_dir: Path | None, print_enabled: bool, print_every: int):
        self.env = env
        self.base = env.unwrapped
        self.log_dir = log_dir
        self.print_enabled = print_enabled
        self.print_every = max(1, print_every)
        self.rows: list[dict[str, Any]] = []
        # Match real-robot logger filename for drop-in use with plot_sim_real_compare.py.
        self.csv_path = (log_dir / "policy_io.csv") if log_dir is not None else None
        self._fieldnames: list[str] | None = None
        self._t0 = time.time()
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _collect_row(
        self,
        obs,
        actions,
        step: int,
        episode: int,
        lstm_h: np.ndarray | None = None,
        lstm_c: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Snapshot the state the policy acted on, before the action is applied."""
        scene = self.base.scene
        # Timing: Unix epoch wall clock (matches real logger); ros_time := sim time.
        sim_dt = float(getattr(self.base, "step_dt", 0.0) or 0.0)
        row: dict[str, Any] = {
            "wall_time": time.time(),
            "ros_time": step * sim_dt,
            "step": step,
            "episode": episode,
        }

        # Flat policy observation vector (obs_0..N).
        obs_arr = _flatten_policy_obs(obs)
        if obs_arr is not None:
            for i, val in enumerate(obs_arr.tolist()):
                row[f"obs_{i}"] = float(val)

        # Raw normalized policy output (sim-only). Absolute action_* / blend_cmd_*
        # are filled in end_step() from joint_pos_target.
        act = _as_numpy_1d(actions[0] if hasattr(actions, "__getitem__") else actions)
        if act is not None:
            for i, val in enumerate(act.tolist()):
                row[f"action_raw_{i}"] = float(val)

        # Recurrent state as deploy ``rnn_*`` (per-layer [h, c], layers concatenated).
        rnn = _pack_rnn_state(lstm_h, lstm_c)
        if rnn is not None:
            for i, val in enumerate(rnn.reshape(-1).tolist()):
                row[f"rnn_{i}"] = float(val)

        robot = scene["robot"]
        joint_names = list(_get_attr(robot.data, "joint_names") or _get_attr(robot, "joint_names") or [])

        joint_pos = _as_numpy_1d(_to_torch(robot.data.joint_pos)[0])
        if joint_pos is not None:
            names = joint_names if joint_names else [str(i) for i in range(len(joint_pos))]
            for name, val in zip(names, joint_pos.tolist()):
                # Reference real logs use ``joint_<name>_pos``.
                row[f"joint_{name}_pos"] = float(val)

        # Goal pose = policy-visible socket (observed override if set, else true socket).
        goal_pos, goal_quat = self._resolve_goal_pose()
        if goal_pos is not None and len(goal_pos) >= 3:
            row["goal_px"], row["goal_py"], row["goal_pz"] = map(float, goal_pos[:3])
        if goal_quat is not None and len(goal_quat) >= 4:
            row["goal_qx"], row["goal_qy"], row["goal_qz"], row["goal_qw"] = map(float, goal_quat[:4])

        eef_name = getattr(self.base.cfg, "end_effector_body_name", "flange")
        body_names = _get_attr(robot.data, "body_names") or _get_attr(robot, "body_names")
        if body_names is not None and eef_name in list(body_names):
            idx = list(body_names).index(eef_name)
            eef_pos = _as_numpy_1d(_to_torch(robot.data.body_pos_w)[0, idx])
            eef_quat = _as_numpy_1d(_to_torch(robot.data.body_quat_w)[0, idx])
            if eef_pos is not None:
                # Reference naming: eef_p{x,y,z} / eef_q{x,y,z,w}
                row["eef_px"], row["eef_py"], row["eef_pz"] = map(float, eef_pos.tolist())
            if eef_quat is not None:
                row["eef_qx"], row["eef_qy"], row["eef_qz"], row["eef_qw"] = map(float, eef_quat.tolist())

        # Extra sim-only ground truth (not in real policy_io.csv, but useful for debugging).
        for asset_key, candidates in (("plug", ["dp_plug"]), ("socket", ["dp_socket"])):
            asset, _ = _get_asset(scene, candidates)
            if asset is None:
                continue
            pos = _as_numpy_1d(_to_torch(asset.data.root_pos_w)[0])
            quat = _as_numpy_1d(_to_torch(asset.data.root_quat_w)[0])
            if pos is not None:
                row[f"{asset_key}_pos_x"], row[f"{asset_key}_pos_y"], row[f"{asset_key}_pos_z"] = map(
                    float, pos.tolist()
                )
            if quat is not None:
                (
                    row[f"{asset_key}_quat_x"],
                    row[f"{asset_key}_quat_y"],
                    row[f"{asset_key}_quat_z"],
                    row[f"{asset_key}_quat_w"],
                ) = map(float, quat.tolist())

        return row

    def _resolve_goal_pose(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (pos_w, quat_xyzw) for the policy goal / socket.

        Uses the live ``socket_pos`` / ``socket_quat`` observation callables when present
        (so ``--observed_socket_*`` overrides are reflected), otherwise the true socket root.
        """
        scene = self.base.scene
        goal_pos = goal_quat = None
        socket, _ = _get_asset(scene, ["dp_socket"])
        if socket is not None:
            goal_pos = _as_numpy_1d(_to_torch(socket.data.root_pos_w)[0])
            goal_quat = _as_numpy_1d(_to_torch(socket.data.root_quat_w)[0])

        try:
            obs_mgr = self.base.observation_manager
            names = list(obs_mgr._group_obs_term_names["policy"])
            cfgs = list(obs_mgr._group_obs_term_cfgs["policy"])
            origins = _as_numpy_1d(_to_torch(scene.env_origins)[0])
            for name, cfg in zip(names, cfgs):
                if name not in ("socket_pos", "socket_quat"):
                    continue
                out = cfg.func(self.base, **cfg.params)
                arr = _as_numpy_1d(out[0] if hasattr(out, "__getitem__") else out)
                if arr is None:
                    continue
                if name == "socket_pos" and arr.shape[0] >= 3:
                    # Observation terms are in env frame; convert to world.
                    goal_pos = arr[:3].copy()
                    if origins is not None:
                        goal_pos = goal_pos + origins[:3]
                elif name == "socket_quat" and arr.shape[0] >= 4:
                    goal_quat = arr[:4].copy()
        except Exception:
            pass
        return goal_pos, goal_quat

    def print_step(
        self,
        obs,
        actions,
        step: int,
        success_info: dict | None,
        lstm_h: np.ndarray | None = None,
        lstm_c: np.ndarray | None = None,
    ):
        if not self.print_enabled or (step % self.print_every) != 0:
            return

        scene = self.base.scene
        print(f"\n================= STEP {step} =================")
        obs_arr = _flatten_policy_obs(obs)
        if obs_arr is not None:
            print(f"[obs] {_fmt(torch.as_tensor(obs_arr))}")
        else:
            print(f"[obs] {_fmt(_to_torch(obs))}")

        print(f"[actions] {_fmt(_to_torch(actions)[0] if hasattr(actions, '__getitem__') else actions)}")

        robot = scene["robot"]
        joint_pos = _to_torch(robot.data.joint_pos)[0]
        joint_names = _get_attr(robot.data, "joint_names") or _get_attr(robot, "joint_names")
        if joint_names is not None:
            pairs = {n: round(float(v), 4) for n, v in zip(list(joint_names), joint_pos.tolist())}
            print(f"[joint_pos] {pairs}")
        else:
            print(f"[joint_pos] {_fmt(joint_pos)}")

        try:
            target = _to_torch(robot.data.joint_pos_target)[0]
            num_arm = int(getattr(self.base.cfg, "num_arm_joints", 7) or 7)
            print(f"[target_joint_pos] {_fmt(target[:num_arm])}")
        except Exception:
            pass

        if lstm_h is not None:
            rnn = _pack_rnn_state(lstm_h, lstm_c)
            if rnn is not None:
                print(f"[rnn] dim={rnn.size} ||rnn||={np.linalg.norm(rnn):.4f}")
            else:
                h = np.asarray(lstm_h, dtype=np.float64).reshape(-1)
                print(f"[lstm] h dim={h.size} ||h||={np.linalg.norm(h):.4f}")

        eef_name = getattr(self.base.cfg, "end_effector_body_name", "flange")
        body_names = _get_attr(robot.data, "body_names") or _get_attr(robot, "body_names")
        if body_names is not None and eef_name in list(body_names):
            idx = list(body_names).index(eef_name)
            print(
                f"[eef '{eef_name}'] pos={_fmt(_to_torch(robot.data.body_pos_w)[0, idx])} "
                f"quat(xyzw)={_fmt(_to_torch(robot.data.body_quat_w)[0, idx])}"
            )

        for candidates in (["dp_plug"], ["dp_socket"]):
            asset, name = _get_asset(scene, candidates)
            if asset is None:
                continue
            print(
                f"[{name}] pos={_fmt(_to_torch(asset.data.root_pos_w)[0])} "
                f"quat(xyzw)={_fmt(_to_torch(asset.data.root_quat_w)[0])}"
            )

        if success_info is not None:
            print(SuccessTracker.format(success_info))

    def print_success(self, step: int, success_info: dict | None):
        if not self.print_enabled or success_info is None or (step % self.print_every) != 0:
            return
        print(SuccessTracker.format(success_info))

    def begin_step(
        self,
        obs,
        actions,
        step: int,
        episode: int,
        lstm_h: np.ndarray | None = None,
        lstm_c: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """Snapshot pre-step state. Call after ``policy(obs)`` but before ``env.step``.

        Keeps ``obs_*`` and the action columns on the same row time-aligned (the action is
        the one the policy produced from that observation), matching the real-robot logger.
        ``lstm_h`` / ``lstm_c`` are packed into ``rnn_*`` (deploy recurrent-state layout).
        """
        if self.csv_path is None:
            return None
        return self._collect_row(obs, actions, step, episode, lstm_h=lstm_h, lstm_c=lstm_c)

    def end_step(self, row: dict[str, Any] | None, success_info: dict | None):
        """Attach post-step command / metrics to a row from :meth:`begin_step`."""
        if row is None:
            return
        robot = self.base.scene["robot"]
        num_arm = int(getattr(self.base.cfg, "num_arm_joints", 7) or 7)
        joint_names = list(_get_attr(robot.data, "joint_names") or _get_attr(robot, "joint_names") or [])
        try:
            # Absolute joint command produced by applying the logged action.
            target = _as_numpy_1d(_to_torch(robot.data.joint_pos_target)[0])
        except Exception:
            target = None
        if target is not None:
            arm_n = min(num_arm, len(target))
            for i in range(arm_n):
                # action_* = absolute policy target (matches real DisplayPort logs).
                row[f"action_{i}"] = float(target[i])
                # Sim-only alias used by some plot helpers.
                row[f"target_joint_pos_{i}"] = float(target[i])
            # blend_cmd_* = command applied this cycle. No safety controller in sim,
            # so this equals the absolute policy target (real blend_cmd may differ).
            names = joint_names[:arm_n] if joint_names else [f"joint{i + 1}" for i in range(arm_n)]
            for name, val in zip(names, target[:arm_n].tolist()):
                row[f"blend_cmd_{name}"] = float(val)

        if success_info is not None:
            row["instant_success_rate"] = success_info["instant_success_rate"]
            row["pos_error_m"] = success_info["pos_error_m"]
            row["keypoint_dist_m"] = success_info["keypoint_dist_m"]
            row["episode_success_rate"] = success_info["episode_success_rate"]
            row["num_episodes"] = success_info["num_episodes"]

        self.rows.append(row)

    def flush(self):
        if self.csv_path is None or not self.rows:
            return
        # Prefer reference column order from policy_io_csv_schema.md, then extras.
        preferred = [
            "wall_time",
            "ros_time",
            "step",
            "episode",
        ]
        keys: list[str] = []
        seen = set()
        for preferred_key in preferred:
            if any(preferred_key in row for row in self.rows):
                keys.append(preferred_key)
                seen.add(preferred_key)
        # Group remaining keys to mirror the reference column order. ``action_`` must be
        # matched exactly so it does not also swallow ``action_raw_``.
        groups = (
            lambda k: k.startswith("joint_"),
            lambda k: re.fullmatch(r"obs_\d+", k) is not None,
            lambda k: re.fullmatch(r"action_\d+", k) is not None,
            lambda k: re.fullmatch(r"rnn_\d+", k) is not None,
            lambda k: k.startswith("goal_"),
            lambda k: k.startswith("eef_"),
            lambda k: k.startswith("blend_cmd_"),
            # Sim-only extras (after the real schema groups).
            lambda k: k.startswith("action_raw_"),
            lambda k: k.startswith("target_joint_pos_"),
            lambda k: k.startswith("safety_cmd_"),
        )
        for matches in groups:
            for row in self.rows:
                for k in row:
                    if k not in seen and matches(k):
                        keys.append(k)
                        seen.add(k)
        for row in self.rows:
            for k in row:
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        self._fieldnames = keys
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"[INFO] Wrote {len(self.rows)} steps to {self.csv_path}")

def _disable_randomization(env_cfg):
    """Make resets deterministic for debugging / replay."""
    zero_range = {
        "x": [0.0, 0.0],
        "y": [0.0, 0.0],
        "z": [0.0, 0.0],
        "roll": [0.0, 0.0],
        "pitch": [0.0, 0.0],
        "yaw": [0.0, 0.0],
    }
    if hasattr(env_cfg.events, "randomize_socket_pose"):
        env_cfg.events.randomize_socket_pose.params["pose_range"] = zero_range
    if hasattr(env_cfg.events, "randomize_plug_pose"):
        env_cfg.events.randomize_plug_pose.params["pose_range"] = zero_range
    if hasattr(env_cfg.events, "reset_plug_curriculum"):
        env_cfg.events.reset_plug_curriculum.params["at_goal_prob"] = 0.0
        env_cfg.events.reset_plug_curriculum.params["normal_pose_range"] = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
        }
    # Disable observation corruption for deterministic inference.
    if hasattr(env_cfg.observations, "policy") and hasattr(env_cfg.observations.policy, "enable_corruption"):
        env_cfg.observations.policy.enable_corruption = False


def _apply_robot_joint_override(env_cfg):
    if args_cli.robot_joint_pos is None:
        return

    overrides = _parse_robot_joint_pos(args_cli.robot_joint_pos, _arm_joint_names(env_cfg))
    robot_cfg = env_cfg.scene.robot
    joint_pos = dict(robot_cfg.init_state.joint_pos)
    joint_pos.update(overrides)
    robot_cfg.init_state.joint_pos = joint_pos
    print(f"[INFO] Manual robot joint override applied: {overrides}")

    grasp_event = getattr(env_cfg.events, "set_robot_to_grasp_pose", None)
    if grasp_event is None:
        print(
            "[WARNING] No 'set_robot_to_grasp_pose' event found; plug will spawn at its configured pose, "
            "not necessarily in the gripper."
        )
        return

    params = grasp_event.params
    env_cfg.events.set_robot_to_grasp_pose = None
    env_cfg.events.place_plug_in_grasp = EventTerm(
        func=place_plug_at_grasp_pose,
        mode="reset",
        params={
            "robot_asset_cfg": params.get("robot_asset_cfg", SceneEntityCfg("robot")),
            "target_object_name": params["target_object_name"],
            "end_effector_body_name": params["end_effector_body_name"],
            "grasp_offset": params.get("grasp_offset", [0.0, 0.0, 0.0]),
            "grasp_rot_offset": params["grasp_rot_offset"],
            "gripper_joint_setter_func": params["gripper_joint_setter_func"],
            "hand_hold_width": env_cfg.hand_hold_width,
            "hand_close_width": env_cfg.hand_close_width,
            "num_arm_joints": params["num_arm_joints"],
        },
    )
    print("[INFO] Replaced IK grasp reset with place_plug_in_grasp (keeps manual joint pose).")


def _apply_socket_override(env_cfg) -> dict[str, Any]:
    """Apply actual socket/plug init poses. Returns resolved pose metadata for logging."""
    socket_cfg = env_cfg.scene.dp_socket
    plug_cfg = env_cfg.scene.dp_plug
    socket_rot = tuple(args_cli.socket_rot) if args_cli.socket_rot is not None else tuple(socket_cfg.init_state.rot)
    meta: dict[str, Any] = {"socket_rot_xyzw": list(socket_rot)}

    if args_cli.socket_pos is not None:
        geometry_pos = tuple(args_cli.socket_pos)
        socket_root = compute_socket_root(geometry_pos, socket_rot)
        plug_root, plug_rot = compute_plug_pose(geometry_pos, socket_rot, z_clearance=args_cli.z_clearance)
        socket_cfg.init_state.pos = socket_root
        socket_cfg.init_state.rot = socket_rot
        plug_cfg.init_state.pos = plug_root
        plug_cfg.init_state.rot = plug_rot
        if hasattr(env_cfg, "fixed_asset_init_pos_center"):
            env_cfg.fixed_asset_init_pos_center = list(geometry_pos)
        meta.update(
            {
                "socket_geometry_pos": list(geometry_pos),
                "socket_root_pos": list(socket_root),
                "plug_root_pos": list(plug_root),
                "plug_rot_xyzw": list(plug_rot),
                "z_clearance": args_cli.z_clearance,
            }
        )
        print("[INFO] Actual socket override applied:")
        print(f"       socket geometry pos = {tuple(round(v, 4) for v in geometry_pos)}")
        print(f"       socket root pos     = {tuple(round(v, 4) for v in socket_root)}")
        print(f"       socket rot (xyzw)   = {tuple(round(v, 4) for v in socket_rot)}")
        print(f"       plug root pos       = {tuple(round(v, 4) for v in plug_root)}")
    else:
        meta["socket_geometry_pos"] = "env_default"
        print("[INFO] No --socket_pos given; using env-configured socket pose (randomization disabled).")

    _disable_randomization(env_cfg)
    return meta


def _apply_observed_socket_override(env) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if args_cli.observed_socket_pos is None and args_cli.observed_socket_rot is None:
        return meta

    base = env.unwrapped
    obs_mgr = base.observation_manager
    term_names = obs_mgr.active_terms["policy"]
    term_cfgs = obs_mgr._group_obs_term_cfgs["policy"]

    for name, term_cfg in zip(term_names, term_cfgs):
        if name == "socket_pos" and args_cli.observed_socket_pos is not None:
            term_cfg.func = _FixedObservedSocketPos(base, args_cli.observed_socket_pos)
            term_cfg.noise = None
            meta["observed_socket_pos"] = list(args_cli.observed_socket_pos)
        elif name == "socket_quat" and args_cli.observed_socket_rot is not None:
            term_cfg.func = _FixedObservedSocketQuat(base, args_cli.observed_socket_rot)
            meta["observed_socket_rot"] = list(args_cli.observed_socket_rot)

    print("[INFO] Observed (policy-visible) socket pose override applied:")
    if args_cli.observed_socket_pos is not None:
        print(f"       observed socket geometry pos = {tuple(round(v, 4) for v in args_cli.observed_socket_pos)}")
    if args_cli.observed_socket_rot is not None:
        print(f"       observed socket rot (xyzw)   = {tuple(round(v, 4) for v in args_cli.observed_socket_rot)}")
    return meta


def _resolve_log_dir(resume_path: str | None) -> Path | None:
    if args_cli.no_log_file:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args_cli.log_dir is not None:
        return Path(args_cli.log_dir).expanduser().resolve() / stamp
    if resume_path is not None:
        return Path(os.path.dirname(resume_path)) / "inference_logs" / stamp
    return Path("logs") / "dp_replay" / stamp


def _write_run_metadata(
    log_dir: Path,
    resume_path: str | None,
    pose_meta: dict[str, Any],
    replay: ReplayTrajectory | None = None,
    leapp_model: str | None = None,
):
    payload = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "leapp_model": leapp_model,
        "replay_csv": str(replay.path) if replay is not None else None,
        "replay_source": replay.source if replay is not None else None,
        "replay_num_steps": replay.num_steps if replay is not None else None,
        "num_envs": args_cli.num_envs,
        "max_steps": args_cli.max_steps,
        "num_episodes": args_cli.num_episodes,
        "seed": args_cli.seed,
        "robot_joint_pos": args_cli.robot_joint_pos,
        "socket_pos": args_cli.socket_pos,
        "socket_rot": args_cli.socket_rot,
        "observed_socket_pos": args_cli.observed_socket_pos,
        "observed_socket_rot": args_cli.observed_socket_rot,
        "z_clearance": args_cli.z_clearance,
        "resolved_poses": pose_meta,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    path = log_dir / "run_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Wrote run config to {path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run DisplayPort insertion inference with controllable poses and logging."""
    with launch_simulation(env_cfg, args_cli):
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "").replace("-ROS-Inference", "")
        replay_mode = args_cli.replay_csv is not None
        leapp_mode = args_cli.leapp_model is not None

        if leapp_mode and replay_mode:
            raise ValueError("Use only one of --leapp_model and --replay_csv.")
        if leapp_mode and (args_cli.checkpoint or args_cli.use_pretrained_checkpoint):
            print("[WARNING] --leapp_model set; ignoring --checkpoint / --use_pretrained_checkpoint.")
        if replay_mode and (args_cli.checkpoint or args_cli.use_pretrained_checkpoint):
            print("[WARNING] --replay_csv set; checkpoint will not be used for actions.")

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        replay_traj: ReplayTrajectory | None = None
        if replay_mode:
            num_arm = int(getattr(env_cfg, "num_arm_joints", 7) or 7)
            replay_traj = load_replay_trajectory(args_cli.replay_csv, num_arm_joints=num_arm)
            _seed_cli_from_replay(replay_traj)
            # Avoid mid-replay resets wiping the open-loop track.
            if hasattr(env_cfg, "episode_length_s"):
                env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), replay_traj.num_steps * 10.0)

        _apply_robot_joint_override(env_cfg)
        pose_meta = _apply_socket_override(env_cfg)

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path: str | None = None
        leapp_yaml: Path | None = None
        if leapp_mode:
            leapp_yaml = resolve_leapp_model_yaml(args_cli.leapp_model)
            resume_path = str(leapp_yaml)
            print(f"[INFO] Using LEAPP model: {leapp_yaml}")
        elif args_cli.use_pretrained_checkpoint:
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] No pre-trained checkpoint is currently available for this task.")
                return
        elif args_cli.checkpoint:
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            resume_path = retrieve_file_path(args_cli.checkpoint)
        elif not replay_mode:
            print(f"[INFO] Loading experiment from directory: {log_root_path}")
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        else:
            print("[INFO] Replay mode: no checkpoint required (policy will not be loaded).")

        if resume_path is not None:
            env_cfg.log_dir = os.path.dirname(resume_path)
        log_dir = _resolve_log_dir(resume_path)

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        observed_meta = _apply_observed_socket_override(env)
        pose_meta.update(observed_meta)
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            _write_run_metadata(
                log_dir,
                resume_path if not leapp_mode else None,
                pose_meta,
                replay=replay_traj,
                leapp_model=str(leapp_yaml) if leapp_yaml is not None else None,
            )

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        policy = None
        policy_nn = None
        leapp_policy: LeappDisplayportPolicy | None = None
        if leapp_mode:
            assert leapp_yaml is not None
            leapp_policy = LeappDisplayportPolicy(env, leapp_yaml, clip_actions=agent_cfg.clip_actions)
            policy = leapp_policy
        elif not replay_mode:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            if version.parse(installed_version) < version.parse("4.0.0"):
                if version.parse(installed_version) >= version.parse("2.3.0"):
                    policy_nn = runner.alg.policy
                else:
                    policy_nn = runner.alg.actor_critic
        else:
            assert replay_traj is not None
            print(
                f"[INFO] Open-loop replay of {replay_traj.num_steps} absolute joint targets "
                f"(source={replay_traj.source}, action_scale={_action_scale(env):.4f})."
            )

        dt = env.unwrapped.step_dt
        logger = InferenceLogger(
            env,
            log_dir=log_dir,
            print_enabled=not args_cli.no_print,
            print_every=args_cli.print_every,
        )
        success_tracker = SuccessTracker(env)

        obs = env.get_observations()
        step_count = 0
        episode_count = 0
        lstm_logged_once = False
        try:
            while True:
                start_time = time.time()
                with torch.inference_mode():
                    if replay_mode:
                        assert replay_traj is not None
                        if step_count >= replay_traj.num_steps:
                            print(f"[INFO] Reached end of replay CSV ({replay_traj.num_steps} steps).")
                            break
                        actions = _absolute_targets_to_relative_actions(
                            env,
                            replay_traj.targets[step_count],
                            clip_actions=agent_cfg.clip_actions,
                        )
                    else:
                        actions = policy(obs)

                    # Post-forward LSTM state (None in replay mode / non-recurrent policies).
                    if replay_mode:
                        lstm_h, lstm_c = None, None
                    elif leapp_policy is not None:
                        lstm_h, lstm_c = leapp_policy.extract_lstm_state()
                    else:
                        lstm_h, lstm_c = _extract_lstm_hidden_state(policy)
                        if lstm_h is None and policy_nn is not None:
                            lstm_h, lstm_c = _extract_lstm_hidden_state(policy_nn)

                    if not replay_mode and not lstm_logged_once:
                        if lstm_h is not None:
                            rnn = _pack_rnn_state(lstm_h, lstm_c)
                            dim = int(rnn.size) if rnn is not None else int(np.asarray(lstm_h).size)
                            print(f"[INFO] Logging recurrent state as rnn_* (dim={dim}).")
                        else:
                            print(
                                "[WARNING] Policy has no accessible LSTM hidden state; "
                                "rnn_* will not be written."
                            )
                        lstm_logged_once = True

                    # Snapshot the state the policy/replay actually saw, before the action lands.
                    pending_row = logger.begin_step(
                        obs, actions, step_count, episode_count, lstm_h=lstm_h, lstm_c=lstm_c
                    )
                    logger.print_step(obs, actions, step_count, None, lstm_h=lstm_h, lstm_c=lstm_c)

                    obs, _, dones, _ = env.step(actions)
                    success_info = success_tracker.update(dones)
                    dones_b = torch.as_tensor(dones).bool().view(-1)
                    if bool(dones_b.any()):
                        episode_count += int(dones_b.sum().item())

                    logger.end_step(pending_row, success_info)
                    logger.print_success(step_count, success_info)
                    step_count += 1

                    if leapp_policy is not None:
                        leapp_policy.reset(dones)
                    elif not replay_mode:
                        if version.parse(installed_version) >= version.parse("4.0.0"):
                            policy.reset(dones)
                        else:
                            policy_nn.reset(dones)
                    elif bool(dones_b.any()):
                        print(
                            "[WARNING] Env reset during open-loop replay; subsequent CSV targets "
                            "no longer match the robot state. Stopping."
                        )
                        break

                if args_cli.max_steps is not None and step_count >= args_cli.max_steps:
                    print(f"[INFO] Reached --max_steps={args_cli.max_steps}.")
                    break
                if args_cli.num_episodes is not None and episode_count >= args_cli.num_episodes:
                    print(f"[INFO] Reached --num_episodes={args_cli.num_episodes}.")
                    break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            logger.flush()
            if success_tracker.enabled and success_tracker.num_episodes > 0:
                print(
                    f"[INFO] Final episode success rate: "
                    f"{success_tracker.num_success_episodes / success_tracker.num_episodes:.3f} "
                    f"({success_tracker.num_success_episodes}/{success_tracker.num_episodes})"
                )
            env.close()


if __name__ == "__main__":
    main()
