# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trace the per-step state of the environment that blows up on rough terrain.

Only one environment out of hundreds goes non-finite, and a different one each run, so a
video of the whole field is useless and the culprit cannot be picked in advance. This
wrapper instead keeps a ring buffer of cheap per-environment scalars, catches the NaN
abort, and prints the trailing history of the offending environment.

The point is to measure how many steps of warning there are before the blow-up: that lead
time decides whether an armed-rendering capture of the final moments is feasible at all.

Environment variables:
    WATCH_STEPS: trailing steps to retain and print (default 40).

Every argument is forwarded verbatim to the training CLI.
"""

import os
import sys
from collections import deque

TAG = "NANWATCH:"
WATCH_STEPS = int(os.environ.get("WATCH_STEPS", "40"))

_STATE: dict = {}
_HISTORY: deque = deque(maxlen=WATCH_STEPS)


def _sample(env) -> None:
    """Append one step of per-environment scalars to the ring buffer."""
    import torch

    robot = env.scene["robot"]
    data = robot.data
    row = {}
    root = data.root_state_w
    row["height"] = root[:, 2].detach().clone()
    row["lin_vel"] = torch.linalg.norm(data.root_lin_vel_w, dim=-1).detach().clone()
    row["ang_vel"] = torch.linalg.norm(data.root_ang_vel_w, dim=-1).detach().clone()
    row["joint_vel_max"] = data.joint_vel.abs().amax(dim=-1).detach().clone()
    row["joint_acc_max"] = data.joint_acc.abs().amax(dim=-1).detach().clone()
    try:
        forces = env.scene["contact_forces"].data.net_forces_w
        row["contact_max"] = torch.linalg.norm(forces, dim=-1).amax(dim=-1).detach().clone()
    except Exception:  # noqa: BLE001
        pass
    _HISTORY.append(row)


def _install() -> None:
    """Sample after every env step."""
    import isaaclab.envs.manager_based_rl_env as manager_based_rl_env

    original = manager_based_rl_env.ManagerBasedRLEnv.step

    def patched(self, action):
        out = original(self, action)
        _STATE["env"] = self
        try:
            _sample(self)
        except Exception as exc:  # noqa: BLE001
            if not _STATE.get("sample_error"):
                _STATE["sample_error"] = True
                print(f"{TAG} sampling failed: {type(exc).__name__}: {exc}", flush=True)
        return out

    manager_based_rl_env.ManagerBasedRLEnv.step = patched


def _report() -> None:
    """Identify the diverged environment and print its trailing history."""
    import torch

    env = _STATE.get("env")
    if env is None or not _HISTORY:
        print(f"{TAG} nothing recorded", flush=True)
        return

    robot = env.scene["robot"]
    root = robot.data.root_state_w
    bad = (~torch.isfinite(root)).any(dim=-1).nonzero(as_tuple=False).flatten().tolist()
    print(f"{TAG} non-finite envs: {bad[:16]} ({len(bad)}/{root.shape[0]})", flush=True)
    if not bad:
        return
    idx = bad[0]
    keys = list(_HISTORY[-1].keys())
    print(f"{TAG} trailing {len(_HISTORY)} steps for env {idx} (most recent last)", flush=True)
    print(f"{TAG} {'step':>6}  " + "  ".join(f"{k:>13}" for k in keys), flush=True)
    n = len(_HISTORY)
    for i, row in enumerate(_HISTORY):
        vals = []
        for k in keys:
            v = row[k][idx].item() if k in row else float("nan")
            vals.append(f"{v:13.4g}")
        print(f"{TAG} {i - n + 1:>6}  " + "  ".join(vals), flush=True)


_install()

from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402

try:
    code = run_train_cli(sys.argv[1:])
except BaseException as exc:  # noqa: BLE001
    print(f"{TAG} aborted: {type(exc).__name__}: {exc}", flush=True)
    _report()
    raise SystemExit(1)

raise SystemExit(code)
