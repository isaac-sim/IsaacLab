# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reproduces the ``ManagerBasedEnv.close`` memory leak fixed by upstream PR #5896.

Builds the ``Isaac-Cartpole-RGB-v0`` task in a tight construct→step→close loop,
then counts live :class:`gymnasium.spaces.Box` instances and sums the bytes they
hold in their ``low`` / ``high`` / ``bounded_below`` / ``bounded_above`` numpy
attributes. With a pre-fix ``close()`` (monkey-patched in by default) the count
stays at one env's worth across cycles — gymnasium's wrapper chain keeps the
last env reachable past close. With the post-PR ``close()`` (``--patched``) the
Boxes and their bounds arrays drop to zero on each close.

Run from the IsaacLab repository root::

    ./isaaclab.sh -p source/isaaclab/test/envs/check_manager_based_env_close_leak.py
    ./isaaclab.sh -p source/isaaclab/test/envs/check_manager_based_env_close_leak.py --patched

Pre-fix expected output (last cycle line)::

    after cycle 04  live Boxes=4   bounds-arrays held: 131.8 MB

Post-fix (``--patched``)::

    after cycle 04  live Boxes=0   bounds-arrays held:   0.0 MB
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--iterations", type=int, default=4, help="Build/close cycles.")
parser.add_argument("--num_envs", type=int, default=4, help="Parallel envs per cycle.")
parser.add_argument("--camera_width", type=int, default=1280, help="Camera image width.")
parser.add_argument("--camera_height", type=int, default=720, help="Camera image height.")
parser.add_argument(
    "--patched",
    action="store_true",
    help="Use the installed ManagerBasedEnv.close. Default monkey-patches it back to the"
    " pre-fix version so the leak reproduces even when the installed IsaacLab is patched.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

simulation_app = AppLauncher(args_cli).app


"""Rest everything follows."""

import gc
import os

import gymnasium as gym
import numpy as np
import psutil
import torch
from gymnasium.spaces import Box

import isaaclab.envs.manager_based_env as manager_based_env_module

import isaaclab_tasks  # noqa: F401 — registers Isaac-* env IDs with gymnasium
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def pre_fix_close(self):
    """Reproduce ``ManagerBasedEnv.close`` as it shipped before the fix in PR #5896."""
    if not self._is_closed:
        self.sim.stop()
        del self.viewport_camera_controller
        del self.action_manager
        del self.observation_manager
        del self.event_manager
        del self.recorder_manager
        del self.scene
        self.sim.clear_instance()
        if self._window is not None:
            self._window = None
        self._is_closed = True


if not args_cli.patched:
    manager_based_env_module.ManagerBasedEnv.close = pre_fix_close
    print("[setup] ManagerBasedEnv.close monkey-patched to PRE-FIX behaviour\n", flush=True)
else:
    print("[setup] Using installed ManagerBasedEnv.close\n", flush=True)


process = psutil.Process(os.getpid())


def snapshot(label: str) -> tuple[float, int, float]:
    """Report host RSS, live Box count, and total bytes pinned by Box bounds arrays.

    NumPy arrays are not gc-tracked (``gc.is_tracked(np.array(...)) is False``), so we
    walk the live Box objects (which are tracked) and sum the bytes of their bounds
    array attributes — the four allocations :class:`gymnasium.spaces.Box`'s ``__init__``
    makes via ``np.full`` and ``astype``.
    """
    gc.collect()
    rss_mb = process.memory_info().rss / 1024 / 1024
    live_boxes = 0
    bounds_bytes = 0
    for obj in gc.get_objects():
        try:
            if type(obj) is Box:
                live_boxes += 1
                for attr in ("low", "high", "bounded_below", "bounded_above"):
                    arr = getattr(obj, attr, None)
                    if isinstance(arr, np.ndarray):
                        bounds_bytes += arr.nbytes
        except Exception:
            continue
    bounds_mb = bounds_bytes / 1024 / 1024
    print(
        f"  {label:>16}  RSS={rss_mb:>8.1f} MB   live Boxes={live_boxes:>3}   bounds-arrays held: {bounds_mb:>7.1f} MB",
        flush=True,
    )
    return rss_mb, live_boxes, bounds_mb


def run_one_cycle(cycle: int) -> None:
    print(f"\n--- cycle {cycle:02d} ---", flush=True)
    env_cfg = parse_env_cfg("Isaac-Cartpole-RGB-v0", num_envs=args_cli.num_envs)
    # Default cartpole image is 100x100 — too small for the leak to be visible above
    # RSS noise. Bump to 720x1280 so each Box's bounds array is ~33 MB.
    env_cfg.scene.tiled_camera.width = args_cli.camera_width
    env_cfg.scene.tiled_camera.height = args_cli.camera_height
    env = gym.make("Isaac-Cartpole-RGB-v0", cfg=env_cfg)
    obs, _ = env.reset()
    action_shape = (env.unwrapped.num_envs, env.action_space.shape[-1])
    zero_action = torch.zeros(action_shape, device=env.unwrapped.device)
    for _ in range(3):
        obs, _, _, _, _ = env.step(zero_action)
    del obs, zero_action
    env.close()
    del env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


history = [snapshot("baseline")]
for cycle in range(1, args_cli.iterations + 1):
    run_one_cycle(cycle)
    history.append(snapshot(f"after cycle {cycle:02d}"))

print("\n=== per-cycle deltas ===", flush=True)
print(f"  {'cycle':>6}  {'ΔRSS (MB)':>11}  {'ΔBox count':>11}  {'Δbounds (MB)':>13}", flush=True)
for cycle_index in range(1, len(history)):
    d_rss = history[cycle_index][0] - history[cycle_index - 1][0]
    d_box = history[cycle_index][1] - history[cycle_index - 1][1]
    d_bounds = history[cycle_index][2] - history[cycle_index - 1][2]
    print(f"  {cycle_index:>6}  {d_rss:>+11.1f}  {d_box:>+11d}  {d_bounds:>+13.1f}", flush=True)

simulation_app.close()
