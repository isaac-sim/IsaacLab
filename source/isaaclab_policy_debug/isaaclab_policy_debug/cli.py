# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path


def configure_policy_debug_args(args) -> None:
    """Validate conflicts and force the interactive Newton visualizer."""
    if not getattr(args, "policy_debug", None):
        return
    conflicts = []
    if getattr(args, "checkpoint", None):
        conflicts.append("--checkpoint")
    if getattr(args, "use_pretrained_checkpoint", False):
        conflicts.append("--use_pretrained_checkpoint")
    if getattr(args, "num_envs", None) is not None:
        conflicts.append("--num_envs")
    if getattr(args, "headless", False):
        conflicts.append("--headless")
    if conflicts:
        raise ValueError(f"--policy_debug cannot be combined with {', '.join(conflicts)}")
    if args.policy_debug_max_policies <= 0:
        raise ValueError("--policy_debug_max_policies must be greater than zero")

    run_dir = Path(args.policy_debug).expanduser().resolve()
    if not run_dir.is_dir():
        raise ValueError(f"--policy_debug folder does not exist or is not a directory: {run_dir}")
    try:
        next(run_dir.iterdir(), None)
    except OSError as exc:
        raise ValueError(f"--policy_debug folder is not readable: {run_dir}: {exc}") from exc
    args.policy_debug = str(run_dir)
    args.visualizer = ["newton_gl"]
    args.visualizer_explicit = True
    args.max_visible_envs = args.policy_debug_max_policies


def find_newton_visualizer(env):
    """Return the initialized Newton visualizer or raise an install hint."""
    for visualizer in env.unwrapped.sim.visualizers:
        if getattr(getattr(visualizer, "cfg", None), "visualizer_type", None) == "newton_gl":
            return visualizer
    raise RuntimeError(
        "Policy debug requires the Newton visualizer. Install it with "
        "./isaaclab.sh -i policy_debug and do not disable visualizers."
    )
