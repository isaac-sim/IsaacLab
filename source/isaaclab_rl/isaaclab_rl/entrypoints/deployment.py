# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in deployment export after one-time task initialization, before training reset."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.envs.common import _export_deployment_scene

if TYPE_CHECKING:
    from argparse import Namespace

    from isaaclab.envs import DirectMARLEnv, DirectRLEnv, ManagerBasedEnv


def export_training_scene(env: DirectRLEnv | DirectMARLEnv | ManagerBasedEnv, args_cli: Namespace) -> Path | None:
    """Export environment zero after the actual task constructor has returned.

    One-time prestartup/startup changes, including sampled values, are retained.
    This function does not reset the scene, apply defaults, step physics or consume RNGs.
    Only global rank zero writes into the current run's log directory.
    """
    rank_key = "JAX_RANK" if getattr(args_cli, "ml_framework", "torch").startswith("jax") else "RANK"
    if int(os.environ.get(rank_key, "0")) != 0:
        return None
    if not env.cfg.log_dir:
        raise ValueError("Deployment export requires the training run log_dir.")
    output = Path(env.cfg.log_dir).resolve() / "deployment.usda"
    return _export_deployment_scene(env.scene, str(output))
