#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expose one physical GPU as logical CUDA device zero before importing Isaac Lab."""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path


def _select_physical_device(local_rank: int) -> str:
    """Resolve the scheduler-visible device token assigned to a local rank."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices:
        return str(local_rank)
    device_tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
    if local_rank >= len(device_tokens):
        raise SystemExit(f"LOCAL_RANK {local_rank} exceeds CUDA_VISIBLE_DEVICES with {len(device_tokens)} entries")
    return device_tokens[local_rank]


def _seed_cache(source: str | None, destination: Path) -> None:
    """Copy a read-only warmup cache into one rank-private runtime directory."""
    if not source:
        return
    source_path = Path(source)
    if source_path.is_dir() and not destination.exists():
        shutil.copytree(source_path, destination)


def _rewrite_reset_replay_seed(arguments: list[str], global_rank: int) -> tuple[list[str], int | None]:
    """Offset the cable reset-bank seed while preserving all other Hydra overrides."""
    prefix = "env.commands.route.reset_replay.seed="
    rewritten = list(arguments)
    matched_indices = [index for index, argument in enumerate(rewritten) if argument.startswith(prefix)]
    if not matched_indices:
        return rewritten, None
    if len(matched_indices) != 1:
        raise SystemExit("Expected exactly one reset-replay seed override")
    index = matched_indices[0]
    try:
        base_seed = int(rewritten[index][len(prefix) :])
    except ValueError as error:
        raise SystemExit("Reset-replay seed override must be an integer") from error
    rank_seed = base_seed + global_rank
    rewritten[index] = f"{prefix}{rank_seed}"
    return rewritten, rank_seed


def main() -> None:
    """Configure rank-private CUDA/native caches and replace this process with training."""
    physical_local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ.get("RANK", "0"))
    physical_device = _select_physical_device(physical_local_rank)
    command_arguments = sys.argv[1:]
    reset_replay_seed = None
    if os.environ.get("OSMO_RESET_REPLAY_SEED_PER_RANK") == "1":
        command_arguments, reset_replay_seed = _rewrite_reset_replay_seed(command_arguments, global_rank)
        if reset_replay_seed is None:
            raise SystemExit("Per-rank reset seeding requires a reset-replay seed override")

    os.environ["CUDA_VISIBLE_DEVICES"] = physical_device
    os.environ["OSMO_ORIGINAL_LOCAL_RANK"] = str(physical_local_rank)
    # Each process now sees exactly one device. RSL-RL, Isaac Lab, Newton, and Warp must
    # all address it as cuda:0 while global RANK remains untouched for collectives.
    os.environ["LOCAL_RANK"] = "0"

    runtime_root = Path(os.environ.get("OSMO_RANK_RUNTIME_ROOT", "/tmp")) / f"isaaclab-rank-{global_rank}"
    runtime_root.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(runtime_root)
    os.environ["TMP"] = str(runtime_root)
    os.environ["TEMP"] = str(runtime_root)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(runtime_root / "torch-extensions")
    os.environ["XDG_CACHE_HOME"] = str(runtime_root / "xdg-cache")

    warp_cache = runtime_root / "warp-cache"
    omniclient_cache = runtime_root / "omniclient-cache"
    _seed_cache(os.environ.get("OSMO_WARP_CACHE_SEED"), warp_cache)
    _seed_cache(os.environ.get("OSMO_OMNICLIENT_CACHE_SEED"), omniclient_cache)
    warp_cache.mkdir(parents=True, exist_ok=True)
    omniclient_cache.mkdir(parents=True, exist_ok=True)
    os.environ["WARP_CACHE_PATH"] = str(warp_cache)
    os.environ["OMNICLIENT_HUB_CACHE_DIR"] = str(omniclient_cache)

    if global_rank != 0:
        os.environ["WANDB_MODE"] = "disabled"

    startup_stagger_s = float(os.environ.get("OSMO_RANK_STARTUP_STAGGER_S", "8"))
    startup_delay = startup_stagger_s * physical_local_rank
    print(
        "rank_device_map "
        f"global_rank={global_rank} physical_local_rank={physical_local_rank} "
        f"visible_device={physical_device} logical_device=cuda:0 "
        f"runtime_dir={runtime_root} startup_delay={startup_delay:.1f}s "
        f"reset_replay_seed={reset_replay_seed}",
        flush=True,
    )
    time.sleep(startup_delay)
    os.execv(sys.executable, [sys.executable, *command_arguments])


if __name__ == "__main__":
    main()
