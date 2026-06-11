#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export reward and success-rate curves from TensorBoard logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

REWARD_TAG_CANDIDATES = (
    "Train/mean_reward",
    "Train/mean_reward/time",
    "Episode/rew_total",
    "Episode_Reward/total",
)
SUCCESS_TAG_CANDIDATES = (
    "Metrics/success_rate",
    "Episode/Metrics/success_rate",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a training checkpoint (e.g., logs/.../model_750.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder for curve images. Defaults to <checkpoint_dir>/training_curves.",
    )
    parser.add_argument(
        "--event-file",
        type=Path,
        default=None,
        help="Optional explicit TensorBoard events file. Defaults to latest events* in checkpoint directory.",
    )
    return parser.parse_args()


def _resolve_run_dir(checkpoint: Path) -> Path:
    checkpoint_path = checkpoint.expanduser().resolve()
    if checkpoint_path.is_dir():
        return checkpoint_path
    if checkpoint_path.is_file():
        return checkpoint_path.parent
    raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")


def _resolve_event_file(run_dir: Path, explicit_event_file: Path | None) -> Path:
    if explicit_event_file is not None:
        event_file = explicit_event_file.expanduser().resolve()
        if not event_file.is_file():
            raise FileNotFoundError(f"Event file does not exist: {event_file}")
        return event_file

    event_files = sorted(run_dir.glob("events*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in: {run_dir}")
    return event_files[-1]


def _pick_tag(candidates: tuple[str, ...], available_tags: list[str]) -> str | None:
    for tag in candidates:
        if tag in available_tags:
            return tag
    return None


def _plot_scalar(
    accumulator: event_accumulator.EventAccumulator,
    tag: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    events = accumulator.Scalars(tag)
    if not events:
        raise RuntimeError(f"No scalar data found for tag: {tag}")

    steps = [event.step for event in events]
    values = [event.value for event in events]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, values, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = _parse_args()

    run_dir = _resolve_run_dir(args.checkpoint)
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (run_dir / "training_curves")
    output_dir.mkdir(parents=True, exist_ok=True)

    event_file = _resolve_event_file(run_dir, args.event_file)
    accumulator = event_accumulator.EventAccumulator(str(event_file))
    accumulator.Reload()

    scalar_tags = accumulator.Tags().get("scalars", [])
    reward_tag = _pick_tag(REWARD_TAG_CANDIDATES, scalar_tags)
    success_tag = _pick_tag(SUCCESS_TAG_CANDIDATES, scalar_tags)

    if reward_tag is None and success_tag is None:
        available = ", ".join(sorted(scalar_tags)) if scalar_tags else "<none>"
        raise RuntimeError(
            "Could not find reward or success-rate tags in TensorBoard logs. "
            f"Available scalar tags: {available}"
        )

    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Event file: {event_file}")
    print(f"[INFO] Output directory: {output_dir}")

    if reward_tag is not None:
        reward_output = output_dir / "reward_curve.png"
        _plot_scalar(
            accumulator,
            reward_tag,
            title=f"Reward Curve ({reward_tag})",
            ylabel="Reward",
            output_path=reward_output,
        )
        print(f"[INFO] Saved reward curve: {reward_output}")
    else:
        print("[WARNING] Reward tag not found. Skipping reward curve export.")

    if success_tag is not None:
        success_output = output_dir / "success_rate_curve.png"
        _plot_scalar(
            accumulator,
            success_tag,
            title=f"Success Rate Curve ({success_tag})",
            ylabel="Success Rate",
            output_path=success_output,
        )
        print(f"[INFO] Saved success-rate curve: {success_output}")
    else:
        print("[WARNING] Success-rate tag not found. Skipping success-rate curve export.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
