# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate Franka Pour reset candidates and write the production reset dataset.

Every candidate is restored into the real task and simulated under a neutral arm command. Grasping
states receive a close-gripper command. The validator rejects non-finite, out-of-bounds, or
particle-workspace failures and requires grasping states to retain bilateral contact after settling.
The output remains balanced across grasp side and broad/near-pour strata.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.contrib.franka_pour.mdp.terminations import source_grasp_milestones
from isaaclab_tasks.contrib.franka_pour.pour_env import FrankaPourResetDatasetValidationEnv
from isaaclab_tasks.contrib.franka_pour.reset_dataset_generator import (
    FRANKA_POUR_RESET_DATASET_TASK_ID,
    GRASPING_CATEGORY,
    NON_GRASPING_CATEGORY,
    RESET_DATASET_GRASPING_COUNT,
    RESET_DATASET_NEAR_POUR_COUNT,
    RESET_DATASET_NON_GRASPING_COUNT,
    FrankaPourResetDatasetGeneratorCfg,
    build_reset_dataset_payload,
    normalize_grasp_objectives,
    save_reset_dataset,
    select_production_reset_rows,
    validate_reset_dataset,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT = _REPO_ROOT / "datasets/franka_pour/reset_dataset_candidates.pt"
_DEFAULT_OUTPUT = _REPO_ROOT / "datasets/franka_pour/reset_dataset.pt"
_DEFAULT_REPORT = _REPO_ROOT / "datasets/franka_pour/reset_dataset_validation.json"
_FAILURE_NAMES = (
    "nonfinite",
    "extreme_rigid_state",
    "particle_out_of_bounds",
    "grasp_lost",
    "near_pour_missed_target",
    "near_pour_excessive_spill",
)


def _never_terminate(
    env,
    dwell_time_s: float | None = None,
    min_lift_height: float | None = None,
    max_tcp_distance: float | None = None,
    max_gripper_width_error: float | None = None,
    max_gripper_command: float | None = None,
    terminate: bool | None = None,
) -> torch.Tensor:
    """Keep validation rows alive for the complete settling window."""
    del (
        dwell_time_s,
        min_lift_height,
        max_tcp_distance,
        max_gripper_width_error,
        max_gripper_command,
        terminate,
    )
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def _parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=FRANKA_POUR_RESET_DATASET_TASK_ID)
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help="Candidate reset dataset.")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="Validated production dataset.")
    parser.add_argument("--report", type=Path, default=_DEFAULT_REPORT, help="Compact JSON validation report.")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument(
        "--steps",
        type=int,
        default=60,
        help="Policy steps simulated per state; 60 is two seconds at 30 Hz.",
    )
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=8,
        help="Initial steps excluded from grasp-retention checks.",
    )
    parser.add_argument(
        "--failure_dwell_steps",
        type=int,
        default=2,
        help="Consecutive failed contact checks required to reject a grasp.",
    )
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0:
        parser.error("num_envs and steps must be positive.")
    if args.settle_steps < 0 or args.settle_steps >= args.steps:
        parser.error("settle_steps must lie in [0, steps).")
    if args.failure_dwell_steps <= 0:
        parser.error("failure_dwell_steps must be positive.")
    return args


def _physical_failure_bits(env) -> torch.Tensor:
    """Return bit-packed physical failures for every environment."""
    bits = (~env.state_finite()).to(torch.uint8)
    bits |= (~env.rigid_state_in_bounds()).to(torch.uint8) << 1
    bits |= (~env.particles_in_workspace()).to(torch.uint8) << 2
    return bits


def _build_validated_payload(
    payload: dict,
    keep: torch.Tensor,
    *,
    steps: int,
    settle_steps: int,
    failure_dwell_steps: int,
    failure_counts: dict[str, int],
    balance_trimmed: int,
) -> dict:
    """Build a self-consistent dataset from the dynamically valid rows."""
    states = {name: value[keep].clone() for name, value in payload["states"].items()}
    grasping = states["category"] == GRASPING_CATEGORY
    states["objective"][grasping] = normalize_grasp_objectives(states["objective_raw"][grasping])

    sampler_cfg = replace(
        FrankaPourResetDatasetGeneratorCfg(**payload["metadata"]["sampler_cfg"]),
        grasping_count=RESET_DATASET_GRASPING_COUNT,
        non_grasping_count=RESET_DATASET_NON_GRASPING_COUNT,
        near_pour_grasp_count=RESET_DATASET_NEAR_POUR_COUNT,
    )
    metadata = dict(payload["metadata"])
    metadata["objective_raw_min_max"] = torch.stack(
        (states["objective_raw"][grasping].min(), states["objective_raw"][grasping].max())
    )
    metadata["dynamic_validation"] = {
        "source_content_sha256": payload["content_sha256"],
        "steps": steps,
        "settle_steps": settle_steps,
        "failure_dwell_steps": failure_dwell_steps,
        "failure_counts": failure_counts,
        "balance_trimmed": balance_trimmed,
    }
    return build_reset_dataset_payload(
        states,
        payload["particle_layouts"]["local_position"],
        metadata,
        sampler_cfg,
    )


def main() -> None:
    """Validate every candidate and write the filtered production dataset."""
    args = _parse_args()
    input_path = args.input.expanduser().resolve()
    payload = torch.load(input_path, map_location="cpu", weights_only=True)
    validate_reset_dataset(payload)
    row_count = int(payload["metadata"]["state_count"])
    env_count = min(args.num_envs, row_count)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=env_count)
    env_cfg.seed = int(payload["metadata"]["seed"])
    env_cfg.reset_dataset_path = str(input_path)
    env_cfg.reset_dataset_content_sha256 = payload["content_sha256"]
    env_cfg.curriculum_freeze = True
    for term_name in (
        "failure",
        "extreme_rigid_state",
        "lost_grasp",
        "spill",
        "particle_out_of_bounds",
        "success",
        "time_out",
    ):
        getattr(env_cfg.terminations, term_name).func = _never_terminate

    failures = torch.zeros(row_count, dtype=torch.uint8)
    with launch_simulation(env_cfg, args):
        env = FrankaPourResetDatasetValidationEnv(env_cfg)
        try:
            task = env
            task.sim._app_control_on_stop_handle = None
            actions = torch.zeros((env_count, task.action_manager.total_action_dim), device=task.device)
            gripper_term_index = task.action_manager.active_terms.index("gripper_action")
            if task.action_manager.action_term_dim[gripper_term_index] != 1:
                raise RuntimeError("Reset validation requires a one-dimensional gripper action.")
            gripper_action_index = sum(task.action_manager.action_term_dim[:gripper_term_index])
            category = payload["states"]["category"].to(task.device)
            grasp_region = payload["states"]["grasp_region"].to(task.device)

            for first_row in range(0, row_count, env_count):
                active_count = min(env_count, row_count - first_row)
                rows = torch.arange(first_row, first_row + active_count, device=task.device)
                padded_rows = rows
                if active_count < env_count:
                    padded_rows = torch.cat((rows, rows[:1].expand(env_count - active_count)))
                task._forced_reset_dataset_row.copy_(padded_rows)
                env.reset()
                if not bool(torch.equal(task.reset_dataset_row_id, padded_rows)):
                    raise RuntimeError("Forced reset rows did not restore the expected dataset entries.")

                actions.zero_()
                grasping = category[padded_rows] == GRASPING_CATEGORY
                near_pour = grasping & (grasp_region[padded_rows] == 1)
                actions[:, gripper_action_index] = torch.where(
                    grasping,
                    actions.new_tensor(-1.0),
                    actions.new_tensor(1.0),
                )
                batch_failures = _physical_failure_bits(task)
                loss_streak = torch.zeros(env_count, device=task.device, dtype=torch.long)
                near_pour_succeeded = ~near_pour
                near_pour_spilled = torch.zeros(env_count, device=task.device, dtype=torch.bool)
                for step in range(args.steps):
                    env.step(actions)
                    batch_failures |= _physical_failure_bits(task)
                    target_fraction = task.count_in_target() / max(task.num_particles, 1)
                    spill_fraction = task.spilled_fraction()
                    within_spill_limit = spill_fraction <= float(task.cfg.max_spill_fraction)
                    near_pour_succeeded |= (
                        near_pour & (target_fraction >= float(task.cfg.pour_target_frac)) & within_spill_limit
                    )
                    near_pour_spilled |= near_pour & ~within_spill_limit
                    if step < args.settle_steps:
                        continue
                    preloaded = source_grasp_milestones(
                        task,
                        min_lift_height=task.cfg.success_min_lift_height,
                        max_tcp_distance=task.cfg.success_max_tcp_distance,
                        max_gripper_width_error=task.cfg.success_max_gripper_width_error,
                        max_gripper_command=task.cfg._resolved_success_max_gripper_command(),
                    )[1]
                    loss_streak = torch.where(
                        grasping & preloaded,
                        torch.zeros_like(loss_streak),
                        torch.where(grasping, loss_streak + 1, torch.zeros_like(loss_streak)),
                    )
                    batch_failures |= (loss_streak >= args.failure_dwell_steps).to(torch.uint8) << 3

                final_preloaded = source_grasp_milestones(
                    task,
                    min_lift_height=task.cfg.success_min_lift_height,
                    max_tcp_distance=task.cfg.success_max_tcp_distance,
                    max_gripper_width_error=task.cfg.success_max_gripper_width_error,
                    max_gripper_command=task.cfg._resolved_success_max_gripper_command(),
                )[1]
                batch_failures |= (grasping & ~final_preloaded).to(torch.uint8) << 3
                batch_failures |= (~near_pour_succeeded).to(torch.uint8) << 4
                batch_failures |= near_pour_spilled.to(torch.uint8) << 5
                failures[first_row : first_row + active_count] = batch_failures[:active_count].cpu()
                passed = int((batch_failures[:active_count] == 0).sum())
                print(f"[RESET VALIDATION] rows {first_row}:{first_row + active_count} passed {passed}/{active_count}")
        finally:
            env.close()

    valid = failures == 0
    keep, balance_trimmed = select_production_reset_rows(payload["states"], valid)
    failure_counts = {name: int(((failures & (1 << bit)) != 0).sum()) for bit, name in enumerate(_FAILURE_NAMES)}
    validated_payload = _build_validated_payload(
        payload,
        keep,
        steps=args.steps,
        settle_steps=args.settle_steps,
        failure_dwell_steps=args.failure_dwell_steps,
        failure_counts=failure_counts,
        balance_trimmed=int(balance_trimmed.sum()),
    )
    output_path = args.output.expanduser().resolve()
    save_reset_dataset(validated_payload, output_path)

    category = payload["states"]["category"]
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "source_content_sha256": payload["content_sha256"],
        "content_sha256": validated_payload["content_sha256"],
        "candidate_count": row_count,
        "dynamically_valid_count": int(valid.sum()),
        "retained_count": int(keep.sum()),
        "balance_trimmed_count": int(balance_trimmed.sum()),
        "failure_counts": failure_counts,
        "category_counts": {
            "non_grasping": int((keep & (category == NON_GRASPING_CATEGORY)).sum()),
            "grasping": int((keep & (category == GRASPING_CATEGORY)).sum()),
        },
    }
    report_path = args.report.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[RESET VALIDATION] retained {report['retained_count']}/{row_count}; "
        f"failures={failure_counts}; dataset={output_path}; report={report_path}"
    )


if __name__ == "__main__":
    main()
