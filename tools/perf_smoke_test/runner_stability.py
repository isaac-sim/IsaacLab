# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Qualify whether FPS variance on the CI runner pool is suitable for gating.

This report combines samples from independent runner allocations at one commit.
It ties the stability verdict to the perf-smoke oracle: a pool qualifies only
when complete, homogeneous evidence shows that FPS regressions greater than 10%
remain outside the effective four-MAD BLOCK band.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gate_config import DEFAULT_K_BLOCK, DEFAULT_K_WARN, MIN_BLOCK_REGRESSION_PCT
from gpu_identity import canonical_gpu_model, gpu_model_config_keys
from task_config import TaskConfig, load_tasks

STABLE = "STABLE_ENOUGH_TO_GATE"
INCONCLUSIVE = "INCONCLUSIVE_DO_NOT_GATE"
UNSTABLE = "UNSTABLE_DO_NOT_GATE"

DEFAULT_EXPECTED_ALLOCATIONS = 5
DEFAULT_MINIMUM_DISTINCT_RUNNERS = 3
DEFAULT_SAMPLES_PER_ALLOCATION = 3
MAX_EFFECTIVE_BLOCK_PCT = 10.0
MAX_POOLED_CV_PCT = 5.0
MAX_RUNNER_MEDIAN_DEVIATION_PCT = 5.0
MAX_SAMPLE_DEVIATION_PCT = 5.0

_FINGERPRINT_FIELDS = (
    "gpu_model",
    "target_branch",
    "commit_sha",
    "launch_config_hash",
    "benchmark_contract_hash",
    "runtime_contract_hash",
    "baseline_epoch",
)


@dataclass(frozen=True)
class StabilityBucket:
    """FPS stability evidence for one task/backend combination."""

    task_id: str
    backend: str
    verdict: str
    reasons: tuple[str, ...]
    fingerprint_count: int
    runtime_contract_hashes: tuple[str, ...]
    allocation_count: int
    runner_count: int
    sample_count: int
    median_fps: float | None
    mad_pct: float | None
    cv_pct: float | None
    range_pct: float | None
    max_sample_deviation_pct: float | None
    max_runner_median_deviation_pct: float | None
    configured_noise_floor_pct: float
    effective_noise_pct: float | None
    effective_warn_regression_pct: float | None
    effective_block_regression_pct: float | None
    allocation_medians_fps: dict[str, float]
    runner_medians_fps: dict[str, float]


def _percent(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _execution_id(record: dict[str, Any]) -> str:
    label = str(record.get("ci_run_label") or "").strip()
    if label:
        return label
    parts = (
        record.get("ci_run_id"),
        record.get("ci_run_attempt"),
        record.get("ci_job"),
        record.get("ci_runner_name"),
    )
    return ":".join(str(part or "unknown") for part in parts)


def _backend(record: dict[str, Any]) -> str:
    return str(record.get("backend") or record.get("backend_key") or "")


def _fingerprint(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record.get(field) for field in _FINGERPRINT_FIELDS)


def _valid_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for record in records:
        fps = record.get("fps")
        if (
            isinstance(fps, (int, float))
            and not isinstance(fps, bool)
            and math.isfinite(float(fps))
            and float(fps) > 0.0
            and record.get("task_id")
            and _backend(record)
        ):
            valid.append(record)
    return valid


def _noise_floor_for_task(task: TaskConfig, gpu_model: str) -> float:
    for key in gpu_model_config_keys(gpu_model):
        value = (task.noise_floor_pct or {}).get(key, {}).get(task.backend_key)
        if value is not None:
            return float(value)
    return 0.0


def select_backends(tasks: list[TaskConfig], backends: str) -> list[TaskConfig]:
    """Narrow the qualification scope to an explicit backend allowlist.

    Args:
        tasks: Every task/backend bucket configured for the gate.
        backends: Comma-separated ``backend_key`` allowlist; empty keeps all.

    Returns:
        The buckets whose backend is in the allowlist.

    Raises:
        ValueError: If a requested backend matches no configured bucket, which
            would otherwise silently shrink the scope and weaken the verdict.
    """
    requested = {backend.strip() for backend in backends.split(",") if backend.strip()}
    if not requested:
        return tasks
    configured = {task.backend_key for task in tasks}
    unknown = sorted(requested - configured)
    if unknown:
        raise ValueError(f"Unknown backend_key(s) {unknown}; configured backends are {sorted(configured)}")
    return [task for task in tasks if task.backend_key in requested]


def configured_scope(
    tasks: list[TaskConfig],
    gpu_model: str,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], float]]:
    """Return expected buckets and configured noise floors for a GPU model."""
    expected = {(task.task_id, task.backend_key) for task in tasks}
    noise_floors = {(task.task_id, task.backend_key): _noise_floor_for_task(task, gpu_model) for task in tasks}
    return expected, noise_floors


def _sample_coverage_reasons(
    bucket_records: list[dict[str, Any]],
    by_allocation: dict[str, list[dict[str, Any]]],
    by_runner: dict[str, list[dict[str, Any]]],
    *,
    expected_allocations: int,
    minimum_distinct_runners: int,
    samples_per_allocation: int,
) -> list[str]:
    """Return fail-closed sample, allocation, and runner coverage reasons."""
    reasons: list[str] = []
    if len(by_allocation) != expected_allocations:
        reasons.append(f"expected {expected_allocations} allocations, observed {len(by_allocation)}")
    if len(by_runner) < minimum_distinct_runners:
        reasons.append(f"expected at least {minimum_distinct_runners} distinct runner names, observed {len(by_runner)}")

    samples_without_runner = sum(1 for record in bucket_records if not str(record.get("ci_runner_name") or "").strip())
    if samples_without_runner:
        reasons.append(f"{samples_without_runner} samples are missing runner identity")

    required_samples = expected_allocations * samples_per_allocation
    if len(bucket_records) != required_samples:
        reasons.append(f"expected {required_samples} samples, observed {len(bucket_records)}")
    sample_ids = [str(record.get("sample_id") or "").strip() for record in bucket_records]
    samples_without_id = sum(not sample_id for sample_id in sample_ids)
    if samples_without_id:
        reasons.append(f"{samples_without_id} samples are missing sample identity")
    present_sample_ids = [sample_id for sample_id in sample_ids if sample_id]
    duplicate_sample_count = len(present_sample_ids) - len(set(present_sample_ids))
    if duplicate_sample_count:
        reasons.append(f"observed {duplicate_sample_count} duplicate sample identities")

    incomplete_allocations = sorted(
        allocation
        for allocation, allocation_records in by_allocation.items()
        if len(allocation_records) != samples_per_allocation
    )
    if incomplete_allocations:
        reasons.append(
            f"allocations without exactly {samples_per_allocation} samples: " + ", ".join(incomplete_allocations)
        )
    expected_sample_indexes = set(range(samples_per_allocation))
    allocations_with_wrong_sample_indexes = sorted(
        allocation
        for allocation, allocation_records in by_allocation.items()
        if {
            record.get("sample_index")
            for record in allocation_records
            if isinstance(record.get("sample_index"), int) and not isinstance(record.get("sample_index"), bool)
        }
        != expected_sample_indexes
    )
    if allocations_with_wrong_sample_indexes:
        reasons.append(
            f"allocations without sample indexes 0..{samples_per_allocation - 1}: "
            + ", ".join(allocations_with_wrong_sample_indexes)
        )
    allocations_without_one_runner = sorted(
        allocation
        for allocation, allocation_records in by_allocation.items()
        if len(
            {
                str(record.get("ci_runner_name") or "").strip()
                for record in allocation_records
                if str(record.get("ci_runner_name") or "").strip()
            }
        )
        != 1
    )
    if allocations_without_one_runner:
        reasons.append("allocations without exactly one runner identity: " + ", ".join(allocations_without_one_runner))
    return reasons


def evaluate_stability(
    records: list[dict[str, Any]],
    *,
    expected_buckets: set[tuple[str, str]],
    noise_floors: dict[tuple[str, str], float] | None = None,
    expected_allocations: int = DEFAULT_EXPECTED_ALLOCATIONS,
    minimum_distinct_runners: int = DEFAULT_MINIMUM_DISTINCT_RUNNERS,
    samples_per_allocation: int = DEFAULT_SAMPLES_PER_ALLOCATION,
    expected_gpu_model: str | None = None,
    expected_target_branch: str | None = None,
    expected_commit: str | None = None,
) -> tuple[str, list[StabilityBucket]]:
    """Evaluate complete runner-pool evidence against the qualification policy."""
    if not expected_buckets:
        return INCONCLUSIVE, []
    if expected_allocations <= 0:
        raise ValueError("expected_allocations must be positive")
    if samples_per_allocation <= 0:
        raise ValueError("samples_per_allocation must be positive")
    if minimum_distinct_runners <= 0 or minimum_distinct_runners > expected_allocations:
        raise ValueError("minimum_distinct_runners must be between 1 and expected_allocations")

    valid = _valid_records(records)
    noise_floors = noise_floors or {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in valid:
        grouped.setdefault((str(record["task_id"]), _backend(record)), []).append(record)

    buckets: list[StabilityBucket] = []
    for task_id, backend in sorted(expected_buckets):
        bucket_records = grouped.get((task_id, backend), [])
        fingerprints = {_fingerprint(record) for record in bucket_records}
        by_allocation: dict[str, list[dict[str, Any]]] = {}
        by_runner: dict[str, list[dict[str, Any]]] = {}
        for record in bucket_records:
            by_allocation.setdefault(_execution_id(record), []).append(record)
            runner_name = str(record.get("ci_runner_name") or "").strip()
            if runner_name:
                by_runner.setdefault(runner_name, []).append(record)

        coverage_reasons: list[str] = []
        observed_gpu_models = {canonical_gpu_model(record.get("gpu_model")) for record in bucket_records}
        if expected_gpu_model and observed_gpu_models != {canonical_gpu_model(expected_gpu_model)}:
            coverage_reasons.append(
                f"expected GPU {canonical_gpu_model(expected_gpu_model)}, observed "
                + (", ".join(sorted(observed_gpu_models)) or "none")
            )
        observed_target_branches = {str(record.get("target_branch") or "") for record in bucket_records}
        if expected_target_branch and observed_target_branches != {expected_target_branch}:
            coverage_reasons.append(
                f"expected target branch {expected_target_branch}, observed "
                + (", ".join(sorted(observed_target_branches)) or "none")
            )
        observed_commits = {str(record.get("commit_sha") or "") for record in bucket_records}
        if expected_commit and observed_commits != {expected_commit}:
            coverage_reasons.append(
                f"expected commit {expected_commit}, observed " + (", ".join(sorted(observed_commits)) or "none")
            )
        missing_fingerprint_fields = sorted(
            field
            for field in _FINGERPRINT_FIELDS
            if any(record.get(field) is None or record.get(field) == "" for record in bucket_records)
        )
        if missing_fingerprint_fields:
            coverage_reasons.append("missing runtime fingerprint fields: " + ", ".join(missing_fingerprint_fields))
        if len(fingerprints) != 1:
            coverage_reasons.append(f"expected one runtime fingerprint, observed {len(fingerprints)}")
        coverage_reasons.extend(
            _sample_coverage_reasons(
                bucket_records,
                by_allocation,
                by_runner,
                expected_allocations=expected_allocations,
                minimum_distinct_runners=minimum_distinct_runners,
                samples_per_allocation=samples_per_allocation,
            )
        )

        values = [float(record["fps"]) for record in bucket_records]
        median_fps: float | None = None
        mad_pct: float | None = None
        cv_pct: float | None = None
        range_pct: float | None = None
        max_sample_deviation_pct: float | None = None
        max_runner_deviation_pct: float | None = None
        effective_noise_pct: float | None = None
        effective_warn_pct: float | None = None
        effective_block_pct: float | None = None
        allocation_medians: dict[str, float] = {}
        runner_medians: dict[str, float] = {}
        metric_reasons: list[str] = []
        noise_floor = float(noise_floors.get((task_id, backend), 0.0))

        if values:
            median_fps = statistics.median(values)
            mean_fps = statistics.mean(values)
            mad_fps = statistics.median(abs(value - median_fps) for value in values)
            stdev_fps = statistics.stdev(values) if len(values) > 1 else 0.0
            mad_pct = _percent(mad_fps, median_fps)
            cv_pct = _percent(stdev_fps, mean_fps)
            range_pct = _percent(max(values) - min(values), median_fps)
            max_sample_deviation_pct = max(abs(_percent(value - median_fps, median_fps)) for value in values)
            allocation_medians = {
                allocation: statistics.median(float(record["fps"]) for record in allocation_records)
                for allocation, allocation_records in sorted(by_allocation.items())
            }
            runner_medians = {
                runner: statistics.median(float(record["fps"]) for record in runner_records)
                for runner, runner_records in sorted(by_runner.items())
            }
            if runner_medians:
                max_runner_deviation_pct = max(
                    abs(_percent(runner_median - median_fps, median_fps)) for runner_median in runner_medians.values()
                )
            effective_noise_pct = max(mad_pct, noise_floor)
            effective_warn_pct = DEFAULT_K_WARN * effective_noise_pct
            effective_block_pct = max(DEFAULT_K_BLOCK * effective_noise_pct, MIN_BLOCK_REGRESSION_PCT)

            if cv_pct > MAX_POOLED_CV_PCT:
                metric_reasons.append(f"pooled CV {cv_pct:.2f}% exceeds {MAX_POOLED_CV_PCT:.2f}%")
            if max_sample_deviation_pct > MAX_SAMPLE_DEVIATION_PCT:
                metric_reasons.append(
                    f"single-sample deviation {max_sample_deviation_pct:.2f}% exceeds {MAX_SAMPLE_DEVIATION_PCT:.2f}%"
                )
            if max_runner_deviation_pct is not None and max_runner_deviation_pct > MAX_RUNNER_MEDIAN_DEVIATION_PCT:
                metric_reasons.append(
                    f"runner median deviation {max_runner_deviation_pct:.2f}% exceeds "
                    f"{MAX_RUNNER_MEDIAN_DEVIATION_PCT:.2f}%"
                )
            if effective_block_pct > MAX_EFFECTIVE_BLOCK_PCT:
                metric_reasons.append(
                    f"effective BLOCK band {effective_block_pct:.2f}% exceeds {MAX_EFFECTIVE_BLOCK_PCT:.2f}%"
                )

        if coverage_reasons:
            verdict = INCONCLUSIVE
            reasons = tuple(coverage_reasons + metric_reasons)
        elif metric_reasons:
            verdict = UNSTABLE
            reasons = tuple(metric_reasons)
        else:
            verdict = STABLE
            reasons = ()

        buckets.append(
            StabilityBucket(
                task_id=task_id,
                backend=backend,
                verdict=verdict,
                reasons=reasons,
                fingerprint_count=len(fingerprints),
                runtime_contract_hashes=tuple(
                    sorted(
                        {
                            str(record.get("runtime_contract_hash") or "")
                            for record in bucket_records
                            if record.get("runtime_contract_hash")
                        }
                    )
                ),
                allocation_count=len(by_allocation),
                runner_count=len(by_runner),
                sample_count=len(bucket_records),
                median_fps=median_fps,
                mad_pct=mad_pct,
                cv_pct=cv_pct,
                range_pct=range_pct,
                max_sample_deviation_pct=max_sample_deviation_pct,
                max_runner_median_deviation_pct=max_runner_deviation_pct,
                configured_noise_floor_pct=noise_floor,
                effective_noise_pct=effective_noise_pct,
                effective_warn_regression_pct=effective_warn_pct,
                effective_block_regression_pct=effective_block_pct,
                allocation_medians_fps=allocation_medians,
                runner_medians_fps=runner_medians,
            )
        )

    if any(bucket.verdict == UNSTABLE for bucket in buckets):
        overall = UNSTABLE
    elif any(bucket.verdict == INCONCLUSIVE for bucket in buckets):
        overall = INCONCLUSIVE
    else:
        overall = STABLE
    return overall, buckets


def _format_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}%"


def build_markdown(
    overall_verdict: str,
    buckets: list[StabilityBucket],
    *,
    records: list[dict[str, Any]],
    expected_allocations: int,
    minimum_distinct_runners: int,
    samples_per_allocation: int,
    expected_gpu_model: str | None,
    expected_target_branch: str | None,
    expected_commit: str | None,
) -> str:
    """Render the qualification decision and supporting FPS evidence."""
    icon = "✅" if overall_verdict == STABLE else "❌"
    decision = (
        "The measured runner pool is stable enough for the perf-smoke FPS gate."
        if overall_verdict == STABLE
        else "Do not enable blocking from this evidence."
    )
    lines = [
        "# L40S FPS runner stability qualification",
        "",
        f"## {icon} {overall_verdict}",
        "",
        decision,
        "",
        "### Evidence scope",
        "",
        f"- GPU: `{canonical_gpu_model(expected_gpu_model) if expected_gpu_model else 'unspecified'}`",
        f"- Target branch: `{expected_target_branch or 'unspecified'}`",
        f"- Commit: `{expected_commit or 'unspecified'}`",
        "",
        "### Qualification policy fixed before measurement",
        "",
        (
            f"- Coverage: {expected_allocations} independent allocations × {samples_per_allocation} FPS samples "
            f"for every configured task/backend bucket, spanning at least "
            f"{minimum_distinct_runners} distinct runner registrations."
        ),
        "- Environment: exactly one commit/runtime fingerprint per bucket.",
        (
            f"- Runner bias: every runner median must remain within "
            f"{MAX_RUNNER_MEDIAN_DEVIATION_PCT:.1f}% of the pooled median."
        ),
        f"- Dispersion: pooled FPS CV must be at most {MAX_POOLED_CV_PCT:.1f}%.",
        f"- Single-run reliability: every FPS sample must remain within {MAX_SAMPLE_DEVIATION_PCT:.1f}% of the median.",
        (
            f"- Gate sensitivity: the configured four-MAD BLOCK band must remain within "
            f"{MAX_EFFECTIVE_BLOCK_PCT:.1f}% FPS regression."
        ),
        "",
        (
            "A passing result qualifies this pool for detecting substantial smoke-test regressions "
            "(greater than 10%); it does not claim microbenchmark-level sensitivity."
        ),
        "",
        "### Per-bucket FPS evidence",
        "",
        "| Task | Backend | Runners | Samples | Median FPS | MAD | CV | Max sample offset | "
        "Max runner offset | Effective WARN | Effective BLOCK | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for bucket in buckets:
        verdict = "PASS" if bucket.verdict == STABLE else bucket.verdict
        lines.append(
            f"| `{bucket.task_id}` | `{bucket.backend}` | {bucket.runner_count} | {bucket.sample_count} | "
            f"{bucket.median_fps:,.1f} | {_format_pct(bucket.mad_pct)} | {_format_pct(bucket.cv_pct)} | "
            f"{_format_pct(bucket.max_sample_deviation_pct)} | "
            f"{_format_pct(bucket.max_runner_median_deviation_pct)} | "
            f"{_format_pct(bucket.effective_warn_regression_pct)} | "
            f"{_format_pct(bucket.effective_block_regression_pct)} | **{verdict}** |"
            if bucket.median_fps is not None
            else (
                f"| `{bucket.task_id}` | `{bucket.backend}` | {bucket.runner_count} | "
                f"{bucket.sample_count} | — | — | — | — | — | — | — | **{verdict}** |"
            )
        )

    failed = [bucket for bucket in buckets if bucket.verdict != STABLE]
    if failed:
        lines.extend(["", "### Why the pool did not qualify", ""])
        for bucket in failed:
            reason = "; ".join(bucket.reasons) or "unknown reason"
            lines.append(f"- `{bucket.task_id}/{bucket.backend}`: {reason}.")
    elif not buckets:
        lines.extend(["", "### Why the pool did not qualify", "", "- No task/backend buckets were configured."])

    allocation_rows: dict[str, dict[str, Any]] = {}
    for record in _valid_records(records):
        allocation = _execution_id(record)
        row = allocation_rows.setdefault(
            allocation,
            {
                "runner": str(record.get("ci_runner_name") or "unknown"),
                "samples": 0,
                "buckets": set(),
            },
        )
        row["samples"] += 1
        row["buckets"].add((str(record["task_id"]), _backend(record)))
    lines.extend(
        [
            "",
            "<details>",
            "<summary>Runner allocation evidence</summary>",
            "",
            "| Allocation | GitHub runner | Successful FPS samples | Buckets covered |",
            "|---|---|---:|---:|",
        ]
    )
    for allocation, row in sorted(allocation_rows.items()):
        lines.append(f"| `{allocation}` | `{row['runner']}` | {row['samples']} | {len(row['buckets'])} |")
    lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def build_report(
    records: list[dict[str, Any]],
    *,
    expected_buckets: set[tuple[str, str]],
    noise_floors: dict[tuple[str, str], float] | None = None,
    expected_allocations: int = DEFAULT_EXPECTED_ALLOCATIONS,
    minimum_distinct_runners: int = DEFAULT_MINIMUM_DISTINCT_RUNNERS,
    samples_per_allocation: int = DEFAULT_SAMPLES_PER_ALLOCATION,
    expected_gpu_model: str | None = None,
    expected_target_branch: str | None = None,
    expected_commit: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Build machine-readable and reviewer-facing runner qualification reports."""
    overall, buckets = evaluate_stability(
        records,
        expected_buckets=expected_buckets,
        noise_floors=noise_floors,
        expected_allocations=expected_allocations,
        minimum_distinct_runners=minimum_distinct_runners,
        samples_per_allocation=samples_per_allocation,
        expected_gpu_model=expected_gpu_model,
        expected_target_branch=expected_target_branch,
        expected_commit=expected_commit,
    )
    report = {
        "schema_version": 1,
        "overall_verdict": overall,
        "policy": {
            "expected_allocations": expected_allocations,
            "minimum_distinct_runners": minimum_distinct_runners,
            "samples_per_allocation": samples_per_allocation,
            "max_effective_block_pct": MAX_EFFECTIVE_BLOCK_PCT,
            "max_pooled_cv_pct": MAX_POOLED_CV_PCT,
            "max_sample_deviation_pct": MAX_SAMPLE_DEVIATION_PCT,
            "max_runner_median_deviation_pct": MAX_RUNNER_MEDIAN_DEVIATION_PCT,
            "gate_k_warn": DEFAULT_K_WARN,
            "gate_k_block": DEFAULT_K_BLOCK,
            "gate_min_block_regression_pct": MIN_BLOCK_REGRESSION_PCT,
            "expected_gpu_model": expected_gpu_model,
            "expected_target_branch": expected_target_branch,
            "expected_commit": expected_commit,
        },
        "input_sample_count": len(records),
        "valid_sample_count": len(_valid_records(records)),
        "buckets": [asdict(bucket) for bucket in buckets],
    }
    markdown = build_markdown(
        overall,
        buckets,
        records=records,
        expected_allocations=expected_allocations,
        minimum_distinct_runners=minimum_distinct_runners,
        samples_per_allocation=samples_per_allocation,
        expected_gpu_model=expected_gpu_model,
        expected_target_branch=expected_target_branch,
        expected_commit=expected_commit,
    )
    return report, markdown


def _load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(record, dict) for record in payload):
            raise ValueError(f"{path} must contain a JSON list of objects")
        records.extend(payload)
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, action="append", type=Path)
    parser.add_argument("--tasks_config", type=Path, default=Path(__file__).with_name("tasks.json"))
    parser.add_argument(
        "--backends",
        default="",
        help="Comma-separated backend_key allowlist to qualify (empty = every configured backend).",
    )
    parser.add_argument("--gpu_model", default="l40s")
    parser.add_argument("--expected_target_branch")
    parser.add_argument("--expected_commit")
    parser.add_argument("--expected_allocations", type=int, default=DEFAULT_EXPECTED_ALLOCATIONS)
    parser.add_argument("--minimum_distinct_runners", type=int, default=DEFAULT_MINIMUM_DISTINCT_RUNNERS)
    parser.add_argument("--samples_per_allocation", type=int, default=DEFAULT_SAMPLES_PER_ALLOCATION)
    parser.add_argument("--json_output", type=Path)
    parser.add_argument("--markdown_output", type=Path)
    parser.add_argument("--github_step_summary", type=Path)
    parser.add_argument("--require_ready", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run runner-pool stability qualification."""
    args = _parse_args()
    records = _load_records(args.records)
    tasks = select_backends(load_tasks(args.tasks_config), args.backends)
    expected, noise_floors = configured_scope(tasks, args.gpu_model)
    report, markdown = build_report(
        records,
        expected_buckets=expected,
        noise_floors=noise_floors,
        expected_allocations=args.expected_allocations,
        minimum_distinct_runners=args.minimum_distinct_runners,
        samples_per_allocation=args.samples_per_allocation,
        expected_gpu_model=args.gpu_model,
        expected_target_branch=args.expected_target_branch,
        expected_commit=args.expected_commit,
    )
    print(markdown)
    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(markdown + "\n", encoding="utf-8")
    if args.github_step_summary:
        with args.github_step_summary.open("a", encoding="utf-8") as summary:
            summary.write(markdown + "\n")
    return int(args.require_ready and report["overall_verdict"] != STABLE)


if __name__ == "__main__":
    raise SystemExit(main())
