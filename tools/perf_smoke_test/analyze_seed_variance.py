# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Summarize run-to-run FPS variance from a baseline seeding run.

The seeder writes one record per successful repetition to ``seed_records.json``.
This tool groups only records with the same commit and gate fingerprint so code or
environment changes are never mistaken for runner noise. The report is advisory:
it exposes the observed distribution without choosing a project-wide stability
threshold or changing the seeder's exit status.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_GROUP_FIELDS = (
    "gpu_model",
    "task_id",
    "backend",
    "target_branch",
    "commit_sha",
    "launch_config_hash",
    "benchmark_contract_hash",
    "runtime_contract_hash",
    "baseline_epoch",
)


@dataclass(frozen=True)
class VarianceBucket:
    """Run-to-run statistics for one exact commit and gate fingerprint."""

    gpu_model: str
    task_id: str
    backend: str
    target_branch: str
    commit_sha: str
    launch_config_hash: str
    benchmark_contract_hash: str
    runtime_contract_hash: str
    baseline_epoch: int | None
    ci_run_count: int
    runner_count: int
    sample_count: int
    median_fps: float
    mean_fps: float
    stdev_fps: float
    mad_fps: float
    mad_pct: float
    cv_pct: float
    range_pct: float
    first_to_last_pct: float


def _percent(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _record_group_key(record: dict[str, Any]) -> tuple[Any, ...]:
    normalized = dict(record)
    normalized["backend"] = record.get("backend") or record.get("backend_key")
    return tuple(normalized.get(field) for field in _GROUP_FIELDS)


def _sample_order(item: tuple[int, dict[str, Any]]) -> tuple[str, str, int, int]:
    position, record = item
    sample_index = record.get("sample_index")
    timestamp = record.get("timestamp")
    ci_run_id = record.get("ci_run_id")
    return (
        str(timestamp) if timestamp else "",
        str(ci_run_id) if ci_run_id else "",
        sample_index if isinstance(sample_index, int) else position,
        position,
    )


def _execution_id(record: dict[str, Any]) -> str:
    label = str(record.get("ci_run_label") or "").strip()
    if label:
        return label
    return ":".join(
        str(record.get(field) or "unknown") for field in ("ci_run_id", "ci_run_attempt", "ci_job", "ci_runner_name")
    )


def analyze_records(records: list[dict[str, Any]]) -> tuple[list[VarianceBucket], int]:
    """Calculate variance statistics for valid FPS records.

    Args:
        records: Seeder summary records in collection order.

    Returns:
        Per-fingerprint statistics and the number of invalid records skipped.
    """
    groups: dict[tuple[Any, ...], list[tuple[int, dict[str, Any]]]] = {}
    skipped = 0
    for position, record in enumerate(records):
        fps = record.get("fps")
        task_id = record.get("task_id")
        backend = record.get("backend") or record.get("backend_key")
        if (
            not isinstance(fps, (int, float))
            or isinstance(fps, bool)
            or not math.isfinite(float(fps))
            or not task_id
            or not backend
        ):
            skipped += 1
            continue
        groups.setdefault(_record_group_key(record), []).append((position, record))

    buckets: list[VarianceBucket] = []
    for key, indexed_records in groups.items():
        ordered = [record for _, record in sorted(indexed_records, key=_sample_order)]
        values = [float(record["fps"]) for record in ordered]
        ci_runs = {_execution_id(record) for record in ordered}
        runners = {
            str(record.get("ci_runner_name")).strip()
            for record in ordered
            if str(record.get("ci_runner_name") or "").strip()
        }
        median_fps = statistics.median(values)
        mean_fps = statistics.mean(values)
        stdev_fps = statistics.stdev(values) if len(values) > 1 else 0.0
        mad_fps = statistics.median(abs(value - median_fps) for value in values)
        first_to_last_pct = _percent(values[-1] - values[0], values[0])
        buckets.append(
            VarianceBucket(
                gpu_model=str(key[0] or ""),
                task_id=str(key[1]),
                backend=str(key[2]),
                target_branch=str(key[3] or ""),
                commit_sha=str(key[4] or ""),
                launch_config_hash=str(key[5] or ""),
                benchmark_contract_hash=str(key[6] or ""),
                runtime_contract_hash=str(key[7] or ""),
                baseline_epoch=key[8] if isinstance(key[8], int) else None,
                ci_run_count=len(ci_runs),
                runner_count=len(runners),
                sample_count=len(values),
                median_fps=median_fps,
                mean_fps=mean_fps,
                stdev_fps=stdev_fps,
                mad_fps=mad_fps,
                mad_pct=_percent(mad_fps, median_fps),
                cv_pct=_percent(stdev_fps, mean_fps),
                range_pct=_percent(max(values) - min(values), median_fps),
                first_to_last_pct=first_to_last_pct,
            )
        )

    buckets.sort(key=lambda bucket: (bucket.task_id, bucket.backend, bucket.commit_sha, bucket.runtime_contract_hash))
    return buckets, skipped


def build_markdown(buckets: list[VarianceBucket], *, valid_samples: int, skipped_samples: int) -> str:
    """Render a concise report for the GitHub Actions step summary."""
    lines = [
        "## Run-to-run variance",
        "",
        (
            f"Analyzed {valid_samples} successful repetition(s) across {len(buckets)} exact "
            "commit/runtime bucket(s). This report is advisory and does not fail CI."
        ),
        "",
        "| Task | Backend | Commit / runtime | Allocations | Runners | Samples | Median FPS | "
        "MAD / median | CV | Min-max / median | First → last |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bucket in buckets:
        context = f"{bucket.commit_sha[:8] or 'unknown'} / {bucket.runtime_contract_hash[:8] or 'unknown'}"
        lines.append(
            f"| `{bucket.task_id}` | `{bucket.backend}` | `{context}` | {bucket.ci_run_count} | "
            f"{bucket.runner_count} | {bucket.sample_count} | {bucket.median_fps:,.1f} | "
            f"{bucket.mad_pct:.2f}% | {bucket.cv_pct:.2f}% | {bucket.range_pct:.2f}% | "
            f"{bucket.first_to_last_pct:+.2f}% |"
        )
    if not buckets:
        lines.append("| _No valid samples_ | — | — | 0 | 0 | 0 | — | — | — | — | — |")
    lines.extend(
        [
            "",
            "- **MAD / median** is the gate-aligned robust noise measure; lower is more stable.",
            "- **CV** is standard deviation divided by mean; it is more sensitive to outliers.",
            "- **Min-max / median** shows the full observed spread.",
            "- **First → last** helps reveal sample-order drift such as cold-start or thermal effects.",
        ]
    )
    if skipped_samples:
        lines.extend(["", f"Skipped {skipped_samples} malformed or non-finite record(s)."])
    return "\n".join(lines) + "\n"


def build_report(records: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """Build machine-readable and Markdown reports from seeder records."""
    buckets, skipped = analyze_records(records)
    valid_samples = sum(bucket.sample_count for bucket in buckets)
    report = {
        "schema_version": 1,
        "input_sample_count": len(records),
        "valid_sample_count": valid_samples,
        "skipped_sample_count": skipped,
        "bucket_count": len(buckets),
        "buckets": [asdict(bucket) for bucket in buckets],
    }
    markdown = build_markdown(buckets, valid_samples=valid_samples, skipped_samples=skipped)
    return report, markdown


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        required=True,
        action="append",
        type=Path,
        help="Seeder seed_records.json file. Repeat to combine independent CI runs.",
    )
    parser.add_argument("--json_output", type=Path, help="Optional machine-readable report path.")
    parser.add_argument("--markdown_output", type=Path, help="Optional Markdown report path.")
    parser.add_argument(
        "--github_step_summary",
        type=Path,
        help="Optional GITHUB_STEP_SUMMARY path to append the Markdown report to.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the variance report CLI."""
    args = _parse_args()
    payload: list[dict[str, Any]] = []
    for records_path in args.records:
        records = json.loads(records_path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
            raise ValueError(f"{records_path} must contain a JSON list of objects")
        payload.extend(
            record if record.get("ci_run_id") else {**record, "ci_run_id": f"records:{records_path}"}
            for record in records
        )

    report, markdown = build_report(payload)
    print(markdown, end="")
    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if args.github_step_summary:
        with args.github_step_summary.open("a", encoding="utf-8") as summary:
            summary.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
