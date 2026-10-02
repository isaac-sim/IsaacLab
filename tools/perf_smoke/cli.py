# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entry points for the performance smoke gate.

The paired comparison stays here as a small, stateless transform over benchmark
bundles. The rolling-history policy lives in :mod:`compare` and :mod:`store`.

The container SAS URL is read from ``$ISAACLAB_BLOB_URL``.

Subcommands:
    ``compare``    compare independent benchmark runs against the baseline store
    ``pair``       describe FPS changes between a PR base and tested merge
    ``write``      record one measurement in the store (develop only)
    ``aggregate``  roll several comparison JSONs into one summary
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from . import compare as compare_mod
from . import contract as contract_mod
from . import metrics as metrics_mod
from . import report as report_mod
from . import store as store_mod

_DEFAULT_THRESHOLDS = Path(__file__).resolve().parent.parent / "perf_smoke_thresholds.json"
_PAIR_SAMPLE_COUNT = 3


def _load_json(path: Path, name: str) -> dict:
    try:
        return metrics_mod.mapping(json.loads(path.read_text(encoding="utf-8")), name)
    except FileNotFoundError as exc:
        raise metrics_mod.PerfSmokeError(f"{name} not found: {path}") from exc
    except OSError as exc:
        # A bad path should surface as a gate error.
        raise metrics_mod.PerfSmokeError(f"{name} could not be read: {path} ({exc})") from exc
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise metrics_mod.PerfSmokeError(f"{name} is not valid UTF-8 JSON: {exc}") from exc


def _cmd_compare(args: argparse.Namespace) -> int:
    # Split into compare and measure stages such that measurements are preserved.
    try:
        bundles = [_load_json(path, "benchmark result") for path in args.benchmark_result]
        key = contract_mod.build(bundles[0])
        if any(not contract_mod.build(bundle).matches(key) for bundle in bundles[1:]):
            raise metrics_mod.PerfSmokeError("candidate runs have different runtime contracts")
        measurements = [metrics_mod.extract(bundle) for bundle in bundles]
        measured = {
            metric.name: statistics.median(row[metric.name] for row in measurements) for metric in metrics_mod.METRICS
        }
    except metrics_mod.PerfSmokeError as exc:
        report = compare_mod.errored(str(exc), label=args.label)
        print(f"::warning::perf-smoke: {exc}", file=sys.stderr)
    else:
        try:
            thresholds = _load_json(args.thresholds, "threshold config")
            if not store_mod.is_configured():
                report = compare_mod.unresolved(
                    key,
                    measured,
                    compare_mod.SKIP,
                    "No baseline store credential is available for this run",
                    label=args.label,
                )
            else:
                rows = store_mod.read(key.hash, compare_mod.MAX_BASELINE_SAMPLES)
                # The storage key is a truncation of the contract digest; need a full match.
                expected_contract = key.as_dict()
                history = [row.metrics for row in rows if row.contract == expected_contract]
                report = compare_mod.compare(
                    key,
                    measurements,
                    history,
                    thresholds,
                    min_samples=args.min_samples,
                    label=args.label,
                )
        except metrics_mod.PerfSmokeError as exc:
            report = compare_mod.unresolved(key, measured, compare_mod.ERROR, str(exc), label=args.label)
            print(f"::warning::perf-smoke: {exc}", file=sys.stderr)

    # An artifact is always written so that a faulted combination still shows in the summary.
    report_mod.write_json(report, args.output_json)
    print(report_mod.render(report), end="")
    # Only a measured regression fails jobs; other errors are non-blocking and exit 0.
    return 1 if report.verdict == compare_mod.FAIL else 0


def _cmd_pair(args: argparse.Namespace) -> int:
    report = {
        "baseline_commit": args.baseline_commit,
        "baseline_source": args.baseline_source,
        "candidate_commit": args.candidate_commit,
        "rows": [_pair_row(args.baseline_dir, args.candidate_dir, leg) for leg in _matrix_legs(args.benchmark_matrix)],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_mod.render_pair(report), end="")
    return 0


def _cmd_write(args: argparse.Namespace) -> int:
    bundle = _load_json(args.benchmark_result, "benchmark result")
    key = contract_mod.build(bundle)
    row = store_mod.BaselineRow(
        contract=key.as_dict(),
        contract_hash=key.hash,
        metrics=metrics_mod.extract(bundle),
        commit=args.commit,
        timestamp=args.timestamp,
        run_id=args.run_id,
    )
    try:
        created = store_mod.write(row)
    except metrics_mod.PerfSmokeError as exc:
        print(f"::warning::perf-smoke: baseline not recorded: {exc}", file=sys.stderr)
        return 0
    action = "Recorded" if created else "Already recorded"
    print(f"{action} baseline for contract {key.hash} at commit {args.commit[:12]}")
    return 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    reports: list[tuple[str, compare_mod.Report]] = []
    for path in sorted(args.comparison_dir.rglob("comparison.json")):
        name = path.parent.name
        try:
            payload = _load_json(path, str(path))
            # A TypeError here means an artifact written by a different version of this tool.
            metrics = tuple(compare_mod.MetricResult(**metric) for metric in payload.get("metrics", []))
        except (metrics_mod.PerfSmokeError, TypeError) as exc:
            print(f"::warning::perf-smoke: {path} could not be read: {exc}", file=sys.stderr)
            reports.append((name, compare_mod.errored(f"comparison artifact could not be read: {exc}", label=name)))
            continue
        reports.append(
            (
                payload.get("label") or name,
                compare_mod.Report(
                    contract=payload.get("contract", {}),
                    contract_hash=payload.get("contract_hash", ""),
                    metrics=metrics,
                    verdict=payload.get("verdict", compare_mod.SKIP),
                    message=payload.get("message", ""),
                    label=payload.get("label", ""),
                ),
            )
        )

    pair_reports = sorted(args.comparison_dir.rglob("pair-comparison.json"))
    pair_summary = ""
    if len(pair_reports) == 1:
        try:
            pair_summary = report_mod.render_pair(_load_json(pair_reports[0], str(pair_reports[0]))) + "\n"
        except metrics_mod.PerfSmokeError as exc:
            print(f"::warning::perf-smoke: paired comparison could not be read: {exc}", file=sys.stderr)
    elif len(pair_reports) > 1:
        print("::warning::perf-smoke: multiple paired comparison artifacts were found", file=sys.stderr)

    summary = pair_summary + report_mod.render_aggregate(reports)
    print(summary, end="")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(summary, encoding="utf-8")

    if not reports:
        # Non-blocking: this is an infra fault (e.g. a flaky artifact download)
        print("::warning::perf-smoke: no comparison artifacts were produced", file=sys.stderr)
        return 0
    return 1 if any(report.verdict == compare_mod.FAIL for _, report in reports) else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(prog="perf_smoke", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser("compare", help="compare a benchmark against the baseline store")
    compare_parser.add_argument("--benchmark_result", type=Path, nargs="+", required=True)
    compare_parser.add_argument("--thresholds", type=Path, default=_DEFAULT_THRESHOLDS)
    compare_parser.add_argument("--output_json", type=Path, required=True)
    compare_parser.add_argument("--min_samples", type=int, default=compare_mod.MIN_BASELINE_SAMPLES)
    compare_parser.add_argument("--label", default="", help="matrix combination name, carried into the artifact")
    compare_parser.set_defaults(func=_cmd_compare)

    pair_parser = subparsers.add_parser("pair", help="compare a PR base and tested merge")
    pair_parser.add_argument("--baseline_dir", type=Path, required=True)
    pair_parser.add_argument("--candidate_dir", type=Path, required=True)
    pair_parser.add_argument("--benchmark_matrix", type=Path, required=True)
    pair_parser.add_argument("--baseline_commit", required=True)
    pair_parser.add_argument("--baseline_source", choices=("fresh", "reused"), default="fresh")
    pair_parser.add_argument("--candidate_commit", required=True)
    pair_parser.add_argument("--output_json", type=Path, required=True)
    pair_parser.set_defaults(func=_cmd_pair)

    write_parser = subparsers.add_parser("write", help="append a measurement to the baseline store")
    write_parser.add_argument("--benchmark_result", type=Path, required=True)
    write_parser.add_argument("--commit", required=True)
    write_parser.add_argument("--timestamp", required=True)
    write_parser.add_argument("--run_id", default="")
    write_parser.set_defaults(func=_cmd_write)

    aggregate_parser = subparsers.add_parser("aggregate", help="roll comparison JSONs into one summary")
    aggregate_parser.add_argument("--comparison_dir", type=Path, required=True)
    aggregate_parser.add_argument("--output_markdown", type=Path)
    aggregate_parser.set_defaults(func=_cmd_aggregate)

    return parser


def _matrix_legs(path: Path) -> list[str]:
    try:
        legs = [line.split("|", 1)[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise metrics_mod.PerfSmokeError(f"benchmark matrix could not be read: {path} ({exc})") from exc
    if not legs or len(legs) != len(set(legs)):
        raise metrics_mod.PerfSmokeError("benchmark matrix must contain unique, non-empty leg names")
    return legs


def _pair_measurement(root: Path, leg: str, side: str) -> tuple[dict | None, str | None]:
    leg_dir = root / leg
    try:
        status = (leg_dir / "status").read_text(encoding="utf-8").strip()
    except OSError:
        return None, f"{side} status is unavailable"
    if status != "ok":
        return None, f"{side} status is {status or 'unavailable'}"

    paths = sorted(leg_dir.rglob("benchmark_runtime_*.json"))
    if len(paths) != _PAIR_SAMPLE_COUNT:
        return None, f"{side} has {len(paths)} samples; expected {_PAIR_SAMPLE_COUNT}"
    try:
        bundles = [_load_json(path, f"{side} benchmark result") for path in paths]
        contracts = [contract_mod.build(bundle) for bundle in bundles]
        if any(not contract.matches(contracts[0]) for contract in contracts[1:]):
            raise metrics_mod.PerfSmokeError(f"{side} samples have different runtime contracts")
        samples = [metrics_mod.extract(bundle)["total_fps"] for bundle in bundles]
    except metrics_mod.PerfSmokeError as exc:
        return None, str(exc)
    return {
        "fps": statistics.median(samples),
        "samples": samples,
        "workload": contracts[0].workload,
        "hardware": {
            "cpu_name": contracts[0].runtime.get("cpu_name"),
            "gpu_model": contracts[0].runtime.get("gpu_model"),
        },
    }, None


def _pair_row(baseline_dir: Path, candidate_dir: Path, leg: str) -> dict:
    baseline, baseline_error = _pair_measurement(baseline_dir, leg, "baseline")
    candidate, candidate_error = _pair_measurement(candidate_dir, leg, "candidate")
    row = {
        "label": leg,
        "status": "not_comparable",
        "baseline_fps": baseline["fps"] if baseline else None,
        "candidate_fps": candidate["fps"] if candidate else None,
        "change_pct": None,
        "baseline_samples": baseline["samples"] if baseline else [],
        "candidate_samples": candidate["samples"] if candidate else [],
        "reason": "; ".join(reason for reason in (baseline_error, candidate_error) if reason),
    }
    if baseline is None or candidate is None:
        return row
    if baseline["workload"] != candidate["workload"]:
        row["reason"] = "baseline and candidate workloads differ"
        return row
    if baseline["hardware"] != candidate["hardware"]:
        row["reason"] = "baseline and candidate hardware differ"
        return row
    if baseline["fps"] == 0:
        row["reason"] = "baseline FPS is zero"
        return row

    row["change_pct"] = 100 * (candidate["fps"] - baseline["fps"]) / baseline["fps"]
    row["status"] = "improved" if row["change_pct"] > 0 else "regressed" if row["change_pct"] < 0 else "unchanged"
    row["reason"] = ""
    return row


def main(argv: list[str] | None = None) -> int:
    """Run the performance smoke CLI."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except metrics_mod.PerfSmokeError as exc:
        # Exit 2: parse error; exit 1: measured regression or empty results.
        print(f"::error::perf-smoke: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
