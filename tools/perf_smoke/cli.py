# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entry points for the performance smoke gate.

Wiring only; every decision lives in :mod:`compare`, :mod:`contract` or
:mod:`store`, so the library stays importable without argparse.

The container SAS URL is read from ``$ISAACLAB_BLOB_URL``.

Subcommands:
    ``compare``    compare one benchmark bundle against the baseline store
    ``write``      record one measurement in the store (develop only)
    ``aggregate``  roll several comparison JSONs into one summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import compare as compare_mod
from . import contract as contract_mod
from . import metrics as metrics_mod
from . import report as report_mod
from . import store as store_mod

_DEFAULT_THRESHOLDS = Path(__file__).resolve().parent.parent / "perf_smoke_thresholds.json"


def _load_json(path: Path, name: str) -> dict:
    try:
        return metrics_mod.mapping(json.loads(path.read_text(encoding="utf-8")), name)
    except FileNotFoundError as exc:
        raise metrics_mod.PerfSmokeError(f"{name} not found: {path}") from exc
    except OSError as exc:
        # A bad path should surface as a gate error.
        raise metrics_mod.PerfSmokeError(f"{name} could not be read: {path} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise metrics_mod.PerfSmokeError(f"{name} is not valid JSON: {exc}") from exc


def _cmd_compare(args: argparse.Namespace) -> int:
    # Split into compare and measure stages such that measurements are preserved.
    try:
        bundle = _load_json(args.benchmark_result, "benchmark result")
        key = contract_mod.build(bundle)
        measured = metrics_mod.extract(bundle)
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
                history = [row.metrics for row in rows if contract_mod.from_dict(row.contract).matches(key)]
                report = compare_mod.compare(
                    key, measured, history, thresholds, min_samples=args.min_samples, label=args.label
                )
        except metrics_mod.PerfSmokeError as exc:
            report = compare_mod.unresolved(key, measured, compare_mod.ERROR, str(exc), label=args.label)
            print(f"::warning::perf-smoke: {exc}", file=sys.stderr)

    # An artifact is always written so that a faulted combination still shows in the summary.
    report_mod.write_json(report, args.output_json)
    print(report_mod.render(report), end="")
    # Only a measured regression fails jobs; other errors are non-blocking and exit 0.
    return 1 if report.verdict == compare_mod.FAIL else 0


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
    created = store_mod.write(row)
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

    summary = report_mod.render_aggregate(reports)
    print(summary, end="")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(summary, encoding="utf-8")

    if not reports:
        # Passes: unconfigured store produces SKIP rows, non-function gate produces zero rows.
        print("::error::perf-smoke: no comparison artifacts were produced", file=sys.stderr)
        return 1
    return 1 if any(report.verdict == compare_mod.FAIL for _, report in reports) else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(prog="perf_smoke", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser("compare", help="compare a benchmark against the baseline store")
    compare_parser.add_argument("--benchmark_result", type=Path, required=True)
    compare_parser.add_argument("--thresholds", type=Path, default=_DEFAULT_THRESHOLDS)
    compare_parser.add_argument("--output_json", type=Path, required=True)
    compare_parser.add_argument("--min_samples", type=int, default=compare_mod.MIN_BASELINE_SAMPLES)
    compare_parser.add_argument("--label", default="", help="matrix combination name, carried into the artifact")
    compare_parser.set_defaults(func=_cmd_compare)

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
