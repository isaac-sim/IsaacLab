# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entry points for the performance smoke gate.

Wiring only; every decision lives in :mod:`compare`, :mod:`contract` or
:mod:`store`, so the library stays importable and testable without argparse.

Subcommands:
    ``compare``    compare one benchmark bundle against the baseline store
    ``write``      append one measurement to the store (develop only)
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
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise metrics_mod.PerfSmokeError(f"{name} not found: {path}") from exc
    except OSError as exc:
        # A bad path should surface as a gate error.
        raise metrics_mod.PerfSmokeError(f"{name} could not be read: {path} ({exc})") from exc
    except json.JSONDecodeError as exc:
        raise metrics_mod.PerfSmokeError(f"{name} is not valid JSON: {exc}") from exc


def _cmd_compare(args: argparse.Namespace) -> int:
    bundle = _load_json(args.benchmark_result, "benchmark result")
    thresholds = _load_json(args.thresholds, "threshold config")
    key = contract_mod.build(bundle)
    measured = metrics_mod.extract(bundle)

    if not args.baseline_uri:
        report = compare_mod._skipped(
            key, measured, "Baseline store is not configured (PERF_BASELINE_URI unset)", label=args.label
        )
    else:
        rows = store_mod.read(args.baseline_uri, key.hash, compare_mod.MAX_BASELINE_SAMPLES)
        # The store keys by contract hash, but verify the full contract too.
        history = [row.metrics for row in rows if contract_mod.from_dict(row.contract).matches(key)]
        report = compare_mod.compare(key, measured, history, thresholds, min_samples=args.min_samples, label=args.label)

    report_mod.write_json(report, args.output_json)
    print(report_mod.render(report), end="")
    # SKIP must not fail the job: a missing baseline is not evidence of a regression.
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
    store_mod.write(args.baseline_uri, row)
    print(f"Recorded baseline for contract {key.hash} at commit {args.commit[:12]}")
    return 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    reports: list[tuple[str, compare_mod.Report]] = []
    for path in sorted(args.comparison_dir.rglob("comparison.json")):
        payload = _load_json(path, str(path))
        metrics = tuple(compare_mod.MetricResult(**metric) for metric in payload.get("metrics", []))
        reports.append(
            (
                payload.get("label") or path.parent.name,
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
    return 1 if any(report.verdict == compare_mod.FAIL for _, report in reports) else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(prog="perf_smoke", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare_parser = subparsers.add_parser("compare", help="compare a benchmark against the baseline store")
    compare_parser.add_argument("--benchmark_result", type=Path, required=True)
    compare_parser.add_argument("--baseline_uri", default="", help="file://, s3:// or az:// store URI")
    compare_parser.add_argument("--thresholds", type=Path, default=_DEFAULT_THRESHOLDS)
    compare_parser.add_argument("--output_json", type=Path, required=True)
    compare_parser.add_argument("--min_samples", type=int, default=compare_mod.MIN_BASELINE_SAMPLES)
    compare_parser.add_argument("--label", default="", help="matrix combination name, carried into the artifact")
    compare_parser.set_defaults(func=_cmd_compare)

    write_parser = subparsers.add_parser("write", help="append a measurement to the baseline store")
    write_parser.add_argument("--benchmark_result", type=Path, required=True)
    write_parser.add_argument("--baseline_uri", required=True)
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
        # Malformed input is corruption.
        print(f"::error::perf-smoke: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
