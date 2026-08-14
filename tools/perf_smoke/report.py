# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Markdown and JSON rendering for the per-combination and aggregate reports.

Kept separate from :mod:`compare` because presentation churns far more often than
verdict logic, and both the per-job summary and the aggregate job render from the
same data.
"""

from __future__ import annotations

import json
from pathlib import Path

from .compare import ERROR, FAIL, PASS, SKIP, WARN, Report

_ICONS = {PASS: "✅", WARN: "⚠️", FAIL: "❌", SKIP: "⏭️", ERROR: "🚫"}


def _num(value: float | None, digits: int = 6) -> str:
    """Format a number for a table cell, or ``-`` when absent."""
    return "-" if value is None else f"{value:.{digits}g}"


def _pct(value: float | None) -> str:
    """Format a signed percentage, or ``-`` when absent."""
    return "-" if value is None else f"{value:+.2f}%"


def _icon(verdict: str) -> str:
    return f"{_ICONS.get(verdict, '')} {verdict}".strip()


def render(report: Report) -> str:
    """Render one combination's comparison as Markdown.

    Advisory (non-gating) metrics are shown with the same detail as gating ones so
    the evidence to promote them accumulates in plain sight.
    """
    lines = [
        f"## Performance smoke: {_icon(report.verdict)}",
        "",
        report.message,
        "",
        "| Metric | Measured | Baseline median | Change | Noise (1σ) | Significance | Warn | Fail | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for metric in report.metrics:
        significance = "-" if metric.significance_sigma is None else f"{metric.significance_sigma:.2f}σ"
        label = metric.label if metric.gating else f"{metric.label} _(advisory)_"
        verdict = _icon(metric.verdict) if metric.gating else f"{metric.verdict} _(advisory)_"
        note = f" — {metric.note}" if metric.note else ""
        lines.append(
            f"| {label} | {_num(metric.measured)} | {_num(metric.reference)} | {_pct(metric.regression_pct)} | "
            f"{_pct(metric.spread_pct)} | {significance} | {_pct(metric.warn_pct)} | {_pct(metric.fail_pct)} | "
            f"{verdict}{note} |"
        )

    samples = max((metric.sample_count for metric in report.metrics), default=0)
    lines += [
        "",
        f"Baseline: {samples} comparable run(s), contract `{report.contract_hash}`.",
        "",
        "Only **Total FPS** gates. The other metrics are recorded and compared so their noise can be "
        "characterised before any of them is trusted to fail a pull request.",
    ]
    return "\n".join(lines) + "\n"


def render_aggregate(reports: list[tuple[str, Report]]) -> str:
    """Render one table covering every combination that reported.

    Args:
        reports: ``(combination name, report)`` pairs.

    Returns:
        Markdown for the aggregate job summary.
    """
    if not reports:
        return "## Performance smoke: no results\n\nNo comparison artifacts were produced.\n"

    worst = PASS
    for _, report in reports:
        if report.verdict == FAIL or (report.verdict == WARN and worst != FAIL):
            worst = report.verdict

    lines = [
        f"## Performance smoke: {_icon(worst)}",
        "",
        "| Combination | Total FPS | Baseline | Change | Startup [s] | GPU mem [GB] | RSS [GB] | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    # Sort on the label only: a duplicate label would otherwise fall through to
    # comparing Report objects, which are frozen dataclasses without ordering.
    for name, report in sorted(reports, key=lambda item: item[0]):
        by_name = {metric.name: metric for metric in report.metrics}
        fps = by_name.get("total_fps")
        lines.append(
            f"| {name} | {_num(fps.measured) if fps else '-'} | {_num(fps.reference) if fps else '-'} | "
            f"{_pct(fps.regression_pct) if fps else '-'} | "
            f"{_num(by_name['startup_time_s'].measured, 4) if 'startup_time_s' in by_name else '-'} | "
            f"{_num(by_name['gpu_mem_peak_gb'].measured, 4) if 'gpu_mem_peak_gb' in by_name else '-'} | "
            f"{_num(by_name['ram_peak_gb'].measured, 4) if 'ram_peak_gb' in by_name else '-'} | "
            f"{_icon(report.verdict)} |"
        )
    lines += [
        "",
        f"{len(reports)} combination(s) reported. Combinations whose benchmark failed to run appear as a failed "
        "job rather than a row here.",
    ]
    return "\n".join(lines) + "\n"


def write_json(report: Report, path: Path) -> None:
    """Write the machine-readable comparison next to the human-readable summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
