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

import html
import json
import re
from pathlib import Path

from .compare import ERROR, FAIL, PASS, SKIP, WARN, Report
from .metrics import METRICS

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
        "| Metric | Measured | Baseline median | Change | Warn | Fail | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for metric in report.metrics:
        label = metric.label if metric.gating else f"{metric.label} _(advisory)_"
        verdict = _icon(metric.verdict) if metric.gating else f"{metric.verdict} _(advisory)_"
        note = f" — {metric.note}" if metric.note else ""
        lines.append(
            f"| {label} | {_num(metric.measured)} | {_num(metric.reference)} | {_pct(metric.regression_pct)} | "
            f"{_pct(metric.warn_pct)} | {_pct(metric.fail_pct)} | "
            f"{verdict}{note} |"
        )

    samples = max((metric.sample_count for metric in report.metrics), default=0)
    gating = ", ".join(f"**{metric.label}**" for metric in METRICS if metric.gating)
    lines += [
        "",
        f"Baseline: {samples} comparable run(s), contract `{report.contract_hash}`.",
        "",
        f"Only {gating} gates. The other metrics are recorded and compared so their noise can be "
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

    # SKIP ranks below PASS so that a run where nothing was compared does not headline as a green
    # pass. A mix still headlines PASS since at least once comparison was made.
    order = {SKIP: 0, PASS: 1, ERROR: 2, WARN: 3, FAIL: 4}
    worst = max((report.verdict for _, report in reports), key=lambda verdict: order.get(verdict, 0))

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
        f"{len(reports)} combination(s) reported. A 🚫 ERROR row is a fault in the gate, not a performance "
        "result, and never blocks a pull request. A combination whose benchmark crashed shows both an ERROR "
        "row here and a failed job.",
    ]
    return "\n".join(lines) + "\n"


def write_json(report: Report, path: Path) -> None:
    """Write the machine-readable comparison to ``path``. The Markdown form goes to stdout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_text(value: object) -> str:
    """Keep artifact-provided labels inside their Markdown table cell or paragraph."""
    result = html.escape(str(value)).replace("\n", " ").replace("\r", " ")
    for character, entity in (("|", "&#124;"), ("`", "&#96;"), ("[", "&#91;"), ("]", "&#93;")):
        result = result.replace(character, entity)
    return result


def _build_identity(label: str, identity: dict | None) -> str:
    if not identity:
        return f"**{label}:** unavailable."
    commit = _build_text(str(identity.get("source_commit", "unknown"))[:12])
    run = _build_text(identity.get("run_id", "unknown"))
    attempt = _build_text(identity.get("run_attempt", "unknown"))
    # Reconstruct links even for unavailable identities recovered from an earlier report.
    repository = identity.get("repository", "")
    origin = f"https://github.com/{repository}" if re.fullmatch(r"[\w.-]+/[\w.-]+", repository) else None
    source = commit
    if origin and re.fullmatch(r"[0-9a-fA-F]{40}", str(identity.get("source_commit", ""))):
        source = f"[{commit}]({origin}/commit/{identity['source_commit']})"
    execution = f"run {run}, attempt {attempt}"
    run_url = None
    if origin and all(str(identity.get(field, "")).isdigit() for field in ("run_id", "run_attempt")):
        run_url = f"{origin}/actions/runs/{identity['run_id']}"
        execution = f"[{execution}]({run_url}/attempts/{identity['run_attempt']})"
    evidence = ""
    if run_url and str(identity.get("artifact_id", "")).isdigit():
        evidence = f" · [source results]({run_url}/artifacts/{identity['artifact_id']})"
    note = ""
    if identity.get("source_provenance") == "github_run_metadata":
        note = " · source SHA recovered from GitHub; tested checkout not independently recorded"
    return f"**{label}:** {source} · {execution}{evidence}{note}."


def render_build_comparison(report: dict) -> str:
    """Render exact-build FPS observations within the existing Performance smoke summary."""
    selection = report.get("selection", {})
    identities = {side: report.get(side) for side in ("baseline", "candidate")}
    unavailable_side = selection.get("unavailable_side")
    if unavailable_side in identities and not identities[unavailable_side]:
        identities[unavailable_side] = selection.get("unavailable_evidence")
    a_label = "A — historical baseline" + (" (evidence unavailable)" if unavailable_side == "baseline" else "")
    b_label = "B — current benchmark" + (" (evidence unavailable)" if unavailable_side == "candidate" else "")
    lines = [
        "### Automatic build comparison",
        "",
        _build_identity(a_label, identities["baseline"]),
        "",
        _build_identity(b_label, identities["candidate"]),
        "",
        "**Baseline selection:** " + _build_text(selection.get("reason", "Selection information is unavailable.")),
        "",
    ]
    if selection.get("reference_branch"):
        anchor = _build_text(str(selection.get("reference_commit") or "unknown")[:12])
        lines += [f"Reference: {_build_text(selection['reference_branch'])} at `{anchor}`.", ""]
    visited = selection.get("visited_commits", [])
    if report.get("baseline") and len(visited) > 1:
        lines += [f"The selected reference is {len(visited) - 1} first-parent commit(s) older than the anchor.", ""]
    candidate = report.get("candidate") or {}
    if candidate.get("run_attempt") and report.get("report_attempt") != candidate["run_attempt"]:
        lines += [
            f"Measurements: attempt {_build_text(candidate['run_attempt'])}; "
            f"report: attempt {_build_text(report.get('report_attempt'))}.",
            "",
        ]
    if selection.get("reason_code"):
        lines += ["**Comparison unavailable:** " + _build_text(selection["reason_code"]) + ".", ""]
    rows = report.get("rows", [])
    if rows:
        lines += [
            "| Workload | A FPS | B FPS | Δ FPS (B − A) | Δ % | Samples A / B | Result |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    for row in rows:
        counts = []
        for side in ("baseline", "candidate"):
            item = row[side]
            expected = item.get("expected_count")
            counts.append(f"{item['count']}/{expected if expected is not None else '?'}")
        details = [row["status"], *row.get("reasons", []), *row.get("notes", [])]
        for difference in row.get("protocol_differences", []) + row.get("context_differences", []):
            details.append(
                f"{difference['field']}: "
                f"{json.dumps(difference['baseline'], sort_keys=True)} → "
                f"{json.dumps(difference['candidate'], sort_keys=True)}"
            )
        delta = row.get("absolute_change")
        absolute = "—" if delta is None else f"{delta:+.6g}"
        lines.append(
            f"| {_build_text(row['label'])} | {_num(row['baseline']['median'])} | "
            f"{_num(row['candidate']['median'])} | {absolute} | {_pct(row.get('change_pct'))} | "
            f"{' / '.join(counts)} | {'<br>'.join(_build_text(item) for item in details)} |"
        )
    if not rows:
        lines.append("No readable workload rows are available for this comparison.")
    lines += [
        "",
        "FPS values are medians of the available samples. Positive Δ means higher observed FPS; "
        "negative Δ means lower observed FPS. Sample counts are valid/expected for A and B.",
        "",
        "The existing rolling-history CI gate above is unchanged. "
        "These exact-build deltas do not determine its verdict.",
        "",
    ]
    for note in dict.fromkeys(report.get("notes", [])):
        lines.append("- " + _build_text(note))
    for issue in selection.get("issues", []):
        lines.append("- Selection note: " + _build_text(json.dumps(issue, sort_keys=True)))
    return "\n".join(lines) + "\n"
