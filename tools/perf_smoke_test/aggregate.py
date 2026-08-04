# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate benchmark artifacts, run the oracle, and update trusted baselines"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
_TOOLS_DIR = _MODULE_DIR.parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
DEFAULT_BASELINE_BRANCH = "perf-baselines"

from baseline_manager import (  # noqa: E402
    BaselineUpdateRecord,
    load_baseline,
    load_baseline_git,
    make_sample_metadata,
    match_context_from_bench_result,
    refresh_baseline_branch,
    update_baseline,
    update_baselines_git,
)
from contracts import BenchResult  # noqa: E402
from gate_config import BASELINE_PUSH_RETRIES, MIN_BASELINE_SAMPLES, load_gate_config  # noqa: E402
from gate_types import FpsMeanThreshold, OracleVerdict  # noqa: E402
from gpu_identity import canonical_gpu_model, gpu_model_config_keys  # noqa: E402
from omni_github import write_artifact as write_omni_github_artifact  # noqa: E402
from oracle import compare  # noqa: E402
from task_config import get_task, load_tasks  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(description="Aggregate bench results and run oracle.")
    parser.add_argument("--artifacts_dir", required=True, type=Path)
    parser.add_argument("--gpu_model", default="L40S")
    parser.add_argument("--gate_config", type=Path, default=_MODULE_DIR / "gate_config.json")
    parser.add_argument("--baseline_branch", default=DEFAULT_BASELINE_BRANCH)
    parser.add_argument(
        "--baseline_remote", default="origin", help="Git remote that owns the baseline branch; empty = local only"
    )
    parser.add_argument("--baseline_push_retries", type=int, default=None)
    parser.add_argument("--baselines_dir", type=Path, default=None, help="Flat-file baseline directory; bypasses git")
    parser.add_argument("--allow_baseline_update", default="false")
    parser.add_argument("--summary_file", default=None)
    parser.add_argument(
        "--omni_github_dir",
        type=Path,
        default=None,
        help="Write the omni-github test-result artifact (manifest + result JSON) to this directory",
    )
    parser.add_argument(
        "--omni_platform", default="linux-x86_64", help="omni-github app.platform for the emitted artifact"
    )
    parser.add_argument(
        "--omni_app_config", default="perf-smoke", help="omni-github app.config for the emitted artifact"
    )
    parser.add_argument("--base_sha", default=None, help="PR base SHA for ancestry-aware baseline matching")
    parser.add_argument("--target_branch", default=None, help="Target protected branch, e.g. main/develop/release/x")
    parser.add_argument("--source_branch", default=None, help="Branch that produced baseline updates")
    parser.add_argument(
        "--trusted_source", default="protected_branch", help="Audit label for baseline samples written by this run"
    )
    return parser.parse_args()


def _find_bench_results(artifacts_dir: Path) -> list[tuple[Path, BenchResult]]:
    found = []
    for path in sorted(artifacts_dir.rglob("perf_smoke_test_result.json")):
        with path.open() as fh:
            found.append((path.parent, BenchResult.from_dict(json.load(fh))))
    return found


def _fmt(value, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}" if value is not None else "N/A"


def _short_sha(value: str | None) -> str:
    return value[:12] if value else "none"


def _bench_gpu_model(bench_result: BenchResult, fallback: str) -> str:
    launch_config = bench_result.launch_config or {}
    gpu_model = canonical_gpu_model(launch_config.get("gpu_model") or launch_config.get("gpu_model_raw"))
    return canonical_gpu_model(fallback) if gpu_model == "unknown_gpu" else gpu_model


def _thresholds(bench_result: BenchResult, gpu_model: str, backend: str) -> list[FpsMeanThreshold]:
    """Resolve configured FPS thresholds, preferring the run's launch_config artifact."""
    launch_config = bench_result.launch_config or {}
    raw = launch_config.get("fps_mean_thresholds")
    if raw is not None:
        return FpsMeanThreshold.from_list(raw, context=f"{bench_result.task_id}/{backend}")
    try:
        task = get_task(bench_result.task_id, backend)
        return task.thresholds_for(gpu_model)
    except Exception:
        return []


def _noise_floor_pct(bench_result: BenchResult, gpu_model: str, backend: str) -> float:
    """Resolve the per-task/GPU/backend noise floor % (0.0 when unconfigured)."""
    launch_config = bench_result.launch_config or {}
    if launch_config.get("noise_floor_pct") is not None:
        return float(launch_config.get("noise_floor_pct") or 0.0)
    try:
        task = get_task(bench_result.task_id, backend)
        for key in gpu_model_config_keys(gpu_model):
            value = (task.noise_floor_pct or {}).get(key, {}).get(backend)
            if value is not None:
                return float(value)
    except Exception:
        pass
    return 0.0


def _render_crossed(crossed: list[dict]) -> list[str]:
    """Render crossed thresholds as compact ``name(verdict)@value`` tags for reporting."""
    parts = []
    for rec in crossed:
        tag = rec.get("threshold_verdict") or "report"
        parts.append(f"crossed:{rec.get('threshold_name')}({tag})@{_fmt(rec.get('threshold'))}")
    return parts


def _fmt_change(value: float | None) -> str:
    """Format a signed percent change where positive means faster."""
    return f"{value:+.2f}%" if value is not None else "N/A"


def _runtime_context(bench_result: BenchResult) -> tuple[str, str]:
    """Return concise GPU and runtime labels for one benchmark result."""
    runtime_resources = bench_result.runtime_resources or {}
    launch_config = bench_result.launch_config or {}
    gpu_name = (
        runtime_resources.get("gpu_name") or launch_config.get("gpu_model_raw") or launch_config.get("gpu_model", "")
    )
    provenance = bench_result.provenance or {}
    software = provenance.get("software") or {}
    runtime = ", ".join(
        part
        for part in (
            f"cuda={runtime_resources.get('cuda_version')}" if runtime_resources.get("cuda_version") else "",
            f"driver={runtime_resources.get('nvidia_driver_version')}"
            if runtime_resources.get("nvidia_driver_version")
            else "",
            f"warp={software.get('warp')}" if software.get("warp") else "",
        )
        if part
    )
    return str(gpu_name or "N/A"), runtime or "N/A"


# --- Source-versus-image dependency skew -------------------------------------
#
# The gate bind-mounts Isaac Lab source over a prebuilt CI image, but the image
# supplies the installed third-party packages (Newton, Warp, Isaac Sim). Between
# a dependency-pin change landing on develop and the next image publish, a PR's
# source can reference a symbol the installed package does not have yet, which
# crashes every affected task before any FPS is measured. That crash says nothing
# about the PR's performance, so it is reported as a stale image and left
# advisory rather than read as a performance failure.

# Packages installed into the CI image rather than bind-mounted from the PR. A
# missing symbol in Isaac Lab's own source is a genuine defect in the change
# under test and still fails.
IMAGE_PROVIDED_PACKAGES: frozenset[str] = frozenset(
    {
        "carb",
        "isaacsim",
        "mujoco",
        "mujoco_warp",
        "newton",
        "omni",
        "pxr",
        "warp",
    }
)

# Python spells "this name is not in the installed package" three ways.
_MISSING_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ImportError: cannot import name ['\"](?P<symbol>\w+)['\"] from ['\"](?P<module>[\w.]+)['\"]"),
    re.compile(r"ModuleNotFoundError: No module named ['\"](?P<module>[\w.]+)['\"]"),
    re.compile(r"AttributeError: module ['\"](?P<module>[\w.]+)['\"] has no attribute ['\"](?P<symbol>\w+)['\"]"),
)


@dataclass(frozen=True)
class DependencySkew:
    """One detected mismatch between the PR's source and the image's packages."""

    package: str
    module: str
    symbol: str | None

    def describe(self) -> str:
        """Return a one-line reviewer-facing description of the mismatch."""
        if self.symbol:
            return f"`{self.module}` in the CI image has no `{self.symbol}`"
        return f"`{self.module}` is not installed in the CI image"


def detect_dependency_skew(log_text: str | None) -> DependencySkew | None:
    """Return the dependency skew a benchmark log indicates, if any.

    Args:
        log_text: Captured benchmark output, typically ``BenchResult.stdout_tail``.

    Returns:
        The detected mismatch, or ``None`` when the log shows no missing symbol
        from an image-provided package.
    """
    if not log_text:
        return None
    for pattern in _MISSING_NAME_PATTERNS:
        match = pattern.search(log_text)
        if match is None:
            continue
        module = match.group("module")
        package = module.split(".", 1)[0]
        if package not in IMAGE_PROVIDED_PACKAGES:
            continue
        return DependencySkew(package=package, module=module, symbol=match.groupdict().get("symbol"))
    return None


def _skewed_rows(rows: list[tuple]) -> list[tuple[str, str, DependencySkew]]:
    """Return ``(task_id, backend, skew)`` for failures caused by a stale CI image."""
    skewed = []
    for result, bench_result in rows:
        if result.verdict != OracleVerdict.HARD_FAILURE:
            continue
        skew = detect_dependency_skew(bench_result.stdout_tail)
        if skew is not None:
            skewed.append((result.task_id, result.backend, skew))
    return skewed


def _row_explanation(result) -> str:
    """Explain one verdict in reviewer-facing language."""
    if result.verdict == OracleVerdict.HARD_FAILURE:
        explanation = (
            f"Benchmark failed during {result.failure_phase}"
            if result.failure_phase
            else "Benchmark produced no usable FPS"
        )
    elif result.threshold_source == "no_baseline":
        explanation = "No compatible baseline yet"
    elif result.threshold_source == "insufficient_window":
        explanation = f"Baseline warming up ({result.baseline_sample_count}/{MIN_BASELINE_SAMPLES} samples)"
    elif result.verdict == OracleVerdict.BLOCK:
        explanation = "Blocking-level slowdown detected"
    elif result.verdict == OracleVerdict.WARN:
        explanation = "Possible slowdown; review recommended"
    else:
        explanation = "No meaningful slowdown"
    if result.was_retried:
        explanation += "; retry also failed" if result.verdict == OracleVerdict.HARD_FAILURE else "; result was retried"
    return explanation


def _build_reviewer_table(rows: list[tuple]) -> str:
    """Build the compact table reviewers see first."""
    verdict_labels = {
        OracleVerdict.PASS: "✅ PASS",
        OracleVerdict.WARN: "⚠️ WARN",
        OracleVerdict.BLOCK: "🚫 BLOCK",
        OracleVerdict.HARD_FAILURE: "❌ HARD FAILURE",
    }
    lines = [
        "| Task | Backend | Result | FPS | Baseline | Change | Samples | What it means |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for result, _ in rows:
        lines.append(
            f"| {result.task_id} | {result.backend} | {verdict_labels[result.verdict]}"
            f" | {_fmt(result.measured_fps)} | {_fmt(result.baseline_fps)} | {_fmt_change(result.regression_pct)}"
            f" | {result.baseline_sample_count} | {_row_explanation(result)} |"
        )
    return "\n".join(lines)


def _build_technical_table(rows: list[tuple]) -> str:
    """Build the complete diagnostic table shown on demand."""
    lines = [
        "| Task | Backend | Verdict | FPS | Baseline | Samples | Regression% | Floor | Threshold | Phase | "
        "Retry | GPU | Runtime | Note |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for result, bench_result in rows:
        gpu_name, runtime = _runtime_context(bench_result)
        # result.note already carries the config_mismatch string on the
        # config-mismatch HARD_FAILURE path; dedupe so it is not shown twice.
        note_parts = list(dict.fromkeys(part for part in (result.note, bench_result.config_mismatch) if part))
        note_parts.extend(_render_crossed(result.crossed_thresholds))
        lines.append(
            f"| {result.task_id} | {result.backend} | {result.verdict.value}"
            f" | {_fmt(result.measured_fps)} | {_fmt(result.baseline_fps)} | {result.baseline_sample_count}"
            f" | {_fmt(result.regression_pct, 2)} | {_fmt(result.hard_floor_fps)} | {result.threshold_source}"
            f" | {result.failure_phase or ''} | {'yes' if result.was_retried else 'no'} | {gpu_name}"
            f" | {runtime} | {'; '.join(note_parts)} |"
        )
    return "\n".join(lines)


def _build_stale_image_section(skewed: list[tuple[str, str, DependencySkew]]) -> str:
    """Explain that a stale CI image, not the PR, caused these failures."""
    packages = sorted({skew.package for _, _, skew in skewed})
    affected = "\n".join(f"- `{task_id}` ({backend}): {skew.describe()}" for task_id, backend, skew in skewed)
    return (
        "### Stale CI image\n\n"
        f"The prebuilt CI image does not match this PR's pinned {' and '.join(packages)} version, so the "
        "tasks below crashed before producing any FPS. This reflects the image, not the change under review, "
        "so these results are advisory and do not fail the check.\n\n"
        f"{affected}\n\n"
        "This resolves itself once the CI image is rebuilt for the current dependency pins. Re-run the gate "
        "after the next image publish to get real numbers for these tasks."
    )


def _coverage(rows: list[tuple]) -> tuple[list[str], int]:
    """Return ``(missing_labels, expected_total)`` for the configured matrix.

    A bucket whose job died before ``build_bench_result`` uploads nothing, so it
    contributes no row. Grading only the rows that arrived would report an
    all-clear over a bucket that was never measured, which is exactly the class
    of silent green this gate exists to catch. Returns ``([], 0)`` when the
    matrix cannot be read, so this can never invent a failure.

    Counts are over distinct buckets, not rows, so a duplicated artifact cannot
    inflate the total.
    """
    try:
        expected = {(task.task_id, task.backend_key) for task in load_tasks()}
    except Exception:
        return [], 0
    reported = {(result.task_id, result.backend) for result, _ in rows}
    missing = sorted(f"{task_id}/{backend}" for task_id, backend in expected - reported)
    return missing, len(expected)


def _build_summary_markdown(
    rows: list[tuple], *, blocking: bool, missing: list[str] | None = None, expected_total: int = 0
) -> str:
    """Build reviewer-first Markdown for the sticky PR comment.

    Args:
        rows: ``(OracleResult, BenchResult)`` pairs for every scored bucket.
        blocking: Whether the gate is in blocking mode, for the mode banner.
        missing: ``task/backend`` labels that produced no result at all. Kept in
            the headline so the comment cannot disagree with the commit status.
        expected_total: Size of the configured matrix, for the "N of M" phrasing.
    """
    missing = missing or []
    counts = {verdict: 0 for verdict in OracleVerdict}
    for result, _ in rows:
        counts[result.verdict] += 1

    skewed = _skewed_rows(rows)
    skewed_keys = {(task_id, backend) for task_id, backend, _ in skewed}
    unexplained_failures = sum(
        1
        for result, _ in rows
        if result.verdict == OracleVerdict.HARD_FAILURE and (result.task_id, result.backend) not in skewed_keys
    )

    if not rows:
        overall = "❌ No benchmark results were produced"
    elif unexplained_failures:
        overall = "❌ One or more benchmarks failed before producing usable performance data"
    elif missing:
        # Ranked above skew/BLOCK/WARN: the rows that did arrive may all be clean,
        # but the change is not covered, so no all-clear may be printed.
        reported_n = expected_total - len(missing)
        overall = f"❌ Only {reported_n} of {expected_total} benchmark buckets reported — coverage is incomplete"
    elif skewed:
        overall = "⚠️ The CI image is stale for this PR, so some tasks could not be measured"
    elif counts[OracleVerdict.BLOCK]:
        overall = "🚫 One or more blocking-level performance regressions were detected"
    elif counts[OracleVerdict.WARN]:
        overall = "⚠️ No confirmed regression, but one or more results need attention"
    else:
        overall = "✅ No meaningful performance regressions detected"

    warnings_label = "warning" if counts[OracleVerdict.WARN] == 1 else "warnings"
    blocks_label = "blocking signal" if counts[OracleVerdict.BLOCK] == 1 else "blocking signals"
    failures_label = "benchmark failure" if counts[OracleVerdict.HARD_FAILURE] == 1 else "benchmark failures"
    count_parts = [
        f"✅ {counts[OracleVerdict.PASS]} passed",
        f"⚠️ {counts[OracleVerdict.WARN]} {warnings_label}",
        f"🚫 {counts[OracleVerdict.BLOCK]} {blocks_label}",
        f"❌ {counts[OracleVerdict.HARD_FAILURE]} {failures_label}",
    ]
    if missing:
        # Name them: "8 of 9" alone leaves the reviewer guessing which task is
        # unmeasured, and a missing bucket is the one a regression can hide in.
        count_parts.append(f"🚫 {len(missing)} did not report ({', '.join(missing)})")
    count_summary = " · ".join(count_parts)
    mode = (
        "**Blocking:** BLOCK and HARD FAILURE results fail the check."
        if blocking
        else (
            "**Advisory:** this job stays green whatever the verdict. Anything that is not a clean pass --"
            " a BLOCK, a HARD FAILURE, or a bucket that never reported -- still shows up as a red"
            " `perf-smoke-test` commit status, and the reason is in the overall result above."
            " It just does not fail the PR."
        )
    )

    contexts = {_runtime_context(bench_result) for _, bench_result in rows}
    if len(contexts) == 1:
        gpu_name, runtime = next(iter(contexts))
    elif contexts:
        gpu_name, runtime = "Multiple; see technical details", "Multiple; see technical details"
    else:
        gpu_name, runtime = "N/A", "N/A"

    sections = [
        f"### Overall result\n\n**{overall}**\n\n{count_summary}\n\n{mode}",
        f"### Run context\n\n- **GPU:** {gpu_name}\n- **Runtime:** {runtime}",
    ]
    if skewed:
        sections.append(_build_stale_image_section(skewed))
    sections.extend(
        (
            "### How to read this\n\n"
            "Start with **BLOCK** and **HARD FAILURE**, then review any **WARN** rows.\n\n"
            "- **✅ PASS:** no meaningful slowdown was detected.\n"
            "- **⚠️ WARN:** the result is uncertain—for example, the baseline is still warming up, the run "
            "needed a retry, or performance is in the warning band.\n"
            "- **🚫 BLOCK:** performance crossed a blocking threshold. This fails the check only when the gate "
            "is in blocking mode.\n"
            "- **❌ HARD FAILURE:** the benchmark did not produce usable FPS, usually because it failed during "
            "import, initialization, or runtime.\n"
            "- **FPS:** current throughput; higher is better. **Baseline:** the median of compatible historical "
            "runs. **Change:** `+` is faster and `-` is slower.\n"
            f"- **Samples:** compatible historical runs. At least {MIN_BASELINE_SAMPLES} are required before "
            "the rolling baseline can make a confident decision.",
            _build_reviewer_table(rows),
            "<details>\n<summary>Technical details</summary>\n\n"
            "Threshold values, failure phases, retries, hardware, runtime versions, and diagnostic notes:\n\n"
            f"{_build_technical_table(rows)}\n\n</details>",
        )
    )
    return "\n\n".join(sections)


def _write_github_output(**values) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if not github_output:
        return
    with open(github_output, "a") as fh:
        for key, value in values.items():
            if value is not None:
                fh.write(f"{key}={value}\n")


def _verdict_outputs(
    rows: list[tuple],
    *,
    has_block: bool,
    has_hard_failure: bool,
    blocking: bool,
    missing: list[str] | None = None,
    expected_total: int = 0,
) -> dict[str, str]:
    """Derive the reported verdict and the `perf-smoke-test` commit status from the rows.

    The verdict is reported through the commit status, the sticky PR comment and
    the job summary -- never through the process exit code. Emitting it as a step
    output is what lets the workflow paint the status red on a real regression
    while the aggregate job itself stays green in advisory mode.

    Args:
        rows: ``(OracleResult, BenchResult)`` pairs for every scored bucket.
        has_block: Whether any bucket crossed a blocking threshold.
        has_hard_failure: Whether any bucket failed to produce a usable measurement,
            excluding failures already excused as a stale CI image.
        blocking: The gate's ``blocking`` setting, used only to label the status.
        missing: ``task/backend`` labels that produced no result. Shared with
            :func:`_build_summary_markdown` so the commit status and the PR
            comment cannot disagree about coverage.

    Returns:
        The ``overall_verdict`` / ``status_state`` / ``status_description`` /
        ``blocking`` step outputs.
    """
    # Derive from the rows, never from the booleans alone. `has_hard_failure` is
    # cleared for crashes excused as CI-image skew, and `main` only bails when
    # there are *zero* artifacts -- so a run where every bucket crashed, or where
    # most bench jobs never uploaded, would otherwise fall through to an
    # affirmative "no regression detected" over measurements that never happened.
    unmeasured = [result for result, _ in rows if result.verdict == OracleVerdict.HARD_FAILURE]
    missing = missing or []

    if has_hard_failure:
        verdict = OracleVerdict.HARD_FAILURE
        description = "perf-smoke: a benchmark failed to produce a usable measurement"
    elif unmeasured:
        # Excused as a stale CI image: not the change's fault, but nothing was
        # measured either, so the gate must not claim the change is clean.
        verdict = OracleVerdict.HARD_FAILURE
        description = f"perf-smoke: no usable measurement for {len(unmeasured)} bucket(s); CI image looks stale"
    elif missing or not rows:
        verdict = OracleVerdict.HARD_FAILURE
        description = f"perf-smoke: only {expected_total - len(missing)} of {expected_total} buckets reported a result"
    elif has_block:
        verdict = OracleVerdict.BLOCK
        description = "perf-smoke: blocking-level performance regression detected"
    elif any(result.verdict == OracleVerdict.WARN for result, _ in rows):
        verdict = OracleVerdict.WARN
        description = "perf-smoke: results need attention (see the verdict comment)"
    else:
        verdict = OracleVerdict.PASS
        description = f"perf-smoke: no meaningful regression across {len(rows)} buckets"

    state = "success" if verdict in (OracleVerdict.PASS, OracleVerdict.WARN) else "failure"
    if not blocking and state == "failure":
        description += " (advisory)"

    return {
        "overall_verdict": verdict.value,
        "status_state": state,
        "status_description": description,
        "blocking": "true" if blocking else "false",
    }


def _exit_code(*, baseline_update_failed: bool, has_block: bool, has_hard_failure: bool, blocking: bool) -> int:
    """Decide the process exit code.

    The exit code answers "did the gate run?", never "what did the gate
    conclude?". The conclusion travels through the ``perf-smoke-test`` commit
    status, the sticky PR comment, the job summary and the omni-github artifact.

    In advisory mode (``blocking: false``) every verdict, including
    HARD_FAILURE, exits 0. A crashed benchmark is still reported loudly -- the
    commit status goes red and the summary names the bucket that died. What
    advisory mode buys is that a registry outage or image drift unrelated to the
    change under test does not paint an unexplained red check on somebody else's
    pull request.

    Flipping ``blocking: true`` is the deliberate rollout step: HARD_FAILURE
    then exits 2 and BLOCK exits 1, failing the aggregate job itself.

    Gate malfunctions are the exception and stay fatal in both modes: a failed
    baseline push here, and (in :func:`main`) no bench artifacts at all or an
    unreadable baseline branch. Those mean no trustworthy verdict was produced.

    Returns:
        ``0`` to pass, ``1`` for a gate malfunction or a blocking regression,
        ``2`` for a blocking execution failure.
    """
    if baseline_update_failed:
        return 1
    if not blocking:
        return 0
    if has_hard_failure:
        return 2
    if has_block:
        return 1
    return 0


def main() -> int:
    args = _parse_args()
    use_flat = args.baselines_dir is not None
    allow_update = args.allow_baseline_update.strip().lower() in ("true", "1", "yes")
    baseline_remote = args.baseline_remote or None

    gate_config = load_gate_config(args.gate_config)
    blocking = bool(gate_config.get("blocking", False))
    min_block_regression_pct = float(gate_config.get("min_block_regression_pct", 3.0))
    baseline_push_retries = int(
        args.baseline_push_retries or gate_config.get("baseline_push_retries", BASELINE_PUSH_RETRIES)
    )

    items = _find_bench_results(args.artifacts_dir)
    if not items:
        print(f"[aggregate] No perf_smoke_test_result.json files found under {args.artifacts_dir}")
        return 1

    baseline_read_sha = None
    baseline_read_ref = None
    if not use_flat:
        try:
            baseline_read_sha = refresh_baseline_branch(
                args.baseline_branch, remote=baseline_remote, allow_missing=True
            )
            baseline_read_ref = baseline_read_sha
        except Exception as exc:
            print(f"::error::Failed to refresh baseline branch before reading: {exc}")
            return 1
        if baseline_read_sha:
            print(f"[aggregate] Baseline read snapshot: {args.baseline_branch}@{_short_sha(baseline_read_sha)}")
        else:
            print(f"[aggregate] Baseline branch {args.baseline_branch!r} not found; treating this as a seed run")

    rows = []
    has_block = False
    has_hard_failure = False
    baselines_updated = False
    baseline_update_failed = False
    pending_git_updates: list[BaselineUpdateRecord] = []
    # Tasks whose current runtime_contract_hash bucket is under-filled despite a
    # valid measurement this run. On a protected develop push these drive a targeted
    # reseed so a freshly opened bucket (new deps -> new runtime_contract_hash) is
    # filled at once instead of over ~MIN_BASELINE_SAMPLES self-seed pushes.
    underfilled_tasks: set[str] = set()

    for artifact_dir, bench_result in items:
        task_id = bench_result.task_id
        backend = bench_result.backend_key or bench_result.backend
        bench_gpu_model = _bench_gpu_model(bench_result, args.gpu_model)
        # The baseline_manager storage layer works with the serialized (dict) form.
        match_context = match_context_from_bench_result(
            bench_result.to_dict(),
            gpu_model=bench_gpu_model,
            base_sha=args.base_sha,
            target_branch=args.target_branch,
        )

        baseline = None
        try:
            if use_flat:
                baseline = load_baseline(
                    args.baselines_dir, bench_gpu_model, task_id, backend, match_context=match_context
                )
            elif baseline_read_ref:
                baseline = load_baseline_git(
                    baseline_read_ref,
                    bench_gpu_model,
                    task_id,
                    backend,
                    None,
                    match_context,
                )
        except Exception as exc:
            print(f"[aggregate] Warning: baseline load failed for {task_id}/{backend}: {exc}")

        oracle_result = compare(
            bench_result=bench_result,
            baseline=baseline,
            fps_mean_thresholds=_thresholds(bench_result, bench_gpu_model, backend),
            min_block_regression_pct=min_block_regression_pct,
            noise_floor_pct=_noise_floor_pct(bench_result, bench_gpu_model, backend),
        )
        rows.append((oracle_result, bench_result))

        # A valid run whose matching baseline window is short of MIN_BASELINE_SAMPLES
        # is an under-filled bucket for this exact runtime_contract_hash (the oracle
        # already scoped the load to it). Measured-fps guard avoids reseeding a task
        # that crashed this run (nothing trustworthy to base a bucket on).
        if oracle_result.measured_fps is not None and oracle_result.baseline_sample_count < MIN_BASELINE_SAMPLES:
            underfilled_tasks.add(task_id)

        crossed_summary = _render_crossed(oracle_result.crossed_thresholds)
        print(
            f"[aggregate] {task_id}/{backend}: {oracle_result.verdict.value}"
            f"  fps={_fmt(oracle_result.measured_fps)}  baseline={_fmt(oracle_result.baseline_fps)}"
            f"  samples={oracle_result.baseline_sample_count}  source={oracle_result.threshold_source}"
            + (f"  {'  '.join(crossed_summary)}" if crossed_summary else "")
        )

        if oracle_result.verdict == OracleVerdict.BLOCK:
            has_block = True
        elif oracle_result.verdict == OracleVerdict.HARD_FAILURE:
            # A crash caused by the image lacking a symbol this source pins is a
            # property of the image, not of the change under test, so it is
            # reported loudly but never fails the PR.
            skew = detect_dependency_skew(bench_result.stdout_tail)
            if skew is None:
                has_hard_failure = True
            else:
                print(f"[aggregate] {task_id}/{backend}: stale CI image; {skew.describe()}")

        if (
            allow_update
            and oracle_result.verdict in (OracleVerdict.PASS, OracleVerdict.WARN)
            and oracle_result.measured_fps is not None
        ):
            sample_metadata = make_sample_metadata(
                gpu_model=bench_gpu_model,
                task_id=task_id,
                backend=backend,
                fps=oracle_result.measured_fps,
                bench_result=bench_result.to_dict(),
                target_branch=args.target_branch,
                source_branch=args.source_branch,
                trusted_source=args.trusted_source,
            )
            if baseline_read_sha:
                sample_metadata["baseline_read_sha"] = baseline_read_sha

            if use_flat:
                try:
                    update_baseline(
                        args.baselines_dir,
                        bench_gpu_model,
                        task_id,
                        backend,
                        oracle_result.measured_fps,
                        sample_metadata=sample_metadata,
                    )
                    baselines_updated = True
                    print(f"[aggregate]   -> baseline updated locally: {oracle_result.measured_fps:.1f} FPS")
                except Exception as exc:
                    baseline_update_failed = True
                    print(f"::error::Baseline update failed for {task_id}/{backend}: {exc}")
            else:
                pending_git_updates.append(
                    BaselineUpdateRecord(
                        gpu_model=bench_gpu_model,
                        task_id=task_id,
                        backend=backend,
                        fps=oracle_result.measured_fps,
                        sample_metadata=sample_metadata,
                    )
                )
                print(f"[aggregate]   -> baseline update queued: {oracle_result.measured_fps:.1f} FPS")

    baseline_push_result = None
    if pending_git_updates:
        try:
            baseline_push_result = update_baselines_git(
                args.baseline_branch,
                pending_git_updates,
                remote=baseline_remote,
                max_retries=baseline_push_retries,
            )
            baselines_updated = baseline_push_result.pushed
            if baseline_push_result.pushed:
                print(
                    f"[aggregate] Baseline push succeeded: {args.baseline_branch}@"
                    f"{_short_sha(baseline_push_result.pushed_sha)} "
                    f"after {baseline_push_result.attempts} attempt(s)"
                )
            else:
                print("[aggregate] Baseline samples were already present; no push needed")
        except Exception as exc:
            baseline_update_failed = True
            print(f"::error::Baseline push failed: {exc}")

    # One source of truth for coverage, shared by the summary/comment and the
    # commit status so the two surfaces cannot contradict each other.
    missing_buckets, expected_total = _coverage(rows)
    summary = _build_summary_markdown(rows, blocking=blocking, missing=missing_buckets, expected_total=expected_total)
    print("\n## Performance Smoke Results\n")
    print(summary)
    print()

    if args.summary_file:
        with open(args.summary_file, "a") as fh:
            fh.write("\n## Performance Smoke Results\n\n")
            if not use_flat:
                fh.write(f"Baseline read SHA: `{_short_sha(baseline_read_sha)}`\n\n")
                if baseline_push_result and baseline_push_result.pushed_sha:
                    fh.write(
                        f"Baseline pushed SHA: `{_short_sha(baseline_push_result.pushed_sha)}` "
                        f"after {baseline_push_result.attempts} attempt(s)\n\n"
                    )
            fh.write(summary)
            fh.write("\n")

    if args.omni_github_dir:
        write_omni_github_artifact(
            rows,
            args.omni_github_dir,
            platform=args.omni_platform,
            app_config=args.omni_app_config,
        )

    output_values = {
        "baseline_read_sha": baseline_read_sha,
        **_verdict_outputs(
            rows,
            has_block=has_block,
            has_hard_failure=has_hard_failure,
            blocking=blocking,
            missing=missing_buckets,
            expected_total=expected_total,
        ),
    }
    if baseline_push_result:
        output_values.update(
            {
                "baselines_updated": "true" if baseline_push_result.pushed else "false",
                "baseline_pushed_sha": baseline_push_result.pushed_sha,
                "baseline_push_attempts": baseline_push_result.attempts,
            }
        )
    elif baselines_updated:
        output_values["baselines_updated"] = "true"
    if underfilled_tasks:
        # Consumed by the gate's reseed job (protected develop push only) to seed
        # exactly these tasks for HEAD. Emitted everywhere for visibility; the
        # workflow gates the actual reseed on the trusted push context.
        reseed_tasks = ",".join(sorted(underfilled_tasks))
        output_values["reseed_tasks"] = reseed_tasks
        output_values["reseed_min_samples"] = MIN_BASELINE_SAMPLES
        print(f"[aggregate] Under-filled buckets flagged for reseed: {reseed_tasks}")
    _write_github_output(**output_values)

    return _exit_code(
        baseline_update_failed=baseline_update_failed,
        has_block=has_block,
        has_hard_failure=has_hard_failure,
        blocking=blocking,
    )


if __name__ == "__main__":
    sys.exit(main())
