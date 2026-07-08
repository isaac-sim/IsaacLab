# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aggregate benchmark artifacts, run the oracle, and update trusted baselines"""

import argparse
import json
import os
import sys
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
from gate_config import BASELINE_PUSH_RETRIES, load_gate_config  # noqa: E402
from gate_types import FpsMeanThreshold, OracleVerdict  # noqa: E402
from gpu_identity import canonical_gpu_model  # noqa: E402
from oracle import compare  # noqa: E402
from task_config import get_task  # noqa: E402


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
    parser.add_argument("--base_sha", default=None, help="PR base SHA for ancestry-aware baseline matching")
    parser.add_argument("--target_branch", default=None, help="Target protected branch, e.g. main/develop/release/x")
    parser.add_argument("--source_branch", default=None, help="Branch that produced baseline updates")
    parser.add_argument(
        "--trusted_source", default="protected_branch", help="Audit label for baseline samples written by this run"
    )
    return parser.parse_args()


def _find_bench_results(artifacts_dir: Path) -> list[tuple[Path, dict]]:
    found = []
    for path in sorted(artifacts_dir.rglob("perf_smoke_test_result.json")):
        with path.open() as fh:
            found.append((path.parent, json.load(fh)))
    return found


def _fmt(value, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}" if value is not None else "N/A"


def _short_sha(value: str | None) -> str:
    return value[:12] if value else "none"


def _bench_gpu_model(bench_result: dict, fallback: str) -> str:
    launch_config = bench_result.get("launch_config") or {}
    gpu_model = canonical_gpu_model(launch_config.get("gpu_model") or launch_config.get("gpu_model_raw"))
    return canonical_gpu_model(fallback) if gpu_model == "unknown_gpu" else gpu_model


def _thresholds(bench_result: dict, gpu_model: str, backend: str) -> list[FpsMeanThreshold]:
    """Resolve configured FPS thresholds, preferring the run's launch_config artifact."""
    launch_config = bench_result.get("launch_config") or {}
    raw = launch_config.get("fps_mean_thresholds")
    if raw is not None:
        return FpsMeanThreshold.from_list(raw, context=f"{bench_result.get('task_id')}/{backend}")
    try:
        task = get_task(bench_result["task_id"], backend)
        return task.thresholds_for(gpu_model)
    except Exception:
        return []


def _render_crossed(crossed: list[dict]) -> list[str]:
    """Render crossed thresholds as compact ``name(verdict)@value`` tags for reporting."""
    parts = []
    for rec in crossed:
        tag = rec.get("threshold_verdict") or "report"
        parts.append(f"crossed:{rec.get('threshold_name')}({tag})@{_fmt(rec.get('threshold'))}")
    return parts


def _build_summary_table(rows: list[tuple]) -> str:
    lines = [
        "| Task | Backend | Verdict | FPS | Baseline | Samples | Regression% | Floor | Threshold | Phase | "
        "Retry | GPU | Runtime | Note |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for result, bench_result in rows:
        gpu_diag = bench_result.get("gpu_diag") or {}
        launch_config = bench_result.get("launch_config") or {}
        gpu_name = gpu_diag.get("gpu_name") or launch_config.get("gpu_model_raw") or launch_config.get("gpu_model", "")
        provenance = bench_result.get("provenance") or {}
        software = provenance.get("software") or {}
        runtime = ", ".join(
            part
            for part in (
                f"cuda={gpu_diag.get('cuda_version')}" if gpu_diag.get("cuda_version") else "",
                f"driver={gpu_diag.get('nvidia_driver_version')}" if gpu_diag.get("nvidia_driver_version") else "",
                f"warp={software.get('warp')}" if software.get("warp") else "",
            )
            if part
        )
        # result.note already carries the config_mismatch string on the
        # config-mismatch HARD_FAILURE path; dedupe so it is not shown twice.
        note_parts = list(dict.fromkeys(part for part in (result.note, bench_result.get("config_mismatch")) if part))
        note_parts.extend(_render_crossed(result.crossed_thresholds))
        lines.append(
            f"| {result.task_id} | {result.backend} | {result.verdict.value}"
            f" | {_fmt(result.measured_fps)} | {_fmt(result.baseline_fps)} | {result.baseline_sample_count}"
            f" | {_fmt(result.regression_pct, 2)} | {_fmt(result.hard_floor_fps)} | {result.threshold_source}"
            f" | {result.failure_phase or ''} | {result.was_retried} | {gpu_name}"
            f" | {runtime} | {'; '.join(note_parts)} |"
        )
    return "\n".join(lines)


def _write_github_output(**values) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if not github_output:
        return
    with open(github_output, "a") as fh:
        for key, value in values.items():
            if value is not None:
                fh.write(f"{key}={value}\n")


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

    for artifact_dir, bench_result in items:
        task_id = bench_result["task_id"]
        backend = bench_result.get("backend_key") or bench_result.get("backend")
        bench_gpu_model = _bench_gpu_model(bench_result, args.gpu_model)
        match_context = match_context_from_bench_result(
            bench_result,
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
        )
        rows.append((oracle_result, bench_result))

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
            has_hard_failure = True

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
                bench_result=bench_result,
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

    table = _build_summary_table(rows)
    print("\n## Performance Smoke Results\n")
    print(table)
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
            fh.write(table)
            fh.write("\n")

    output_values = {"baseline_read_sha": baseline_read_sha}
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
    _write_github_output(**output_values)

    if baseline_update_failed:
        return 1

    if blocking:
        if has_block:
            return 1
        if has_hard_failure:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
