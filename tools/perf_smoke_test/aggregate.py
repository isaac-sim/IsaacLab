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
    parser.add_argument(
        "--confirm_rerun_mode",
        choices=["none", "local", "docker"],
        default="none",
        help="Re-run BLOCK cells before finalizing the verdict: 'local' (./isaaclab.sh), 'docker' (CI image), or 'none'",
    )
    parser.add_argument(
        "--confirm_block_reruns",
        type=int,
        default=0,
        help="Number of extra reruns per BLOCK cell when confirm_rerun_mode != none (median of all attempts decides)",
    )
    parser.add_argument("--ci_image_tag", default=None, help="CI image tag for confirm_rerun_mode=docker")
    parser.add_argument("--workspace", type=Path, default=None, help="Workspace root for confirm_rerun_mode=docker")
    return parser.parse_args()


def _find_bench_results(artifacts_dir: Path) -> list[tuple[Path, dict]]:
    found = []
    for path in sorted(artifacts_dir.rglob("perf_smoke_test_result.json")):
        with path.open() as fh:
            found.append((path.parent, json.load(fh)))
    return found


def _excluded_frames(bench_result: dict) -> frozenset[int]:
    launch_config = bench_result.get("launch_config") or {}
    raw = launch_config.get("excluded_frames_raw")
    if raw is None:
        raw = (bench_result.get("task_config_snapshot") or {}).get("excluded_frames_raw", [])
    indices: set[int] = set()
    for entry in raw or []:
        if isinstance(entry, list):
            indices.update(range(int(entry[0]), int(entry[1]) + 1))
        else:
            indices.add(int(entry))
    return frozenset(indices)


def _fmt(value, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}" if value is not None else "N/A"


def _fmt_pct(value, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}%" if value is not None else "N/A"


def _fmt_signed_pct(value, decimals: int = 2) -> str:
    return f"{value:+.{decimals}f}%" if value is not None else "N/A"


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


def _noise_floor_pct(bench_result: dict, gpu_model: str, backend: str) -> float:
    launch_config = bench_result.get("launch_config") or {}
    if launch_config.get("noise_floor_pct") is not None:
        return float(launch_config.get("noise_floor_pct") or 0.0)
    try:
        task = get_task(bench_result["task_id"], backend)
        for key in gpu_model_config_keys(gpu_model):
            value = task.noise_floor_pct.get(key, {}).get(backend)
            if value is not None:
                return float(value)
        return 0.0
    except Exception:
        return 0.0


def _runtime_label(bench_result: dict) -> str:
    """Return the compact runtime label shown in the sticky summary."""
    gpu_diag = bench_result.get("gpu_diag") or {}
    provenance = bench_result.get("provenance") or {}
    software = provenance.get("software") or {}
    return ", ".join(
        part
        for part in (
            f"cuda={gpu_diag.get('cuda_version')}" if gpu_diag.get("cuda_version") else "",
            f"driver={gpu_diag.get('nvidia_driver_version')}" if gpu_diag.get("nvidia_driver_version") else "",
            f"warp={software.get('warp')}" if software.get("warp") else "",
        )
        if part
    )


def _collapse_values(values: list[str]) -> str:
    """Render a list of possibly repeated values as one summary value."""
    unique = sorted({value for value in values if value})
    if not unique:
        return "N/A"
    if len(unique) == 1:
        return unique[0]
    return "varies: " + "; ".join(unique)


def _uniform_value(values: list[str]) -> str | None:
    """Return the shared value when every row agrees, otherwise ``None``."""
    unique = {value for value in values if value}
    return next(iter(unique)) if len(unique) == 1 else None


def _threshold_sources(rows: list[tuple]) -> list[str]:
    return [result.threshold_source for result, _ in rows]


def _threshold_policy_label(source: str) -> str:
    """Return a reader-friendly explanation of a shared threshold policy."""
    if source == "rolling_window":
        return (
            "rolling baseline per row. Each task/backend is compared against its own compatible baseline "
            "samples using median/MAD."
        )
    if source == "n/a":
        return "not applicable because no usable FPS was produced for baseline comparison."
    return f"{source} for every row."


def _build_run_context(rows: list[tuple]) -> str:
    """Return run-wide fields that should not be repeated in every table row."""
    gpu_names = []
    runtimes = []
    for _result, bench_result in rows:
        gpu_diag = bench_result.get("gpu_diag") or {}
        launch_config = bench_result.get("launch_config") or {}
        gpu_names.append(gpu_diag.get("gpu_name") or launch_config.get("gpu_model_raw") or launch_config.get("gpu_model", ""))
        runtimes.append(_runtime_label(bench_result))

    lines = [
        "### Run context",
        "",
        f"- **GPU:** {_collapse_values(gpu_names)}",
        f"- **Runtime:** {_collapse_values(runtimes)}",
    ]
    shared_threshold = _uniform_value(_threshold_sources(rows))
    if shared_threshold is not None:
        lines.append(f"- **Threshold policy:** {_threshold_policy_label(shared_threshold)}")
    return "\n".join(lines)


def _build_summary_table(rows: list[tuple]) -> str:
    # Threshold only earns a column when it actually differs across tasks; if every
    # task shares the same threshold source it is reported once in the run context.
    show_threshold = _uniform_value(_threshold_sources(rows)) is None

    header = ["Task", "Backend", "Verdict", "FPS", "Baseline", "Delta (+ faster / - slower)", "Noise", "Samples"]
    if show_threshold:
        header.append("Threshold")
    header += ["Phase", "Notes", "Retried"]

    aligns = ["---", "---", "---", "---:", "---:", "---:", "---:", "---:"]
    if show_threshold:
        aligns.append("---")
    aligns += ["---", "---", "---"]

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(aligns) + "|"]
    for result, bench_result in rows:
        note_parts = [part for part in (result.note, bench_result.get("config_mismatch")) if part]
        note_parts.extend(_render_crossed(result.crossed_thresholds))
        if bench_result.get("p99_over_median") is not None:
            note_parts.append(f"p99/med={bench_result['p99_over_median']}")
        if bench_result.get("outlier_count") is not None:
            note_parts.append(f"outliers={bench_result['outlier_count']}")
        cells = [
            result.task_id,
            result.backend,
            result.verdict.value,
            _fmt(result.measured_fps),
            _fmt(result.baseline_fps),
            _fmt_signed_pct(result.regression_pct),
            _fmt_pct(result.effective_noise_pct if result.effective_noise_pct is not None else result.baseline_noise_pct),
            str(result.baseline_sample_count),
        ]
        if show_threshold:
            cells.append(result.threshold_source)
        cells += [
            result.failure_phase or "",
            "; ".join(note_parts),
            "yes" if result.was_retried else "no",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _build_summary_notes() -> str:
    return "\n".join(
        [
            "### How to read this table",
            "",
            "- **PASS**: no meaningful slowdown was detected.",
            "- **WARN**: suspicious or uncertain result, such as insufficient baselines, retry-only success, or a"
            " slowdown in the WARN band.",
            "- **BLOCK**: a blocking-level regression signal was detected. In this POC, BLOCK is advisory (it flags a"
            " regression) and only fails the check when the gate is explicitly configured as blocking.",
            "- **HARD_FAILURE**: the benchmark did not produce usable FPS data, for example an import/init/runtime"
            " failure or config mismatch.",
            "- **Delta**: change in FPS versus the baseline. ``+`` means faster than baseline (speedup);"
            " ``-`` means slower than baseline (slowdown).",
            "- **Threshold policy**: explains which comparison rule produced the verdict. ``rolling_window`` means"
            " each row uses its own compatible baseline median/MAD window, not one shared FPS cutoff.",
            "- **Noise**: baseline MAD as a percent of baseline FPS. Higher noise means the task/backend naturally"
            " varies more from run to run. If a calibrated noise floor is configured, this shows the effective"
            " noise used for WARN/BLOCK thresholds.",
            "- **Phase**: the stage a HARD_FAILURE happened in (e.g. ``import``, ``init``, ``runtime``); blank when"
            " the benchmark ran to completion.",
            "- **Retried**: ``yes`` if the cell only passed after an automatic re-run.",
        ]
    )


def _write_github_output(**values) -> None:
    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if not github_output:
        return
    with open(github_output, "a") as fh:
        for key, value in values.items():
            if value is not None:
                fh.write(f"{key}={value}\n")


def _evaluate_cell(bench_result: dict, baseline, bench_gpu_model: str, backend: str, artifact_dir: Path, min_block_regression_pct: float):
    """Run the oracle for one cell with hard-floor and noise-floor config applied."""
    return compare(
        bench_result=bench_result,
        baseline=baseline,
        fps_mean_thresholds=_thresholds(bench_result, bench_gpu_model, backend),
        excluded_frames=_excluded_frames(bench_result),
        artifact_dir=artifact_dir,
        min_block_regression_pct=min_block_regression_pct,
        noise_floor_pct=_noise_floor_pct(bench_result, bench_gpu_model, backend),
    )


def _make_rerun_fn(args):
    """Build the confirm-on-BLOCK rerun backend, or None when confirmation is off."""
    mode = (args.confirm_rerun_mode or "none").lower()
    if mode == "none" or args.confirm_block_reruns <= 0:
        return None
    if mode == "local":
        from confirm import make_local_rerun

        repo_root = _TOOLS_DIR.parent
        return make_local_rerun(repo_root, repo_root / "scripts" / "benchmarks" / "benchmark_non_rl.py")
    if mode == "docker":
        if not args.ci_image_tag or not args.workspace:
            print("::error::confirm_rerun_mode=docker requires --ci_image_tag and --workspace; skipping confirmation")
            return None
        from confirm import make_docker_rerun

        return make_docker_rerun(Path(args.workspace), args.ci_image_tag)
    return None


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

    has_block = False
    has_hard_failure = False
    baselines_updated = False
    baseline_update_failed = False
    pending_git_updates: list[BaselineUpdateRecord] = []

    # Pass 1: evaluate every cell. Keep the per-cell context so BLOCK cells can be
    # re-scored after confirmation without re-reading baselines.
    cells: list[dict] = []
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

        oracle_result = _evaluate_cell(
            bench_result, baseline, bench_gpu_model, backend, artifact_dir, min_block_regression_pct
        )
        cells.append(
            {
                "artifact_dir": artifact_dir,
                "bench_result": bench_result,
                "task_id": task_id,
                "backend": backend,
                "bench_gpu_model": bench_gpu_model,
                "baseline": baseline,
                "result": oracle_result,
            }
        )

        crossed_summary = _render_crossed(oracle_result.crossed_thresholds)
        print(
            f"[aggregate] {task_id}/{backend}: {oracle_result.verdict.value}"
            f"  fps={_fmt(oracle_result.measured_fps)}  baseline={_fmt(oracle_result.baseline_fps)}"
            f"  samples={oracle_result.baseline_sample_count}  source={oracle_result.threshold_source}"
            + (f"  {'  '.join(crossed_summary)}" if crossed_summary else "")
        )

    # Confirm-on-BLOCK: re-run only the cells that initially blocked, then re-score
    # them against the median of all attempts (see confirm.py / oracle.py).
    rerun_fn = _make_rerun_fn(args)
    block_cells = [cell for cell in cells if cell["result"].verdict == OracleVerdict.BLOCK]
    if rerun_fn is not None and block_cells:
        from confirm import confirm_block_cell

        print(f"[aggregate] confirm-on-BLOCK: re-running {len(block_cells)} cell(s) x{args.confirm_block_reruns}")
        for cell in block_cells:
            confirm_block_cell(
                cell["bench_result"],
                cell["artifact_dir"],
                cell["artifact_dir"] / "perf_smoke_test_result.json",
                _excluded_frames(cell["bench_result"]),
                rerun_fn,
                args.confirm_block_reruns,
            )
            cell["result"] = _evaluate_cell(
                cell["bench_result"],
                cell["baseline"],
                cell["bench_gpu_model"],
                cell["backend"],
                cell["artifact_dir"],
                min_block_regression_pct,
            )
            print(
                f"[aggregate] {cell['task_id']}/{cell['backend']}: confirmed verdict"
                f" {cell['result'].verdict.value} ({cell['result'].note})"
            )

    # Finalize: build the verdict rows and apply baseline updates from FINAL verdicts.
    rows = []
    for cell in cells:
        oracle_result = cell["result"]
        bench_result = cell["bench_result"]
        task_id = cell["task_id"]
        backend = cell["backend"]
        bench_gpu_model = cell["bench_gpu_model"]
        rows.append((oracle_result, bench_result))

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
    print(_build_run_context(rows))
    print()
    print(_build_summary_notes())
    print()
    print(table)
    print()

    if args.summary_file:
        with open(args.summary_file, "a") as fh:
            fh.write("## Performance Smoke Results\n\n")
            if not use_flat:
                fh.write(f"Baseline read SHA: `{_short_sha(baseline_read_sha)}`\n\n")
                if baseline_push_result and baseline_push_result.pushed_sha:
                    fh.write(
                        f"Baseline pushed SHA: `{_short_sha(baseline_push_result.pushed_sha)}` "
                        f"after {baseline_push_result.attempts} attempt(s)\n\n"
                    )
            fh.write(_build_run_context(rows))
            fh.write("\n\n")
            fh.write(_build_summary_notes())
            fh.write("\n\n")
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
