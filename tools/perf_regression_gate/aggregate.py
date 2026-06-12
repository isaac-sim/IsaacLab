# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CI aggregate script: load per-task perf_regression_gate_result.json, run oracle, update baselines.

Scans the local (downloaded) artifacts directory for perf_regression_gate_result.json files, runs the
oracle for each, writes a GitHub Step Summary table, and optionally updates the baselines branch.

Per-task threshold overrides come from ``baseline_overrides.json`` (committed with
the PR), and baselines are bucketed by an environment ``--fingerprint`` resolved
through a fallback chain so a dependency/driver bump still gates against the
nearest compatible history.

Usage::

    python3 tools/perf_regression_gate/aggregate.py \\
        --artifacts_dir artifacts/ \\
        --gpu_model "NVIDIA L40S" \\
        --gate_config tools/perf_regression_gate/gate_config.json \\
        --baseline_branch perf-baselines \\
        --fingerprint "warp1.12.0/<runtime_hash>/<code_fingerprint>" \\
        --allow_baseline_update true \\
        --summary_file "$GITHUB_STEP_SUMMARY"

For offline/test use, pass ``--baselines_dir`` to read/write flat files instead
of the git baselines branch::

    python3 tools/perf_regression_gate/aggregate.py \\
        --artifacts_dir artifacts/ \\
        --gpu_model "NVIDIA L40S" \\
        --baselines_dir local_baselines/
"""

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
    load_baseline_git_resolved,
    load_baseline_resolved,
    update_baseline,
    update_baseline_git,
)
from gate_config import load_gate_config  # noqa: E402
from oracle import OracleVerdict, compare  # noqa: E402
from task_config import get_task  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser(description="Aggregate bench results and run oracle.")
    p.add_argument(
        "--artifacts_dir",
        required=True,
        type=Path,
        help="Root directory containing per-task artifact subdirectories",
    )
    p.add_argument("--gpu_model", default="NVIDIA L40S")
    p.add_argument("--gate_config", type=Path, default=_MODULE_DIR / "gate_config.json")
    p.add_argument(
        "--baseline_branch",
        default=DEFAULT_BASELINE_BRANCH,
        help=f"Git branch for baseline storage (default: {DEFAULT_BASELINE_BRANCH})",
    )
    p.add_argument(
        "--baselines_dir",
        type=Path,
        default=None,
        help="Flat-file baseline directory; bypasses git (use for offline testing)",
    )
    p.add_argument(
        "--overrides",
        type=Path,
        default=_MODULE_DIR / "baseline_overrides.json",
        help="Path to baseline_overrides.json (manual per-task threshold overrides, committed with the PR)",
    )
    p.add_argument(
        "--fingerprint",
        default=None,
        help="Environment fingerprint bucket ({backend_version}/{runtime_hash}/{code_fingerprint}); "
        "loads resolve outward through looser buckets, writes target this exact bucket",
    )
    p.add_argument(
        "--allow_baseline_update",
        default="false",
        help="Update baselines for PASS/WARN results ('true'/'false', default: false)",
    )
    p.add_argument(
        "--summary_file",
        default=None,
        help="Append step-summary markdown to this path (set to $GITHUB_STEP_SUMMARY in CI)",
    )
    return p.parse_args()


def _find_bench_results(artifacts_dir: Path) -> list[tuple[Path, dict]]:
    """Return list of (artifact_dir: Path, perf_regression_gate_result: dict) sorted by task_id."""
    found = []
    for p in sorted(artifacts_dir.rglob("perf_regression_gate_result.json")):
        with p.open() as fh:
            bench_result = json.load(fh)
        found.append((p.parent, bench_result))
    return found


def _excluded_frames(bench_result: dict) -> frozenset:
    """Expand excluded_frames_raw from the task_config_snapshot into a frozenset."""
    raw = (bench_result.get("task_config_snapshot") or {}).get("excluded_frames_raw", [])
    indices: set[int] = set()
    for entry in raw:
        if isinstance(entry, list):
            indices.update(range(entry[0], entry[1] + 1))
        else:
            indices.add(int(entry))
    return frozenset(indices)


def _overrides_for(overrides: dict, task: str, gpu_key: str) -> dict:
    """Merge global defaults < per-task < per-task/gpu override blocks.

    Reserved top-level keys (``_defaults``) apply everywhere; a task block's scalar
    keys apply to all GPUs for that task, and a nested ``<gpu>`` block refines them.
    """
    merged: dict[str, object] = {}
    defaults = overrides.get("_defaults")
    if isinstance(defaults, dict):
        merged.update(defaults)
    task_block = overrides.get(task)
    if isinstance(task_block, dict):
        merged.update({k: v for k, v in task_block.items() if not isinstance(v, dict)})
        for key, block in task_block.items():
            if isinstance(block, dict) and (key == gpu_key or key in gpu_key or gpu_key in key):
                merged.update(block)
                break
    return merged


def _fmt(v, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}" if v is not None else "N/A"


def _build_summary_table(rows: list[tuple]) -> str:
    lines = [
        "| Task | Backend | Verdict | FPS (mean) | Center | Spread | Regression% | Source | Wall (s) | Note |",
        "|------|---------|---------|------------|--------|--------|-------------|--------|----------|------|",
    ]
    for r, matched_fp in rows:
        source = r.threshold_source
        if matched_fp:
            source = f"{source} @ {matched_fp}"
        lines.append(
            f"| {r.task_id} | {r.backend} | {r.verdict.value}"
            f" | {_fmt(r.measured_fps)} | {_fmt(r.baseline_fps)} | {_fmt(r.spread_fps)}"
            f" | {_fmt(r.regression_pct, 2)} | {source} | {_fmt(r.wall_time_s, 1)} | {r.note or ''} |"
        )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    use_flat = args.baselines_dir is not None
    allow_update = args.allow_baseline_update.strip().lower() in ("true", "1", "yes")

    gate_config = load_gate_config(args.gate_config)
    blocking = gate_config.get("blocking", False)

    overrides_doc: dict = {}
    if args.overrides and Path(args.overrides).exists():
        with open(args.overrides) as fh:
            overrides_doc = json.load(fh)

    items = _find_bench_results(args.artifacts_dir)
    if not items:
        print(f"[aggregate] No perf_regression_gate_result.json files found under {args.artifacts_dir}")
        return 1

    rows: list[tuple] = []
    has_block = False
    has_hard_failure = False
    baselines_updated = False

    for artifact_dir, bench_result in items:
        task_id = bench_result["task_id"]
        backend = bench_result.get("backend_key")

        # Load rolling baseline through the fingerprint fallback chain.
        baseline = None
        matched_fp = None
        try:
            if use_flat:
                baseline, matched_fp = load_baseline_resolved(
                    args.baselines_dir, args.gpu_model, task_id, backend, args.fingerprint
                )
            else:
                baseline, matched_fp = load_baseline_git_resolved(
                    args.baseline_branch, args.gpu_model, task_id, backend, args.fingerprint
                )
        except Exception as exc:
            print(f"[aggregate] Warning: baseline load failed for {task_id}/{backend}: {exc}")

        # Relative catastrophic floor (fps_floor_pct% of the per-GPU ref_fps); 0 disables it.
        try:
            task = get_task(task_id, backend)
            fps_mean_floor = task.fps_floor(args.gpu_model)
        except Exception:
            fps_mean_floor = 0.0

        ov = _overrides_for(overrides_doc, task_id, args.gpu_model)

        oracle_result = compare(
            bench_result=bench_result,
            baseline=baseline,
            fps_mean_floor=fps_mean_floor,
            excluded_frames=_excluded_frames(bench_result),
            artifact_dir=artifact_dir,
            overrides=ov,
        )
        rows.append((oracle_result, matched_fp))

        bucket = f" (bucket={matched_fp})" if matched_fp else ""
        print(
            f"[aggregate] {task_id}/{backend}: {oracle_result.verdict.value}"
            f"  fps={_fmt(oracle_result.measured_fps)}  center={_fmt(oracle_result.baseline_fps)}"
            f"  src={oracle_result.threshold_source}{bucket}"
        )

        if oracle_result.verdict == OracleVerdict.BLOCK:
            has_block = True
        elif oracle_result.verdict == OracleVerdict.HARD_FAILURE:
            has_hard_failure = True

        # Update baseline only for measured PASS/WARN results (never for BLOCK/HARD_FAILURE).
        # Writes always target the exact --fingerprint bucket (loads relax outward).
        if (
            allow_update
            and oracle_result.verdict in (OracleVerdict.PASS, OracleVerdict.WARN)
            and oracle_result.measured_fps is not None
        ):
            try:
                if use_flat:
                    update_baseline(
                        args.baselines_dir,
                        args.gpu_model,
                        task_id,
                        backend,
                        oracle_result.measured_fps,
                        fingerprint=args.fingerprint,
                    )
                else:
                    update_baseline_git(
                        args.baseline_branch,
                        args.gpu_model,
                        task_id,
                        backend,
                        oracle_result.measured_fps,
                        args.fingerprint,
                    )
                baselines_updated = True
                print(f"[aggregate]   -> baseline updated: {oracle_result.measured_fps:.1f} FPS")
            except Exception as exc:
                print(f"[aggregate] Warning: baseline update failed for {task_id}/{backend}: {exc}")

    table = _build_summary_table(rows)
    print("\n## Performance Gate Results\n")
    print(table)
    print()

    if args.summary_file:
        with open(args.summary_file, "a") as fh:
            fh.write("\n## Performance Gate Results\n\n")
            fh.write(table)
            fh.write("\n")

    # Signal baseline push to the calling workflow step
    if baselines_updated:
        github_output = os.environ.get("GITHUB_OUTPUT", "")
        if github_output:
            with open(github_output, "a") as fh:
                fh.write("baselines_updated=true\n")
        print(f"[aggregate] Baselines updated; workflow will push {args.baseline_branch!r}")

    if blocking:  # from gate_config.py, explicit PR to make gate blocking
        if has_block:
            return 1
        if has_hard_failure:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
