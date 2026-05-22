# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare an OmniPerf benchmark JSON against a per-GPU baseline.

Used by the Phase 1 perf-smoke CI gate. Runs the comparison logic and exits
with one of three codes that the workflow translates into a PR signal:

* ``0`` PASS              -- measured FPS within or above the baseline threshold.
* ``1`` REGRESSION        -- measured FPS dropped beyond ``max_regression_pct``.
* ``2`` HARD_FAILURE      -- structural problem (missing/malformed result, missing
  baseline entry, NaN/zero FPS, GPU model not in baseline, etc.).

The split between regression and hard failure is intentional: it lets us
distinguish "your code got slower" from "the environment is broken" without
relying on log-grep heuristics.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

EXIT_PASS = 0
EXIT_REGRESSION = 1
EXIT_HARD_FAILURE = 2

METRIC_PHASE = "runtime"
METRIC_NAME = "Mean Environment step effective FPS"

DEFAULT_GLOB_TEMPLATE = "benchmark_non_rl_{task}*.json"


class CompareError(Exception):
    """Raised for any structural problem that should map to ``HARD_FAILURE``."""


def _emit(result: str, **fields: object) -> None:
    """Print a structured single-line summary, optionally to ``$GITHUB_STEP_SUMMARY``.

    Args:
        result: One of ``PASS``, ``REGRESSION``, ``HARD_FAILURE``.
        **fields: Additional ``key=value`` pairs to include on the line.
    """
    parts = [f"RESULT={result}"] + [f"{k}={v}" for k, v in fields.items()]
    line = " ".join(parts)
    print(line)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(f"### Perf Smoke: {result}\n\n```\n{line}\n```\n")
        except OSError:
            # Summary is best-effort; never let it mask the real exit code.
            pass


def _resolve_results(results_dir: str, glob_pattern: str, allow_multiple: bool) -> Path:
    """Resolve the result JSON path within ``results_dir``.

    Args:
        results_dir: Directory the benchmark wrote into.
        glob_pattern: Glob pattern relative to ``results_dir``.
        allow_multiple: When ``True`` and multiple files match, return the most
            recent by filename sort. When ``False``, multiple matches is a hard
            failure.

    Returns:
        Path to the chosen result JSON.

    Raises:
        CompareError: When no files match, or multiple files match without
            ``allow_multiple``.
    """
    matches = sorted(glob.glob(os.path.join(results_dir, glob_pattern)))
    if not matches:
        raise CompareError(f"no_results_found dir={results_dir!r} glob={glob_pattern!r}")
    if len(matches) > 1 and not allow_multiple:
        raise CompareError(f"multiple_results n={len(matches)}")
    return Path(matches[-1])


def _load_json(path: Path) -> dict:
    """Load a JSON file as a dict.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed top-level dict.

    Raises:
        CompareError: When the file is missing, unreadable, malformed, or the
            top-level value is not an object.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise CompareError(f"file_not_found path={path}")
    except json.JSONDecodeError as e:
        raise CompareError(f"malformed_json path={path} line={e.lineno} col={e.colno}")
    if not isinstance(data, dict):
        raise CompareError(f"json_not_object path={path}")
    return data


def _extract_fps(result: dict) -> float:
    """Pull the effective-FPS metric out of the OmniPerf result document.

    Args:
        result: Top-level dict loaded from the OmniPerf JSON.

    Returns:
        Mean environment-step effective FPS, as a positive float.

    Raises:
        CompareError: When the runtime phase is missing, the metric is absent,
            or the value is not a finite positive number.
    """
    phase_data = result.get(METRIC_PHASE)
    if not isinstance(phase_data, dict):
        raise CompareError(f"missing_phase phase={METRIC_PHASE}")
    if METRIC_NAME not in phase_data:
        raise CompareError(f"missing_metric phase={METRIC_PHASE} metric={METRIC_NAME!r}")
    value = phase_data[METRIC_NAME]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CompareError(f"invalid_metric_type metric={METRIC_NAME!r} value={value!r}")
    if value != value:  # NaN
        raise CompareError(f"nan_metric metric={METRIC_NAME!r}")
    if value <= 0:
        raise CompareError(f"non_positive_metric metric={METRIC_NAME!r} value={value}")
    return float(value)


def _extract_gpu_name(result: dict) -> str | None:
    """Read the runner's GPU model name from the result JSON's hardware metadata.

    Args:
        result: Top-level dict loaded from the OmniPerf JSON.

    Returns:
        The current device's reported name (e.g. ``"NVIDIA L40"``), or ``None``
        when hardware metadata is absent or malformed.
    """
    hw = result.get("hardware_info")
    if not isinstance(hw, dict):
        return None
    devices = hw.get("gpu_devices")
    if not isinstance(devices, dict) or not devices:
        return None
    current = str(hw.get("gpu_current_device", "0"))
    device = devices.get(current) or next(iter(devices.values()), None)
    if isinstance(device, dict):
        name = device.get("name")
        return name if isinstance(name, str) and name else None
    return None


def _match_gpu(per_gpu: dict, gpu_key: str) -> tuple[str, dict]:
    """Find the baseline entry whose key is a (sub)string match for ``gpu_key``.

    The baseline can use either the exact ``torch`` device name (e.g.
    ``"NVIDIA L40"``) or a coarser tag (e.g. ``"L40"``). Substring matching in
    either direction lets a single baseline cover minor naming variations.

    Args:
        per_gpu: Mapping from baseline key to baseline entry.
        gpu_key: GPU identifier read from the result (or override).

    Returns:
        Tuple of ``(matched_key, entry)``.

    Raises:
        CompareError: When no key in ``per_gpu`` matches ``gpu_key``.
    """
    for key, entry in per_gpu.items():
        if key == gpu_key or key in gpu_key or gpu_key in key:
            if not isinstance(entry, dict):
                raise CompareError(f"malformed_baseline_entry gpu={key!r}")
            return key, entry
    raise CompareError(f"baseline_gpu_mismatch gpu={gpu_key!r} known={sorted(per_gpu)}")


def _resolve_baseline(baseline: dict, task: str, gpu_name: str | None, gpu_override: str | None) -> tuple[str, dict]:
    """Look up the baseline entry for the given task and GPU.

    Args:
        baseline: Top-level baseline document.
        task: Task name (e.g. ``"Isaac-Cartpole-Direct-v0"``).
        gpu_name: GPU name read from the result JSON (may be ``None``).
        gpu_override: Explicit GPU key supplied on the CLI; takes precedence
            over ``gpu_name`` when set.

    Returns:
        Tuple of ``(matched_gpu_key, entry)``.

    Raises:
        CompareError: When the task is missing, the per-GPU map is absent, no
            GPU identifier is available, no baseline entry matches the GPU, or
            a required entry field is missing.
    """
    task_entry = baseline.get(task)
    if not isinstance(task_entry, dict):
        raise CompareError(f"missing_baseline_task task={task!r}")
    per_gpu = task_entry.get("per_gpu")
    if not isinstance(per_gpu, dict) or not per_gpu:
        raise CompareError(f"missing_per_gpu task={task!r}")
    gpu_key = gpu_override or gpu_name
    if not gpu_key:
        raise CompareError(f"unknown_gpu task={task!r}")
    matched_key, entry = _match_gpu(per_gpu, gpu_key)
    for field in ("baseline_fps", "max_regression_pct"):
        if field not in entry:
            raise CompareError(f"missing_baseline_field task={task!r} gpu={matched_key!r} field={field}")
    return matched_key, entry


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--task", required=True, help="Task name, e.g. Isaac-Cartpole-Direct-v0.")
    parser.add_argument("--results-dir", required=True, help="Directory containing the benchmark JSON.")
    parser.add_argument("--baseline", required=True, help="Path to baseline.json.")
    parser.add_argument(
        "--results-glob",
        default=None,
        help=f"Glob for the result file (defaults to {DEFAULT_GLOB_TEMPLATE!r}).",
    )
    parser.add_argument(
        "--gpu-override",
        default=None,
        help="Override the GPU name read from the result JSON's hardware_info phase.",
    )
    parser.add_argument(
        "--allow-multiple",
        action="store_true",
        help="Permit multiple result files; pick the most recent by sort order.",
    )
    args = parser.parse_args(argv)

    glob_pattern = args.results_glob or DEFAULT_GLOB_TEMPLATE.format(task=args.task)

    try:
        result_path = _resolve_results(args.results_dir, glob_pattern, args.allow_multiple)
        result = _load_json(result_path)
        baseline = _load_json(Path(args.baseline))
        measured_fps = _extract_fps(result)
        gpu_name = _extract_gpu_name(result)
        gpu_key, entry = _resolve_baseline(baseline, args.task, gpu_name, args.gpu_override)
    except CompareError as e:
        _emit("HARD_FAILURE", reason=str(e), task=args.task)
        return EXIT_HARD_FAILURE

    baseline_fps = float(entry["baseline_fps"])
    threshold_pct = float(entry["max_regression_pct"])
    delta_pct = (measured_fps - baseline_fps) / baseline_fps * 100.0

    common: dict[str, object] = {
        "task": args.task,
        "gpu": gpu_key,
        "baseline_fps": f"{baseline_fps:.0f}",
        "measured_fps": f"{measured_fps:.0f}",
        "delta_pct": f"{delta_pct:+.2f}",
        "threshold_pct": f"{threshold_pct:.2f}",
    }
    if delta_pct < -threshold_pct:
        _emit("REGRESSION", **common)
        return EXIT_REGRESSION
    _emit("PASS", **common)
    return EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())
