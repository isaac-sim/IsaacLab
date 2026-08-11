# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare an Isaac Lab runtime benchmark with an authoritative baseline row."""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class ComparisonError(ValueError):
    """Raised when benchmark or baseline input cannot produce one comparison."""


@dataclass(frozen=True)
class MetricResult:
    """Result for one compared performance metric."""

    name: str
    measured: float | None
    reference: float | None
    regression_pct: float | None
    warn_regression_pct: float | None
    fail_regression_pct: float | None
    verdict: str
    message: str
    measured_samples: int | None = None
    reference_samples: int | None = None
    significance_metric: str | None = None
    significance_measured: float | None = None
    significance_reference: float | None = None
    significance_measured_std: float | None = None
    significance_reference_std: float | None = None
    significance_sigma: float | None = None
    statistically_significant: bool | None = None


@dataclass(frozen=True)
class ComparisonReport:
    """Complete performance comparison report."""

    identity: dict[str, object]
    metrics: tuple[MetricResult, ...]
    verdict: str
    message: str


_METRICS = (
    ("total_fps", "Total FPS", (("runtime", "total_fps", "mean"),), False),
    (
        "startup_time_s",
        "Total startup time [s]",
        (
            ("runtime", "startup_time_s", "app_launch"),
            ("runtime", "startup_time_s", "python_imports"),
            ("runtime", "startup_time_s", "task_config"),
            ("runtime", "startup_time_s", "env_creation"),
            ("runtime", "startup_time_s", "first_step"),
        ),
        True,
    ),
    ("gpu_mem_peak_gb", "Peak GPU memory [GB]", (("resources", "gpu_mem_gb", "peak"),), True),
    ("ram_peak_gb", "Peak process RSS [GB]", (("resources", "ram_gb", "peak"),), True),
)
_LABELS = {name: label for name, label, _, _ in _METRICS}
_MIN_SIGNIFICANCE_SIGMA = 2.0


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ComparisonError(f"{name} must be an object")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{name} must be a number")
    try:
        number = float(value)
    except OverflowError:
        raise ComparisonError(f"{name} must be finite") from None
    if not math.isfinite(number):
        raise ComparisonError(f"{name} must be finite")
    return number


def _sample_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 1:
        raise ComparisonError(f"{name} must be an integer greater than one")
    _number(value, name)
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ComparisonError(f"{name} must be a positive integer")
    _number(value, name)
    return value


def _nested_number(data: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return _number(value, ".".join(path))


def _startup_time(data: dict[str, Any]) -> float | None:
    runtime = _mapping(data.get("runtime"), "runtime")
    startup = _mapping(runtime.get("startup_time_s"), "runtime.startup_time_s")
    values: list[float] = []
    for name in ("app_launch", "env_creation", "first_step"):
        values.append(_number(startup.get(name), f"runtime.startup_time_s.{name}"))
    for name in ("python_imports", "task_config"):
        value = startup.get(name)
        if value is not None:
            values.append(_number(value, f"runtime.startup_time_s.{name}"))
    if any(value < 0 for value in values):
        raise ComparisonError("runtime startup phases must be non-negative")
    return _number(sum(values), "total startup time")


def _metric_value(data: dict[str, Any], name: str, paths: tuple[tuple[str, ...], ...]) -> float:
    if name == "startup_time_s":
        return _startup_time(data)
    values = [_nested_number(data, path) for path in paths]
    value = values[0] if len(values) == 1 else sum(item for item in values if item is not None)
    if value is None:
        raise ComparisonError(f"Benchmark result does not contain a measured value for {name}")
    if value < 0:
        raise ComparisonError(f"Measured {name} must be non-negative")
    return value


def _fps_statistics(
    bundle: dict[str, Any], measured_fps: float, expected_steps_per_iteration: int
) -> tuple[float, float, int]:
    if measured_fps <= 0:
        raise ComparisonError("Measured total_fps must be greater than zero")
    runtime = _mapping(bundle.get("runtime"), "runtime")
    std = _nested_number(bundle, ("runtime", "total_fps", "std"))
    if std is None:
        raise ComparisonError("Benchmark result does not contain throughput statistics for total_fps")
    if std < 0:
        raise ComparisonError("Throughput standard deviation must be non-negative")
    samples = _sample_count(runtime.get("iterations_completed"), "measured total_fps sample count")
    steps = _positive_int(runtime.get("steps_per_iteration"), "runtime.steps_per_iteration")
    if steps != expected_steps_per_iteration:
        raise ComparisonError("runtime.steps_per_iteration must match run.num_envs")
    return measured_fps, std, samples


def _normalize_gpu_model(value: str) -> str:
    value = re.sub(r"^nvidia\s+", "", value.strip(), flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _runtime_version(versions: dict[str, Any]) -> str:
    for provider in ("isaacsim", "newton"):
        value = versions.get(provider)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ComparisonError(f"versions.{provider} must be a non-empty string when present")
        return f"{provider}:{value.strip()}"
    raise ComparisonError("versions must contain an Isaac Sim or Newton runtime version")


def _identity(bundle: dict[str, Any]) -> dict[str, object]:
    run = _mapping(bundle.get("run"), "run")
    config = _mapping(run.get("config"), "run.config")
    versions = _mapping(bundle.get("versions"), "versions")
    hardware = _mapping(bundle.get("hardware"), "hardware")
    gpu_devices = hardware.get("gpu_devices")
    if not isinstance(gpu_devices, list) or not gpu_devices:
        raise ComparisonError("hardware.gpu_devices must contain at least one device")
    gpu = _mapping(gpu_devices[0], "hardware.gpu_devices[0]")
    gpu_name = gpu.get("name")
    presets = config.get("presets", [])
    if not isinstance(gpu_name, str) or not gpu_name.strip():
        raise ComparisonError("hardware.gpu_devices[0].name must be a non-empty string")
    if not isinstance(presets, list) or not all(isinstance(item, str) for item in presets):
        raise ComparisonError("run.config.presets must be a list of strings")

    identity = {
        "task": run.get("task"),
        "physics_backend": config.get("physics_backend"),
        "rendering_backend": config.get("rendering_backend"),
        "presets": sorted(presets),
        "gpu_model": _normalize_gpu_model(gpu_name),
        "runtime_version": _runtime_version(versions),
        "num_envs": _positive_int(run.get("num_envs"), "benchmark identity field 'num_envs'"),
    }
    for key in ("task", "physics_backend", "rendering_backend"):
        if not isinstance(identity[key], str) or not identity[key]:
            raise ComparisonError(f"benchmark identity field {key!r} must be a non-empty string")
    return identity


def _matching_row(table: dict[str, Any], identity: dict[str, object]) -> dict[str, Any]:
    if table.get("schema_version") != 1:
        raise ComparisonError("Baseline schema_version must be 1")
    rows = table.get("baselines")
    if not isinstance(rows, list):
        raise ComparisonError("baselines must be a list")

    matches: list[dict[str, Any]] = []
    for index, value in enumerate(rows):
        row = _mapping(value, f"baselines[{index}]")
        row_presets = row.get("presets", [])
        if not isinstance(row_presets, list) or not all(isinstance(item, str) for item in row_presets):
            raise ComparisonError(f"baselines[{index}].presets must be a list of strings")
        row_identity = {key: row.get(key) for key in identity}
        row_identity["presets"] = sorted(row_presets)
        if isinstance(row_identity.get("gpu_model"), str):
            row_identity["gpu_model"] = _normalize_gpu_model(row_identity["gpu_model"])
        if row_identity == identity:
            matches.append(row)

    if not matches:
        raise ComparisonError(f"No baseline row matches {json.dumps(identity, sort_keys=True)}")
    if len(matches) > 1:
        raise ComparisonError(f"Multiple baseline rows match {json.dumps(identity, sort_keys=True)}")
    return matches[0]


def _metric_result(
    name: str,
    measured: float | None,
    config: dict[str, Any],
    *,
    higher_is_worse: bool,
    measured_samples: Any = None,
    require_significance: bool = False,
    significance_measured: float | None = None,
    significance_measured_std: float | None = None,
) -> MetricResult:
    config = _mapping(config, f"metrics.{name}")
    if measured is None:
        raise ComparisonError(f"Benchmark result does not contain a measured value for {name}")

    reference = _number(config.get("reference"), f"metrics.{name}.reference")
    warn = _number(config.get("warn_regression_pct"), f"metrics.{name}.warn_regression_pct")
    fail = _number(config.get("fail_regression_pct"), f"metrics.{name}.fail_regression_pct")
    if reference <= 0:
        raise ComparisonError(f"metrics.{name}.reference must be greater than zero")
    if warn < 0 or fail < 0:
        raise ComparisonError(f"metrics.{name} thresholds must be non-negative")
    if fail < warn:
        raise ComparisonError(f"metrics.{name}.fail_regression_pct must be greater than or equal to warning")

    change = _number((measured - reference) / reference * 100.0, f"derived {name} percentage change")
    regression = change if higher_is_worse else -change
    reference_samples: int | None = None
    significance_metric: str | None = None
    significance_reference: float | None = None
    significance_reference_std: float | None = None
    significance_sigma: float | None = None
    statistically_significant: bool | None = None
    if require_significance:
        if measured_samples is None:
            raise ComparisonError(f"Benchmark result does not contain a measured sample count for {name}")
        measured_samples = _sample_count(measured_samples, f"measured {name} sample count")
        reference_samples = _sample_count(config.get("reference_samples"), f"metrics.{name}.reference_samples")
        if significance_measured is None or significance_measured_std is None:
            raise ComparisonError(f"Benchmark result does not contain throughput statistics for {name}")
        significance_metric = name
        significance_measured = _number(significance_measured, "runtime.total_fps.mean")
        significance_measured_std = _number(significance_measured_std, "runtime.total_fps.std")
        significance_reference = reference
        significance_reference_std = _number(config.get("reference_std"), f"metrics.{name}.reference_std")
        if significance_measured <= 0 or significance_reference <= 0:
            raise ComparisonError("throughput means must be greater than zero")
        if significance_measured_std < 0 or significance_reference_std < 0:
            raise ComparisonError("throughput standard deviations must be non-negative")
        standard_error = _number(
            math.hypot(
                significance_measured_std / math.sqrt(measured_samples),
                significance_reference_std / math.sqrt(reference_samples),
            ),
            f"derived metrics.{name} standard error",
        )
        difference = _number(
            abs(significance_measured - significance_reference), f"derived metrics.{name} timing difference"
        )
        if standard_error == 0.0:
            statistically_significant = difference > 0.0
        else:
            significance_sigma = _number(difference / standard_error, f"derived metrics.{name} significance")
            statistically_significant = significance_sigma >= _MIN_SIGNIFICANCE_SIGMA

    crosses_fail = regression > 0 and regression >= fail
    crosses_warn = regression > 0 and regression >= warn
    threshold_is_significant = statistically_significant is not False
    if crosses_fail and threshold_is_significant:
        verdict = "FAIL"
    elif crosses_warn and threshold_is_significant:
        verdict = "WARN"
    else:
        verdict = "PASS"
    message = f"Measured {measured:g} versus {reference:g} ({regression:+.2f}% regression)"
    if statistically_significant is not None:
        sigma = "infinite" if significance_sigma is None else f"{significance_sigma:.2f}"
        message += f", {sigma} sigma ({'significant' if statistically_significant else 'not significant'})"
    return MetricResult(
        name,
        measured,
        reference,
        regression,
        warn,
        fail,
        verdict,
        message,
        measured_samples=measured_samples,
        reference_samples=reference_samples,
        significance_metric=significance_metric,
        significance_measured=significance_measured,
        significance_reference=significance_reference,
        significance_measured_std=significance_measured_std,
        significance_reference_std=significance_reference_std,
        significance_sigma=significance_sigma,
        statistically_significant=statistically_significant,
    )


def compare(runtime_bundle: dict[str, object], baseline_table: dict[str, object]) -> ComparisonReport:
    """Compare one runtime benchmark bundle with its exact baseline row."""
    bundle = _mapping(runtime_bundle, "benchmark result")
    table = _mapping(baseline_table, "baseline table")
    identity = _identity(bundle)
    row = _matching_row(table, identity)
    metric_configs = _mapping(row.get("metrics"), "baseline metrics")
    missing_metrics = [name for name, _, _, _ in _METRICS if name not in metric_configs]
    if missing_metrics:
        raise ComparisonError(f"The matching baseline row is missing required metrics: {', '.join(missing_metrics)}")
    values = {name: _metric_value(bundle, name, paths) for name, _, paths, _ in _METRICS}
    fps_mean, fps_std, fps_samples = _fps_statistics(bundle, values["total_fps"], identity["num_envs"])

    metrics = tuple(
        _metric_result(
            name,
            values[name],
            metric_configs[name],
            higher_is_worse=higher_is_worse,
            measured_samples=fps_samples if name == "total_fps" else None,
            require_significance=name == "total_fps",
            significance_measured=fps_mean if name == "total_fps" else None,
            significance_measured_std=fps_std if name == "total_fps" else None,
        )
        for name, _, paths, higher_is_worse in _METRICS
    )
    verdict = "FAIL" if any(item.verdict == "FAIL" for item in metrics) else "WARN"
    if verdict == "WARN" and not any(item.verdict == "WARN" for item in metrics):
        verdict = "PASS"
    return ComparisonReport(identity, metrics, verdict, f"Performance comparison {verdict.lower()}")


def skipped_report(runtime_bundle: dict[str, object], reason: str) -> ComparisonReport:
    """Build a report that records unavailable baseline configuration."""
    bundle = _mapping(runtime_bundle, "benchmark result")
    identity = _identity(bundle)
    values = {name: _metric_value(bundle, name, paths) for name, _, paths, _ in _METRICS}
    fps_mean, fps_std, fps_samples = _fps_statistics(bundle, values["total_fps"], identity["num_envs"])
    metrics = tuple(
        MetricResult(
            name,
            values[name],
            None,
            None,
            None,
            None,
            "SKIP",
            reason,
            measured_samples=fps_samples if name == "total_fps" else None,
            significance_metric="total_fps" if name == "total_fps" else None,
            significance_measured=fps_mean if name == "total_fps" else None,
            significance_measured_std=fps_std if name == "total_fps" else None,
        )
        for name, _, paths, _ in _METRICS
    )
    return ComparisonReport(identity, metrics, "SKIP", reason)


def _format_value(value: float | None) -> str:
    return "-" if value is None else f"{value:.6g}"


def write_outputs(
    report: ComparisonReport,
    markdown_path: Path,
    json_path: Path,
    junit_path: Path,
) -> None:
    """Write Markdown, JSON, and JUnit representations of a comparison report."""
    for path in (markdown_path, json_path, junit_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"## Performance comparison: {report.verdict}",
        "",
        report.message,
        "",
        "| Metric | Measured | Reference | Regression | Measured FPS std | Reference FPS std | "
        "Significance | Warn | Fail | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for metric in report.metrics:
        regression = "-" if metric.regression_pct is None else f"{metric.regression_pct:+.2f}%"
        warn = "-" if metric.warn_regression_pct is None else f"{metric.warn_regression_pct:.2f}%"
        fail = "-" if metric.fail_regression_pct is None else f"{metric.fail_regression_pct:.2f}%"
        if metric.statistically_significant is None:
            significance = "-"
        elif metric.significance_sigma is None:
            significance = "infinite" if metric.statistically_significant else "0.00 sigma"
        else:
            significance = f"{metric.significance_sigma:.2f} sigma"
        measured_timing_std = _format_value(metric.significance_measured_std)
        reference_timing_std = _format_value(metric.significance_reference_std)
        lines.append(
            f"| {_LABELS.get(metric.name, metric.name)} | {_format_value(metric.measured)} | "
            f"{_format_value(metric.reference)} | {regression} | {measured_timing_std} | "
            f"{reference_timing_std} | {significance} | {warn} | {fail} | {metric.verdict} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    test_metrics = report.metrics or (
        MetricResult("comparison", None, None, None, None, None, report.verdict, report.message),
    )
    failures = sum(metric.verdict in {"FAIL", "ERROR"} for metric in test_metrics)
    skipped = sum(metric.verdict == "SKIP" for metric in test_metrics)
    suite = ET.Element(
        "testsuite",
        name="performance-comparison",
        tests=str(len(test_metrics)),
        failures=str(failures),
        skipped=str(skipped),
    )
    for metric in test_metrics:
        case = ET.SubElement(suite, "testcase", name=metric.name, classname="performance-comparison")
        if metric.verdict in {"FAIL", "ERROR"}:
            ET.SubElement(case, "failure", message=metric.message).text = metric.message
        elif metric.verdict == "SKIP":
            ET.SubElement(case, "skipped", message=metric.message)
        ET.SubElement(case, "system-out").text = metric.message
    ET.ElementTree(suite).write(junit_path, encoding="unicode", xml_declaration=True)


def _error_report(message: str) -> ComparisonReport:
    metric = MetricResult("comparison", None, None, None, None, None, "ERROR", message)
    return ComparisonReport({}, (metric,), "ERROR", message)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_result", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the performance comparison CLI."""
    args = _parser().parse_args(argv)
    try:
        bundle = json.loads(args.benchmark_result.read_text(encoding="utf-8"))
        if args.baseline is None:
            report = skipped_report(bundle, "PERF_BASELINE_S3_URI is not configured")
        else:
            table = json.loads(args.baseline.read_text(encoding="utf-8"))
            report = compare(bundle, table)
    except (ComparisonError, json.JSONDecodeError, OSError, TypeError) as exc:
        report = _error_report(str(exc))

    write_outputs(report, args.markdown, args.output_json, args.junit)
    print(args.markdown.read_text(encoding="utf-8"), end="")
    return int(report.verdict in {"FAIL", "ERROR"})


if __name__ == "__main__":
    raise SystemExit(main())
