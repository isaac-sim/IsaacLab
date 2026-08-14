# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare one runtime benchmark against the recent history of comparable runs.

Pure functions only. Everything this module needs is passed in.

The reference is the median of the last K comparable develop runs, and the
noise band is their median absolute deviation, not the within-run per-step spread.

A regression must clear both its percentage threshold and the historical noise
band. A hard floor is separate and fires regardless of significance.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from .contract import Contract, backend_key
from .metrics import METRICS, METRICS_BY_NAME, Metric, PerfSmokeError, mapping, number

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"
ERROR = "ERROR"

#: Deviation a change must exceed before it can alter the verdict.
MIN_SIGNIFICANCE_SIGMA = 2.0

#: Comparable runs required before the gate will render a verdict.
MIN_BASELINE_SAMPLES = 3

#: Rows read from the store.
MAX_BASELINE_SAMPLES = 10

#: Scale factor making the median absolute deviation a consistent estimator of the
#: standard deviation for normally distributed data.
_MAD_TO_SIGMA = 1.4826

_SEVERITY = {PASS: 0, SKIP: 0, WARN: 1, FAIL: 2}


@dataclass(frozen=True)
class Thresholds:
    """Resolved gating policy for one metric of one benchmark combination."""

    warn_pct: float
    fail_pct: float
    hard_floor: float | None = None


@dataclass(frozen=True)
class MetricResult:
    """Outcome for one compared metric."""

    name: str
    label: str
    measured: float
    reference: float | None = None
    regression_pct: float | None = None
    warn_pct: float | None = None
    fail_pct: float | None = None
    hard_floor: float | None = None
    sample_count: int = 0
    spread_pct: float | None = None
    significance_sigma: float | None = None
    significant: bool | None = None
    verdict: str = SKIP
    gating: bool = False
    note: str | None = None


@dataclass(frozen=True)
class Report:
    """Complete comparison for one benchmark combination."""

    contract: dict[str, Any] = field(default_factory=dict)
    contract_hash: str = ""
    metrics: tuple[MetricResult, ...] = ()
    verdict: str = SKIP
    message: str = ""
    #: Matrix combination name, carried in the artifact so the aggregate job can
    #: label rows without parsing artifact directory names.
    label: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return the report as a plain, JSON-serialisable dict."""
        return asdict(self)


def _clean(config: Any, name: str) -> dict[str, Any]:
    """Return a config mapping with documentation keys (``_comment``, ``_todo``) removed."""
    return {key: value for key, value in mapping(config, name).items() if not key.startswith("_")}


def resolve_thresholds(config: Any, gpu_model: str, task: str, key: str) -> dict[str, Thresholds]:
    """Resolve gating policy for every metric of one combination.

    Args:
        config: Parsed ``perf_smoke_thresholds.json``.
        gpu_model: Canonical GPU slug (see :func:`~tools.perf_smoke.contract.normalize_gpu_model`).
        task: Gym task id.
        key: ``{physics}`` or ``{physics}_{render}``.

    Returns:
        Mapping of metric name to its resolved :class:`Thresholds`.

    Raises:
        PerfSmokeError: If the config is malformed.
    """
    root = _clean(config, "threshold config")
    defaults = _clean(root.get("defaults", {}), "threshold config defaults")
    warn = number(defaults.get("warn_regression_pct", 5.0), "defaults.warn_regression_pct")
    fail = number(defaults.get("fail_regression_pct", 10.0), "defaults.fail_regression_pct")

    per_task = _clean(root.get("per_task_regression_pct", {}), "per_task_regression_pct")
    override = _clean(per_task.get(task, {}), f"per_task_regression_pct.{task}")
    if "warn_regression_pct" in override:
        warn = number(override["warn_regression_pct"], f"per_task_regression_pct.{task}.warn_regression_pct")
    if "fail_regression_pct" in override:
        fail = number(override["fail_regression_pct"], f"per_task_regression_pct.{task}.fail_regression_pct")
    if fail < warn:
        raise PerfSmokeError(f"fail_regression_pct must be >= warn_regression_pct for {task}")

    floors = _clean(root.get("hard_floor_fps", {}), "hard_floor_fps")
    by_task = _clean(floors.get(gpu_model, {}), f"hard_floor_fps.{gpu_model}")
    by_key = _clean(by_task.get(task, {}), f"hard_floor_fps.{gpu_model}.{task}")
    raw_floor = by_key.get(key)
    # A floor of 0.0 is the documented None value, not a real floor.
    floor = number(raw_floor, f"hard_floor_fps.{gpu_model}.{task}.{key}") if raw_floor is not None else None
    if floor is not None and floor <= 0:
        floor = None

    resolved: dict[str, Thresholds] = {}
    for metric in METRICS:
        resolved[metric.name] = Thresholds(warn, fail, floor if metric.name == "total_fps" else None)
    return resolved


def _robust_spread(values: list[float], center: float) -> float:
    """Return the MAD-derived robust sigma for ``values`` about ``center``."""
    if len(values) < 2:
        return 0.0
    return _MAD_TO_SIGMA * statistics.median([abs(value - center) for value in values])


def _evaluate(
    metric: Metric,
    measured: float,
    history: list[float],
    thresholds: Thresholds,
) -> MetricResult:
    """Compare one metric against its history and resolved policy."""
    reference = statistics.median(history)
    sigma = _robust_spread(history, reference)

    if reference == 0:
        return MetricResult(
            name=metric.name,
            label=metric.label,
            measured=measured,
            reference=reference,
            sample_count=len(history),
            verdict=SKIP,
            gating=metric.gating,
            note="baseline median is zero",
        )

    change_pct = (measured - reference) / reference * 100.0
    # Normalize such that positive regression means worse
    regression_pct = change_pct if metric.higher_is_worse else -change_pct

    difference = abs(measured - reference)
    if sigma == 0.0:
        significant = difference > 0.0
        sigma_count = None
    else:
        sigma_count = difference / sigma
        significant = sigma_count >= MIN_SIGNIFICANCE_SIGMA

    note: str | None = None
    if thresholds.hard_floor is not None and measured < thresholds.hard_floor:
        verdict = FAIL
        note = f"below hard floor {thresholds.hard_floor:g}"
    elif regression_pct >= thresholds.fail_pct and significant:
        verdict = FAIL
    elif regression_pct >= thresholds.warn_pct and significant:
        verdict = WARN
    elif regression_pct >= thresholds.warn_pct:
        verdict = PASS
        note = "regression within historical noise"
    else:
        verdict = PASS

    return MetricResult(
        name=metric.name,
        label=metric.label,
        measured=measured,
        reference=reference,
        regression_pct=regression_pct,
        warn_pct=thresholds.warn_pct,
        fail_pct=thresholds.fail_pct,
        hard_floor=thresholds.hard_floor,
        sample_count=len(history),
        spread_pct=(sigma / reference * 100.0) if reference else None,
        significance_sigma=sigma_count,
        significant=significant,
        verdict=verdict,
        gating=metric.gating,
        note=note,
    )


def _skipped(contract: Contract, measured: dict[str, float], reason: str, history: int = 0, label: str = "") -> Report:
    """Build a report that records why no verdict could be rendered."""
    metrics = tuple(
        MetricResult(
            name=metric.name,
            label=metric.label,
            measured=measured[metric.name],
            sample_count=history,
            verdict=SKIP,
            gating=metric.gating,
        )
        for metric in METRICS
    )
    return Report(contract.as_dict(), contract.hash, metrics, SKIP, reason, label)


def compare(
    contract: Contract,
    measured: dict[str, float],
    history: list[dict[str, float]],
    threshold_config: Any,
    *,
    min_samples: int = MIN_BASELINE_SAMPLES,
    label: str = "",
) -> Report:
    """Compare a measurement against the history of comparable runs.

    Args:
        contract: Comparability contract for this run.
        measured: Metric values from :func:`~tools.perf_smoke.metrics.extract`.
        history: Metric mappings from prior comparable runs, oldest first.
        threshold_config: Parsed ``perf_smoke_thresholds.json``.
        min_samples: Comparable runs required before a verdict is rendered.

    Returns:
        The comparison report. Its verdict is the worst among gating metrics;
        non-gating metrics are evaluated and reported as advisory.
    """
    if not history:
        return _skipped(contract, measured, "No baseline recorded yet for this runtime contract", label=label)
    if len(history) < min_samples:
        return _skipped(
            contract,
            measured,
            f"Baseline warming up ({len(history)}/{min_samples} comparable runs)",
            history=len(history),
            label=label,
        )

    thresholds = resolve_thresholds(
        threshold_config,
        str(contract.runtime.get("gpu_model", "")),
        str(contract.workload.get("task", "")),
        backend_key(contract),
    )

    results: list[MetricResult] = []
    for metric in METRICS:
        series = [row[metric.name] for row in history if metric.name in row]
        if len(series) < min_samples:
            results.append(
                MetricResult(
                    name=metric.name,
                    label=metric.label,
                    measured=measured[metric.name],
                    sample_count=len(series),
                    verdict=SKIP,
                    gating=metric.gating,
                    note="insufficient history for this metric",
                )
            )
            continue
        results.append(_evaluate(metric, measured[metric.name], series, thresholds[metric.name]))

    gating = [result for result in results if result.gating]
    if any(result.verdict == FAIL for result in gating):
        verdict = FAIL
    elif any(result.verdict == WARN for result in gating):
        verdict = WARN
    elif gating and all(result.verdict == SKIP for result in gating):
        verdict = SKIP
    elif gating:
        verdict = PASS
    else:
        verdict = SKIP
    worst = max(gating, key=lambda result: _SEVERITY[result.verdict], default=None)
    if verdict == FAIL:
        message = f"Performance regression in {worst.label}" if worst else "Performance regression"
    elif verdict == WARN:
        message = f"Possible performance regression in {worst.label}" if worst else "Possible regression"
    elif verdict == SKIP:
        message = "No gating metric could be compared"
    else:
        message = "No performance regression detected"
    return Report(contract.as_dict(), contract.hash, tuple(results), verdict, message, label)


def advisory_names() -> tuple[str, ...]:
    """Return the metrics that are measured and reported but never gate."""
    return tuple(metric.name for metric in METRICS if not metric.gating)


def gating_names() -> tuple[str, ...]:
    """Return the metrics that contribute to the overall verdict."""
    return tuple(metric.name for metric in METRICS if metric.gating)


def metric_for(name: str) -> Metric:
    """Return the :class:`~tools.perf_smoke.metrics.Metric` registered under ``name``."""
    if name not in METRICS_BY_NAME:
        raise PerfSmokeError(f"Unknown metric {name!r}")
    return METRICS_BY_NAME[name]
