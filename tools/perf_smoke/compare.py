# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare one runtime benchmark against the recent history of comparable runs.

ASV decides relative regressions and statistical significance. Isaac Lab applies
the warmup, advisory-metric, and absolute-floor policies around that comparison.
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .contract import Contract, backend_key, valid_backend_keys
from .metrics import METRICS, Metric, PerfSmokeError, mapping, number

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"
ERROR = "ERROR"

#: Comparable runs required before the gate will render a verdict.
MIN_BASELINE_SAMPLES = 3

#: Rows read from the store.
MAX_BASELINE_SAMPLES = 20

_SEVERITY = {PASS: 0, SKIP: 0, ERROR: 0, WARN: 1, FAIL: 2}


@dataclass(frozen=True)
class Thresholds:
    """Resolved gating policy for one metric of one benchmark combination."""

    warn_pct: float
    fail_pct: float
    hard_floor: float | None = None
    gating: bool = True


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


def _warn(message: str) -> None:
    """Report a policy problem that must not stop the comparison."""
    print(f"::warning::perf-smoke: {message}", file=sys.stderr)


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
    if not 0 <= warn <= fail < 100:
        raise PerfSmokeError(f"regression percentages must satisfy 0 <= warn <= fail < 100 for {task}")
    advisory_only = override.get("advisory_only", False)
    if not isinstance(advisory_only, bool):
        raise PerfSmokeError(f"per_task_regression_pct.{task}.advisory_only must be a boolean")

    floors = _clean(root.get("hard_floor_fps", {}), "hard_floor_fps")
    by_task = _clean(floors.get(gpu_model, {}), f"hard_floor_fps.{gpu_model}")
    by_key = _clean(by_task.get(task, {}), f"hard_floor_fps.{gpu_model}.{task}")
    if floors and gpu_model not in floors:
        _warn(f"no hard floors are configured for GPU {gpu_model!r}; configured: {sorted(floors)}")
    unreachable = sorted(set(by_key) - valid_backend_keys())
    if unreachable:
        _warn(
            f"hard_floor_fps.{gpu_model}.{task} has key(s) {unreachable} that no run can report,"
            f" so they never apply; expected one of {sorted(valid_backend_keys())}"
        )
    raw_floor = by_key.get(key)
    # A non-positive floor means disabled/no floor. Active floors must be positive.
    floor = number(raw_floor, f"hard_floor_fps.{gpu_model}.{task}.{key}") if raw_floor is not None else None
    if floor is not None and floor <= 0:
        floor = None

    resolved: dict[str, Thresholds] = {}
    for metric in METRICS:
        resolved[metric.name] = Thresholds(
            warn,
            fail,
            floor if metric.name == "total_fps" else None,
            gating=metric.gating and not advisory_only,
        )
    return resolved


def _evaluate(
    metric: Metric,
    candidates: list[float],
    history: list[float],
    thresholds: Thresholds,
    min_samples: int,
) -> MetricResult:
    """Apply absolute floors, then ASV's significance and strict relative thresholds."""
    result = MetricResult(
        name=metric.name,
        label=metric.label,
        measured=statistics.median(candidates),
        warn_pct=thresholds.warn_pct,
        fail_pct=thresholds.fail_pct,
        hard_floor=thresholds.hard_floor,
        sample_count=len(history),
        gating=thresholds.gating,
    )
    if thresholds.hard_floor is not None and min(candidates) < thresholds.hard_floor:
        return replace(result, verdict=FAIL, gating=True, note=f"below hard floor {thresholds.hard_floor:g}")

    if len(history) < min_samples or len(candidates) < 2:
        return replace(result, note="insufficient independent runs for ASV significance testing")

    reference = statistics.median(history)
    if reference == 0:
        return replace(result, reference=reference, note="baseline median is zero")

    # Lazy loading keeps aggregation dependency-free. The internal ASV helper is
    # version-pinned and covered by the comparison tests.
    try:
        from asv.commands.compare import _is_result_better
        from asv_runner.statistics import compute_stats
    except ImportError as exc:
        raise PerfSmokeError("ASV is unavailable; install tools/perf_smoke/requirements.txt") from exc

    # ASV assumes lower is better. Reversing FPS comparisons preserves the
    # original samples and medians, including zero throughput and even counts.
    before, after = (history, candidates) if metric.higher_is_worse else (candidates, history)
    before_value, before_stats = compute_stats(before, 1)
    after_value, after_stats = compute_stats(after, 1)
    warned, failed = (
        _is_result_better(
            before_value,
            after_value,
            (before_stats, before),
            (after_stats, after),
            factor=1 + pct / 100 if metric.higher_is_worse else 1 / (1 - pct / 100),
        )
        for pct in (thresholds.warn_pct, thresholds.fail_pct)
    )
    change_pct = (result.measured - reference) / reference * 100.0
    return replace(
        result,
        reference=reference,
        regression_pct=change_pct if metric.higher_is_worse else -change_pct,
        verdict=FAIL if failed else WARN if warned else PASS,
    )


def unresolved(
    contract: Contract,
    measured: dict[str, float],
    verdict: str,
    reason: str,
    label: str = "",
) -> Report:
    """Build a report that records measurements but no comparison.

    The run itself succeeded; only the verdict is missing. :data:`SKIP` means there was
    nothing to compare against, :data:`ERROR` means the comparison could not be attempted.
    Either way the measurements are carried through: they are the expensive part of the
    job and remain valid on their own.

    Args:
        contract: The run's comparability contract.
        measured: Measured value per metric name.
        verdict: :data:`SKIP` or :data:`ERROR`.
        reason: Operator-facing explanation.
        label: Matrix combination name.

    Returns:
        A report carrying every measurement, with no comparison against a baseline.
    """
    metrics = tuple(
        MetricResult(
            name=metric.name,
            label=metric.label,
            measured=measured[metric.name],
            verdict=verdict,
            gating=metric.gating,
        )
        for metric in METRICS
    )
    return Report(contract.as_dict(), contract.hash, metrics, verdict, reason, label)


def errored(reason: str, label: str = "") -> Report:
    """Build a report for a gate failure that happened before anything was measured.

    Non-blocking and distinct from :data:`SKIP`. Use :func:`unresolved` with
    :data:`ERROR` when the run *was* measured and only the comparison failed; this one is
    for a bundle that could not be parsed, where there is nothing to report but the fault.

    Args:
        reason: Operator-facing explanation of error.
        label: Matrix combination name.

    Returns:
        A report carrying no metrics and the :data:`ERROR` verdict.
    """
    return Report(contract={}, contract_hash="", metrics=(), verdict=ERROR, message=reason, label=label)


def compare(
    contract: Contract,
    measurements: list[dict[str, float]],
    history: list[dict[str, float]],
    threshold_config: Any,
    *,
    min_samples: int = MIN_BASELINE_SAMPLES,
    label: str = "",
) -> Report:
    """Compare independent candidate runs against comparable baseline history.

    Args:
        contract: Comparability contract for this run.
        measurements: Metric mappings from independent candidate runs.
        history: Metric mappings from prior comparable runs, oldest first.
        threshold_config: Parsed ``perf_smoke_thresholds.json``.
        min_samples: Comparable runs required before a verdict is rendered.
        label: Matrix combination name.

    Returns:
        The comparison report: FAIL if any gating metric failed, WARN if any warned,
        SKIP only if every gating metric was skipped, otherwise PASS. Non-gating
        metrics are evaluated and reported, but never change the verdict.
    """
    if not measurements:
        raise PerfSmokeError("at least one candidate run is required")
    if min_samples < 2:
        raise PerfSmokeError("min_samples must be at least 2 for ASV significance testing")
    thresholds = resolve_thresholds(
        threshold_config,
        str(contract.runtime.get("gpu_model", "")),
        str(contract.workload.get("task", "")),
        backend_key(contract),
    )

    results = tuple(
        _evaluate(
            metric,
            [row[metric.name] for row in measurements],
            [row[metric.name] for row in history if metric.name in row],
            thresholds[metric.name],
            min_samples,
        )
        for metric in METRICS
    )

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
    if verdict == FAIL and worst is not None and worst.hard_floor is not None and worst.note is not None:
        message = f"{worst.label} below hard floor"
    elif verdict == FAIL:
        message = f"Performance regression in {worst.label}" if worst else "Performance regression"
    elif verdict == WARN:
        message = f"Possible performance regression in {worst.label}" if worst else "Possible regression"
    elif verdict == SKIP and len(history) < min_samples:
        message = (
            "No baseline recorded yet for this runtime contract"
            if not history
            else f"Baseline warming up ({len(history)}/{min_samples} comparable runs)"
        )
    elif verdict == SKIP:
        message = "No gating metric could be compared"
    else:
        message = "No performance regression detected"
    return Report(contract.as_dict(), contract.hash, results, verdict, message, label)


def gating_names() -> tuple[str, ...]:
    """Return the names of the metrics whose verdict can fail a pull request."""
    return tuple(metric.name for metric in METRICS if metric.gating)
