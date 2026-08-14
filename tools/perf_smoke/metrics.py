# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Metric vocabulary and validated extraction from a runtime benchmark bundle.

This is the only module that knows the shape of a schema-v1 ``RuntimeBundle``.
:mod:`contract`, :mod:`store`, :mod:`compare` and :mod:`report` all speak the
vocabulary defined here, so a schema change lands in one place.

Every metric flows through the whole pipeline -- extracted, stored, compared and
displayed. :attr:`Metric.gating` is consulted in exactly one place (the verdict
rollup in :mod:`compare`), so a non-gating metric still produces a visible,
advisory verdict. Promoting one to gating is a one-line change with historical
data already behind it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class PerfSmokeError(ValueError):
    """Raised when benchmark, baseline, or threshold input cannot be used."""


@dataclass(frozen=True)
class Metric:
    """One compared performance metric.

    Args:
        name: Stable key used in baseline rows and JSON reports.
        label: Human-readable column heading, including units.
        path: Dotted path into the bundle, or ``None`` when the value is derived
            (see :func:`total_startup_time_s`).
        higher_is_worse: ``True`` when an increase is a regression (memory, time);
            ``False`` when a decrease is a regression (throughput).
        gating: Whether this metric contributes to the overall verdict. Non-gating
            metrics are still measured, stored, and reported as advisory.
    """

    name: str
    label: str
    path: tuple[str, ...] | None
    higher_is_worse: bool
    gating: bool


#: Every metric the gate measures. Only ``total_fps`` gates today: startup time
#: was measured varying up to 16x on JIT/shader cache warmth alone, and the memory
#: metrics have no established noise model yet. All four are recorded so the
#: evidence to promote them accumulates before any threshold is trusted.
METRICS: tuple[Metric, ...] = (
    Metric("total_fps", "Total FPS", ("runtime", "total_fps", "mean"), higher_is_worse=False, gating=True),
    Metric("startup_time_s", "Total startup time [s]", None, higher_is_worse=True, gating=False),
    Metric(
        "gpu_mem_peak_gb",
        "Peak GPU memory [GB]",
        ("resources", "gpu_mem_gb", "peak"),
        higher_is_worse=True,
        gating=False,
    ),
    Metric("ram_peak_gb", "Peak process RSS [GB]", ("resources", "ram_gb", "peak"), higher_is_worse=True, gating=False),
)

METRICS_BY_NAME: dict[str, Metric] = {metric.name: metric for metric in METRICS}

#: Startup phases summed into ``startup_time_s``. The first three are required;
#: a bundle missing one is malformed rather than merely sparse.
_REQUIRED_STARTUP_PHASES = ("app_launch", "env_creation", "first_step")
_OPTIONAL_STARTUP_PHASES = ("python_imports", "task_config")


def mapping(value: Any, name: str) -> dict[str, Any]:
    """Return ``value`` as a dict, or raise if it is not an object."""
    if not isinstance(value, dict):
        raise PerfSmokeError(f"{name} must be an object")
    return value


def number(value: Any, name: str) -> float:
    """Return ``value`` as a finite float, rejecting bools and non-finite values."""
    # bool is an int subclass; accepting it would silently turn True into 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerfSmokeError(f"{name} must be a number")
    try:
        result = float(value)
    except OverflowError:
        raise PerfSmokeError(f"{name} must be finite") from None
    if not math.isfinite(result):
        raise PerfSmokeError(f"{name} must be finite")
    return result


def positive_int(value: Any, name: str) -> int:
    """Return ``value`` as an ``int`` greater than zero."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PerfSmokeError(f"{name} must be a positive integer")
    return value


def nested_number(data: dict[str, Any], path: tuple[str, ...]) -> float | None:
    """Return the finite number at ``path``, or ``None`` when the path is absent."""
    value: Any = data
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return number(value, ".".join(path))


def total_startup_time_s(bundle: dict[str, Any]) -> float:
    """Return total startup wall time [s]: the sum of the bundle's startup phases.

    Args:
        bundle: A schema-v1 runtime bundle.

    Returns:
        Summed startup duration [s].

    Raises:
        PerfSmokeError: If a required phase is missing or any phase is negative.
    """
    runtime = mapping(bundle.get("runtime"), "runtime")
    startup = mapping(runtime.get("startup_time_s"), "runtime.startup_time_s")
    values = [number(startup.get(phase), f"runtime.startup_time_s.{phase}") for phase in _REQUIRED_STARTUP_PHASES]
    for phase in _OPTIONAL_STARTUP_PHASES:
        if startup.get(phase) is not None:
            values.append(number(startup[phase], f"runtime.startup_time_s.{phase}"))
    if any(value < 0 for value in values):
        raise PerfSmokeError("runtime startup phases must be non-negative")
    return number(sum(values), "total startup time")


def extract(bundle: dict[str, Any]) -> dict[str, float]:
    """Extract every metric in :data:`METRICS` from a runtime bundle.

    Args:
        bundle: A schema-v1 runtime bundle.

    Returns:
        Mapping of metric name to measured value.

    Raises:
        PerfSmokeError: If any metric is missing, non-numeric, or negative.
    """
    values: dict[str, float] = {}
    for metric in METRICS:
        if metric.path is None:
            values[metric.name] = total_startup_time_s(bundle)
            continue
        value = nested_number(bundle, metric.path)
        if value is None:
            raise PerfSmokeError(f"Benchmark result does not contain a measured value for {metric.name}")
        if value < 0:
            raise PerfSmokeError(f"Measured {metric.name} must be non-negative")
        values[metric.name] = value
    return values
