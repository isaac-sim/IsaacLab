# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared model states for the performance smoke test"""

from dataclasses import dataclass
from enum import Enum


class OracleVerdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"
    HARD_FAILURE = "HARD_FAILURE"


# Severity ordering used to combine independent verdict signals; higher is worse.
_VERDICT_SEVERITY: dict[OracleVerdict, int] = {
    OracleVerdict.PASS: 0,
    OracleVerdict.WARN: 1,
    OracleVerdict.BLOCK: 2,
    OracleVerdict.HARD_FAILURE: 3,
}

# Verdicts a configured FPS threshold is allowed to declare (gating verdicts).
# ``PASS``/``HARD_FAILURE`` are not thresholdable: PASS gates nothing and
# HARD_FAILURE is reserved for benchmark-level failures, not FPS floors.
THRESHOLD_VERDICTS: frozenset[OracleVerdict] = frozenset({OracleVerdict.WARN, OracleVerdict.BLOCK})


def verdict_severity(verdict: OracleVerdict) -> int:
    """Return the severity rank of a verdict; higher means worse."""
    return _VERDICT_SEVERITY[verdict]


def worst_verdict(*verdicts: OracleVerdict) -> OracleVerdict:
    """Return the most severe verdict among the arguments."""
    return max(verdicts, key=verdict_severity)


class BisectVerdict(str, Enum):
    GOOD = "GOOD"
    BAD = "BAD"
    SKIP = "SKIP"


class FailurePhase(str, Enum):
    IMPORT = "import"
    INIT = "init"
    RUNTIME = "runtime"
    OOM = "oom"
    HANG = "hang"
    DRIVER = "driver"
    CONFIG_MISMATCH = "config_mismatch"


class ThresholdSource(str, Enum):
    NO_BASELINE = "no_baseline"
    INSUFFICIENT_WINDOW = "insufficient_window"
    ROLLING_WINDOW = "rolling_window"
    THRESHOLD = "threshold"
    NOT_APPLICABLE = "n/a"


@dataclass(frozen=True)
class FpsMeanThreshold:
    """A single configured mean-FPS threshold for one task/backend/GPU.

    A threshold is *crossed* when the measured mean FPS falls below :attr:`value`.
    A crossed threshold whose :attr:`verdict` is set contributes that verdict to
    the gate outcome; a threshold with :attr:`verdict` = ``None`` is *reporting-only*
    and is surfaced in outputs without ever changing the verdict.

    Args:
        name: Informative label for the threshold (e.g. ``"IsaacLab-2.0"``).
        value: Mean-FPS floor [FPS]. ``0.0`` is a valid, effectively non-gating floor.
        verdict: Gating verdict to raise when crossed, or ``None`` for reporting-only.
    """

    name: str
    value: float
    verdict: OracleVerdict | None

    @property
    def is_gating(self) -> bool:
        """Whether crossing this threshold can change the gate verdict."""
        return self.verdict is not None

    def crosses(self, mean_fps: float) -> bool:
        """Return whether ``mean_fps`` [FPS] falls below this threshold."""
        return mean_fps < self.value

    def to_dict(self) -> dict:
        """Serialize to the ``launch_config.json`` / ``tasks.json`` entry shape."""
        return {
            "threshold_verdict": self.verdict.value if self.verdict is not None else None,
            "threshold_name": self.name,
            "threshold": self.value,
        }

    @classmethod
    def from_dict(cls, raw: dict, *, context: str = "") -> "FpsMeanThreshold | None":
        """Parse and validate one raw threshold entry.

        Returns ``None`` when the entry has no ``threshold`` value (skipped).

        Raises:
            TypeError: If ``raw`` is not an object.
            ValueError: If ``threshold_name`` is missing/empty, ``threshold`` is
                non-numeric, or ``threshold_verdict`` is not a gating verdict.
        """
        where = f" ({context})" if context else ""
        if not isinstance(raw, dict):
            raise TypeError(f"fps_mean_thresholds entry{where} must be an object, got {type(raw).__name__}")

        name = raw.get("threshold_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"fps_mean_thresholds entry{where} must define a non-empty 'threshold_name'")
        name = name.strip()

        value = raw.get("threshold")
        if value is None:
            return None
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"fps_mean_thresholds entry {name!r} has non-numeric 'threshold': {value!r}")

        verdict_raw = raw.get("threshold_verdict")
        if verdict_raw is None:
            verdict: OracleVerdict | None = None
        else:
            allowed = sorted(v.value for v in THRESHOLD_VERDICTS)
            try:
                verdict = OracleVerdict(verdict_raw)
            except ValueError:
                raise ValueError(
                    f"fps_mean_thresholds entry {name!r} has invalid 'threshold_verdict' {verdict_raw!r}; "
                    f"must be one of {allowed} (or omitted for reporting-only)"
                )
            if verdict not in THRESHOLD_VERDICTS:
                raise ValueError(
                    f"fps_mean_thresholds entry {name!r} has non-gating 'threshold_verdict' {verdict_raw!r}; "
                    f"must be one of {allowed} (or omitted for reporting-only)"
                )
        return cls(name=name, value=threshold, verdict=verdict)

    @classmethod
    def from_list(cls, raw_list, *, context: str = "") -> "list[FpsMeanThreshold]":
        """Parse a leaf list of raw threshold entries, dropping value-less entries."""
        if raw_list is None:
            return []
        if not isinstance(raw_list, list):
            raise TypeError(f"fps_mean_thresholds leaf ({context}) must be a list, got {type(raw_list).__name__}")
        parsed: list[FpsMeanThreshold] = []
        for i, entry in enumerate(raw_list):
            threshold = cls.from_dict(entry, context=f"{context}[{i}]" if context else f"[{i}]")
            if threshold is not None:
                parsed.append(threshold)
        return parsed
