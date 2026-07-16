# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac-Sim-free metric helpers for benchmark bundles: TensorBoard parsing, convergence, EMA, MeanStd aggregation,
and success-rate tracking."""

from __future__ import annotations

import glob
import logging
import os
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from tensorboard.backend.event_processing import event_accumulator

from isaaclab.benchmark.schema import Framework, MeanStd

SUCCESS_RATE_LOG_TAGS = ("Metrics/success_rate", "Episode/Metrics/success_rate")


@dataclass(frozen=True)
class RLLibraryDescriptor:
    """TensorBoard locations and tags for one RL library benchmark integration.

    Attributes:
        framework: Schema framework id.
        tfevents_pattern: Glob relative to the run log directory matching TensorBoard event files.
        reward_tag: TensorBoard scalar tag for mean reward per iteration.
        ep_length_tag: TensorBoard scalar tag for mean episode length per iteration.
    """

    framework: Framework
    tfevents_pattern: str
    reward_tag: str
    ep_length_tag: str


RL_LIBRARY_DESCRIPTORS: dict[Framework, RLLibraryDescriptor] = {
    "rsl_rl": RLLibraryDescriptor(
        framework="rsl_rl",
        tfevents_pattern="events*",
        reward_tag="Train/mean_reward",
        ep_length_tag="Train/mean_episode_length",
    ),
    "rl_games": RLLibraryDescriptor(
        framework="rl_games",
        tfevents_pattern="summaries/events*",
        reward_tag="rewards/iter",
        ep_length_tag="episode_lengths/iter",
    ),
    "skrl": RLLibraryDescriptor(
        framework="skrl",
        tfevents_pattern="events*",
        reward_tag="Reward / Total reward (mean)",
        ep_length_tag="Episode / Total timesteps (mean)",
    ),
    "sb3": RLLibraryDescriptor(
        framework="sb3",
        tfevents_pattern="PPO_*/events*",
        reward_tag="rollout/ep_rew_mean",
        ep_length_tag="rollout/ep_len_mean",
    ),
}


def parse_tf_logs(log_dir: str, pattern: str = "events*") -> dict[str, list[float]]:
    """Load the latest TensorBoard events file under *log_dir* into ``{tag: [values]}``.

    Args:
        log_dir: Directory (glob root) to search for event files.
        pattern: Glob relative to *log_dir* matching the event files, e.g.
            ``"events*"`` (root), ``"summaries/events*"`` (rl_games), or
            ``"PPO_*/events*"`` (sb3).

    Returns:
        Mapping of each scalar tag to its per-iteration value list; empty when no event file matched.
    """
    list_of_files = glob.glob(os.path.join(log_dir, pattern))
    if not list_of_files:
        logging.getLogger(__name__).warning(
            "No TensorBoard event files matched %r under %r; returning empty log data.", pattern, log_dir
        )
        return {}
    latest_file = max(list_of_files, key=os.path.getmtime)
    ea = event_accumulator.EventAccumulator(latest_file)
    ea.Reload()
    log_data: dict[str, list[float]] = {}
    for tag in ea.Tags()["scalars"]:
        log_data[tag] = [event.value for event in ea.Scalars(tag)]
    return log_data


def get_success_rate_log(log_data: dict[str, list[float]]) -> list[float] | None:
    """Return the per-iteration success-rate series from parsed TensorBoard *log_data*.

    Looks up the first present tag from :data:`SUCCESS_RATE_LOG_TAGS` in *log_data* (the
    ``{tag: [values]}`` mapping produced by :func:`parse_tf_logs`) and returns its value
    series, or ``None`` when no success tag was logged.

    Args:
        log_data: Parsed TensorBoard scalars mapping each tag to its per-iteration list.

    Returns:
        The per-iteration success-rate values, or ``None`` if no success tag is present.
    """
    for tag in SUCCESS_RATE_LOG_TAGS:
        if tag in log_data:
            return log_data[tag]
    return None


def success_rate_step_value(extras_log: dict) -> float | None:
    """Return the scalar success-rate value for one env step from a live ``extras['log']`` dict.

    Unlike :func:`get_success_rate_log` (which returns a per-iteration *series* from parsed
    TensorBoard logs), this reads the single scalar logged per step in the live environment
    ``extras['log']`` mapping, coercing a tensor to ``float`` via ``.item()``.

    Args:
        extras_log: The ``extras['log']`` mapping from one environment step.

    Returns:
        The step's success-rate value as a ``float``, or ``None`` if no success tag is present.
    """
    for tag in SUCCESS_RATE_LOG_TAGS:
        if tag in extras_log:
            val = extras_log[tag]
            return float(val.item()) if hasattr(val, "item") else float(val)
    return None


def check_convergence(
    rewards: list[float],
    threshold: float,
    window_pct: float = 0.2,
    cv_threshold: float = 20.0,
) -> dict[str, float | bool]:
    """Check whether training rewards have converged.

    Passes when the trailing window mean exceeds *threshold* and the
    coefficient of variation (CV) is below *cv_threshold*.

    Args:
        rewards: Per-iteration mean reward values.
        threshold: Minimum reward to pass.
        window_pct: Fraction of iterations for the trailing window.
        cv_threshold: Maximum CV (%) for stable convergence.

    Returns:
        Dict with ``tail_mean``, ``cv``, and ``passed``.
    """
    if not rewards:
        return {"tail_mean": 0.0, "cv": 999.9, "passed": False}
    window = max(1, int(len(rewards) * window_pct))
    tail = rewards[-window:]
    tail_mean = statistics.mean(tail)
    tail_std = statistics.stdev(tail) if len(tail) > 1 else 0.0
    cv = (tail_std / abs(tail_mean) * 100) if tail_mean != 0 else 999.9
    passed = tail_mean >= threshold and cv <= cv_threshold
    return {"tail_mean": round(tail_mean, 2), "cv": round(cv, 1), "passed": passed}


class SuccessRateTracker:
    """Accumulates a per-iteration success-rate metric and checks trailing-window convergence.

    Args:
        threshold: Minimum value to consider a pass.
        window: Consecutive iterations above *threshold* to trigger convergence.
        num_steps_per_env: Steps per RL iteration (for boundary detection).
    """

    def __init__(self, threshold: float, window: int, num_steps_per_env: int):
        self.threshold = threshold
        self.window = window
        self.num_steps_per_env = num_steps_per_env

        self.history: list[float] = []
        self._step_count = 0
        self._iter_sum = 0.0
        self._iter_count = 0

    def record_step(self, extras: dict) -> None:
        """Record one env step."""
        val = success_rate_step_value(extras.get("log", {}))
        if val is not None:
            self._iter_sum += val
            self._iter_count += 1
        self._step_count += 1

    def end_iteration(self) -> float | None:
        """Finalize the current iteration. Returns mean metric, or ``None`` if no data."""
        if self._iter_count == 0:
            return None
        mean = self._iter_sum / self._iter_count
        self.history.append(mean)
        self._iter_sum = 0.0
        self._iter_count = 0
        return mean

    @property
    def at_iteration_boundary(self) -> bool:
        """Whether the tracker has seen exactly a full iteration's worth of steps.

        Assumes :meth:`record_step` is called exactly once per env step. This property is
        used only by the rsl_rl wrapper (whose patched ``env.step`` calls
        :meth:`record_step` once per env step); the rl_games observer ends iterations
        directly via ``after_steps`` and does not use this property.
        Integrations that call :meth:`record_step` more or fewer times per env step will
        break iteration accounting.
        """
        return self.num_steps_per_env > 0 and self._step_count % self.num_steps_per_env == 0

    @property
    def converged(self) -> bool:
        if len(self.history) < self.window:
            return False
        return all(v >= self.threshold for v in self.history[-self.window :])

    @property
    def current_iteration(self) -> int:
        return len(self.history)

    @property
    def tail_mean(self) -> float:
        if not self.history:
            return 0.0
        tail = self.history[-self.window :] if len(self.history) >= self.window else self.history
        return statistics.mean(tail)


def mean_std_peak(values: Sequence[float]) -> MeanStd:
    """Aggregate *values* into a :class:`MeanStd` with ``peak`` = max. Empty -> all zeros.

    Args:
        values: Per-sample values [same unit as the field being aggregated].
    """
    vals = list(values)
    if not vals:
        return MeanStd(mean=0.0, std=0.0, peak=0.0)
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return MeanStd(mean=mean, std=std, peak=max(vals))


def mean_std(values: Sequence[float]) -> MeanStd:
    """Aggregate *values* into a :class:`MeanStd` with ``peak`` omitted (``None``).

    Use for quantities where a peak is uninformative (e.g. utilisation). Empty -> zeros.

    Args:
        values: Per-sample values.
    """
    vals = list(values)
    if not vals:
        return MeanStd(mean=0.0, std=0.0, peak=None)
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return MeanStd(mean=mean, std=std, peak=None)


def ema(series: Sequence[float], alpha: float) -> float:
    """Exponential-moving-average final value of *series* (``e_0 = x_0``). Empty -> ``0.0``.

    Args:
        series: Per-iteration values.
        alpha: Smoothing factor in [0, 1]; higher weights recent values more.
    """
    vals = list(series)
    if not vals:
        return 0.0
    e = float(vals[0])
    for x in vals[1:]:
        e = alpha * float(x) + (1.0 - alpha) * e
    return e
