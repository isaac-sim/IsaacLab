# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-metric tracking and early stopping for training benchmarks."""

from __future__ import annotations

import argparse
import os
from typing import TYPE_CHECKING

from isaaclab.benchmark import SingleMeasurement
from isaaclab.benchmark.metrics import SuccessRateTracker, get_success_rate_log

if TYPE_CHECKING:
    from types import TracebackType

    from rl_games.common.algo_observer import AlgoObserver
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

DEFAULT_SUCCESS_THRESHOLD = 0.3
DEFAULT_SUCCESS_WINDOW = 20


class EarlyStopConverged(Exception):
    """Raised by :class:`RslRlEarlyStopWrapper` when the metric has converged."""


class RslRlEarlyStopWrapper:
    """Track RSL-RL success metrics and optionally stop on convergence.

    Args:
        env: ``RslRlVecEnvWrapper`` instance.
        runner: RSL-RL runner instance.
        threshold: Minimum metric value to pass.
        window: Consecutive iterations above threshold to trigger stop.
        num_steps_per_env: Steps per RL iteration.
        stop_on_convergence: If ``True``, raise :class:`EarlyStopConverged` when the metric converges.
            If ``False``, only track the metric without interrupting training.
    """

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        runner: OnPolicyRunner | DistillationRunner,
        threshold: float,
        window: int,
        num_steps_per_env: int,
        stop_on_convergence: bool = True,
    ):
        self.env = env
        self.runner = runner
        self.tracker = SuccessRateTracker(threshold, window, num_steps_per_env)
        self.stop_on_convergence = stop_on_convergence
        self._orig_step = env.step

    def __enter__(self) -> RslRlEarlyStopWrapper:
        self.env.step = self._step
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self.env.step = self._orig_step
        if exc_type is EarlyStopConverged:
            self._runner_cleanup()
            print(
                f"[INFO] Early stop: success rate converged at iteration "
                f"{self.tracker.current_iteration} (tail mean {self.tracker.tail_mean:.4f})"
            )
            return True
        return False

    def _step(self, actions) -> tuple:
        result = self._orig_step(actions)
        self.tracker.record_step(result[3])  # rsl_rl: (obs, rew, dones, extras)
        if self.tracker.at_iteration_boundary:
            self.tracker.end_iteration()
            if self.stop_on_convergence and self.tracker.converged:
                raise EarlyStopConverged()
        return result

    def _runner_cleanup(self):
        """Save final checkpoint and flush the TensorBoard writer."""
        if self.runner.logger.writer is not None:
            it = self.runner.current_learning_iteration
            self.runner.save(os.path.join(self.runner.logger.log_dir, f"model_{it}.pt"))
            self.runner.logger.stop_logging_writer()

    @property
    def framework_iteration_count(self) -> int:
        """Return completed runner iterations, including an active rollout."""
        return self.runner.current_learning_iteration + 1


class SuccessRateTrackerWrapper:
    """Track success metrics from a raw Gymnasium environment.

    Args:
        env: Environment whose ``step`` method returns Gymnasium's five-tuple.
        threshold: Minimum metric value to pass.
        window: Consecutive iterations above the threshold to converge.
        num_steps_per_env: Environment steps per training iteration.
    """

    def __init__(
        self,
        env,
        threshold: float,
        window: int,
        num_steps_per_env: int,
    ) -> None:
        self.env = env
        self.tracker = SuccessRateTracker(threshold, window, num_steps_per_env)
        self._orig_step = env.step

    def __enter__(self) -> SuccessRateTrackerWrapper:
        self.env.step = self._step
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.env.step = self._orig_step

    def _step(self, actions) -> tuple:
        result = self._orig_step(actions)
        self.tracker.record_step(result[4])
        if self.tracker.at_iteration_boundary:
            self.tracker.end_iteration()
        return result


class RlGamesEarlyStopObserver:
    """Track RL-Games success metrics and optionally stop on convergence.

    Args:
        base_observer: Original ``AlgoObserver`` to delegate to.
        threshold: Minimum metric value to pass.
        window: Consecutive iterations above threshold to trigger stop.
        stop_on_convergence: If ``True``, set ``algo.max_epochs`` when the metric converges.
            If ``False``, only track the metric without interrupting training.
    """

    def __init__(
        self,
        base_observer: AlgoObserver,
        threshold: float,
        window: int,
        stop_on_convergence: bool = True,
    ):
        self._base = base_observer
        self.threshold = threshold
        self.window = window
        self.stop_on_convergence = stop_on_convergence
        self.algo = None
        self.tracker: SuccessRateTracker | None = None

    def before_init(self, base_name: str, config: dict, experiment_name: str) -> None:
        self._base.before_init(base_name, config, experiment_name)

    def after_init(self, algo) -> None:
        self._base.after_init(algo)
        self.algo = algo
        num_steps = getattr(algo, "horizon_length", algo.config.get("horizon_length", 16))
        self.tracker = SuccessRateTracker(self.threshold, self.window, num_steps)

    def process_infos(self, infos, done_indices) -> None:
        self._base.process_infos(infos, done_indices)
        if self.tracker is not None and isinstance(infos, dict) and "episode" in infos:
            # rl_games remaps extras["log"] to extras["episode"]
            self.tracker.record_step({"log": infos["episode"]})

    def after_steps(self) -> None:
        self._base.after_steps()
        if self.tracker is None:
            return
        self.tracker.end_iteration()
        if self.stop_on_convergence and self.tracker.converged and self.algo is not None:
            print(
                f"[INFO] Early stop: success rate converged at iteration "
                f"{self.tracker.current_iteration} (tail mean {self.tracker.tail_mean:.4f})"
            )
            self.algo.max_epochs = self.tracker.current_iteration

    def after_clear_stats(self) -> None:
        if hasattr(self._base, "after_clear_stats"):
            self._base.after_clear_stats()

    def after_print_stats(self, frame: int, epoch_num: int, total_time: float) -> None:
        self._base.after_print_stats(frame, epoch_num, total_time)

    @property
    def framework_iteration_count(self) -> int | None:
        """Return the RL-Games epoch count, or ``None`` before initialization."""
        return None if self.algo is None else self.algo.epoch_num


def add_success_cli_args(parser: argparse.ArgumentParser, *, include_check_success: bool = True) -> None:
    """Register the success-metric CLI args on *parser*.

    Args:
        parser: Parser receiving the success-metric arguments.
        include_check_success: Whether to register the early-stop flag. Integrations
            without live success tracking still accept threshold and window overrides
            for post-hoc diagnostics.
    """
    if include_check_success:
        parser.add_argument(
            "--check_success", action="store_true", help="Early-stop when the normalized success metric converges."
        )
    else:
        parser.set_defaults(check_success=False)
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=None,
        help=f"Override the success threshold (default: {DEFAULT_SUCCESS_THRESHOLD}).",
    )
    parser.add_argument(
        "--success_window",
        type=int,
        default=None,
        help=f"Override the convergence window (default: {DEFAULT_SUCCESS_WINDOW}).",
    )


def build_success_kwargs(args_cli: argparse.Namespace) -> dict[str, float | int | bool]:
    """Resolve success-metric CLI args for the live tracker integrations.

    Returns:
        A dict with ``threshold``, ``window``, and ``stop_on_convergence`` keys.
    """
    return {
        "threshold": (
            args_cli.success_threshold if args_cli.success_threshold is not None else DEFAULT_SUCCESS_THRESHOLD
        ),
        "window": args_cli.success_window if args_cli.success_window is not None else DEFAULT_SUCCESS_WINDOW,
        "stop_on_convergence": getattr(args_cli, "check_success", False),
    }


def get_success_tracker(
    args_cli: argparse.Namespace,
    live_tracker: SuccessRateTracker | None,
    log_data: dict[str, list[float]],
) -> SuccessRateTracker | None:
    """Return a tracker with recorded history, or ``None`` if neither source has data.

    Prefer logged per-iteration history; fall back to the live tracker.

    Args:
        args_cli: Parsed arg namespace with the ``--success_*`` flags.
        live_tracker: Tracker attached to the early-stop wrapper/observer (or ``None``).
        log_data: Mapping of TB tag -> list of scalars for the current run.

    Returns:
        A :class:`SuccessRateTracker` populated with history, or ``None`` if no data is available.
    """
    history = get_success_rate_log(log_data)
    if history:
        tracker = live_tracker
        if tracker is None:
            kwargs = build_success_kwargs(args_cli)
            tracker = SuccessRateTracker(kwargs["threshold"], kwargs["window"], num_steps_per_env=0)
        tracker.history = list(history)
        return tracker
    if live_tracker is not None and live_tracker.history:
        return live_tracker
    return None


def success_measurements(tracker: SuccessRateTracker | None) -> list[SingleMeasurement]:
    """Build the established flat-output success diagnostics.

    Args:
        tracker: Success-rate tracker populated during or after training.

    Returns:
        Success tail mean, convergence iteration, and pass status, or an empty
        list when no success metric was recorded.
    """
    if tracker is None or not tracker.history:
        return []

    converged = tracker.converged
    return [
        SingleMeasurement(name="Success Rate (tail mean)", value=round(tracker.tail_mean, 4), unit="float"),
        SingleMeasurement(
            name="Success Converged At Iter",
            value=tracker.current_iteration if converged else -1,
            unit="int",
        ),
        SingleMeasurement(name="Success Passed", value=int(converged), unit="bool"),
    ]
