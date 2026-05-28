# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Early stopping for benchmark training based on a success metric.

Framework-specific implementations that monitor a metric from ``extras["log"]``
and stop training when it stabilizes above a threshold:

- **rsl_rl**: ``env.step`` wrapper + exception (no callback API in rsl_rl).
- **rl_games**: ``AlgoObserver`` subclass, sets ``max_epochs`` for clean exit.
"""

from __future__ import annotations

import argparse
import os
import statistics
from typing import TYPE_CHECKING

from scripts.benchmarks.utils import get_success_rate_log

if TYPE_CHECKING:
    from rl_games.common.algo_observer import AlgoObserver
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

DEFAULT_SUCCESS_THRESHOLD = 0.3
DEFAULT_SUCCESS_WINDOW = 20


class EarlyStopConverged(Exception):
    """Raised by :class:`RslRlEarlyStopWrapper` when the metric has converged."""


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
        val = get_success_rate_log(extras.get("log", {}))
        if val is not None:
            self._iter_sum += val.item() if hasattr(val, "item") else float(val)
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

        Assumes :meth:`record_step` is called exactly once per env step. This holds for
        all current framework integrations (rsl_rl's patched ``env.step`` and rl_games'
        ``AlgoObserver.process_infos``) — both pair a single step with a single record.
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


class RslRlEarlyStopWrapper:
    """Context manager that wraps ``env.step`` to track a success metric during rsl_rl training.

    Always records the metric into :attr:`tracker` so the caller can log the tail mean / converged-at
    iteration regardless of whether early stopping is enabled. When ``stop_on_convergence=True``, the
    wrapper also raises :class:`EarlyStopConverged` on the first iteration where the trailing window
    is above threshold, performs runner cleanup (checkpoint save + logger flush), and suppresses the
    exception so the caller sees a normal return from :meth:`rsl_rl.runners.OnPolicyRunner.learn`.

    Args:
        env: ``RslRlVecEnvWrapper`` instance.
        runner: ``OnPolicyRunner`` instance.
        threshold: Minimum metric value to pass.
        window: Consecutive iterations above threshold to trigger stop.
        num_steps_per_env: Steps per RL iteration.
        stop_on_convergence: If ``True``, raise :class:`EarlyStopConverged` when the metric converges.
            If ``False``, only track the metric without interrupting training.
    """

    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        runner: OnPolicyRunner,
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

    def __enter__(self):
        self.env.step = self._step
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.step = self._orig_step
        if exc_type is EarlyStopConverged:
            self._runner_cleanup()
            print(
                f"[INFO] Early stop: success rate converged at iteration "
                f"{self.tracker.current_iteration} (tail mean {self.tracker.tail_mean:.4f})"
            )
            return True
        return False

    def _step(self, actions):
        result = self._orig_step(actions)
        self.tracker.record_step(result[3])  # rsl_rl: (obs, rew, dones, extras)
        if self.tracker.at_iteration_boundary:
            self.tracker.end_iteration()
            if self.stop_on_convergence and self.tracker.converged:
                # relies on rsl_rl's rollout loop not catching arbitrary exceptions; if upstream
                # ever wraps env.step in a broad except, this exception will be swallowed
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
        """Number of training iterations the rsl_rl runner has recorded as completed.

        Note: ``current_learning_iteration`` is set AFTER rollout + policy update, so mid-rollout
        (including the instant our early-stop exception fires) this counter lags :attr:`tracker`
        by 1 iteration.
        """
        return self.runner.current_learning_iteration + 1


class RlGamesEarlyStopObserver:
    """``AlgoObserver`` that tracks a success metric during rl_games training.

    Always records the metric into :attr:`tracker` so the caller can log the tail mean / converged-at
    iteration regardless of whether early stopping is enabled. When ``stop_on_convergence=True``, the
    observer also sets ``algo.max_epochs`` on the first iteration where the trailing window is above
    threshold, which forces a clean exit from :meth:`rl_games.torch_runner.Runner.run`. All other
    observer calls are delegated to *base_observer*.

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

    def before_init(self, base_name, config, experiment_name):
        self._base.before_init(base_name, config, experiment_name)

    def after_init(self, algo):
        self._base.after_init(algo)
        self.algo = algo
        num_steps = getattr(algo, "horizon_length", algo.config.get("horizon_length", 16))
        self.tracker = SuccessRateTracker(self.threshold, self.window, num_steps)

    def process_infos(self, infos, done_indices):
        self._base.process_infos(infos, done_indices)
        if self.tracker is not None and isinstance(infos, dict) and "episode" in infos:
            # rl_games remaps extras["log"] → extras["episode"]
            self.tracker.record_step({"log": infos["episode"]})

    def after_steps(self):
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

    def after_clear_stats(self):
        self._base.after_clear_stats()

    def after_print_stats(self, frame, epoch_num, total_time):
        self._base.after_print_stats(frame, epoch_num, total_time)

    @property
    def framework_iteration_count(self) -> int | None:
        """Number of training iterations the rl_games algo has recorded.

        rl_games increments ``algo.epoch_num`` at the start of each iteration, so after iter N
        completes this value equals N (matching :attr:`tracker`'s count exactly). Returns
        ``None`` before :meth:`after_init` has attached to an algo.
        """
        return None if self.algo is None else self.algo.epoch_num


def add_success_cli_args(parser: argparse.ArgumentParser) -> None:
    """Register the success-metric CLI args on *parser*.

    Adds ``--check_success``, ``--success_threshold``, and ``--success_window``. Use
    :func:`build_success_kwargs` to resolve the parsed values into a kwargs dict for
    the wrapper constructors.
    """
    parser.add_argument(
        "--check_success", action="store_true", help="Early-stop when the normalized success metric converges."
    )
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


def build_success_kwargs(args_cli: argparse.Namespace) -> dict:
    """Resolve success-metric CLI args into kwargs for the wrapper constructors.

    Returns a dict with ``threshold``, ``window``, and ``stop_on_convergence``, suitable
    to splat into :class:`RslRlEarlyStopWrapper` or :class:`RlGamesEarlyStopObserver`.
    """
    return {
        "threshold": (
            args_cli.success_threshold if args_cli.success_threshold is not None else DEFAULT_SUCCESS_THRESHOLD
        ),
        "window": args_cli.success_window if args_cli.success_window is not None else DEFAULT_SUCCESS_WINDOW,
        "stop_on_convergence": args_cli.check_success,
    }


def get_success_tracker(
    args_cli: argparse.Namespace,
    live_tracker: SuccessRateTracker | None,
    log_data: dict[str, list[float]],
) -> SuccessRateTracker | None:
    """Return a tracker with recorded history, or ``None`` if neither source has data.

    Prefers *live_tracker* (from the training wrapper/observer). If it never ran or recorded
    no iterations, falls back to building a post-hoc tracker by replaying the success metric
    series out of TensorBoard *log_data* (from :func:`scripts.benchmarks.utils.parse_tf_logs`).

    Args:
        args_cli: Parsed arg namespace with the ``--success_*`` flags.
        live_tracker: Tracker attached to the early-stop wrapper/observer (or ``None``).
        log_data: Mapping of TB tag -> list of scalars for the current run.
    """
    if live_tracker is not None and live_tracker.history:
        return live_tracker
    history = get_success_rate_log(log_data)
    if not history:
        return None
    kwargs = build_success_kwargs(args_cli)
    tracker = SuccessRateTracker(kwargs["threshold"], kwargs["window"], num_steps_per_env=0)
    tracker.history = list(history)
    return tracker
