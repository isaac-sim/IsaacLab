# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the benchmark success-metric early-stopping helpers."""

from __future__ import annotations

import argparse

import pytest

from scripts.benchmarks.early_stop import (
    DEFAULT_SUCCESS_THRESHOLD,
    DEFAULT_SUCCESS_WINDOW,
    RlGamesEarlyStopObserver,
    RslRlEarlyStopWrapper,
    SuccessRateTracker,
    add_success_cli_args,
    build_success_kwargs,
    get_success_tracker,
)
from scripts.benchmarks.utils import SUCCESS_RATE_LOG_TAGS, log_success

DEFAULT_SUCCESS_TAG = SUCCESS_RATE_LOG_TAGS[0]

# -- fakes ------------------------------------------------------------------


class _FakeTensor:
    """Stand-in for ``torch.Tensor`` with only the ``.item()`` path exercised."""

    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


class _FakeBenchmark:
    def __init__(self):
        self.measurements: list[tuple[str, str, object, str]] = []

    def add_measurement(self, phase, measurement):
        self.measurements.append((phase, measurement.name, measurement.value, measurement.unit))

    def by_name(self, name: str):
        return next(m for m in self.measurements if m[1] == name)


class _FakeLogger:
    def __init__(self, has_writer: bool = True):
        self.writer = object() if has_writer else None
        self.log_dir = "/tmp/fake_log_dir"
        self.stopped = False

    def stop_logging_writer(self):
        self.stopped = True


class _FakeRunner:
    def __init__(self, has_writer: bool = True):
        self.logger = _FakeLogger(has_writer=has_writer)
        self.current_learning_iteration = 7
        self.saved: list[str] = []

    def save(self, path: str):
        self.saved.append(path)


class _FakeEnv:
    def __init__(self, extras_sequence):
        self._seq = list(extras_sequence)
        self.step_calls = 0

    def step(self, actions):
        extras = self._seq[self.step_calls] if self.step_calls < len(self._seq) else self._seq[-1]
        self.step_calls += 1
        return (None, None, None, extras)


class _FakeBaseObserver:
    def __init__(self):
        self.calls: list[str] = []

    def before_init(self, base_name, config, experiment_name):
        self.calls.append("before_init")

    def after_init(self, algo):
        self.calls.append("after_init")

    def process_infos(self, infos, done_indices):
        self.calls.append("process_infos")

    def after_steps(self):
        self.calls.append("after_steps")

    def after_clear_stats(self):
        self.calls.append("after_clear_stats")

    def after_print_stats(self, frame, epoch_num, total_time):
        self.calls.append("after_print_stats")


class _FakeAlgo:
    def __init__(self, horizon_length: int | None = None, config_horizon: int | None = 16, epoch_num: int = 0):
        self.max_epochs = 999
        self.epoch_num = epoch_num
        if horizon_length is not None:
            self.horizon_length = horizon_length
        self.config = {"horizon_length": config_horizon} if config_horizon is not None else {}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    add_success_cli_args(p)
    return p


# -- SuccessRateTracker -----------------------------------------------------


class TestSuccessRateTracker:
    """Test cases for the per-iteration metric accumulator and convergence check."""

    def test_records_metric_from_extras_log(self):
        """Test that a present metric is accumulated into the iteration sum."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.record_step({"log": {DEFAULT_SUCCESS_TAG: 0.9}})
        assert t._iter_sum == pytest.approx(0.9)
        assert t._iter_count == 1

    def test_ignores_missing_metric_key(self):
        """Test that a foreign key in extras["log"] is ignored."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.record_step({"log": {"other": 1.0}})
        assert t._iter_count == 0

    def test_missing_log_subdict_does_not_raise(self):
        """Test that an extras dict without a "log" sub-dict is handled gracefully."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.record_step({})
        assert t._iter_count == 0
        assert t._step_count == 1

    def test_tensor_value_uses_item_method(self):
        """Test that tensor-like values are extracted via ``.item()``."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.record_step({"log": {DEFAULT_SUCCESS_TAG: _FakeTensor(0.7)}})
        assert t._iter_sum == pytest.approx(0.7)

    def test_step_count_increments_even_without_metric(self):
        """Test that ``_step_count`` tracks every call regardless of metric presence."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.record_step({})
        t.record_step({"log": {"other": 1.0}})
        assert t._step_count == 2
        assert t._iter_count == 0

    def test_end_iteration_averages_and_resets(self):
        """Test that ``end_iteration`` averages recorded values and resets counters."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.record_step({"log": {DEFAULT_SUCCESS_TAG: 0.4}})
        t.record_step({"log": {DEFAULT_SUCCESS_TAG: 0.6}})
        assert t.end_iteration() == pytest.approx(0.5)
        assert t.history == [pytest.approx(0.5)]
        assert t._iter_sum == 0.0
        assert t._iter_count == 0

    def test_end_iteration_no_data_returns_none_without_recording(self):
        """Test that ``end_iteration`` returns None and skips history append when no data was seen."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        assert t.end_iteration() is None
        assert t.history == []

    def test_at_iteration_boundary_respects_num_steps_per_env(self):
        """Test that the boundary flag fires only after exactly ``num_steps_per_env`` calls."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        for _ in range(3):
            t.record_step({"log": {DEFAULT_SUCCESS_TAG: 0.1}})
        assert t.at_iteration_boundary is False
        t.record_step({"log": {DEFAULT_SUCCESS_TAG: 0.1}})
        assert t.at_iteration_boundary is True

    def test_at_iteration_boundary_false_when_num_steps_zero(self):
        """Test that a post-hoc tracker (``num_steps_per_env=0``) never reports a boundary."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=0)
        t.record_step({"log": {DEFAULT_SUCCESS_TAG: 0.1}})
        assert t.at_iteration_boundary is False

    def test_not_converged_when_history_shorter_than_window(self):
        """Test that convergence is False when there aren't yet enough history entries."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.history = [0.9, 0.9]
        assert t.converged is False

    def test_not_converged_when_history_empty(self):
        """Test that convergence is False on a freshly-created tracker."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        assert t.history == []
        assert t.converged is False

    def test_converged_when_window_all_above_threshold(self):
        """Test that convergence is True when the trailing window is all above threshold."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.history = [0.1, 0.9, 0.9, 0.9]
        assert t.converged is True

    def test_converged_when_history_length_equals_window(self):
        """Test the window boundary: history length == window (minimum qualifying case)."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.history = [0.9, 0.9, 0.9]
        assert t.converged is True

    def test_converged_at_exact_threshold(self):
        """Test the threshold boundary: values equal to the threshold satisfy ``>= threshold``."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.history = [0.5, 0.5, 0.5]
        assert t.converged is True

    def test_not_converged_when_any_window_value_below(self):
        """Test that a single sub-threshold value in the trailing window blocks convergence."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.history = [0.9, 0.9, 0.4]
        assert t.converged is False

    def test_converged_with_window_of_one(self):
        """Test the degenerate ``window=1`` case: only the last value matters."""
        t = SuccessRateTracker(0.5, 1, num_steps_per_env=4)
        t.history = [0.1, 0.2, 0.9]
        assert t.converged is True
        t.history = [0.9, 0.9, 0.1]
        assert t.converged is False

    def test_tail_mean_empty_history_is_zero(self):
        """Test that ``tail_mean`` returns 0.0 for an empty history."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        assert t.tail_mean == 0.0

    def test_tail_mean_shorter_than_window_uses_all_values(self):
        """Test that ``tail_mean`` averages the full history when it's shorter than the window."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.history = [0.2, 0.4]
        assert t.tail_mean == pytest.approx(0.3)

    def test_tail_mean_longer_than_window_uses_tail(self):
        """Test that ``tail_mean`` averages only the last ``window`` entries."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.history = [0.9, 0.9, 0.1, 0.2, 0.3]
        assert t.tail_mean == pytest.approx(0.2)

    def test_current_iteration_equals_history_length(self):
        """Test that ``current_iteration`` reports the history length."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.history = [0.1, 0.2, 0.3]
        assert t.current_iteration == 3


# -- CLI helpers ------------------------------------------------------------


class TestCliHelpers:
    """Test cases for the ``--success_*`` CLI registration and kwargs resolution."""

    def test_defaults_parse_to_none_and_false(self):
        """Test that unset args resolve to None / False."""
        args = _parser().parse_args([])
        assert args.check_success is False
        assert args.success_threshold is None
        assert args.success_window is None

    def test_overrides_parse(self):
        """Test that explicit ``--success_*`` values round-trip through argparse."""
        args = _parser().parse_args(
            [
                "--check_success",
                "--success_threshold",
                "0.75",
                "--success_window",
                "50",
            ]
        )
        assert args.check_success is True
        assert args.success_threshold == 0.75
        assert args.success_window == 50

    def test_build_success_kwargs_uses_defaults_when_unset(self):
        """Test that ``build_success_kwargs`` substitutes library defaults for unset args."""
        kwargs = build_success_kwargs(_parser().parse_args([]))
        assert kwargs == {
            "threshold": DEFAULT_SUCCESS_THRESHOLD,
            "window": DEFAULT_SUCCESS_WINDOW,
            "stop_on_convergence": False,
        }

    def test_build_success_kwargs_applies_overrides(self):
        """Test that CLI overrides flow through into the kwargs dict."""
        args = _parser().parse_args(
            [
                "--check_success",
                "--success_threshold",
                "0.1",
                "--success_window",
                "5",
            ]
        )
        kwargs = build_success_kwargs(args)
        assert kwargs["threshold"] == pytest.approx(0.1)
        assert kwargs["window"] == 5
        assert kwargs["stop_on_convergence"] is True

    def test_zero_threshold_is_respected_not_treated_as_unset(self):
        """Test that ``--success_threshold 0`` is preserved (``is not None`` check, not truthy)."""
        args = _parser().parse_args(["--success_threshold", "0"])
        assert build_success_kwargs(args)["threshold"] == 0.0


# -- get_success_tracker ----------------------------------------------------


class TestGetSuccessTracker:
    """Test cases for the live-vs-post-hoc tracker resolution helper."""

    def test_prefers_live_tracker_with_history(self):
        """Test that a non-empty live tracker is returned as-is."""
        live = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        live.history = [0.9, 0.9]
        assert get_success_tracker(_parser().parse_args([]), live, {}) is live

    def test_falls_back_to_post_hoc_when_live_tracker_empty(self):
        """Test that an empty live tracker falls back to TensorBoard replay."""
        live = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        log_data = {DEFAULT_SUCCESS_TAG: [0.1, 0.2, 0.3]}
        result = get_success_tracker(_parser().parse_args([]), live, log_data)
        assert result is not live
        assert result.history == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]

    def test_falls_back_to_post_hoc_when_live_tracker_none(self):
        """Test that a missing live tracker falls back to TensorBoard replay."""
        log_data = {DEFAULT_SUCCESS_TAG: [0.5, 0.6, 0.7]}
        result = get_success_tracker(_parser().parse_args([]), None, log_data)
        assert result is not None
        assert result.history == [pytest.approx(0.5), pytest.approx(0.6), pytest.approx(0.7)]

    def test_returns_none_when_no_data_anywhere(self):
        """Test that both sources missing resolves to ``None``."""
        assert get_success_tracker(_parser().parse_args([]), None, {}) is None

    def test_returns_none_when_tag_absent_from_log_data(self):
        """Test that unrelated TensorBoard tags don't satisfy the fallback."""
        assert get_success_tracker(_parser().parse_args([]), None, {"Metrics/other": [1.0]}) is None

    def test_post_hoc_honors_override_threshold_and_window(self):
        """Test that CLI threshold/window overrides are applied to the post-hoc tracker."""
        args = _parser().parse_args(["--success_threshold", "0.2", "--success_window", "2"])
        log_data = {DEFAULT_SUCCESS_TAG: [0.3, 0.3]}
        result = get_success_tracker(args, None, log_data)
        assert result.threshold == pytest.approx(0.2)
        assert result.window == 2
        assert result.converged is True

    def test_post_hoc_tracker_has_no_iteration_boundary(self):
        """Test that post-hoc trackers use ``num_steps_per_env=0`` so ``at_iteration_boundary`` never fires."""
        result = get_success_tracker(_parser().parse_args([]), None, {DEFAULT_SUCCESS_TAG: [0.9]})
        assert result.num_steps_per_env == 0
        assert result.at_iteration_boundary is False


# -- RslRlEarlyStopWrapper --------------------------------------------------


class TestRslRlEarlyStopWrapper:
    """Test cases for the rsl_rl env.step monkey-patch context manager."""

    def test_records_every_step_and_restores_on_exit(self):
        """Test that wrapped env.step records, and original step is restored on normal exit."""
        env = _FakeEnv([{"log": {DEFAULT_SUCCESS_TAG: 0.9}}] * 5)
        runner = _FakeRunner()
        with RslRlEarlyStopWrapper(env, runner, 0.5, 3, num_steps_per_env=2) as ctx:
            env.step(None)
            assert ctx.tracker._iter_sum == pytest.approx(0.9)
        # after exit, env.step no longer routes through the tracker
        env.step(None)
        assert ctx.tracker._iter_sum == pytest.approx(0.9)
        assert env.step_calls == 2

    def test_raises_and_cleans_up_on_convergence_by_default(self):
        """Test that convergence triggers cleanup (checkpoint + flush) and suppresses the exception."""
        env = _FakeEnv([{"log": {DEFAULT_SUCCESS_TAG: 0.9}}] * 100)
        runner = _FakeRunner()
        # num_steps_per_env=2, window=2 -> converges on step 4 (iter 2)
        with RslRlEarlyStopWrapper(env, runner, 0.5, 2, num_steps_per_env=2) as ctx:
            for _ in range(10):
                env.step(None)
        assert ctx.tracker.converged is True
        assert env.step_calls == 4
        assert len(runner.saved) == 1
        assert runner.logger.stopped is True

    def test_does_not_raise_when_stop_on_convergence_false(self):
        """Test that ``stop_on_convergence=False`` lets training run past convergence."""
        env = _FakeEnv([{"log": {DEFAULT_SUCCESS_TAG: 0.9}}] * 100)
        runner = _FakeRunner()
        with RslRlEarlyStopWrapper(
            env,
            runner,
            0.5,
            2,
            num_steps_per_env=2,
            stop_on_convergence=False,
        ) as ctx:
            for _ in range(10):
                env.step(None)
        assert env.step_calls == 10
        assert ctx.tracker.converged is True
        assert runner.saved == []
        assert runner.logger.stopped is False

    def test_does_not_suppress_other_exceptions(self):
        """Test that non-EarlyStopConverged exceptions propagate out of the ``with`` block."""
        env = _FakeEnv([{"log": {}}])
        runner = _FakeRunner()
        with pytest.raises(ValueError):
            with RslRlEarlyStopWrapper(env, runner, 0.5, 2, num_steps_per_env=2):
                raise ValueError("not an early stop")

    def test_env_step_restored_after_early_stop_exception(self):
        """Test that env.step is unwrapped after an early-stop exception suppressed by __exit__."""
        env = _FakeEnv([{"log": {DEFAULT_SUCCESS_TAG: 0.9}}] * 100)
        runner = _FakeRunner()
        with RslRlEarlyStopWrapper(env, runner, 0.5, 2, num_steps_per_env=2) as ctx:
            for _ in range(10):
                env.step(None)  # converges & raises at step 4, suppressed
        sum_at_exit = ctx.tracker._iter_sum
        env.step(None)
        assert ctx.tracker._iter_sum == sum_at_exit  # post-exit step bypassed the tracker

    def test_env_step_restored_after_unrelated_exception(self):
        """Test that env.step is unwrapped even when a non-EarlyStopConverged exception propagates."""
        env = _FakeEnv([{"log": {DEFAULT_SUCCESS_TAG: 0.9}}] * 10)
        runner = _FakeRunner()
        try:
            with RslRlEarlyStopWrapper(env, runner, 0.5, 2, num_steps_per_env=2) as ctx:
                env.step(None)
                raise ValueError("boom")
        except ValueError:
            pass
        sum_at_exit = ctx.tracker._iter_sum
        env.step(None)
        assert ctx.tracker._iter_sum == sum_at_exit

    def test_cleanup_not_called_on_unrelated_exceptions(self):
        """Test that only EarlyStopConverged triggers checkpoint save + logger flush."""
        env = _FakeEnv([{"log": {}}])
        runner = _FakeRunner()
        try:
            with RslRlEarlyStopWrapper(env, runner, 0.5, 2, num_steps_per_env=2):
                raise ValueError("boom")
        except ValueError:
            pass
        assert runner.saved == []
        assert runner.logger.stopped is False

    def test_cleanup_skipped_when_runner_has_no_writer(self):
        """Test that cleanup skips both save and flush when ``runner.logger.writer`` is ``None``."""
        env = _FakeEnv([{"log": {DEFAULT_SUCCESS_TAG: 0.9}}] * 100)
        runner = _FakeRunner(has_writer=False)
        with RslRlEarlyStopWrapper(env, runner, 0.5, 2, num_steps_per_env=2):
            for _ in range(10):
                env.step(None)
        assert runner.saved == []
        assert runner.logger.stopped is False

    def test_framework_iteration_count_reflects_runner(self):
        """Test that the framework-counter property reports ``current_learning_iteration + 1``."""
        env = _FakeEnv([{"log": {DEFAULT_SUCCESS_TAG: 0.0}}])
        runner = _FakeRunner()
        runner.current_learning_iteration = 42
        wrapper = RslRlEarlyStopWrapper(env, runner, 0.5, 3, num_steps_per_env=2)
        assert wrapper.framework_iteration_count == 43


# -- RlGamesEarlyStopObserver -----------------------------------------------


class TestRlGamesEarlyStopObserver:
    """Test cases for the rl_games AlgoObserver that tracks success and forces max_epochs."""

    def test_delegates_every_call_to_base(self):
        """Test that all observer lifecycle calls are forwarded to the wrapped base observer."""
        base = _FakeBaseObserver()
        obs = RlGamesEarlyStopObserver(base, 0.5, 3)
        obs.before_init("name", {}, "exp")
        obs.after_init(_FakeAlgo(horizon_length=8))
        obs.process_infos({"episode": {}}, [])
        obs.after_steps()
        obs.after_clear_stats()
        obs.after_print_stats(0, 0, 0)
        assert base.calls == [
            "before_init",
            "after_init",
            "process_infos",
            "after_steps",
            "after_clear_stats",
            "after_print_stats",
        ]

    def test_tracker_uses_horizon_length_attribute(self):
        """Test that the tracker pulls ``num_steps_per_env`` from ``algo.horizon_length`` when present."""
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 3)
        obs.after_init(_FakeAlgo(horizon_length=24))
        assert obs.tracker.num_steps_per_env == 24

    def test_tracker_falls_back_to_config_horizon_length(self):
        """Test that the tracker falls back to ``algo.config['horizon_length']`` when the attr is missing."""
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 3)
        obs.after_init(_FakeAlgo(horizon_length=None, config_horizon=32))
        assert obs.tracker.num_steps_per_env == 32

    def test_process_infos_records_from_episode_key(self):
        """Test that ``infos["episode"]`` is remapped to the tracker's extras["log"] shape."""
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 3)
        obs.after_init(_FakeAlgo(horizon_length=2))
        obs.process_infos({"episode": {DEFAULT_SUCCESS_TAG: 0.8}}, [])
        assert obs.tracker._iter_sum == pytest.approx(0.8)

    def test_process_infos_is_noop_before_after_init(self):
        """Test that ``process_infos`` before ``after_init`` does not raise (tracker is None)."""
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 3)
        obs.process_infos({"episode": {DEFAULT_SUCCESS_TAG: 0.8}}, [])
        assert obs.tracker is None

    def test_process_infos_ignores_non_dict_infos(self):
        """Test that non-dict ``infos`` are skipped gracefully without mutating the tracker."""
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 3)
        obs.after_init(_FakeAlgo(horizon_length=2))
        obs.process_infos([], [])
        assert obs.tracker._iter_sum == 0.0

    def test_after_steps_sets_max_epochs_on_convergence(self):
        """Test that convergence on iteration N sets ``algo.max_epochs = N`` for clean exit."""
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 2)
        algo = _FakeAlgo(horizon_length=1)
        obs.after_init(algo)
        obs.process_infos({"episode": {DEFAULT_SUCCESS_TAG: 0.9}}, [])
        obs.after_steps()
        obs.process_infos({"episode": {DEFAULT_SUCCESS_TAG: 0.9}}, [])
        obs.after_steps()
        assert algo.max_epochs == 2

    def test_after_steps_leaves_max_epochs_alone_when_stop_disabled(self):
        """Test that ``stop_on_convergence=False`` preserves the caller's ``algo.max_epochs``."""
        obs = RlGamesEarlyStopObserver(
            _FakeBaseObserver(),
            0.5,
            2,
            stop_on_convergence=False,
        )
        algo = _FakeAlgo(horizon_length=1)
        original_max_epochs = algo.max_epochs
        obs.after_init(algo)
        obs.process_infos({"episode": {DEFAULT_SUCCESS_TAG: 0.9}}, [])
        obs.after_steps()
        obs.process_infos({"episode": {DEFAULT_SUCCESS_TAG: 0.9}}, [])
        obs.after_steps()
        assert algo.max_epochs == original_max_epochs

    def test_after_steps_noop_before_after_init(self):
        """Test that ``after_steps`` before ``after_init`` does not raise (tracker is None)."""
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 2)
        obs.after_steps()
        assert obs.tracker is None

    def test_each_after_steps_appends_one_iteration(self):
        """Test that each ``after_steps`` call finalizes exactly one iteration in the tracker."""
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 5)
        obs.after_init(_FakeAlgo(horizon_length=1))
        for i in range(4):
            obs.process_infos({"episode": {DEFAULT_SUCCESS_TAG: 0.9}}, [])
            obs.after_steps()
            assert obs.tracker.current_iteration == i + 1

    def test_after_steps_does_not_converge_with_insufficient_history(self):
        """Test that a trailing window shorter than ``window`` does not trigger early stop."""
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 5)
        algo = _FakeAlgo(horizon_length=1)
        obs.after_init(algo)
        for _ in range(4):
            obs.process_infos({"episode": {DEFAULT_SUCCESS_TAG: 0.9}}, [])
            obs.after_steps()
        assert algo.max_epochs == 999  # unchanged: tracker.converged is still False

    def test_framework_iteration_count_returns_none_before_after_init(self):
        """Test that the framework-counter property returns ``None`` before an algo is attached."""
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 2)
        assert obs.framework_iteration_count is None

    def test_framework_iteration_count_reflects_algo_epoch_num(self):
        """Test that the framework-counter property mirrors ``algo.epoch_num``."""
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 2)
        obs.after_init(_FakeAlgo(horizon_length=1, epoch_num=7))
        assert obs.framework_iteration_count == 7


# -- log_success (scripts.benchmarks.utils) ---------------------------------


class TestLogSuccess:
    """Test cases for the benchmark-side success-metric logging helper."""

    def _tracker_with(self, history: list[float]) -> SuccessRateTracker:
        """Build a tracker with a pre-populated history for testing."""
        t = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        t.history = history
        return t

    def test_noop_when_tracker_is_none(self):
        """Test that ``log_success`` emits nothing when no tracker is supplied."""
        bench = _FakeBenchmark()
        log_success(bench, None)
        assert bench.measurements == []

    def test_noop_when_history_empty(self):
        """Test that an empty tracker history is a silent no-op."""
        bench = _FakeBenchmark()
        log_success(bench, self._tracker_with([]))
        assert bench.measurements == []

    def test_logs_full_measurement_set(self):
        """Test that a populated tracker produces the full measurement set."""
        bench = _FakeBenchmark()
        log_success(bench, self._tracker_with([0.9, 0.9, 0.9]))
        names = {m[1] for m in bench.measurements}
        assert names == {"Success Rate (tail mean)", "Success Converged At Iter", "Success Passed"}

    def test_converged_path(self):
        """Test that a converged run reports ``Passed=1`` with the true converged iter + tail mean."""
        bench = _FakeBenchmark()
        log_success(bench, self._tracker_with([0.9, 0.9, 0.9]))
        assert bench.by_name("Success Passed")[2] == 1
        assert bench.by_name("Success Converged At Iter")[2] == 3
        assert bench.by_name("Success Rate (tail mean)")[2] == pytest.approx(0.9)

    def test_failed_path(self):
        """Test that a non-converged run reports ``Passed=0`` and ``Converged At Iter=-1``."""
        bench = _FakeBenchmark()
        log_success(bench, self._tracker_with([0.1, 0.2, 0.3]))
        assert bench.by_name("Success Passed")[2] == 0
        assert bench.by_name("Success Converged At Iter")[2] == -1

    def test_cadence_warning_fires_on_cadence_violation(self, capsys):
        """Test that a 2x tracker/framework ratio triggers the cadence warning."""
        bench = _FakeBenchmark()
        log_success(bench, self._tracker_with([0.5] * 100), framework_iteration_count=50)
        captured = capsys.readouterr().out
        assert "[WARN]" in captured
        assert "check record_step cadence" in captured

    def test_no_cadence_warning_on_exact_agreement(self, capsys):
        """Test that an exact tracker-vs-framework match (rl_games case) is silent."""
        bench = _FakeBenchmark()
        log_success(bench, self._tracker_with([0.5] * 50), framework_iteration_count=50)
        assert "[WARN]" not in capsys.readouterr().out

    def test_no_cadence_warning_on_rsl_rl_early_stop_offset(self, capsys):
        """Test that the rsl_rl early-stop +1 offset (tracker=51, framework=50) is within slack."""
        bench = _FakeBenchmark()
        log_success(bench, self._tracker_with([0.5] * 51), framework_iteration_count=50)
        assert "[WARN]" not in capsys.readouterr().out

    def test_no_cadence_warning_when_framework_count_not_provided(self, capsys):
        """Test that the cadence check is skipped entirely when no framework count is supplied."""
        bench = _FakeBenchmark()
        log_success(bench, self._tracker_with([0.5] * 999))
        assert "[WARN]" not in capsys.readouterr().out

    def test_cadence_violation_end_to_end_via_wrapper(self, capsys):
        """Test that a simulated 2x env.step bug manifests as an overcounted tracker and is caught.

        The wrapper can't distinguish "2 env.step calls that should have been 1" from normal
        traffic — but the tracker overcounts iterations by 2x, and comparing against the
        runner's independent counter catches the discrepancy.
        """
        env = _FakeEnv([{"log": {DEFAULT_SUCCESS_TAG: 0.5}}] * 100)
        runner = _FakeRunner()
        runner.current_learning_iteration = 9  # rsl_rl thinks 10 iterations completed
        with RslRlEarlyStopWrapper(
            env,
            runner,
            0.5,
            3,
            num_steps_per_env=2,
            stop_on_convergence=False,
        ) as ctx:
            # simulate the bug: upstream calls env.step 2x per real rollout step
            for _ in range(10 * 2 * 2):  # 10 iters * 2 steps/iter * 2x-bug
                env.step(None)
        # 40 calls with num_steps_per_env=2 => tracker.current_iteration = 20
        assert ctx.tracker.current_iteration == 20
        # framework's counter is independent: reports 10 iterations actually ran
        assert ctx.framework_iteration_count == 10
        bench = _FakeBenchmark()
        log_success(bench, ctx.tracker, framework_iteration_count=ctx.framework_iteration_count)
        captured = capsys.readouterr().out
        assert "[WARN]" in captured
