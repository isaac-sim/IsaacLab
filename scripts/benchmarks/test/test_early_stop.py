# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the benchmark success-metric early-stopping helpers."""

from __future__ import annotations

import argparse

import pytest

from isaaclab.test.benchmark.metrics import SUCCESS_RATE_LOG_TAGS

from scripts.benchmarks import early_stop
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

DEFAULT_SUCCESS_TAG = SUCCESS_RATE_LOG_TAGS[0]


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


class _FakeGymEnv:
    def __init__(self, extras_sequence):
        self._seq = list(extras_sequence)
        self.step_calls = 0

    def step(self, actions):
        extras = self._seq[self.step_calls] if self.step_calls < len(self._seq) else self._seq[-1]
        self.step_calls += 1
        return (None, None, None, None, extras)


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


class TestCliHelpers:
    def test_build_success_kwargs_uses_defaults_when_unset(self):
        kwargs = build_success_kwargs(_parser().parse_args([]))
        assert kwargs == {
            "threshold": DEFAULT_SUCCESS_THRESHOLD,
            "window": DEFAULT_SUCCESS_WINDOW,
            "stop_on_convergence": False,
        }

    def test_build_success_kwargs_applies_overrides(self):
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
        args = _parser().parse_args(["--success_threshold", "0"])
        assert build_success_kwargs(args)["threshold"] == 0.0

    def test_registration_can_exclude_early_stop(self):
        parser = argparse.ArgumentParser()
        add_success_cli_args(parser, include_check_success=False)

        args = parser.parse_args(["--success_threshold", "0.2"])

        assert build_success_kwargs(args)["stop_on_convergence"] is False
        with pytest.raises(SystemExit):
            parser.parse_args(["--check_success"])


class TestGetSuccessTracker:
    def test_prefers_live_tracker_with_history(self):
        live = SuccessRateTracker(0.5, 3, num_steps_per_env=4)
        live.history = [0.9, 0.9]
        assert get_success_tracker(_parser().parse_args([]), live, {}) is live

    def test_falls_back_to_post_hoc_when_live_tracker_none(self):
        log_data = {DEFAULT_SUCCESS_TAG: [0.5, 0.6, 0.7]}
        result = get_success_tracker(_parser().parse_args([]), None, log_data)
        assert result is not None
        assert result.history == [pytest.approx(0.5), pytest.approx(0.6), pytest.approx(0.7)]

    def test_returns_none_when_no_data_anywhere(self):
        assert get_success_tracker(_parser().parse_args([]), None, {}) is None

    def test_post_hoc_honors_override_threshold_and_window(self):
        args = _parser().parse_args(["--success_threshold", "0.2", "--success_window", "2"])
        log_data = {DEFAULT_SUCCESS_TAG: [0.3, 0.3]}
        result = get_success_tracker(args, None, log_data)
        assert result.threshold == pytest.approx(0.2)
        assert result.window == 2
        assert result.converged is True

    def test_success_measurements_preserve_flat_output_fields(self):
        tracker = SuccessRateTracker(0.3, 2, num_steps_per_env=0)
        tracker.history = [0.1, 0.2]

        measurements = early_stop.success_measurements(tracker)

        assert {measurement.name: measurement.value for measurement in measurements} == {
            "Success Rate (tail mean)": 0.15,
            "Success Converged At Iter": -1,
            "Success Passed": 0,
        }


class TestSuccessRateTrackerWrapper:
    def test_tracks_raw_env_success_by_iteration(self):
        env = _FakeGymEnv(
            [
                {"log": {DEFAULT_SUCCESS_TAG: 0.2}},
                {"log": {DEFAULT_SUCCESS_TAG: 0.6}},
                {"log": {DEFAULT_SUCCESS_TAG: 0.8}},
                {"log": {DEFAULT_SUCCESS_TAG: 1.0}},
            ]
        )
        wrapper = early_stop.SuccessRateTrackerWrapper(env, 0.5, 2, num_steps_per_env=2)

        with wrapper:
            for _ in range(4):
                env.step(None)

        assert wrapper.tracker.history == [pytest.approx(0.4), pytest.approx(0.9)]


class TestRslRlEarlyStopWrapper:
    def test_records_every_step_and_restores_on_exit(self):
        env = _FakeEnv([{"log": {DEFAULT_SUCCESS_TAG: 0.9}}] * 5)
        runner = _FakeRunner()
        with RslRlEarlyStopWrapper(env, runner, 0.5, 3, num_steps_per_env=2) as ctx:
            env.step(None)
            assert ctx.tracker._iter_sum == pytest.approx(0.9)
        env.step(None)
        assert ctx.tracker._iter_sum == pytest.approx(0.9)
        assert env.step_calls == 2

    def test_raises_and_cleans_up_on_convergence_by_default(self):
        env = _FakeEnv([{"log": {DEFAULT_SUCCESS_TAG: 0.9}}] * 100)
        runner = _FakeRunner()
        with RslRlEarlyStopWrapper(env, runner, 0.5, 2, num_steps_per_env=2) as ctx:
            for _ in range(10):
                env.step(None)
        assert ctx.tracker.converged is True
        assert env.step_calls == 4
        assert len(runner.saved) == 1
        assert runner.logger.stopped is True
        tracked_iterations = ctx.tracker.current_iteration
        env.step(None)
        assert ctx.tracker.current_iteration == tracked_iterations

    def test_does_not_raise_when_stop_on_convergence_false(self):
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
        env = _FakeEnv([{"log": {}}])
        runner = _FakeRunner()
        wrapper = RslRlEarlyStopWrapper(env, runner, 0.5, 2, num_steps_per_env=2)
        with pytest.raises(ValueError):
            with wrapper:
                raise ValueError("not an early stop")
        env.step(None)
        assert wrapper.tracker._step_count == 0
        assert runner.saved == []
        assert runner.logger.stopped is False

    def test_cleanup_skipped_when_runner_has_no_writer(self):
        env = _FakeEnv([{"log": {DEFAULT_SUCCESS_TAG: 0.9}}] * 100)
        runner = _FakeRunner(has_writer=False)
        with RslRlEarlyStopWrapper(env, runner, 0.5, 2, num_steps_per_env=2):
            for _ in range(10):
                env.step(None)
        assert runner.saved == []
        assert runner.logger.stopped is False

    def test_framework_iteration_count_reflects_runner(self):
        env = _FakeEnv([{"log": {DEFAULT_SUCCESS_TAG: 0.0}}])
        runner = _FakeRunner()
        runner.current_learning_iteration = 42
        wrapper = RslRlEarlyStopWrapper(env, runner, 0.5, 3, num_steps_per_env=2)
        assert wrapper.framework_iteration_count == 43


class TestRlGamesEarlyStopObserver:
    def test_delegates_every_call_to_base(self):
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
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 3)
        obs.after_init(_FakeAlgo(horizon_length=24))
        assert obs.tracker.num_steps_per_env == 24

    def test_tracker_falls_back_to_config_horizon_length(self):
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 3)
        obs.after_init(_FakeAlgo(horizon_length=None, config_horizon=32))
        assert obs.tracker.num_steps_per_env == 32

    def test_process_infos_records_from_episode_key(self):
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 3)
        obs.after_init(_FakeAlgo(horizon_length=2))
        obs.process_infos({"episode": {DEFAULT_SUCCESS_TAG: 0.8}}, [])
        assert obs.tracker._iter_sum == pytest.approx(0.8)

    def test_after_steps_sets_max_epochs_on_convergence(self):
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 2)
        algo = _FakeAlgo(horizon_length=1)
        obs.after_init(algo)
        obs.process_infos({"episode": {DEFAULT_SUCCESS_TAG: 0.9}}, [])
        obs.after_steps()
        assert algo.max_epochs == 999
        assert obs.tracker.current_iteration == 1
        obs.process_infos({"episode": {DEFAULT_SUCCESS_TAG: 0.9}}, [])
        obs.after_steps()
        assert algo.max_epochs == 2

    def test_after_steps_leaves_max_epochs_alone_when_stop_disabled(self):
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

    def test_framework_iteration_count_reflects_algo_epoch_num(self):
        obs = RlGamesEarlyStopObserver(_FakeBaseObserver(), 0.5, 2)
        assert obs.framework_iteration_count is None
        obs.after_init(_FakeAlgo(horizon_length=1, epoch_num=7))
        assert obs.framework_iteration_count == 7
