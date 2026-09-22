# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the benchmark bundle builders."""

import statistics

import pytest

from isaaclab.benchmark import builders
from isaaclab.benchmark.schema import PlayBundle, RuntimeBundle, StartupTime, TrainingBundle

pytestmark = pytest.mark.benchmark

_STARTUP = StartupTime(app_launch=1.0, env_creation=2.0, first_step=0.5)


def test_run_config_and_identity():
    assert builders.build_run_config("physx").presets == []
    assert builders.build_run_config("newton_mjwarp", presets=["rgb"]).presets == ["rgb"]

    run = builders.build_run_identity(
        run_id="x",
        framework="rsl_rl",
        config=builders.build_run_config("newton_mjwarp"),
        task="t",
        seed=0,
        start_utc="2026-04-22T13:15:00+00:00",
        end_utc="2026-04-22T13:15:30+00:00",
        num_envs=4096,
        max_iterations=500,
    )
    assert run.duration_s == pytest.approx(30.0)
    assert run.status == "completed"


def test_build_runtime_aggregates():
    rt = builders.build_runtime(
        startup_time_s=_STARTUP,
        iteration_times_s=[1.0, 1.0, 2.0],
        collection_fps=[100.0, 110.0],
        total_fps=[90.0, 95.0],
        steps_per_iteration=24,
    )
    assert rt.iterations_completed == 3
    assert rt.total_wall_time_s == pytest.approx(4.0)
    assert rt.total_fps.peak == pytest.approx(95.0)
    assert rt.iterations_per_s.mean == pytest.approx(statistics.mean([1.0, 1.0, 0.5]))
    assert rt.environment_step_timing is None


def test_build_runtime_effective_aggregate_throughput():
    """Aggregate throughput divides total work by total wall time while std stays the sample deviation."""
    fps = [8.0, 8.0 / 3.0]
    rt = builders.build_runtime(
        startup_time_s=_STARTUP,
        iteration_times_s=[1.0, 3.0],
        collection_fps=fps,
        total_fps=fps,
        steps_per_iteration=8,
        aggregate_throughput=True,
    )
    assert rt.total_fps.mean == pytest.approx(4.0)
    assert rt.collection_fps.mean == pytest.approx(4.0)
    assert rt.total_fps.std == pytest.approx(statistics.stdev(fps))
    assert rt.iterations_per_s.mean == pytest.approx(0.5)


def test_build_runtime_environment_step_timing():
    serialized = builders.build_runtime(
        startup_time_s=_STARTUP,
        iteration_times_s=[2.0],
        collection_fps=[4.0],
        total_fps=[4.0],
        steps_per_iteration=8,
        frames_per_environment_step=8,
        environment_step_warmup_steps=3,
        environment_step_times_s=[1.0, 2.0],
        simulation_step_times_s=[0.5, 0.5],
        simulation_step_calls=8,
    ).environment_step_timing
    assert serialized.measurement_mode == "serialized_synchronized"
    assert serialized.environment_step_time_s.mean == pytest.approx(1.5)
    assert serialized.environment_step_time_s.std == pytest.approx(statistics.stdev([1.0, 2.0]))
    # Effective rate: 2 steps of 8 frames over 3 seconds; std is the sample deviation of per-step rates.
    assert serialized.environment_step_fps.mean == pytest.approx(16.0 / 3.0)
    assert serialized.environment_step_fps.std == pytest.approx(statistics.stdev([8.0, 4.0]))
    assert serialized.simulation_step_time_s.mean == pytest.approx(0.5)
    assert serialized.outside_simulation_step_time_s.mean == pytest.approx(1.0)
    assert serialized.outside_simulation_step_fraction == pytest.approx(2.0 / 3.0)
    assert (serialized.environment_step_calls, serialized.simulation_step_calls, serialized.warmup_steps) == (2, 8, 3)

    host_return = builders.build_runtime(
        startup_time_s=_STARTUP,
        iteration_times_s=[2.0],
        collection_fps=[4.0],
        total_fps=[4.0],
        steps_per_iteration=8,
        frames_per_environment_step=8,
        environment_step_times_s=[1.0, 2.0],
    ).environment_step_timing
    assert host_return.measurement_mode == "host_return"
    assert host_return.environment_step_fps.mean == pytest.approx(16.0 / 3.0)
    assert host_return.simulation_step_time_s is None
    assert host_return.simulation_step_calls is None


@pytest.mark.parametrize(
    ("timing", "message"),
    [
        (
            {"environment_step_times_s": [1.0, 2.0], "simulation_step_times_s": [1.1, 1.5], "simulation_step_calls": 2},
            "simulation time cannot exceed",
        ),
        (
            {"environment_step_times_s": [1.0, 2.0], "simulation_step_times_s": [], "simulation_step_calls": 2},
            "simulation_step_times_s must contain only positive",
        ),
        (
            {"environment_step_times_s": [1.0, 2.0], "simulation_step_times_s": [0.0, 0.5], "simulation_step_calls": 2},
            "simulation_step_times_s must contain only positive",
        ),
        (
            {"environment_step_times_s": [1.0], "simulation_step_times_s": [0.5], "simulation_step_calls": 0},
            "simulation_step_calls must be greater than zero",
        ),
        ({"environment_step_times_s": []}, "No environment-step timing samples remained"),
        ({"environment_step_times_s": [0.0]}, "environment_step_times_s must contain only positive"),
        ({"environment_step_times_s": [-1.0]}, "environment_step_times_s must contain only positive"),
        ({"simulation_step_times_s": [0.5]}, "environment_step_times_s is required with simulation timing"),
        ({"simulation_step_calls": 1}, "environment_step_times_s is required with simulation timing"),
        (
            {"environment_step_times_s": [1.0], "frames_per_environment_step": 0},
            "frames_per_environment_step must be greater than zero",
        ),
    ],
)
def test_build_runtime_rejects_inconsistent_step_timing(timing: dict, message: str):
    kwargs = {"frames_per_environment_step": 8, **timing}
    with pytest.raises(ValueError, match=message):
        builders.build_runtime(
            startup_time_s=_STARTUP,
            iteration_times_s=[1.0],
            collection_fps=[8.0],
            total_fps=[8.0],
            steps_per_iteration=8,
            **kwargs,
        )


def test_build_learning():
    learning = builders.build_learning(
        reward_series=[1.0, 2.0, 3.0],
        ep_length_series=[10.0, 20.0],
        success_rate_series=[0.1, 0.5, 0.9],
        ema_alpha=0.5,
    )
    assert learning.reward.final_raw == pytest.approx(3.0)
    assert learning.reward.series_per_iter == [1.0, 2.0, 3.0]
    assert learning.success_rate.final_raw == pytest.approx(0.9)
    assert learning.success_rate.final_ema == pytest.approx(0.6)

    empty = builders.build_learning(reward_series=[], ep_length_series=[], ema_alpha=0.1)
    assert (empty.reward.final_raw, empty.reward.final_ema, empty.reward.series_per_iter) == (0.0, 0.0, [])
    assert empty.success_rate is None
    assert (
        builders.build_learning(
            reward_series=[1.0], ep_length_series=[1.0], success_rate_series=[], ema_alpha=0.1
        ).success_rate
        is None
    )

    compact = builders.build_learning(
        reward_series=[1.0, 2.0],
        ep_length_series=[10.0],
        success_rate_series=[0.25, 0.75],
        ema_alpha=0.1,
        keep_series=False,
    )
    assert compact.reward.series_per_iter is None
    assert compact.success_rate.final_raw == pytest.approx(0.75)
    assert compact.success_rate.series_per_iter is None


def test_bundle_builders_forward_fields(run_identity, versions, hardware, runtime, resources):
    common = {
        "run": run_identity,
        "versions": versions,
        "hardware": hardware,
        "runtime": runtime,
        "resources": resources,
    }
    learning = builders.build_learning(reward_series=[1.0], ep_length_series=[1.0], ema_alpha=0.1)

    training = builders.build_training_bundle(**common, learning=learning, success_rate=0.9, checkpoint_path="m.pt")
    assert isinstance(training, TrainingBundle)
    assert (training.learning, training.success_rate, training.checkpoint_path) == (learning, 0.9, "m.pt")

    play = builders.build_play_bundle(
        **common, reward=resources.ram_gb, ep_length=resources.gpu_mem_gb, checkpoint_path="m.pt"
    )
    assert isinstance(play, PlayBundle)
    assert (play.reward, play.ep_length, play.success_rate) == (resources.ram_gb, resources.gpu_mem_gb, None)

    assert isinstance(builders.build_runtime_bundle(**common), RuntimeBundle)
    startup = builders.build_startup_bundle(
        run=run_identity, versions=versions, hardware=hardware, phases={}, top_n=5, whitelist=None, extra={"a": 1}
    )
    assert (startup.config.top_n, startup.config.whitelist, startup.extra) == (5, None, {"a": 1})
