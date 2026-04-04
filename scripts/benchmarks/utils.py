# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import glob
import os
import statistics
import sys

from tensorboard.backend.event_processing import event_accumulator

from isaaclab.test.benchmark import BaseIsaacLabBenchmark, DictMeasurement, ListMeasurement, SingleMeasurement

# Path to configs.yaml and the config loader.
_BENCHMARKING_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "source", "isaaclab_tasks", "test", "benchmarking"
)
_CONFIGS_YAML = os.path.join(_BENCHMARKING_DIR, "configs.yaml")


def get_backend_type(cli_backend: str) -> str:
    """Map old CLI backend names to new backend types.

    Args:
        cli_backend: The backend name from CLI arguments.

    Returns:
        The new backend type string.
    """
    mapping = {
        "OmniPerfKPIFile": "omniperf",
        "JSONFileMetrics": "json",
        "OsmoKPIFile": "osmo",
        "LocalLogMetrics": "json",
        "omniperf": "omniperf",
        "json": "json",
        "osmo": "osmo",
        "summary": "summary",
    }
    return mapping.get(cli_backend, "omniperf")


def parse_tf_logs(log_dir: str):
    """Search for the latest tfevents file in log_dir folder and returns
    the tensorboard logs in a dictionary.

    Args:
        log_dir: directory used to search for tfevents files
    """

    # search log directory for latest log file
    list_of_files = glob.glob(f"{log_dir}/events*")  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)

    log_data = {}
    ea = event_accumulator.EventAccumulator(latest_file)
    ea.Reload()
    tags = ea.Tags()["scalars"]
    for tag in tags:
        log_data[tag] = []
        for event in ea.Scalars(tag):
            log_data[tag].append(event.value)

    return log_data


#############################
# logging benchmark metrics #
#############################


def log_min_max_mean_stats(benchmark: BaseIsaacLabBenchmark, values: dict):
    for k, v in values.items():
        unit = "FPS" if "FPS" in k else "ms" if "Time" in k or "time" in k else ""
        measurement = SingleMeasurement(name=f"Min {k}", value=min(v), unit=unit)
        benchmark.add_measurement("runtime", measurement=measurement)
        measurement = SingleMeasurement(name=f"Max {k}", value=max(v), unit=unit)
        benchmark.add_measurement("runtime", measurement=measurement)
        measurement = SingleMeasurement(name=f"Mean {k}", value=sum(v) / len(v), unit=unit)
        benchmark.add_measurement("runtime", measurement=measurement)


def log_app_start_time(benchmark: BaseIsaacLabBenchmark, value: float):
    measurement = SingleMeasurement(name="App Launch Time", value=value, unit="ms")
    benchmark.add_measurement("startup", measurement=measurement)


def log_python_imports_time(benchmark: BaseIsaacLabBenchmark, value: float):
    measurement = SingleMeasurement(name="Python Imports Time", value=value, unit="ms")
    benchmark.add_measurement("startup", measurement=measurement)


def log_task_start_time(benchmark: BaseIsaacLabBenchmark, value: float):
    measurement = SingleMeasurement(name="Task Creation and Start Time", value=value, unit="ms")
    benchmark.add_measurement("startup", measurement=measurement)


def log_scene_creation_time(benchmark: BaseIsaacLabBenchmark, value: float):
    measurement = SingleMeasurement(name="Scene Creation Time", value=value, unit="ms")
    benchmark.add_measurement("startup", measurement=measurement)


def log_simulation_start_time(benchmark: BaseIsaacLabBenchmark, value: float):
    measurement = SingleMeasurement(name="Simulation Start Time", value=value, unit="ms")
    benchmark.add_measurement("startup", measurement=measurement)


def log_total_start_time(benchmark: BaseIsaacLabBenchmark, value: float):
    measurement = SingleMeasurement(name="Total Start Time (Launch to Train)", value=value, unit="ms")
    benchmark.add_measurement("startup", measurement=measurement)


def log_runtime_step_times(benchmark: BaseIsaacLabBenchmark, value: dict, compute_stats=True):
    measurement = DictMeasurement(name="Step Frametimes", value=value)
    benchmark.add_measurement("runtime", measurement=measurement)
    if compute_stats:
        log_min_max_mean_stats(benchmark, value)


def get_preset_string(hydra_args: list[str]) -> str:
    """Extract the active preset string from CLI hydra args or an environment variable.

    Checks (in order):
        1. ``presets=...`` in *hydra_args* (e.g. ``presets=physx,ovrtx_renderer,rgb``)
        2. ``ISAACLAB_BENCHMARK_PRESET`` environment variable
        3. Falls back to ``"default"``
    """
    for arg in hydra_args:
        if arg.startswith("presets="):
            value = arg.split("=", 1)[1]
            return value if value else "default"
    return os.environ.get("ISAACLAB_BENCHMARK_PRESET", "") or "default"


def log_rl_policy_rewards(benchmark: BaseIsaacLabBenchmark, value: list):
    measurement = ListMeasurement(name="Rewards", value=value)
    benchmark.add_measurement("train", measurement=measurement)
    # log max reward
    measurement = SingleMeasurement(name="Max Rewards", value=max(value), unit="float")
    benchmark.add_measurement("train", measurement=measurement)


def log_rl_policy_episode_lengths(benchmark: BaseIsaacLabBenchmark, value: list):
    measurement = ListMeasurement(name="Episode Lengths", value=value)
    benchmark.add_measurement("train", measurement=measurement)
    # log max episode length
    measurement = SingleMeasurement(name="Max Episode Lengths", value=max(value), unit="float")
    benchmark.add_measurement("train", measurement=measurement)


def check_convergence(
    rewards: list[float],
    threshold: float,
    window_pct: float = 0.2,
    cv_threshold: float = 20.0,
) -> dict:
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


def log_convergence(
    benchmark: BaseIsaacLabBenchmark,
    rewards: list[float],
    task: str,
    workflow: str = "",
    should_check_convergence: bool = False,
    reward_threshold: float | None = None,
    convergence_config: str = "full",
):
    """Check reward convergence and log results to the benchmark backend.

    No-op unless *check_convergence* is True. When enabled, the threshold
    is loaded from ``configs.yaml``. *reward_threshold* overrides the config.

    Args:
        benchmark: Benchmark instance to log measurements to.
        rewards: Per-iteration mean reward values.
        task: Task name for config lookup.
        workflow: RL workflow name (``rsl_rl``, ``rl_games``, etc.).
        should_check_convergence: Whether ``--check_convergence`` was passed.
        reward_threshold: Explicit threshold override.
        convergence_config: Config section for threshold lookup (default: ``full``).
    """
    if not should_check_convergence:
        return

    threshold = reward_threshold
    if threshold is None and os.path.exists(_CONFIGS_YAML):
        if _BENCHMARKING_DIR not in sys.path:
            sys.path.insert(0, _BENCHMARKING_DIR)
        try:
            from env_benchmark_test_utils import get_env_config, get_env_configs

            entry = get_env_config(get_env_configs(_CONFIGS_YAML), convergence_config, workflow, task)
        except (ImportError, ValueError):
            entry = None
        if entry:
            threshold = entry.get("lower_thresholds", {}).get("reward")

    if threshold is None:
        print(
            f"[WARNING] No reward threshold found for '{task}'"
            f" in configs.yaml [{convergence_config}]. Skipping convergence check."
        )
        return

    result = check_convergence(rewards, threshold)
    benchmark.add_measurement(
        "train", SingleMeasurement(name="Mean Reward (Converged)", value=result["tail_mean"], unit="float")
    )
    benchmark.add_measurement("train", SingleMeasurement(name="Reward CV %", value=result["cv"], unit="%"))
    benchmark.add_measurement(
        "train", SingleMeasurement(name="Convergence Passed", value=int(result["passed"]), unit="bool")
    )
