# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common utility functions for benchmark scripts running in standalone or kit mode."""

import os

from isaaclab.utils.timer import Timer


def create_kit_logging_functions():
    """Create logging functions for kit mode that use isaacsim.benchmark.services.

    Returns:
        A dictionary containing all the logging function implementations for kit mode.
    """
    from scripts.benchmarks.utils.utils import (
        get_isaaclab_version,
        get_mujoco_warp_version,
        get_newton_version,
        log_app_start_time,
        log_newton_finalize_builder_time,
        log_newton_initialize_solver_time,
        log_python_imports_time,
        log_rl_policy_episode_lengths,
        log_rl_policy_rewards,
        log_runtime_step_times,
        log_scene_creation_time,
        log_simulation_start_time,
        log_task_start_time,
        log_total_start_time,
        parse_tf_logs,
    )

    return {
        "get_isaaclab_version": get_isaaclab_version,
        "get_mujoco_warp_version": get_mujoco_warp_version,
        "get_newton_version": get_newton_version,
        "log_app_start_time": log_app_start_time,
        "log_python_imports_time": log_python_imports_time,
        "log_task_start_time": log_task_start_time,
        "log_scene_creation_time": log_scene_creation_time,
        "log_simulation_start_time": log_simulation_start_time,
        "log_newton_finalize_builder_time": log_newton_finalize_builder_time,
        "log_newton_initialize_solver_time": log_newton_initialize_solver_time,
        "log_total_start_time": log_total_start_time,
        "log_runtime_step_times": log_runtime_step_times,
        "log_rl_policy_rewards": log_rl_policy_rewards,
        "log_rl_policy_episode_lengths": log_rl_policy_episode_lengths,
        "parse_tf_logs": parse_tf_logs,
    }


def create_standalone_logging_functions():  # noqa: C901
    """Create logging functions for standalone mode that use standalone_benchmark.

    Returns:
        A dictionary containing all the logging function implementations for standalone mode.
    """
    from scripts.benchmarks.utils.standalone_benchmark import DictMeasurement, ListMeasurement, SingleMeasurement

    def get_isaaclab_version():
        try:
            import isaaclab

            return {"version": isaaclab.__version__, "commit": None, "branch": None}
        except Exception:
            return {"version": None, "commit": None, "branch": None}

    def get_mujoco_warp_version():
        try:
            import mujoco_warp

            return {"version": getattr(mujoco_warp, "__version__", None), "commit": None, "branch": None}
        except Exception:
            return {"version": None, "commit": None, "branch": None}

    def get_newton_version():
        try:
            import newton

            return {"version": newton.__version__, "commit": None, "branch": None}
        except Exception:
            return {"version": None, "commit": None, "branch": None}

    def log_app_start_time(benchmark, value):
        measurement = SingleMeasurement(name="App Launch Time", value=value, unit="ms")
        benchmark.store_custom_measurement("startup", measurement)

    def log_python_imports_time(benchmark, value):
        measurement = SingleMeasurement(name="Python Imports Time", value=value, unit="ms")
        benchmark.store_custom_measurement("startup", measurement)

    def log_task_start_time(benchmark, value):
        measurement = SingleMeasurement(name="Task Creation and Start Time", value=value, unit="ms")
        benchmark.store_custom_measurement("startup", measurement)

    def log_scene_creation_time(benchmark, value):
        if value is not None:
            measurement = SingleMeasurement(name="Scene Creation Time", value=value, unit="ms")
            benchmark.store_custom_measurement("startup", measurement)

    def log_simulation_start_time(benchmark, value):
        if value is not None:
            measurement = SingleMeasurement(name="Simulation Start Time", value=value, unit="ms")
            benchmark.store_custom_measurement("startup", measurement)

    def log_newton_finalize_builder_time(benchmark, value):
        if value is not None:
            measurement = SingleMeasurement(name="Newton Finalize Builder Time", value=value, unit="ms")
            benchmark.store_custom_measurement("startup", measurement)

    def log_newton_initialize_solver_time(benchmark, value):
        if value is not None:
            measurement = SingleMeasurement(name="Newton Initialize Solver Time", value=value, unit="ms")
            benchmark.store_custom_measurement("startup", measurement)

    def log_total_start_time(benchmark, value):
        measurement = SingleMeasurement(name="Total Start Time (Launch to Train)", value=value, unit="ms")
        benchmark.store_custom_measurement("startup", measurement)

    def log_runtime_step_times(benchmark, value, compute_stats=True):
        measurement = DictMeasurement(name="Step Frametimes", value=value)
        benchmark.store_custom_measurement("runtime", measurement)
        if compute_stats:
            for k, v in value.items():
                if isinstance(v, list) and len(v) > 0:
                    measurement = SingleMeasurement(name=f"Min {k}", value=min(v), unit="ms")
                    benchmark.store_custom_measurement("runtime", measurement)
                    measurement = SingleMeasurement(name=f"Max {k}", value=max(v), unit="ms")
                    benchmark.store_custom_measurement("runtime", measurement)
                    measurement = SingleMeasurement(name=f"Mean {k}", value=sum(v) / len(v), unit="ms")
                    benchmark.store_custom_measurement("runtime", measurement)

    def log_rl_policy_rewards(benchmark, value):
        measurement = ListMeasurement(name="Rewards", value=value)
        benchmark.store_custom_measurement("train", measurement)
        if len(value) > 0:
            measurement = SingleMeasurement(name="Max Rewards", value=max(value), unit="float")
            benchmark.store_custom_measurement("train", measurement)
            measurement = SingleMeasurement(name="Last Reward", value=value[-1], unit="float")
            benchmark.store_custom_measurement("train", measurement)

    def log_rl_policy_episode_lengths(benchmark, value):
        measurement = ListMeasurement(name="Episode Lengths", value=value)
        benchmark.store_custom_measurement("train", measurement)
        if len(value) > 0:
            measurement = SingleMeasurement(name="Max Episode Lengths", value=max(value), unit="float")
            benchmark.store_custom_measurement("train", measurement)
            measurement = SingleMeasurement(name="Last Episode Length", value=value[-1], unit="float")
            benchmark.store_custom_measurement("train", measurement)

    def parse_tf_logs(log_dir: str):
        """Parse tensorboard logs."""
        import glob

        from tensorboard.backend.event_processing import event_accumulator

        list_of_files = glob.glob(f"{log_dir}/events*")
        if not list_of_files:
            return {}
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

    return {
        "get_isaaclab_version": get_isaaclab_version,
        "get_mujoco_warp_version": get_mujoco_warp_version,
        "get_newton_version": get_newton_version,
        "log_app_start_time": log_app_start_time,
        "log_python_imports_time": log_python_imports_time,
        "log_task_start_time": log_task_start_time,
        "log_scene_creation_time": log_scene_creation_time,
        "log_simulation_start_time": log_simulation_start_time,
        "log_newton_finalize_builder_time": log_newton_finalize_builder_time,
        "log_newton_initialize_solver_time": log_newton_initialize_solver_time,
        "log_total_start_time": log_total_start_time,
        "log_runtime_step_times": log_runtime_step_times,
        "log_rl_policy_rewards": log_rl_policy_rewards,
        "log_rl_policy_episode_lengths": log_rl_policy_episode_lengths,
        "parse_tf_logs": parse_tf_logs,
    }


def get_timer_value(timer_name: str) -> float:
    """Safely get timer value, returning 0 if not available.

    Args:
        timer_name: Name of the timer to retrieve.

    Returns:
        Timer value in seconds, or 0 if not available.
    """
    value = Timer.get_timer_info(timer_name)
    return value if value is not None else 0
