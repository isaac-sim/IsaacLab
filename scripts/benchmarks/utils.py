# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import glob
import importlib
import os
import subprocess

from isaacsim.benchmark.services import BaseIsaacBenchmark
from isaacsim.benchmark.services.metrics.measurements import DictMeasurement, ListMeasurement, SingleMeasurement
from tensorboard.backend.event_processing import event_accumulator


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


def log_min_max_mean_stats(benchmark: BaseIsaacBenchmark, values: dict):
    for k, v in values.items():
        measurement = SingleMeasurement(name=f"Min {k}", value=min(v), unit="ms")
        benchmark.store_custom_measurement("runtime", measurement)
        measurement = SingleMeasurement(name=f"Max {k}", value=max(v), unit="ms")
        benchmark.store_custom_measurement("runtime", measurement)
        measurement = SingleMeasurement(name=f"Mean {k}", value=sum(v) / len(v), unit="ms")
        benchmark.store_custom_measurement("runtime", measurement)


def log_app_start_time(benchmark: BaseIsaacBenchmark, value: float):
    measurement = SingleMeasurement(name="App Launch Time", value=value, unit="ms")
    benchmark.store_custom_measurement("startup", measurement)


def log_python_imports_time(benchmark: BaseIsaacBenchmark, value: float):
    measurement = SingleMeasurement(name="Python Imports Time", value=value, unit="ms")
    benchmark.store_custom_measurement("startup", measurement)


def log_task_start_time(benchmark: BaseIsaacBenchmark, value: float):
    measurement = SingleMeasurement(name="Task Creation and Start Time", value=value, unit="ms")
    benchmark.store_custom_measurement("startup", measurement)


def log_scene_creation_time(benchmark: BaseIsaacBenchmark, value: float):
    measurement = SingleMeasurement(name="Scene Creation Time", value=value, unit="ms")
    benchmark.store_custom_measurement("startup", measurement)


def log_simulation_start_time(benchmark: BaseIsaacBenchmark, value: float):
    measurement = SingleMeasurement(name="Simulation Start Time", value=value, unit="ms")
    benchmark.store_custom_measurement("startup", measurement)


def log_total_start_time(benchmark: BaseIsaacBenchmark, value: float):
    measurement = SingleMeasurement(name="Total Start Time (Launch to Train)", value=value, unit="ms")
    benchmark.store_custom_measurement("startup", measurement)


def log_runtime_step_times(benchmark: BaseIsaacBenchmark, value: dict, compute_stats=True):
    measurement = DictMeasurement(name="Step Frametimes", value=value)
    benchmark.store_custom_measurement("runtime", measurement)
    if compute_stats:
        log_min_max_mean_stats(benchmark, value)


def ema(value: list, alpha: float):
    """Compute the exponential moving average of a list of values."""
    ema_value = value[0]
    for i in range(1, len(value)):
        ema_value = alpha * value[i] + (1 - alpha) * ema_value
    return ema_value


def log_rl_policy_rewards(benchmark: BaseIsaacBenchmark, value: list):
    measurement = ListMeasurement(name="Rewards", value=value)
    benchmark.store_custom_measurement("train", measurement)
    # log max reward
    measurement = SingleMeasurement(name="Max Rewards", value=max(value), unit="float")
    benchmark.store_custom_measurement("train", measurement)
    # log last reward
    measurement = SingleMeasurement(name="Last Reward", value=value[-1], unit="float")
    benchmark.store_custom_measurement("train", measurement)
    # log EMA 0.95 reward
    measurement = SingleMeasurement(name="EMA 0.95 Reward", value=ema(value, 0.95), unit="float")
    benchmark.store_custom_measurement("train", measurement)


def log_rl_policy_episode_lengths(benchmark: BaseIsaacBenchmark, value: list):
    measurement = ListMeasurement(name="Episode Lengths", value=value)
    benchmark.store_custom_measurement("train", measurement)
    # log max episode length
    measurement = SingleMeasurement(name="Max Episode Lengths", value=max(value), unit="float")
    benchmark.store_custom_measurement("train", measurement)
    # log last episode length
    measurement = SingleMeasurement(name="Last Episode Length", value=value[-1], unit="float")
    benchmark.store_custom_measurement("train", measurement)
    # log EMA 0.95 episode length
    measurement = SingleMeasurement(name="EMA 0.95 Episode Length", value=ema(value, 0.95), unit="float")
    benchmark.store_custom_measurement("train", measurement)


def get_newton_version() -> dict[str, str | None]:
    """Get Newton version."""
    try:
        import newton

        version = newton.__version__
        commit = get_git_commit_from_module("newton")
        branch = get_git_branch_from_module("newton")
        return {"version": version, "commit": commit, "branch": branch}
    except Exception as e:
        print(f"[ERROR] Error getting Newton version: {e}")
        return {"version": None, "commit": None, "branch": None}


def get_isaaclab_version() -> dict[str, str | None]:
    """Get Isaac Lab version."""
    try:
        import isaaclab

        version = isaaclab.__version__
        commit = get_git_commit_from_module("isaaclab")
        branch = get_git_branch_from_module("isaaclab")
        return {"version": version, "commit": commit, "branch": branch}
    except Exception as e:
        print(f"[ERROR] Error getting Isaac Lab version: {e}")
        return {"version": None, "commit": None, "branch": None}


def get_mujoco_warp_version() -> dict[str, str | None]:
    """Get Mujoco Warp version."""
    try:
        import mujoco_warp

        try:
            version = mujoco_warp.__version__
        except Exception:
            version = None
        commit = get_git_commit_from_module("mujoco_warp")
        branch = get_git_branch_from_module("mujoco_warp")
        return {"version": version, "commit": commit, "branch": branch}
    except Exception as e:
        print(f"[ERROR] Error getting Mujoco Warp version: {e}")
        return {"version": None, "commit": None, "branch": None}


def get_git_commit_from_module(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    module_path = spec.origin
    repo_path = os.path.abspath(os.path.join(module_path, ".."))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path).decode().strip()
        return commit
    except Exception:
        return None


def get_git_branch_from_module(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    module_path = spec.origin
    repo_path = os.path.abspath(os.path.join(module_path, ".."))
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_path).decode().strip()
        return branch
    except Exception:
        return None
