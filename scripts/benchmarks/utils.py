# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import glob
import os

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


def log_rl_policy_rewards(benchmark: BaseIsaacBenchmark, value: list):
    measurement = ListMeasurement(name="Rewards", value=value)
    benchmark.store_custom_measurement("train", measurement)
    # log max reward
    measurement = SingleMeasurement(name="Max Rewards", value=max(value), unit="float")
    benchmark.store_custom_measurement("train", measurement)


def log_rl_policy_episode_lengths(benchmark: BaseIsaacBenchmark, value: list):
    measurement = ListMeasurement(name="Episode Lengths", value=value)
    benchmark.store_custom_measurement("train", measurement)
    # log max episode length
    measurement = SingleMeasurement(name="Max Episode Lengths", value=max(value), unit="float")
    benchmark.store_custom_measurement("train", measurement)
