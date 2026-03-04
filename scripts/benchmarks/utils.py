# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import glob
import os

from tensorboard.backend.event_processing import event_accumulator

from isaaclab.test.benchmark import BaseIsaacLabBenchmark, DictMeasurement, ListMeasurement, SingleMeasurement


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
        measurement = SingleMeasurement(name=f"Min {k}", value=min(v), unit="ms")
        benchmark.add_measurement("runtime", measurement=measurement)
        measurement = SingleMeasurement(name=f"Max {k}", value=max(v), unit="ms")
        benchmark.add_measurement("runtime", measurement=measurement)
        measurement = SingleMeasurement(name=f"Mean {k}", value=sum(v) / len(v), unit="ms")
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


# Preset names that indicate kitless physics (no Kit/AppLauncher required).
# The renderer must also be kitless for the full pipeline to skip Kit.
KITLESS_PHYSICS_PRESETS = {"newton"}
KITLESS_RENDERER_PRESETS = {"newton_renderer"}
KIT_RENDERER_PRESETS = {"ovrtx_renderer"}


def needs_kit(hydra_args: list[str]) -> bool:
    """Return True if the active presets require Kit (AppLauncher).

    Kit is skipped only when BOTH the physics backend AND renderer (if
    specified) are kitless.  When no renderer preset is given the default
    renderer is assumed, which requires Kit.
    """
    active = set(get_preset_string(hydra_args).split(","))
    has_kitless_physics = bool(active & KITLESS_PHYSICS_PRESETS)
    has_kit_renderer = bool(active & KIT_RENDERER_PRESETS)
    if not has_kitless_physics:
        return True
    if has_kit_renderer:
        return True
    return False


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
