# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import contextlib
import glob
import os
import time as _time

import numpy as _np
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


########################################
# step-time recording via monkey-patch #
########################################


class StepTimeRecorder:
    """Records physics-sim and scene-update step times via instance-level monkey-patching.

    Patches ``sim.step``, ``sim.render``, and ``scene.update`` at the instance level so
    class methods are not modified.  Call :meth:`uninstall` to restore the originals.
    """

    def __init__(self):
        self._sim_step_times_ns: list[int] = []
        self._sim_render_times_ns: list[int] = []
        self._scene_update_times_ns: list[int] = []
        self._sim = None
        self._scene = None

    def install(self, sim, scene) -> None:
        """Install timing wrappers on *sim* and *scene* instances."""
        self._sim = sim
        self._scene = scene

        # -- sim.step --
        orig_sim_step = sim.step
        _buf = self._sim_step_times_ns

        def _timed_sim_step(render: bool = True) -> None:
            t0 = _time.perf_counter_ns()
            orig_sim_step(render=render)
            _buf.append(_time.perf_counter_ns() - t0)

        sim.step = _timed_sim_step

        # -- sim.render (optional) --
        if callable(getattr(sim, "render", None)):
            orig_sim_render = sim.render
            _rbuf = self._sim_render_times_ns

            def _timed_sim_render() -> None:
                t0 = _time.perf_counter_ns()
                orig_sim_render()
                _rbuf.append(_time.perf_counter_ns() - t0)

            sim.render = _timed_sim_render

        # -- scene.update --
        orig_scene_update = scene.update
        _sbuf = self._scene_update_times_ns

        def _timed_scene_update(dt: float) -> None:
            t0 = _time.perf_counter_ns()
            orig_scene_update(dt=dt)
            _sbuf.append(_time.perf_counter_ns() - t0)

        scene.update = _timed_scene_update

    def uninstall(self) -> None:
        """Remove instance-level patches, restoring the original class methods."""
        if self._sim is not None:
            for attr in ("step", "render"):
                with contextlib.suppress(AttributeError):
                    delattr(self._sim, attr)
        if self._scene is not None:
            with contextlib.suppress(AttributeError):
                delattr(self._scene, "update")

    def get_step_times_ms(self) -> dict[str, list[float]]:
        """Return collected step times in milliseconds keyed by measurement name."""
        result: dict[str, list[float]] = {}
        if self._sim_step_times_ns:
            result["Physics Step Time"] = (_np.array(self._sim_step_times_ns) / 1e6).tolist()
        if self._sim_render_times_ns:
            result["Render Step Time"] = (_np.array(self._sim_render_times_ns) / 1e6).tolist()
        if self._scene_update_times_ns:
            result["Scene Update Time"] = (_np.array(self._scene_update_times_ns) / 1e6).tolist()
        return result


def log_step_time_breakdown(benchmark: BaseIsaacLabBenchmark, recorder: StepTimeRecorder) -> None:
    """Log per-step physics, render, and scene-update times from *recorder* into *benchmark*."""
    step_times = recorder.get_step_times_ms()
    if step_times:
        measurement = DictMeasurement(name="Step Time Breakdown", value=step_times)
        benchmark.add_measurement("runtime", measurement=measurement)
        log_min_max_mean_stats(benchmark, step_times)
