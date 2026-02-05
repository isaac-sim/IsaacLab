# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import math
import os
import platform

import psutil

from isaaclab.test.benchmark.interfaces import MeasurementData, MeasurementDataRecorder
from isaaclab.test.benchmark.measurements import IntMetadata, SingleMeasurement, StringMetadata


class CPUInfoRecorder(MeasurementDataRecorder):
    def __init__(self):
        # Empty dictionaries to store hardware and runtime information
        self._cpu_hardware_info = {}
        self._cpu_runtime_info = {}
        # Welford's algorithm for computing the mean and standard deviation
        self._mean = 0
        self._std = 0
        self._n = 0
        self._m2 = 0
        # CPU usage
        self._process = psutil.Process(os.getpid())
        self._get_hardware_info()

    def _get_hardware_info(self) -> None:
        # CPU info
        self._cpu_hardware_info["physical_cores"] = os.cpu_count()
        self._cpu_hardware_info["name"] = platform.processor() or "Unknown"
        with contextlib.suppress(Exception):
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
                for line in cpuinfo.split("\n"):
                    if "model name" in line:
                        self._cpu_hardware_info["name"] = line.split(":")[1].strip()
                        break

    def _get_runtime_info(self) -> None:
        process_cpu_percent = self._process.cpu_percent(interval=None)
        # Welford's algorithm for computing the mean and standard deviation
        self._n += 1
        delta = process_cpu_percent - self._mean
        self._mean += delta / self._n
        delta2 = process_cpu_percent - self._mean
        self._m2 += delta * delta2
        if self._n > 1:
            self._std = math.sqrt(self._m2 / (self._n - 1))
        self._cpu_runtime_info["mean"] = self._mean
        self._cpu_runtime_info["std"] = self._std
        self._cpu_runtime_info["n"] = self._n

    def update(self) -> None:
        self._get_runtime_info()

    def get_initial_data(self) -> dict:
        return {
            "cpu_metadata": self._cpu_hardware_info,
        }

    def get_runtime_data(self) -> dict:
        return {
            "cpu_utilization": self._cpu_runtime_info,
        }

    def get_data(self) -> MeasurementData:
        return MeasurementData(
            measurements=[
                SingleMeasurement(name="CPU Utilization", value=self._cpu_runtime_info["mean"], unit="%"),
                SingleMeasurement(name="CPU Utilization std", value=self._cpu_runtime_info["std"], unit="%"),
                SingleMeasurement(name="CPU Utilization n", value=self._cpu_runtime_info["n"], unit=""),
            ],
            metadata=[
                StringMetadata(name="cpu_name", data=self._cpu_hardware_info["name"]),
                IntMetadata(name="physical_cores", data=self._cpu_hardware_info["physical_cores"]),
            ],
        )
