# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

import psutil

from isaaclab.test.benchmark.interfaces import MeasurementDataRecorder, MeasurementData
from isaaclab.test.benchmark.measurements import FloatMetadata, SingleMeasurement


class MemoryInfoRecorder(MeasurementDataRecorder):
    def __init__(self):
        # Empty dictionaries to store hardware and runtime information
        self._memory_hardware_info = {}
        self._memory_runtime_info = {}
        # Welford's algorithm for process RSS
        self._proc_mean = 0
        self._proc_std = 0
        self._proc_n = 0
        self._proc_m2 = 0
        # Welford's algorithm for system available memory
        self._sys_mean = 0
        self._sys_std = 0
        self._sys_n = 0
        self._sys_m2 = 0
        # Process handle
        self._process = psutil.Process(os.getpid())
        self._get_hardware_info()

    def _get_hardware_info(self) -> None:
        mem = psutil.virtual_memory()
        self._memory_hardware_info["total_ram_gb"] = round(mem.total / (1024**3), 2)

    def _get_runtime_info(self) -> None:
        # Process RSS memory
        process_rss = self._process.memory_info().rss
        self._proc_n += 1
        delta = process_rss - self._proc_mean
        self._proc_mean += delta / self._proc_n
        delta2 = process_rss - self._proc_mean
        self._proc_m2 += delta * delta2
        if self._proc_n > 1:
            self._proc_std = math.sqrt(self._proc_m2 / (self._proc_n - 1))

        self._memory_runtime_info["process_rss_mean_bytes"] = self._proc_mean
        self._memory_runtime_info["process_rss_std_bytes"] = self._proc_std
        self._memory_runtime_info["process_n"] = self._proc_n

        # System available memory
        sys_available = psutil.virtual_memory().available
        self._sys_n += 1
        delta = sys_available - self._sys_mean
        self._sys_mean += delta / self._sys_n
        delta2 = sys_available - self._sys_mean
        self._sys_m2 += delta * delta2
        if self._sys_n > 1:
            self._sys_std = math.sqrt(self._sys_m2 / (self._sys_n - 1))

        self._memory_runtime_info["system_available_mean_bytes"] = self._sys_mean
        self._memory_runtime_info["system_available_std_bytes"] = self._sys_std
        self._memory_runtime_info["system_n"] = self._sys_n

    def update(self) -> None:
        self._get_runtime_info()

    def get_initial_data(self) -> dict:
        return {
            "memory_metadata": self._memory_hardware_info,
        }

    def get_runtime_data(self) -> dict:
        return {
            "memory_utilization": self._memory_runtime_info,
        }

    def get_data(self) -> MeasurementData:
        return MeasurementData(
            measurements=[
                SingleMeasurement(
                    name="process_rss", value=self._memory_runtime_info.get("process_rss_mean_bytes", 0), unit="bytes"
                ),
                SingleMeasurement(
                    name="process_rss_std", value=self._memory_runtime_info.get("process_rss_std_bytes", 0), unit="bytes"
                ),
                SingleMeasurement(name="process_rss_n", value=self._memory_runtime_info.get("process_n", 0), unit=""),
                SingleMeasurement(
                    name="system_available", value=self._memory_runtime_info.get("system_available_mean_bytes", 0), unit="bytes"
                ),
                SingleMeasurement(
                    name="system_available_std", value=self._memory_runtime_info.get("system_available_std_bytes", 0), unit="bytes"
                ),
                SingleMeasurement(name="system_available_n", value=self._memory_runtime_info.get("system_n", 0), unit=""),
            ],
            metadata=[
                FloatMetadata(name="total_ram_gb", data=self._memory_hardware_info["total_ram_gb"]),
            ],
        )
