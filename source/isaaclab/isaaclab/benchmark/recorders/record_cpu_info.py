# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import os
import platform

import psutil

from ..interfaces import MeasurementData, MeasurementDataRecorder
from ..measurements import IntMetadata, SingleMeasurement, StringMetadata
from ._stats import RunningStats


class CPUInfoRecorder(MeasurementDataRecorder):
    """Record the host CPU model and this process's CPU utilization."""

    def __init__(self):
        self._cpu_hardware_info = {"physical_cores": os.cpu_count(), "name": platform.processor() or "Unknown"}
        with contextlib.suppress(Exception), open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    self._cpu_hardware_info["name"] = line.split(":")[1].strip()
                    break
        self._utilization = RunningStats()
        self._process = psutil.Process(os.getpid())

    def update(self) -> None:
        self._utilization.update(self._process.cpu_percent(interval=None))

    def get_initial_data(self) -> dict:
        return {"cpu_metadata": self._cpu_hardware_info}

    def get_runtime_data(self) -> dict:
        if self._utilization.n == 0:
            return {"cpu_utilization": {}}
        stats = self._utilization
        return {"cpu_utilization": {"mean": stats.mean, "std": stats.std, "n": stats.n}}

    def get_data(self) -> MeasurementData:
        stats = self._utilization
        return MeasurementData(
            measurements=[
                SingleMeasurement(name="CPU Utilization", value=stats.mean, unit="%"),
                SingleMeasurement(name="CPU Utilization std", value=stats.std, unit="%"),
                SingleMeasurement(name="CPU Utilization n", value=stats.n, unit=""),
            ],
            metadata=[
                StringMetadata(name="cpu_name", data=self._cpu_hardware_info["name"]),
                IntMetadata(name="physical_cores", data=self._cpu_hardware_info["physical_cores"]),
            ],
        )
