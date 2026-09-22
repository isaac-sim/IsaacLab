# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import os

import psutil

from ..interfaces import MeasurementData, MeasurementDataRecorder
from ..measurements import FloatMetadata, SingleMeasurement
from ._stats import RunningStats, bytes_to_gb


class MemoryInfoRecorder(MeasurementDataRecorder):
    """Record total host RAM and this process's resident, virtual, and unique memory."""

    def __init__(self):
        self._memory_hardware_info = {"total_ram_gb": bytes_to_gb(psutil.virtual_memory().total)}
        # Resident set, virtual size, and unique set sizes [bytes].
        self._stats = {"rss": RunningStats(), "vms": RunningStats(), "uss": RunningStats()}
        self._process = psutil.Process(os.getpid())

    def update(self) -> None:
        mem_info = self._process.memory_info()
        self._stats["rss"].update(mem_info.rss)
        self._stats["vms"].update(mem_info.vms)
        # USS is not available on every platform.
        with contextlib.suppress(psutil.AccessDenied, AttributeError):
            self._stats["uss"].update(self._process.memory_full_info().uss)

    def get_initial_data(self) -> dict:
        return {"memory_metadata": self._memory_hardware_info}

    def get_runtime_data(self) -> dict:
        runtime_info = {}
        for key, stats in self._stats.items():
            if stats.n > 0:
                runtime_info[f"{key}_mean"] = stats.mean
                runtime_info[f"{key}_std"] = stats.std
                runtime_info[f"{key}_n"] = stats.n
                runtime_info[f"{key}_peak"] = stats.peak
        return {"memory_utilization": runtime_info}

    def get_data(self) -> MeasurementData:
        measurements = []
        for key, stats in self._stats.items():
            if key == "uss" and stats.n == 0:
                continue
            label = f"System Memory {key.upper()}"
            measurements.extend(
                [
                    SingleMeasurement(name=label, value=bytes_to_gb(stats.mean), unit="GB"),
                    SingleMeasurement(name=f"{label} std", value=bytes_to_gb(stats.std), unit="GB"),
                    SingleMeasurement(name=f"{label} peak", value=bytes_to_gb(stats.peak), unit="GB"),
                    SingleMeasurement(name=f"{label} n", value=stats.n, unit=""),
                ]
            )
        return MeasurementData(
            measurements=measurements,
            metadata=[FloatMetadata(name="total_ram_gb", data=self._memory_hardware_info["total_ram_gb"])],
        )
