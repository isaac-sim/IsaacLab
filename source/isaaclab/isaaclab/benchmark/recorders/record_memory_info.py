# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

import psutil

from ..interfaces import MeasurementData, MeasurementDataRecorder
from ..measurements import FloatMetadata, SingleMeasurement


class MemoryInfoRecorder(MeasurementDataRecorder):
    def __init__(self):
        # Empty dictionaries to store hardware and runtime information
        self._memory_hardware_info = {}
        self._memory_runtime_info = {}

        # Welford's algorithm stats for RSS (Resident Set Size)
        self._rss_mean = 0
        self._rss_m2 = 0
        self._rss_n = 0

        # Welford's algorithm stats for VMS (Virtual Memory Size)
        self._vms_mean = 0
        self._vms_m2 = 0
        self._vms_n = 0

        # Peak (running max) alongside the Welford mean/std. Initialised to
        # 0.0 so emit-before-record returns a meaningful zero.
        self._rss_peak = 0.0
        self._vms_peak = 0.0

        # Process handle
        self._process = psutil.Process(os.getpid())
        self._get_hardware_info()

    def _get_hardware_info(self) -> None:
        mem = psutil.virtual_memory()
        self._memory_hardware_info["total_ram_gb"] = round(mem.total / (1024**3), 2)

    def _update_welford(self, value: float, mean: float, m2: float, n: int) -> tuple[float, float, int, float]:
        """Update Welford's online algorithm for mean and variance.

        Returns:
            Tuple of (new_mean, new_m2, new_n, std)
        """
        n += 1
        delta = value - mean
        mean += delta / n
        delta2 = value - mean
        m2 += delta * delta2
        std = math.sqrt(m2 / (n - 1)) if n > 1 else 0
        return mean, m2, n, std

    def _get_runtime_info(self) -> None:
        mem_info = self._process.memory_info()

        # RSS (Resident Set Size) - physical memory used
        self._rss_mean, self._rss_m2, self._rss_n, rss_std = self._update_welford(
            mem_info.rss, self._rss_mean, self._rss_m2, self._rss_n
        )
        self._rss_peak = max(self._rss_peak, float(mem_info.rss))
        self._memory_runtime_info["rss_mean"] = self._rss_mean
        self._memory_runtime_info["rss_std"] = rss_std
        self._memory_runtime_info["rss_n"] = self._rss_n
        self._memory_runtime_info["rss_peak"] = self._rss_peak

        # VMS (Virtual Memory Size) - total virtual memory
        self._vms_mean, self._vms_m2, self._vms_n, vms_std = self._update_welford(
            mem_info.vms, self._vms_mean, self._vms_m2, self._vms_n
        )
        self._vms_peak = max(self._vms_peak, float(mem_info.vms))
        self._memory_runtime_info["vms_mean"] = self._vms_mean
        self._memory_runtime_info["vms_std"] = vms_std
        self._memory_runtime_info["vms_n"] = self._vms_n
        self._memory_runtime_info["vms_peak"] = self._vms_peak

        # USS (Unique Set Size) is deliberately not sampled here.
        # ``psutil.Process.memory_full_info()`` walks the process page tables on every
        # call, which costs hundreds of milliseconds once the process is a few GB
        # resident. This recorder is driven by
        # :class:`~isaaclab.benchmark.BenchmarkMonitor` from a background thread while
        # the benchmark is being timed, so that walk perturbs the very workload the run
        # is measuring. ``memory_info()`` above reads cheap kernel counters and is
        # unaffected.

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

    def _bytes_to_gb(self, bytes_value: float) -> float:
        """Convert bytes to gigabytes, rounded to 2 decimal places."""
        return round(bytes_value / (1024**3), 2)

    def get_data(self) -> MeasurementData:
        measurements = [
            # RSS (Resident Set Size)
            SingleMeasurement(
                name="System Memory RSS",
                value=self._bytes_to_gb(self._memory_runtime_info.get("rss_mean", 0)),
                unit="GB",
            ),
            SingleMeasurement(
                name="System Memory RSS std",
                value=self._bytes_to_gb(self._memory_runtime_info.get("rss_std", 0)),
                unit="GB",
            ),
            SingleMeasurement(
                name="System Memory RSS peak",
                value=self._bytes_to_gb(self._memory_runtime_info.get("rss_peak", 0)),
                unit="GB",
            ),
            SingleMeasurement(name="System Memory RSS n", value=self._memory_runtime_info.get("rss_n", 0), unit=""),
            # VMS (Virtual Memory Size)
            SingleMeasurement(
                name="System Memory VMS",
                value=self._bytes_to_gb(self._memory_runtime_info.get("vms_mean", 0)),
                unit="GB",
            ),
            SingleMeasurement(
                name="System Memory VMS std",
                value=self._bytes_to_gb(self._memory_runtime_info.get("vms_std", 0)),
                unit="GB",
            ),
            SingleMeasurement(
                name="System Memory VMS peak",
                value=self._bytes_to_gb(self._memory_runtime_info.get("vms_peak", 0)),
                unit="GB",
            ),
            SingleMeasurement(name="System Memory VMS n", value=self._memory_runtime_info.get("vms_n", 0), unit=""),
        ]

        return MeasurementData(
            measurements=measurements,
            metadata=[
                FloatMetadata(name="total_ram_gb", data=self._memory_hardware_info["total_ram_gb"]),
            ],
        )
