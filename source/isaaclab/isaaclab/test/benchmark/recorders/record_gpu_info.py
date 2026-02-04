# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import math

import torch

from isaaclab.test.benchmark.interfaces import MeasurementDataRecorder, MeasurementData
from isaaclab.test.benchmark.measurements import (
    DictMetadata,
    IntMetadata,
    StringMetadata,
)


class GPUInfoRecorder(MeasurementDataRecorder):
    def __init__(self):
        # Hardware and runtime information
        self._gpu_hardware_info = {}
        self._gpu_runtime_info = {}
        self._device_count = 0

        # Per-device Welford stats for memory (bytes)
        self._mem_mean = []
        self._mem_std = []
        self._mem_n = []
        self._mem_m2 = []

        # Per-device Welford stats for utilization (%)
        self._util_mean = []
        self._util_std = []
        self._util_n = []
        self._util_m2 = []

        # pynvml device handles (one per GPU)
        self._handles = []
        self._nvml_available = False

        self._get_hardware_info()

    def _get_hardware_info(self) -> None:
        if not torch.cuda.is_available():
            self._gpu_hardware_info["available"] = False
            return

        self._gpu_hardware_info["available"] = True
        self._device_count = torch.cuda.device_count()
        self._gpu_hardware_info["device_count"] = self._device_count
        self._gpu_hardware_info["current_device"] = torch.cuda.current_device()

        # Collect info for all devices
        self._gpu_hardware_info["devices"] = []
        for i in range(self._device_count):
            gpu_props = torch.cuda.get_device_properties(i)
            device_info = {
                "index": i,
                "name": gpu_props.name,
                "total_memory_gb": round(gpu_props.total_memory / (1024**3), 2),
                "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                "multi_processor_count": gpu_props.multi_processor_count,
            }
            self._gpu_hardware_info["devices"].append(device_info)

            # Initialize Welford stats for this device
            self._mem_mean.append(0)
            self._mem_std.append(0)
            self._mem_n.append(0)
            self._mem_m2.append(0)
            self._util_mean.append(0)
            self._util_std.append(0)
            self._util_n.append(0)
            self._util_m2.append(0)

        # CUDA version
        with contextlib.suppress(Exception):
            import torch.version as torch_version

            cuda_version = getattr(torch_version, "cuda", None)
            self._gpu_hardware_info["cuda_version"] = cuda_version if cuda_version else "Unknown"

        # Initialize pynvml for GPU utilization monitoring (all devices)
        with contextlib.suppress(Exception):
            import pynvml

            pynvml.nvmlInit()
            for i in range(self._device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self._handles.append(handle)
            self._nvml_available = True

    def _get_runtime_info(self) -> None:
        if not torch.cuda.is_available():
            return

        # Initialize runtime info structure if needed
        if "devices" not in self._gpu_runtime_info:
            self._gpu_runtime_info["devices"] = [{} for _ in range(self._device_count)]

        for i in range(self._device_count):
            # GPU memory usage per device
            memory_allocated = torch.cuda.memory_allocated(i)
            self._mem_n[i] += 1
            delta = memory_allocated - self._mem_mean[i]
            self._mem_mean[i] += delta / self._mem_n[i]
            delta2 = memory_allocated - self._mem_mean[i]
            self._mem_m2[i] += delta * delta2
            if self._mem_n[i] > 1:
                self._mem_std[i] = math.sqrt(self._mem_m2[i] / (self._mem_n[i] - 1))

            self._gpu_runtime_info["devices"][i]["memory_allocated_mean_bytes"] = self._mem_mean[i]
            self._gpu_runtime_info["devices"][i]["memory_allocated_std_bytes"] = self._mem_std[i]
            self._gpu_runtime_info["devices"][i]["memory_n"] = self._mem_n[i]

            # GPU utilization from pynvml
            if self._nvml_available and i < len(self._handles):
                with contextlib.suppress(Exception):
                    import pynvml

                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handles[i])
                    gpu_util = util.gpu

                    self._util_n[i] += 1
                    delta = gpu_util - self._util_mean[i]
                    self._util_mean[i] += delta / self._util_n[i]
                    delta2 = gpu_util - self._util_mean[i]
                    self._util_m2[i] += delta * delta2
                    if self._util_n[i] > 1:
                        self._util_std[i] = math.sqrt(self._util_m2[i] / (self._util_n[i] - 1))

                    self._gpu_runtime_info["devices"][i]["utilization_mean_percent"] = self._util_mean[i]
                    self._gpu_runtime_info["devices"][i]["utilization_std_percent"] = self._util_std[i]
                    self._gpu_runtime_info["devices"][i]["utilization_n"] = self._util_n[i]

    def update(self) -> None:
        self._get_runtime_info()

    def get_initial_data(self) -> dict:
        return {
            "gpu_metadata": self._gpu_hardware_info,
        }

    def get_runtime_data(self) -> dict:
        return {
            "gpu_utilization": self._gpu_runtime_info,
        }

    def get_data(self) -> MeasurementData:
        measurements = []
        metadata = []

        if not self._gpu_hardware_info.get("available", False):
            return MeasurementData(measurements=measurements, metadata=metadata)

        # Global metadata
        metadata.append(IntMetadata(name="gpu_device_count", data=self._device_count))
        metadata.append(IntMetadata(name="gpu_current_device", data=self._gpu_hardware_info["current_device"]))
        metadata.append(StringMetadata(name="cuda_version", data=self._gpu_hardware_info.get("cuda_version", "Unknown")))

        # Per-device info as a dict
        devices_data = {}
        for i in range(self._device_count):
            device_hw = self._gpu_hardware_info.get("devices", [{}])[i] if i < len(self._gpu_hardware_info.get("devices", [])) else {}
            device_runtime = self._gpu_runtime_info.get("devices", [{}] * self._device_count)[i]

            device_data = {
                # Hardware info
                "name": device_hw.get("name", "Unknown"),
                "total_memory_gb": device_hw.get("total_memory_gb", 0),
                "compute_capability": device_hw.get("compute_capability", "Unknown"),
                "multi_processor_count": device_hw.get("multi_processor_count", 0),
                # Runtime measurements
                "memory_allocated_mean_bytes": device_runtime.get("memory_allocated_mean_bytes", 0),
                "memory_allocated_std_bytes": device_runtime.get("memory_allocated_std_bytes", 0),
                "memory_n": device_runtime.get("memory_n", 0),
            }

            # Add utilization if available
            if "utilization_mean_percent" in device_runtime:
                device_data["utilization_mean_percent"] = device_runtime["utilization_mean_percent"]
                device_data["utilization_std_percent"] = device_runtime["utilization_std_percent"]
                device_data["utilization_n"] = device_runtime["utilization_n"]

            devices_data[str(i)] = device_data

        metadata.append(DictMetadata(name="gpu_devices", data=devices_data))

        return MeasurementData(measurements=measurements, metadata=metadata)
