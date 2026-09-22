# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import subprocess

import torch

from ..interfaces import MeasurementData, MeasurementDataRecorder
from ..measurements import DictMetadata, IntMetadata, SingleMeasurement, StringMetadata
from ._stats import RunningStats, bytes_to_gb

_NVIDIA_SMI_QUERY = ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"]


class GPUInfoRecorder(MeasurementDataRecorder):
    """Record CUDA device properties plus per-device memory use and utilization.

    Samples come from pynvml when it is importable, then from ``nvidia-smi``, and finally from
    PyTorch's own allocation counter, which only sees PyTorch tensors and reports no utilization.
    """

    def __init__(self):
        self._gpu_hardware_info: dict = {"available": torch.cuda.is_available()}
        self._device_count = 0
        self._memory: list[RunningStats] = []
        self._utilization: list[RunningStats] = []
        self._nvml = None
        self._handles: list = []
        self._nvidia_smi_available = False
        if self._gpu_hardware_info["available"]:
            self._get_hardware_info()

    def _get_hardware_info(self) -> None:
        self._device_count = torch.cuda.device_count()
        self._gpu_hardware_info["device_count"] = self._device_count
        self._gpu_hardware_info["current_device"] = torch.cuda.current_device()
        self._gpu_hardware_info["devices"] = []
        for index in range(self._device_count):
            props = torch.cuda.get_device_properties(index)
            self._gpu_hardware_info["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": bytes_to_gb(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count,
                }
            )
        self._memory = [RunningStats() for _ in range(self._device_count)]
        self._utilization = [RunningStats() for _ in range(self._device_count)]
        self._gpu_hardware_info["cuda_version"] = torch.version.cuda or "Unknown"

        with contextlib.suppress(Exception):
            import pynvml

            pynvml.nvmlInit()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(self._device_count)]
            self._nvml = pynvml
        if self._nvml is None:
            with contextlib.suppress(Exception):
                result = subprocess.run(_NVIDIA_SMI_QUERY, capture_output=True, text=True, timeout=5)
                self._nvidia_smi_available = result.returncode == 0

    def _sample_devices(self) -> list[tuple[float | None, float | None]]:
        """Return one ``(memory_bytes, utilization_percent)`` pair per device, with ``None`` for unavailable values."""
        samples: list[tuple[float | None, float | None]] = [(None, None)] * self._device_count
        if self._nvml is not None:
            for index, handle in enumerate(self._handles):
                memory = utilization = None
                with contextlib.suppress(Exception):
                    memory = self._nvml.nvmlDeviceGetMemoryInfo(handle).used
                with contextlib.suppress(Exception):
                    utilization = self._nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                samples[index] = (memory, utilization)
        elif self._nvidia_smi_available:
            with contextlib.suppress(Exception):
                result = subprocess.run(_NVIDIA_SMI_QUERY, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for index, line in enumerate(result.stdout.strip().split("\n")[: self._device_count]):
                        memory_mb, utilization = (float(part.strip()) for part in line.split(",")[:2])
                        samples[index] = (memory_mb * 1024 * 1024, utilization)
        return samples

    def update(self) -> None:
        if not self._gpu_hardware_info["available"]:
            return
        for index, (memory, utilization) in enumerate(self._sample_devices()):
            self._memory[index].update(torch.cuda.memory_allocated(index) if memory is None else memory)
            if utilization is not None:
                self._utilization[index].update(utilization)

    def get_initial_data(self) -> dict:
        return {"gpu_metadata": self._gpu_hardware_info}

    def get_runtime_data(self) -> dict:
        if not any(stats.n for stats in self._memory):
            return {"gpu_utilization": {}}
        devices = []
        for memory, utilization in zip(self._memory, self._utilization):
            device: dict[str, float | int] = {
                "memory_used_mean_bytes": memory.mean,
                "memory_used_std_bytes": memory.std,
                "memory_used_peak_bytes": memory.peak,
                "memory_n": memory.n,
            }
            if utilization.n:
                device["utilization_mean_percent"] = utilization.mean
                device["utilization_std_percent"] = utilization.std
                device["utilization_n"] = utilization.n
            devices.append(device)
        return {"gpu_utilization": {"devices": devices}}

    def get_data(self) -> MeasurementData:
        if not self._gpu_hardware_info["available"]:
            return MeasurementData()

        devices_data = {
            str(device["index"]): {
                key: device[key] for key in ("name", "total_memory_gb", "compute_capability", "multi_processor_count")
            }
            for device in self._gpu_hardware_info["devices"]
        }
        metadata = [
            IntMetadata(name="gpu_device_count", data=self._device_count),
            IntMetadata(name="gpu_current_device", data=self._gpu_hardware_info["current_device"]),
            StringMetadata(name="cuda_version", data=self._gpu_hardware_info["cuda_version"]),
            DictMetadata(name="gpu_devices", data=devices_data),
        ]

        measurements = []
        for index, (memory, utilization) in enumerate(zip(self._memory, self._utilization)):
            prefix = f"GPU {index} " if self._device_count > 1 else "GPU "
            if memory.n:
                measurements.extend(
                    [
                        SingleMeasurement(name=f"{prefix}Memory Used", value=bytes_to_gb(memory.mean), unit="GB"),
                        SingleMeasurement(name=f"{prefix}Memory Used std", value=bytes_to_gb(memory.std), unit="GB"),
                        SingleMeasurement(name=f"{prefix}Memory Used n", value=memory.n, unit=""),
                    ]
                )
            # The peak is always reported; it stays at zero until the first sample.
            measurements.append(
                SingleMeasurement(name=f"{prefix}Memory Used peak", value=bytes_to_gb(memory.peak), unit="GB")
            )
            if utilization.n:
                measurements.extend(
                    [
                        SingleMeasurement(name=f"{prefix}Utilization", value=round(utilization.mean, 2), unit="%"),
                        SingleMeasurement(name=f"{prefix}Utilization std", value=round(utilization.std, 2), unit="%"),
                        SingleMeasurement(name=f"{prefix}Utilization n", value=utilization.n, unit=""),
                    ]
                )
        return MeasurementData(measurements=measurements, metadata=metadata)
