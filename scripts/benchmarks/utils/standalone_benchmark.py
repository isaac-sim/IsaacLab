# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Standalone benchmark services that can run without Isaac Sim.

This module provides a lightweight benchmarking system that mimics the functionality
of isaacsim.benchmark.services but does not depend on Isaac Sim being available.
"""

import json
import multiprocessing
import os
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path

# Optional dependencies for system metrics
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("[WARNING] psutil not available. System metrics will not be collected. Install with: pip install psutil")

# Try GPU libraries
GPUTIL_AVAILABLE = False
PYNVML_AVAILABLE = False

try:
    import GPUtil

    GPUTIL_AVAILABLE = True
    print("[INFO] GPUtil available for GPU metrics")
except ImportError:
    pass

if not GPUTIL_AVAILABLE:
    try:
        import pynvml

        PYNVML_AVAILABLE = True
        print("[INFO] pynvml available for GPU metrics")
    except ImportError:
        pass

if not GPUTIL_AVAILABLE and not PYNVML_AVAILABLE:
    print("[INFO] GPUtil/pynvml not available, will use nvidia-smi if available")


#############################
# Measurement Data Classes  #
#############################


@dataclass
class Measurement:
    """Base measurement class."""

    name: str


@dataclass
class SingleMeasurement(Measurement):
    """Represents a single float measurement."""

    value: float
    unit: str
    type: str = "single"


@dataclass
class BooleanMeasurement(Measurement):
    """Represents a boolean measurement."""

    bvalue: bool
    type: str = "boolean"


@dataclass
class DictMeasurement(Measurement):
    """Represents a dictionary measurement."""

    value: dict
    type: str = "dict"


@dataclass
class ListMeasurement(Measurement):
    """Represents a list measurement."""

    value: list
    type: str = "list"

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, length={len(self.value)})"


@dataclass
class MetadataBase:
    """Base metadata class."""

    name: str


@dataclass
class StringMetadata(MetadataBase):
    """String metadata."""

    data: str
    type: str = "string"


@dataclass
class IntMetadata(MetadataBase):
    """Integer metadata."""

    data: int
    type: str = "int"


@dataclass
class FloatMetadata(MetadataBase):
    """Float metadata."""

    data: float
    type: str = "float"


@dataclass
class DictMetadata(MetadataBase):
    """Dictionary metadata."""

    data: dict
    type: str = "dict"


@dataclass
class TestPhase:
    """Represents a single test phase which may have many metrics associated with it."""

    phase_name: str
    measurements: list[Measurement] = field(default_factory=list)
    metadata: list[MetadataBase] = field(default_factory=list)

    def get_metadata_field(self, name, default=KeyError):
        """Get a metadata field's value.

        Args:
            name: Field name (case-insensitive).
            default: Default value if not found. If KeyError, raises exception.

        Returns:
            The metadata value.

        Raises:
            KeyError: If the field is not found and default is KeyError.
        """
        name = name.lower()
        for m in self.metadata:
            name2 = m.name.replace(self.phase_name, "").strip().lower()
            if name == name2:
                return m.data

        if default is KeyError:
            raise KeyError(name)
        return default

    @classmethod
    def metadata_from_dict(cls, m: dict) -> list[MetadataBase]:
        """Create metadata from dictionary.

        Args:
            m: Dictionary containing metadata list.

        Returns:
            List of MetadataBase objects.
        """
        metadata = []
        metadata_mapping = {str: StringMetadata, int: IntMetadata, float: FloatMetadata, dict: DictMetadata}
        for meas in m["metadata"]:
            if "data" in meas:
                metadata_type = metadata_mapping.get(type(meas["data"]))
                if metadata_type:
                    curr_meta = metadata_type(name=meas["name"], data=meas["data"])
                    metadata.append(curr_meta)
        return metadata


class TestPhaseEncoder(json.JSONEncoder):
    """JSON encoder for TestPhase objects."""

    def default(self, o):
        return o.__dict__


#############################
# Backend Implementation    #
#############################


class OmniPerfKPIFile:
    """Prints metrics into a JSON document compatible with OmniPerfKPIFile format."""

    def __init__(self):
        self._test_phases = []

    def add_metrics(self, test_phase: TestPhase) -> None:
        """Add a test phase to the backend.

        Args:
            test_phase: The test phase to add.
        """
        self._test_phases.append(test_phase)

    def finalize(self, metrics_output_folder: str, randomize_filename_prefix: bool = False) -> None:
        """Write metrics to output file.

        Args:
            metrics_output_folder: Output folder for metrics file.
            randomize_filename_prefix: Whether to randomize the filename prefix.
        """
        if not self._test_phases:
            print("[WARNING] No test phases to write. Skipping metrics file generation.")
            return

        workflow_data = {"timestamp": dt.now().isoformat()}

        test_name = None
        for test_phase in self._test_phases:
            # Retrieve useful metadata from test_phase
            test_name = test_phase.get_metadata_field("workflow_name")
            phase_name = test_phase.get_metadata_field("phase")

            phase_data = {}
            log_statements = [f"{phase_name} Metrics:"]
            # Add metadata as metrics
            for metadata in test_phase.metadata:
                phase_data[metadata.name] = metadata.data
                log_statements.append(f"  {metadata.name}: {metadata.data}")
            # Add measurements as metrics
            for measurement in test_phase.measurements:
                if isinstance(measurement, SingleMeasurement):
                    log_statements.append(f"  {measurement.name}: {measurement.value} {measurement.unit}")
                    phase_data[measurement.name] = measurement.value
                elif isinstance(measurement, (DictMeasurement, ListMeasurement)):
                    # For dict and list measurements, store them as-is
                    phase_data[measurement.name] = measurement.value
            # Log all metrics to console
            print("\n".join(log_statements))

            workflow_data[phase_name] = phase_data

        # Generate the output filename
        if randomize_filename_prefix:
            _, metrics_filename_out = tempfile.mkstemp(
                dir=metrics_output_folder, prefix=f"kpis_{test_name}", suffix=".json"
            )
        else:
            metrics_filename_out = Path(metrics_output_folder) / f"kpis_{test_name}.json"
        # Dump key-value pairs to the JSON document
        json_data = json.dumps(workflow_data, indent=4)
        with open(metrics_filename_out, "w") as f:
            print(f"[INFO] Writing metrics to {metrics_filename_out}")
            f.write(json_data)


class OsmoKPIFile:
    """Print metrics into separate JSON documents for each phase."""

    def __init__(self):
        self._test_phases = []

    def add_metrics(self, test_phase: TestPhase) -> None:
        """Add a test phase to the backend.

        Args:
            test_phase: The test phase to add.
        """
        self._test_phases.append(test_phase)

    def finalize(self, metrics_output_folder: str, randomize_filename_prefix: bool = False) -> None:
        """Write metrics to output files.

        Args:
            metrics_output_folder: Output folder for metrics files.
            randomize_filename_prefix: Whether to randomize the filename prefix.
        """
        for test_phase in self._test_phases:
            # Retrieve useful metadata from test_phase
            test_name = test_phase.get_metadata_field("workflow_name")
            phase_name = test_phase.get_metadata_field("phase")

            osmo_kpis = {}
            log_statements = [f"{phase_name} KPIs:"]
            # Add metadata as KPIs
            for metadata in test_phase.metadata:
                osmo_kpis[metadata.name] = metadata.data
                log_statements.append(f"  {metadata.name}: {metadata.data}")
            # Add single measurements as KPIs
            for measurement in test_phase.measurements:
                if isinstance(measurement, SingleMeasurement):
                    osmo_kpis[measurement.name] = measurement.value
                    log_statements.append(f"  {measurement.name}: {measurement.value} {measurement.unit}")
            # Log all KPIs to console
            print("\n".join(log_statements))
            # Generate the output filename
            if randomize_filename_prefix:
                _, metrics_filename_out = tempfile.mkstemp(
                    dir=metrics_output_folder, prefix=f"kpis_{test_name}_{phase_name}", suffix=".json"
                )
            else:
                metrics_filename_out = Path(metrics_output_folder) / f"kpis_{test_name}_{phase_name}.json"
            # Dump key-value pairs to the JSON document
            json_data = json.dumps(osmo_kpis, indent=4)
            with open(metrics_filename_out, "w") as f:
                f.write(json_data)


class JSONFileMetrics:
    """Dump all metrics to a single JSON file."""

    def __init__(self):
        self.data = []
        self.test_name = None

    def add_metrics(self, test_phase: TestPhase) -> None:
        """Add a test phase to the backend.

        Args:
            test_phase: The test phase to add.
        """
        self.data.append(test_phase)

    def finalize(self, metrics_output_folder: str, randomize_filename_prefix: bool = False) -> None:
        """Write metrics to output file.

        Args:
            metrics_output_folder: Output folder for metrics file.
            randomize_filename_prefix: Whether to randomize the filename prefix.
        """
        if not self.data:
            print("[WARNING] No test data to write. Skipping metrics file generation.")
            return

        # Get test name
        for test_phase in self.data:
            test_name = test_phase.get_metadata_field("workflow_name")
            if test_name != self.test_name:
                if self.test_name:
                    print(
                        f"[WARNING] Nonempty test name {self.test_name} different from name {test_name} provided by"
                        " test phase."
                    )
                self.test_name = test_name

            phase_name = test_phase.get_metadata_field("phase")
            for m in test_phase.measurements:
                m.name = f"{test_name} {phase_name} {m.name}"

            for m in test_phase.metadata:
                m.name = f"{test_name} {phase_name} {m.name}"

        json_data = json.dumps(self.data, indent=4, cls=TestPhaseEncoder)

        # Generate the output filename
        if randomize_filename_prefix:
            _, metrics_filename_out = tempfile.mkstemp(
                dir=metrics_output_folder, prefix=f"metrics_{self.test_name}", suffix=".json"
            )
        else:
            metrics_filename_out = Path(metrics_output_folder) / f"metrics_{self.test_name}.json"

        with open(metrics_filename_out, "w") as f:
            print(f"[INFO] Writing metrics to {metrics_filename_out}")
            f.write(json_data)

        self.data.clear()


class LocalLogMetrics:
    """Simple backend that just logs metrics to console."""

    def __init__(self):
        self._test_phases = []

    def add_metrics(self, test_phase: TestPhase) -> None:
        """Add a test phase to the backend.

        Args:
            test_phase: The test phase to add.
        """
        self._test_phases.append(test_phase)

    def finalize(self, metrics_output_folder: str, randomize_filename_prefix: bool = False) -> None:
        """Log metrics to console.

        Args:
            metrics_output_folder: Not used for this backend.
            randomize_filename_prefix: Not used for this backend.
        """
        for test_phase in self._test_phases:
            test_name = test_phase.get_metadata_field("workflow_name")
            phase_name = test_phase.get_metadata_field("phase")

            print(f"\n{'=' * 60}")
            print(f"Benchmark: {test_name} - Phase: {phase_name}")
            print(f"{'=' * 60}")

            print("\nMetadata:")
            for metadata in test_phase.metadata:
                print(f"  {metadata.name}: {metadata.data}")

            print("\nMeasurements:")
            for measurement in test_phase.measurements:
                if isinstance(measurement, SingleMeasurement):
                    print(f"  {measurement.name}: {measurement.value} {measurement.unit}")
                elif isinstance(measurement, (DictMeasurement, ListMeasurement)):
                    print(
                        f"  {measurement.name}: {type(measurement.value).__name__} with {len(measurement.value)} items"
                    )


class MetricsBackend:
    """Factory for creating metrics backends."""

    @staticmethod
    def get_instance(instance_type: str):
        """Get a metrics backend instance.

        Args:
            instance_type: Type of backend ("OmniPerfKPIFile", "OsmoKPIFile", "JSONFileMetrics", "LocalLogMetrics").

        Returns:
            An instance of the requested backend.

        Raises:
            ValueError: If instance_type is not recognized.
        """
        if instance_type == "OmniPerfKPIFile":
            return OmniPerfKPIFile()
        elif instance_type == "OsmoKPIFile":
            return OsmoKPIFile()
        elif instance_type == "JSONFileMetrics":
            return JSONFileMetrics()
        elif instance_type == "LocalLogMetrics":
            return LocalLogMetrics()
        else:
            raise ValueError(f"Unknown backend type: {instance_type}")


#############################
# Benchmark Class           #
#############################


class StandaloneBenchmark:
    """Standalone benchmark class that works without Isaac Sim.

    This class mimics the functionality of BaseIsaacBenchmark but does not
    depend on Isaac Sim or its benchmark services extension.
    """

    def __init__(
        self,
        benchmark_name: str = "StandaloneBenchmark",
        backend_type: str = "OmniPerfKPIFile",
        workflow_metadata: dict = {},
        output_folder: str | None = None,
        randomize_filename_prefix: bool = False,
        collect_system_metrics: bool = True,
    ):
        """Initialize the standalone benchmark.

        Args:
            benchmark_name: Name of the benchmark.
            backend_type: Type of backend to use for metrics collection.
            workflow_metadata: Metadata describing the benchmark.
            output_folder: Output folder for metrics files. If None, uses temp directory.
            randomize_filename_prefix: Whether to randomize the filename prefix.
            collect_system_metrics: Whether to collect system metrics (CPU, memory, GPU).
        """
        self.benchmark_name = benchmark_name
        self._test_phases = []
        self._current_phase = None
        self._collect_system_metrics = collect_system_metrics and PSUTIL_AVAILABLE

        # System metrics tracking
        self._phase_start_time = None
        self._process = psutil.Process() if PSUTIL_AVAILABLE else None

        # Check if nvidia-smi is available
        self._nvidia_smi_available = self._check_nvidia_smi()

        # Get metrics backend
        self._metrics = MetricsBackend.get_instance(instance_type=backend_type)

        # Set output folder
        if output_folder is None:
            self._metrics_output_folder = tempfile.gettempdir()
        else:
            self._metrics_output_folder = output_folder

        self._randomize_filename_prefix = randomize_filename_prefix

        # Generate workflow-level metadata
        self._metadata = [StringMetadata(name="workflow_name", data=self.benchmark_name)]
        if "metadata" in workflow_metadata:
            self._metadata.extend(TestPhase.metadata_from_dict(workflow_metadata))
        elif workflow_metadata:
            print(
                "[WARNING] workflow_metadata provided, but missing expected 'metadata' entry. Metadata will not be"
                " read."
            )

        print(f"[INFO] Benchmark initialized: {self.benchmark_name}")
        print(f"[INFO] Output folder: {self._metrics_output_folder}")
        print(f"[INFO] Backend type: {backend_type}")
        print(f"[INFO] System metrics collection: {'enabled' if self._collect_system_metrics else 'disabled'}")
        if self._nvidia_smi_available:
            print("[INFO] nvidia-smi available for direct GPU queries")
        self.benchmark_start_time = time.time()

    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available.

        Returns:
            True if nvidia-smi is available and working.
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=2
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return False

    def _get_gpu_info_nvidia_smi(self) -> dict[str, any] | None:
        """Get GPU information using nvidia-smi directly.

        Returns:
            Dictionary with GPU info or None if failed.
        """
        try:
            # Query multiple fields at once
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if result.returncode == 0:
                line = result.stdout.strip().split("\n")[0]  # Get first GPU
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    return {
                        "name": parts[0],
                        "memory_used_mb": float(parts[1]),
                        "memory_total_mb": float(parts[2]),
                        "utilization": float(parts[3]),
                    }
        except Exception as e:
            print(f"[WARNING] nvidia-smi query failed: {e}")

        return None

    def _collect_system_info(self) -> list[MetadataBase]:
        """Collect system information as metadata.

        Returns:
            List of metadata objects with system information.
        """
        metadata = []

        if not self._collect_system_metrics:
            return metadata

        # CPU count
        metadata.append(IntMetadata(name="num_cpus", data=multiprocessing.cpu_count()))

        # GPU information - try multiple methods
        gpu_detected = False

        # Method 1: Try GPUtil
        if GPUTIL_AVAILABLE and not gpu_detected:
            try:
                gpus = GPUtil.getGPUs()
                if gpus and len(gpus) > 0:
                    metadata.append(StringMetadata(name="gpu_device_name", data=gpus[0].name))
                    gpu_detected = True
                    print(f"[INFO] GPU detected via GPUtil: {gpus[0].name}")
            except Exception as e:
                print(f"[WARNING] Failed to get GPU info via GPUtil: {e}")

        # Method 2: Try pynvml
        if PYNVML_AVAILABLE and not gpu_detected:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(gpu_name, bytes):
                        gpu_name = gpu_name.decode("utf-8")
                    metadata.append(StringMetadata(name="gpu_device_name", data=gpu_name))
                    gpu_detected = True
                    print(f"[INFO] GPU detected via pynvml: {gpu_name}")
                pynvml.nvmlShutdown()
            except Exception as e:
                print(f"[WARNING] Failed to get GPU info via pynvml: {e}")
                with suppress(Exception):
                    pynvml.nvmlShutdown()

        # Method 3: Try nvidia-smi directly (most reliable, bypasses driver mismatch)
        if self._nvidia_smi_available and not gpu_detected:
            try:
                gpu_info = self._get_gpu_info_nvidia_smi()
                if gpu_info:
                    metadata.append(StringMetadata(name="gpu_device_name", data=gpu_info["name"]))
                    gpu_detected = True
                    print(f"[INFO] GPU detected via nvidia-smi: {gpu_info['name']}")
            except Exception as e:
                print(f"[WARNING] Failed to get GPU info via nvidia-smi: {e}")

        if not gpu_detected:
            print("[WARNING] No GPU detected. GPU metrics will not be available.")
            print(
                "[INFO] This is likely due to NVIDIA driver issues (Driver/library version mismatch or error code 18)."
            )
            print(
                "[INFO] To fix: 1) sudo reboot (reloads drivers) 2) Check nvidia-smi manually 3) Reinstall NVIDIA"
                " drivers"
            )
            print("[INFO] The benchmark will continue without GPU metrics.")

        return metadata

    def _collect_runtime_metrics(self) -> list[Measurement]:
        """Collect runtime system metrics.

        Returns:
            List of measurement objects with runtime metrics.
        """
        measurements = []

        if not self._collect_system_metrics:
            return measurements

        try:
            # Memory metrics (in GB)
            mem_info = self._process.memory_info()
            measurements.append(SingleMeasurement(name="System Memory RSS", value=mem_info.rss / (1024**3), unit="GB"))
            measurements.append(SingleMeasurement(name="System Memory VMS", value=mem_info.vms / (1024**3), unit="GB"))

            # USS (Unique Set Size) if available
            try:
                mem_full = self._process.memory_full_info()
                measurements.append(
                    SingleMeasurement(name="System Memory USS", value=mem_full.uss / (1024**3), unit="GB")
                )
            except (AttributeError, psutil.AccessDenied):
                pass

            # CPU usage
            cpu_times = psutil.cpu_times_percent(interval=0.1)
            measurements.append(SingleMeasurement(name="System CPU user", value=cpu_times.user, unit="%"))
            measurements.append(SingleMeasurement(name="System CPU system", value=cpu_times.system, unit="%"))
            measurements.append(SingleMeasurement(name="System CPU idle", value=cpu_times.idle, unit="%"))
            if hasattr(cpu_times, "iowait"):
                measurements.append(SingleMeasurement(name="System CPU iowait", value=cpu_times.iowait, unit="%"))

            # Runtime duration
            if self._phase_start_time is not None:
                runtime_ms = (time.time() - self._phase_start_time) * 1000
                measurements.append(SingleMeasurement(name="Runtime", value=runtime_ms, unit="ms"))

            # GPU metrics - try multiple methods
            gpu_metrics_collected = False

            # Method 1: Try GPUtil
            if GPUTIL_AVAILABLE and not gpu_metrics_collected:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus and len(gpus) > 0:
                        gpu = gpus[0]
                        measurements.append(
                            SingleMeasurement(name="GPU Memory Tracked", value=gpu.memoryUsed / 1024, unit="GB")
                        )
                        measurements.append(
                            SingleMeasurement(name="GPU Memory Dedicated", value=gpu.memoryTotal / 1024, unit="GB")
                        )
                        measurements.append(SingleMeasurement(name="GPU Utilization", value=gpu.load * 100, unit="%"))
                        gpu_metrics_collected = True
                except Exception as e:
                    print(f"[WARNING] Failed to collect GPU metrics via GPUtil: {e}")

            # Method 2: Try pynvml
            if PYNVML_AVAILABLE and not gpu_metrics_collected:
                try:
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    if device_count > 0:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        measurements.append(
                            SingleMeasurement(name="GPU Memory Tracked", value=mem_info.used / (1024**3), unit="GB")
                        )
                        measurements.append(
                            SingleMeasurement(name="GPU Memory Dedicated", value=mem_info.total / (1024**3), unit="GB")
                        )
                        measurements.append(SingleMeasurement(name="GPU Utilization", value=util_rates.gpu, unit="%"))
                        gpu_metrics_collected = True
                    pynvml.nvmlShutdown()
                except Exception as e:
                    print(f"[WARNING] Failed to collect GPU metrics via pynvml: {e}")
                    with suppress(Exception):
                        pynvml.nvmlShutdown()

            # Method 3: Try nvidia-smi directly (most reliable, bypasses driver mismatch)
            if self._nvidia_smi_available and not gpu_metrics_collected:
                try:
                    gpu_info = self._get_gpu_info_nvidia_smi()
                    if gpu_info:
                        measurements.append(
                            SingleMeasurement(
                                name="GPU Memory Tracked", value=gpu_info["memory_used_mb"] / 1024, unit="GB"
                            )
                        )
                        measurements.append(
                            SingleMeasurement(
                                name="GPU Memory Dedicated", value=gpu_info["memory_total_mb"] / 1024, unit="GB"
                            )
                        )
                        measurements.append(
                            SingleMeasurement(name="GPU Utilization", value=gpu_info["utilization"], unit="%")
                        )
                        gpu_metrics_collected = True
                        print("[INFO] Collected GPU metrics via nvidia-smi")
                except Exception as e:
                    print(f"[WARNING] Failed to collect GPU metrics via nvidia-smi: {e}")

            if not gpu_metrics_collected:
                if GPUTIL_AVAILABLE or PYNVML_AVAILABLE or self._nvidia_smi_available:
                    print("[WARNING] GPU libraries/tools available but all methods failed to collect metrics.")
                    print("[INFO] This usually indicates NVIDIA driver issues (Driver/library mismatch or GPU lost).")
                    print("[INFO] To fix: sudo reboot (reloads drivers and libraries)")
                else:
                    print("[INFO] No GPU libraries available. GPU metrics will not be collected.")

        except Exception as e:
            print(f"[WARNING] Failed to collect runtime metrics: {e}")
            import traceback

            traceback.print_exc()

        return measurements

    def set_phase(
        self, phase: str, start_recording_frametime: bool = True, start_recording_runtime: bool = True
    ) -> None:
        """Set the current benchmarking phase.

        Args:
            phase: Name of the phase.
            start_recording_frametime: Not used in standalone version (for API compatibility).
            start_recording_runtime: Not used in standalone version (for API compatibility).
        """
        print(f"[INFO] Starting phase: {phase}")
        self._current_phase = phase
        self._phase_start_time = time.time()

    def store_measurements(self) -> None:
        """Store measurements for the current phase.

        This method should be called after completing work in a phase and before
        setting a new phase or calling stop().
        """
        if self._current_phase is None:
            print("[WARNING] No phase set. Call set_phase() before store_measurements().")
            return

        # Create a new test phase
        test_phase = TestPhase(phase_name=self._current_phase, measurements=[], metadata=[])

        # Collect system info metadata (only for first phase or if explicitly needed)
        if len(self._test_phases) == 0:
            system_metadata = self._collect_system_info()
            test_phase.metadata.extend(system_metadata)

        # Collect runtime metrics
        runtime_measurements = self._collect_runtime_metrics()
        test_phase.measurements.extend(runtime_measurements)

        # Update test phase metadata with phase name and benchmark metadata
        test_phase.metadata.extend(self._metadata)
        test_phase.metadata.append(StringMetadata(name="phase", data=self._current_phase))
        self._test_phases.append(test_phase)

        print(f"[INFO] Stored measurements for phase: {self._current_phase}")

    def store_custom_measurement(self, phase_name: str, custom_measurement: Measurement) -> None:
        """Store a custom measurement for a specific phase.

        Args:
            phase_name: Name of the phase.
            custom_measurement: The measurement to store.
        """
        # Check if the phase already exists
        existing_phase = next((phase for phase in self._test_phases if phase.phase_name == phase_name), None)

        if existing_phase:
            # Add the custom measurement to the existing phase
            existing_phase.measurements.append(custom_measurement)
        else:
            # If the phase does not exist, create a new test phase
            new_test_phase = TestPhase(phase_name=phase_name, measurements=[custom_measurement], metadata=[])
            # Update test phase metadata with phase name and benchmark metadata
            new_test_phase.metadata.extend(self._metadata)
            new_test_phase.metadata.append(StringMetadata(name="phase", data=phase_name))

            # Add the new test phase to the list of test phases
            self._test_phases.append(new_test_phase)

    def stop(self):
        """Stop benchmarking and write accumulated metrics to file."""
        print("[INFO] Stopping benchmark")

        if not self._test_phases:
            print(
                "[WARNING] No test phases collected. After set_phase(), store_measurements() should be called. "
                "No metrics will be written."
            )
            return

        # Create output folder if it doesn't exist
        if not os.path.exists(self._metrics_output_folder):
            os.makedirs(self._metrics_output_folder, exist_ok=True)

        print("[INFO] Writing metrics data.")

        # Finalize by adding all test phases to the backend metrics
        for test_phase in self._test_phases:
            self._metrics.add_metrics(test_phase)

        self._metrics.finalize(self._metrics_output_folder, self._randomize_filename_prefix)

        elapsed_time = time.time() - self.benchmark_start_time
        print(f"[INFO] Benchmark completed in {elapsed_time:.2f} seconds")
