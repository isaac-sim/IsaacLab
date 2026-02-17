# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import json
import logging
import os
import textwrap
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from .measurements import SingleMeasurement, StatisticalMeasurement, TestPhase, TestPhaseEncoder

logger = logging.getLogger(__name__)


def get_default_output_filename(prefix: str = "benchmark") -> str:
    """Generate default output filename with current date and time.

    Args:
        prefix: Prefix for the filename (e.g., "articulation_benchmark").

    Returns:
        Filename string with timestamp (without extension).
    """
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{datetime_str}"


class MetricsBackendInterface(ABC):
    """Abstract base class for metrics backends."""

    @abstractmethod
    def add_metrics(self, test_phase: TestPhase) -> None:
        """Add metrics from a test phase.

        Args:
            test_phase: Test phase containing metrics to add.
        """
        pass

    @abstractmethod
    def finalize(self, output_path: str, **kwargs) -> None:
        """Finalize and write metrics to output.

        Args:
            output_path: Path to write output file(s).
            **kwargs: Additional backend-specific options.
        """
        pass


class MetricsBackend:
    """Factory for creating metrics backend instances."""

    _instances: dict[str, MetricsBackendInterface] = {}

    @classmethod
    def get_instance(cls, instance_type: str) -> MetricsBackendInterface:
        """Get or create a backend instance by type name.

        Args:
            instance_type: Type of backend to create ("json", "osmo", or "omniperf").

        Returns:
            Backend instance of the requested type.

        Raises:
            ValueError: If the instance_type is not recognized.
        """
        if instance_type not in cls._instances:
            backend_map = {
                "json": JSONFileMetrics,
                "osmo": OsmoKPIFile,
                "omniperf": OmniPerfKPIFile,
                "summary": SummaryMetrics,
            }
            if instance_type not in backend_map:
                raise ValueError(f"Unknown backend type: {instance_type}. Available: {list(backend_map.keys())}")
            cls._instances[instance_type] = backend_map[instance_type]()
        return cls._instances[instance_type]

    @classmethod
    def reset_instances(cls) -> None:
        """Reset all cached backend instances. Useful for testing."""
        cls._instances.clear()


class JSONFileMetrics(MetricsBackendInterface):
    """Write metrics to a JSON file at the end of a session."""

    def __init__(self) -> None:
        self.data: list[TestPhase] = []
        self.test_name = ""

    def add_metrics(self, test_phase: TestPhase) -> None:
        """Accumulate a test phase for later serialization.

        Args:
            test_phase: Test phase to add.

        Example:

        .. code-block:: python

            backend.add_metrics(test_phase)
        """
        self.data.append(copy.deepcopy(test_phase))

    def finalize(self, output_path: str, output_filename: str, **kwargs) -> None:
        """Write metrics data to a JSON file.

        Args:
            output_path: Output path in which metrics file will be stored.
            output_filename: Output filename.
            **kwargs: Additional backend-specific options.

        Example:

        .. code-block:: python

            backend.finalize("/tmp/metrics", "metrics")
        """
        if not self.data:
            logger.warning("No test data to write. Skipping metrics file generation.")
            return

        # Append test name to measurement name as OVAT needs to uniquely identify
        for test_phase in self.data:
            test_name = test_phase.get_metadata_field("workflow_name")
            # Store the test name
            if test_name != self.test_name:
                if self.test_name:
                    logger.warning(
                        f"Nonempty test name {self.test_name} different from name {test_name} provided by test phase."
                    )
                self.test_name = test_name
                logger.info(f"Setting test name to {self.test_name}")

            phase_name = test_phase.get_metadata_field("phase")
            for measurement in test_phase.measurements:
                measurement.name = f"{test_name} {phase_name} {measurement.name}"

            for metadata in test_phase.metadata:
                metadata.name = f"{test_name} {phase_name} {metadata.name}"

        json_data = json.dumps(self.data, indent=4, cls=TestPhaseEncoder)

        metrics_path = os.path.join(output_path, f"{output_filename}.json")
        with open(metrics_path, "w") as f:
            f.write(json_data)
        print(f"Results written to: {metrics_path}")

        self.data.clear()


class SummaryMetrics(MetricsBackendInterface):
    """Print a human-readable summary and write JSON metrics."""

    def __init__(self) -> None:
        """Initialize internal phase storage and JSON backend."""
        self._phases: list[TestPhase] = []
        self._json_backend = JSONFileMetrics()
        self._report_width = 86

    def add_metrics(self, test_phase: TestPhase) -> None:
        """Add metrics from a test phase; store for summary and forward to JSON backend.

        Args:
            test_phase: Test phase containing measurements and metadata.
        """
        self._phases.append(copy.deepcopy(test_phase))
        self._json_backend.add_metrics(test_phase)

    def finalize(self, output_path: str, output_filename: str, **kwargs) -> None:
        """Write JSON output and print human-readable summary to console.

        Args:
            output_path: Path to write output file(s).
            output_filename: Base filename for the JSON file.
            **kwargs: Additional options passed to the JSON backend.
        """
        self._json_backend.finalize(output_path, output_filename, **kwargs)
        if self._phases:
            self._print_summary()
        self._phases.clear()

    def _print_summary(self) -> None:
        """Format and print the boxed summary report to stdout."""
        phases = self._merge_phases()
        benchmark_info = phases.get("benchmark_info")
        runtime_phase = phases.get("runtime")
        startup_phase = phases.get("startup")
        train_phase = phases.get("train")
        frametime_phase = phases.get("frametime")
        hardware_info = phases.get("hardware_info")
        version_info = phases.get("version_info")

        benchmark_meta = self._metadata_map(benchmark_info)
        hardware_meta = self._metadata_map(hardware_info)
        version_meta = self._metadata_map(version_info)
        dev_meta = version_meta.get("dev", {}) if isinstance(version_meta.get("dev"), dict) else {}

        workflow_name = benchmark_meta.get("workflow_name")
        timestamp = benchmark_meta.get("timestamp")
        task = benchmark_meta.get("task")
        seed = benchmark_meta.get("seed")
        num_envs = benchmark_meta.get("num_envs")
        max_iterations = benchmark_meta.get("max_iterations")
        num_cpus = hardware_meta.get("physical_cores")
        commit = dev_meta.get("commit_hash_short") or dev_meta.get("commit_hash")
        branch = dev_meta.get("branch")

        gpu_name, gpu_total_mem = self._get_gpu_summary(hardware_meta)

        print()
        self._print_box_separator()
        self._print_box_line("Summary Report".center(self._report_width - 4))
        self._print_box_separator()
        self._print_box_kv("workflow_name", workflow_name)
        self._print_box_kv("timestamp", timestamp)
        self._print_box_kv("task", task)
        self._print_box_kv("seed", seed)
        self._print_box_kv("num_envs", num_envs)
        self._print_box_kv("max_iterations", max_iterations)
        self._print_box_kv("num_cpus", num_cpus)
        self._print_box_kv("commit", commit)
        self._print_box_kv("branch", branch)
        self._print_box_kv("gpu_name", gpu_name)
        if gpu_total_mem is not None:
            self._print_box_kv("gpu_total_memory_gb", gpu_total_mem)
        self._print_box_separator()

        if runtime_phase:
            runtime_rows = self._summarize_runtime_metrics(runtime_phase.measurements)
            self._print_box_line("Phase: runtime")
            for row in runtime_rows:
                self._print_box_line(row)
            self._print_box_separator()

        if startup_phase:
            self._print_box_line("Phase: startup")
            self._print_optional_measurement(startup_phase, "App Launch Time")
            self._print_optional_measurement(startup_phase, "Python Imports Time")
            self._print_optional_measurement(startup_phase, "Task Creation and Start Time")
            self._print_optional_measurement(startup_phase, "Scene Creation Time")
            self._print_optional_measurement(startup_phase, "Simulation Start Time")
            self._print_optional_measurement(startup_phase, "Total Start Time (Launch to Train)")
            self._print_box_separator()

        if train_phase:
            self._print_box_line("Phase: train")
            self._print_optional_measurement(train_phase, "Max Rewards", unit_fallback="float")
            self._print_optional_measurement(train_phase, "Max Episode Lengths", unit_fallback="float")
            self._print_optional_measurement(train_phase, "Last Reward", unit_fallback="float")
            self._print_optional_measurement(train_phase, "Last Episode Length", unit_fallback="float")
            self._print_optional_measurement(train_phase, "EMA 0.95 Reward", unit_fallback="float")
            self._print_optional_measurement(train_phase, "EMA 0.95 Episode Length", unit_fallback="float")
            self._print_box_separator()

        if frametime_phase and frametime_phase.measurements:
            self._print_box_line("Phase: frametime")
            for measurement in frametime_phase.measurements:
                label = measurement.name
                if isinstance(measurement, StatisticalMeasurement):
                    unit_str = f" {measurement.unit.strip()}" if (measurement.unit and measurement.unit.strip()) else ""
                    value = f"{self._format_scalar(measurement.mean)}{unit_str}"
                elif isinstance(measurement, SingleMeasurement):
                    unit_str = f" {measurement.unit.strip()}" if (measurement.unit and measurement.unit.strip()) else ""
                    value = f"{self._format_scalar(measurement.value)}{unit_str}"
                else:
                    continue
                self._print_box_line(f"{label}: {value}")
            self._print_box_separator()

        if hardware_meta:
            self._print_box_line("System:")
            self._print_box_kv("cpu_name", hardware_meta.get("cpu_name"))
            self._print_box_kv("physical_cores", hardware_meta.get("physical_cores"))
            self._print_box_kv("total_ram_gb", hardware_meta.get("total_ram_gb"))
            self._print_box_kv("gpu_device_count", hardware_meta.get("gpu_device_count"))
            self._print_box_kv("cuda_version", hardware_meta.get("cuda_version"))
            self._print_box_separator()

    def _merge_phases(self) -> dict[str, TestPhase]:
        """Merge all stored phases by name, combining measurements and metadata.

        Returns:
            Dictionary mapping phase name to a single merged TestPhase.
        """
        merged: dict[str, TestPhase] = {}
        for phase in self._phases:
            name = phase.phase_name
            if name not in merged:
                merged[name] = copy.deepcopy(phase)
            else:
                merged[name].measurements.extend(phase.measurements)
                merged[name].metadata.extend(phase.metadata)
        return merged

    def _metadata_map(self, phase: TestPhase | None) -> dict[str, Any]:
        """Build a name -> data map from a phase's metadata list.

        Args:
            phase: Test phase, or None.

        Returns:
            Dictionary of metadata names to their data values.
        """
        if not phase:
            return {}
        metadata: dict[str, Any] = {}
        for item in phase.metadata:
            if hasattr(item, "data"):
                metadata[item.name] = item.data
        return metadata

    def _get_gpu_summary(self, hardware_meta: dict[str, Any]) -> tuple[str | None, float | None]:
        """Extract GPU name and total memory (GB) from hardware metadata.

        Args:
            hardware_meta: Metadata dict from the hardware_info phase.

        Returns:
            (gpu_name, total_memory_gb) or (None, None) if not available.
        """
        gpu_devices = hardware_meta.get("gpu_devices")
        current_device = hardware_meta.get("gpu_current_device", 0)
        if isinstance(gpu_devices, dict):
            device = gpu_devices.get(str(current_device)) or next(iter(gpu_devices.values()), {})
            name = device.get("name")
            total_mem = device.get("total_memory_gb")
            return name, total_mem
        return None, None

    def _print_optional_measurement(self, phase: TestPhase, name: str, unit_fallback: str | None = None) -> None:
        """Print a single measurement line if present in the phase.

        Args:
            phase: Test phase to look up the measurement.
            name: Measurement name.
            unit_fallback: Unit string to use when measurement has no unit.
        """
        measurement = self._get_single_measurement(phase, name)
        if measurement is None:
            return
        unit = (measurement.unit or unit_fallback or "").strip()
        suffix = f" {unit}" if unit else ""
        self._print_box_line(f"{name}: {self._format_scalar(measurement.value)}{suffix}")

    def _get_single_measurement(self, phase: TestPhase, name: str) -> SingleMeasurement | None:
        """Return the first SingleMeasurement in the phase with the given name.

        Args:
            phase: Test phase to search.
            name: Measurement name.

        Returns:
            The matching SingleMeasurement, or None.
        """
        for measurement in phase.measurements:
            if isinstance(measurement, SingleMeasurement) and measurement.name == name:
                return measurement
        return None

    def _summarize_runtime_metrics(self, measurements: list) -> list[str]:
        """Build min/mean/max summary rows from SingleMeasurement runtime metrics.

        Args:
            measurements: List of measurements (typically from the runtime phase).

        Returns:
            List of formatted lines, grouped by category (Collection, Learning, etc.).
        """
        series: dict[str, dict[str, float]] = {}
        units: dict[str, str | None] = {}
        for measurement in measurements:
            if not isinstance(measurement, SingleMeasurement):
                continue
            name = measurement.name
            value = measurement.value
            unit = measurement.unit
            if not isinstance(value, (int, float)):
                continue
            if name.startswith("Min "):
                base = name[len("Min ") :]
                series.setdefault(base, {})["min"] = float(value)
                units.setdefault(base, unit)
            elif name.startswith("Max "):
                base = name[len("Max ") :]
                series.setdefault(base, {})["max"] = float(value)
                units.setdefault(base, unit)
            elif name.startswith("Mean "):
                base = name[len("Mean ") :]
                series.setdefault(base, {})["mean"] = float(value)
                units.setdefault(base, unit)

        category_order = ["Collection", "Learning", "Step Times", "Throughput", "Other"]
        categorized: dict[str, list[str]] = {key: [] for key in category_order}
        for base, stats in series.items():
            raw_unit = units.get(base)
            unit = (raw_unit or "").strip() if isinstance(raw_unit, str) else ""
            unit_suffix = f" {unit}" if unit else ""
            min_val = self._format_scalar(stats.get("min", 0.0))
            mean_val = self._format_scalar(stats.get("mean", 0.0))
            max_val = self._format_scalar(stats.get("max", 0.0))
            row = f"{base} (min/mean/max): {min_val} / {mean_val} / {max_val}{unit_suffix}"

            if "Collection" in base:
                categorized["Collection"].append(row)
            elif "Learning" in base:
                categorized["Learning"].append(row)
            elif "step time" in base.lower():
                categorized["Step Times"].append(row)
            elif "FPS" in base or "Throughput" in base:
                categorized["Throughput"].append(row)
            else:
                categorized["Other"].append(row)

        rows: list[str] = []
        for category in category_order:
            if not categorized[category]:
                continue
            rows.append(f"{category}:")
            rows.extend(f"  {entry}" for entry in categorized[category])
        if not rows:
            rows.append("No runtime metrics available.")
        return rows

    def _print_box_separator(self) -> None:
        """Print a horizontal rule line for the summary box."""
        print("|" + "-" * (self._report_width - 2) + "|")

    def _print_box_line(self, text: str) -> None:
        """Print a line of text inside the box, wrapping if needed."""
        inner_width = self._report_width - 4
        if not text:
            print(f"| {' ' * inner_width} |")
            return
        for line in textwrap.wrap(text, width=inner_width, break_long_words=False, break_on_hyphens=False):
            print(f"| {line.ljust(inner_width)} |")

    def _print_box_kv(self, key: str, value: Any) -> None:
        """Print a key-value line; skip if value is None."""
        if value is None:
            return
        if isinstance(value, float):
            value = self._format_scalar(value)
        self._print_box_line(f"{key}: {value}")

    def _format_scalar(self, value: float | int) -> str:
        """Format a numeric value for display (two decimal places for floats)."""
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)


class OsmoKPIFile(MetricsBackendInterface):
    """Write per-phase KPI documents for Osmo ingestion.

    Only SingleMeasurement metrics and metadata are written as key-value pairs.
    """

    def __init__(self) -> None:
        self._test_phases: list[TestPhase] = []

    def add_metrics(self, test_phase: TestPhase) -> None:
        """Adds provided test_phase to internal list of test_phases.

        Args:
            test_phase: Current test phase.

        Example:

        .. code-block:: python

            backend.add_metrics(test_phase)
        """
        self._test_phases.append(test_phase)

    def finalize(self, output_path: str, output_filename: str, **kwargs) -> None:
        """Write metrics to output file(s).

        Each test phase's SingleMeasurement metrics and metadata are written to an output JSON file, at path
        `[output_path]/[output_filename].json`.

        Args:
            output_path: Output path in which metrics files will be stored.
            output_filename: Output filename.
            **kwargs: Additional backend-specific options.

        Example:

        .. code-block:: python

            backend.finalize("/tmp/metrics", "kpis")
        """
        for test_phase in self._test_phases:
            # Retrieve useful metadata from test_phase
            phase_name = test_phase.get_metadata_field("phase")

            osmo_kpis: dict[str, object] = {}
            log_statements = [f"{phase_name} KPIs:"]
            # Add metadata as KPIs
            for metadata in test_phase.metadata:
                osmo_kpis[metadata.name] = metadata.data
                log_statements.append(f"{metadata.name}: {metadata.data}")
            # Add single measurements as KPIs
            for measurement in test_phase.measurements:
                if isinstance(measurement, SingleMeasurement):
                    osmo_kpis[measurement.name] = measurement.value
                    log_statements.append(f"{measurement.name}: {measurement.value} {measurement.unit}")
            # Generate the output filename with timestamp
            metrics_path = os.path.join(output_path, f"{output_filename}.json")
            # Dump key-value pairs (fields) to the JSON document
            json_data = json.dumps(osmo_kpis, indent=4)
            with open(metrics_path, "w") as f:
                f.write(json_data)
            print(f"Results written to: {metrics_path}")


class OmniPerfKPIFile(MetricsBackendInterface):
    """Write KPI metrics for upload to a PostgreSQL database."""

    def __init__(self) -> None:
        self._test_phases: list[TestPhase] = []

    def add_metrics(self, test_phase: TestPhase) -> None:
        """Adds provided test_phase to internal list of test_phases.

        Args:
            test_phase: Current test phase.

        Example:

        .. code-block:: python

            backend.add_metrics(test_phase)
        """
        self._test_phases.append(test_phase)

    def finalize(self, output_path: str, output_filename: str, **kwargs) -> None:
        """Write metrics to output file(s).

        Measurement metrics and metadata are written to an output JSON file, at path
        `[output_path]/[output_filename].json`.

        Args:
            output_path: Output path in which metrics file will be stored.
            output_filename: Output filename.
            **kwargs: Additional backend-specific options.

        Example:

        .. code-block:: python

            backend.finalize("/tmp/metrics", "omniperf")
        """
        if not self._test_phases:
            logger.warning("No test phases to write. Skipping metrics file generation.")
            return

        workflow_data: dict[str, object] = {}

        for test_phase in self._test_phases:
            # Retrieve useful metadata from test_phase
            phase_name = test_phase.get_metadata_field("phase")

            phase_data: dict[str, object] = {}
            log_statements = [f"{phase_name} Metrics:"]
            # Add metadata as metrics
            for metadata in test_phase.metadata:
                phase_data[metadata.name] = metadata.data
                log_statements.append(f"{metadata.name}: {metadata.data}")
            # Add measurements as metrics
            for measurement in test_phase.measurements:
                if isinstance(measurement, StatisticalMeasurement):
                    log_statements.append(
                        f"{measurement.name}: {measurement.mean:.2f} ± {measurement.std:.2f} "
                        f"{measurement.unit} (n={measurement.n})"
                    )
                    phase_data[f"{measurement.name}_mean"] = measurement.mean
                    phase_data[f"{measurement.name}_std"] = measurement.std
                    phase_data[f"{measurement.name}_n"] = measurement.n
                elif type(measurement).__name__ == "SingleMeasurement":
                    log_statements.append(f"{measurement.name}: {measurement.value} {measurement.unit}")
                    phase_data[measurement.name] = measurement.value
            workflow_data[phase_name] = phase_data

        metrics_path = os.path.join(output_path, f"{output_filename}.json")
        # Dump key-value pairs (fields) to the JSON document
        json_data = json.dumps(workflow_data, indent=4)
        with open(metrics_path, "w") as f:
            f.write(json_data)
        print(f"Results written to: {metrics_path}")
