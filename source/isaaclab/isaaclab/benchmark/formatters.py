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
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .measurements import SingleMeasurement, StatisticalMeasurement, TestPhase, TestPhaseEncoder

if TYPE_CHECKING:
    from .schema import PlayBundle, RuntimeBundle, StartupBundle, TrainingBundle

logger = logging.getLogger(__name__)


_SUMMARY_STARTUP_ROWS = (
    "App Launch Time",
    "Python Imports Time",
    "Task Creation and Start Time",
    "Scene Creation Time",
    "Simulation Start Time",
    "Total Start Time (Launch to Train)",
)
_SUMMARY_TRAIN_ROWS = (
    "Max Rewards",
    "Max Episode Lengths",
    "Last Reward",
    "Last Episode Length",
    "EMA 0.95 Reward",
    "EMA 0.95 Episode Length",
)
_SUMMARY_KNOWN_PHASES = {"benchmark_info", "runtime", "startup", "train", "frametime", "hardware_info", "version_info"}


def _write_json(output_path: str, output_filename: str, data: object, **dump_kwargs) -> None:
    """Write ``data`` to ``<output_path>/<output_filename>.json`` and report the location."""
    metrics_path = os.path.join(output_path, f"{output_filename}.json")
    with open(metrics_path, "w") as f:
        f.write(json.dumps(data, indent=4, **dump_kwargs))
    print(f"Results written to: {metrics_path}")


def get_default_output_filename(prefix: str = "benchmark") -> str:
    """Generate a unique default output filename with the current date and time.

    Args:
        prefix: Prefix for the filename (e.g., "articulation_benchmark").

    Returns:
        Filename string with a timestamp and unique suffix (without extension).
    """
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    unique_suffix = uuid4().hex[:8]
    return f"{prefix}_{datetime_str}_{unique_suffix}"


class MetricsFormatterInterface(ABC):
    """Abstract base class for metrics Formatters."""

    @abstractmethod
    def add_metrics(self, test_phase: TestPhase) -> None:
        """Add metrics from a test phase.

        Args:
            test_phase: Test phase containing metrics to add.
        """

    @abstractmethod
    def finalize(self, output_path: str, **kwargs) -> None:
        """Finalize and write metrics to output.

        Args:
            output_path: Path to write output file(s).
            **kwargs: Additional formatter-specific options.
        """


_FORMATTER_CLASSES: dict[str, type[MetricsFormatterInterface]] = {}
"""Formatter classes keyed by type name, filled in below the class definitions."""


class MetricsFormatter:
    """Factory for creating metrics formatter instances."""

    _instances: dict[str, MetricsFormatterInterface] = {}

    @classmethod
    def get_instance(cls, instance_type: str) -> MetricsFormatterInterface:
        """Get or create a formatter instance by type name.

        Args:
            instance_type: Type of formatter to create ("json", "osmo", "omniperf", "summary", or "schema").

        Returns:
            Formatter instance of the requested type.

        Raises:
            ValueError: If the instance_type is not recognized.
        """
        if instance_type not in cls._instances:
            if instance_type not in _FORMATTER_CLASSES:
                raise ValueError(f"Unknown formatter type: {instance_type}. Available: {list(_FORMATTER_CLASSES)}")
            cls._instances[instance_type] = _FORMATTER_CLASSES[instance_type]()
        return cls._instances[instance_type]

    @classmethod
    def reset_instances(cls) -> None:
        """Reset all cached formatter instances. Useful for testing."""
        cls._instances.clear()


class JSONFileMetrics(MetricsFormatterInterface):
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

            formatter.add_metrics(test_phase)
        """
        self.data.append(copy.deepcopy(test_phase))

    def finalize(self, output_path: str, output_filename: str, **kwargs) -> None:
        """Write metrics data to a JSON file.

        Args:
            output_path: Output path in which metrics file will be stored.
            output_filename: Output filename.
            **kwargs: Additional formatter-specific options.

        Example:

        .. code-block:: python

            formatter.finalize("/tmp/metrics", "metrics")
        """
        if not self.data:
            logger.warning("No test data to write. Skipping metrics file generation.")
            return

        # OVAT identifies measurements by their fully qualified "<test> <phase> <name>" label.
        for test_phase in self.data:
            test_name = test_phase.get_metadata_field("workflow_name")
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

        _write_json(output_path, output_filename, self.data, cls=TestPhaseEncoder)
        self.data.clear()


class SummaryMetrics(MetricsFormatterInterface):
    """Print a human-readable summary and write JSON metrics."""

    def __init__(self) -> None:
        """Initialize internal phase storage and JSON formatter."""
        self._phases: list[TestPhase] = []
        self._json_formatter = JSONFileMetrics()
        self._report_width = 86

    def add_metrics(self, test_phase: TestPhase) -> None:
        """Add metrics from a test phase; store for summary and forward to JSON formatter.

        Args:
            test_phase: Test phase containing measurements and metadata.
        """
        self._phases.append(copy.deepcopy(test_phase))
        self._json_formatter.add_metrics(test_phase)

    def finalize(self, output_path: str, output_filename: str, **kwargs) -> None:
        """Write JSON output and print human-readable summary to console.

        Args:
            output_path: Path to write output file(s).
            output_filename: Base filename for the JSON file.
            **kwargs: Additional options passed to the JSON formatter.
        """
        self._json_formatter.finalize(output_path, output_filename, **kwargs)
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
            for name in _SUMMARY_STARTUP_ROWS:
                self._print_optional_measurement(startup_phase, name)
            self._print_box_separator()

        if train_phase:
            self._print_box_line("Phase: train")
            for name in _SUMMARY_TRAIN_ROWS:
                self._print_optional_measurement(train_phase, name, unit_fallback="float")
            self._print_box_separator()

        if frametime_phase and frametime_phase.measurements:
            self._print_phase_measurements("frametime", frametime_phase)

        # Phases not handled above, e.g. the profiling phases of the startup benchmark.
        for phase_name, phase in phases.items():
            if phase_name not in _SUMMARY_KNOWN_PHASES and phase.measurements:
                self._print_phase_measurements(phase_name, phase)

        if hardware_meta:
            self._print_box_line("System:")
            self._print_box_kv("cpu_name", hardware_meta.get("cpu_name"))
            self._print_box_kv("physical_cores", hardware_meta.get("physical_cores"))
            self._print_box_kv("total_ram_gb", hardware_meta.get("total_ram_gb"))
            self._print_box_kv("gpu_device_count", hardware_meta.get("gpu_device_count"))
            self._print_box_kv("cuda_version", hardware_meta.get("cuda_version"))
            self._print_box_separator()

    def _print_phase_measurements(self, phase_name: str, phase: TestPhase) -> None:
        """Print every scalar measurement of a phase as its own row."""
        self._print_box_line(f"Phase: {phase_name}")
        for measurement in phase.measurements:
            if isinstance(measurement, StatisticalMeasurement):
                value = measurement.mean
            elif isinstance(measurement, SingleMeasurement):
                value = measurement.value
            else:
                continue
            unit = (measurement.unit or "").strip()
            suffix = f" {unit}" if unit else ""
            self._print_box_line(f"{measurement.name}: {self._format_scalar(value)}{suffix}")
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
        """Build summary rows from scalar runtime statistics.

        Args:
            measurements: List of measurements (typically from the runtime phase).

        Returns:
            List of formatted lines, grouped by category (Collection, Learning, etc.).
        """
        series: dict[str, dict[str, float]] = {}
        units: dict[str, str | None] = {}
        for measurement in measurements:
            if not isinstance(measurement, SingleMeasurement) or not isinstance(measurement.value, (int, float)):
                continue
            statistic, _, base = measurement.name.partition(" ")
            if statistic in ("Min", "Max", "Mean", "Std") and base:
                series.setdefault(base, {})[statistic.lower()] = float(measurement.value)
                units.setdefault(base, measurement.unit)

        category_order = ["Collection", "Learning", "Step Times", "Throughput", "Other"]
        categorized: dict[str, list[str]] = {key: [] for key in category_order}
        for base, stats in series.items():
            raw_unit = units.get(base)
            unit = (raw_unit or "").strip() if isinstance(raw_unit, str) else ""
            unit_suffix = f" {unit}" if unit else ""
            available = [statistic for statistic in ("min", "mean", "std", "max") if statistic in stats]
            values = " / ".join(self._format_scalar(stats[statistic]) for statistic in available)
            labels = "/".join(available)
            row = f"{base} ({labels}): {values}{unit_suffix}"

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


class OsmoKPIFile(MetricsFormatterInterface):
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

            formatter.add_metrics(test_phase)
        """
        self._test_phases.append(test_phase)

    def finalize(self, output_path: str, output_filename: str, **kwargs) -> None:
        """Write metrics to output file(s).

        A single phase is written to ``[output_path]/[output_filename].json``. Multiple phases
        are written separately with the phase name appended to each filename.

        Args:
            output_path: Output path in which metrics files will be stored.
            output_filename: Output filename.
            **kwargs: Additional formatter-specific options.

        Example:

        .. code-block:: python

            formatter.finalize("/tmp/metrics", "kpis")
        """
        multi_phase = len(self._test_phases) > 1
        for test_phase in self._test_phases:
            phase_name = test_phase.get_metadata_field("phase")
            osmo_kpis: dict[str, object] = {metadata.name: metadata.data for metadata in test_phase.metadata}
            for measurement in test_phase.measurements:
                if isinstance(measurement, SingleMeasurement):
                    osmo_kpis[measurement.name] = measurement.value
            filename = f"{output_filename}_{phase_name}" if multi_phase else output_filename
            _write_json(output_path, filename, osmo_kpis)
        self._test_phases.clear()


class OmniPerfKPIFile(MetricsFormatterInterface):
    """Write KPI metrics for upload to a PostgreSQL database."""

    def __init__(self) -> None:
        self._test_phases: list[TestPhase] = []

    def add_metrics(self, test_phase: TestPhase) -> None:
        """Adds provided test_phase to internal list of test_phases.

        Args:
            test_phase: Current test phase.

        Example:

        .. code-block:: python

            formatter.add_metrics(test_phase)
        """
        self._test_phases.append(test_phase)

    def finalize(self, output_path: str, output_filename: str, **kwargs) -> None:
        """Write metrics to output file(s).

        Measurement metrics and metadata are written to an output JSON file, at path
        `[output_path]/[output_filename].json`.

        Args:
            output_path: Output path in which metrics file will be stored.
            output_filename: Output filename.
            **kwargs: Additional formatter-specific options.

        Example:

        .. code-block:: python

            formatter.finalize("/tmp/metrics", "omniperf")
        """
        if not self._test_phases:
            logger.warning("No test phases to write. Skipping metrics file generation.")
            return

        workflow_data: dict[str, object] = {}
        for test_phase in self._test_phases:
            phase_data: dict[str, object] = {metadata.name: metadata.data for metadata in test_phase.metadata}
            for measurement in test_phase.measurements:
                if isinstance(measurement, StatisticalMeasurement):
                    phase_data[f"{measurement.name}_mean"] = measurement.mean
                    phase_data[f"{measurement.name}_std"] = measurement.std
                    phase_data[f"{measurement.name}_n"] = measurement.n
                # Matched by class name so Isaac Sim's SingleMeasurement is accepted as well.
                elif type(measurement).__name__ == "SingleMeasurement":
                    phase_data[measurement.name] = measurement.value
            workflow_data[test_phase.get_metadata_field("phase")] = phase_data

        _write_json(output_path, output_filename, workflow_data)
        self._test_phases.clear()


class SchemaBundleFile(MetricsFormatterInterface):
    """Serialize a typed benchmark bundle to schema-v1 JSON.

    Unlike the other formatters, this one does not consume the flat measurement
    phases collected during a run. Instead it serializes the typed bundle
    attached via :meth:`~isaaclab.benchmark.benchmark_core.BaseIsaacLabBenchmark.attach_bundle`.
    """

    def add_metrics(self, test_phase: TestPhase) -> None:
        """Ignore the provided test phase.

        This formatter serializes the typed bundle attached via
        :meth:`~isaaclab.benchmark.benchmark_core.BaseIsaacLabBenchmark.attach_bundle`,
        not the flat measurement phases, so accumulated phases are ignored by design.

        Args:
            test_phase: Test phase to ignore.
        """
        pass

    def finalize(
        self,
        output_path: str,
        output_filename: str,
        bundle: "RuntimeBundle | TrainingBundle | StartupBundle | PlayBundle | None" = None,
        **kwargs,
    ) -> None:
        """Write the attached bundle to a schema-v1 JSON file.

        Args:
            output_path: Output path in which the schema file will be stored.
            output_filename: Output filename (without extension).
            bundle: Typed benchmark bundle to serialize.
            **kwargs: Additional formatter-specific options (ignored).
        """
        if bundle is None:
            raise RuntimeError("The schema formatter requires a benchmark bundle.")

        # Lazy import keeps formatters.py free of the schema layer at module import time.
        from .serialize import write_bundle_file

        path = os.path.join(output_path, f"{output_filename}.json")
        write_bundle_file(bundle, path)
        logger.info("Wrote schema bundle to %s", path)


_FORMATTER_CLASSES.update(
    json=JSONFileMetrics,
    osmo=OsmoKPIFile,
    omniperf=OmniPerfKPIFile,
    summary=SummaryMetrics,
    schema=SchemaBundleFile,
)
