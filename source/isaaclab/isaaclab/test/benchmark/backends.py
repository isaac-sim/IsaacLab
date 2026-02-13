# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime

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
