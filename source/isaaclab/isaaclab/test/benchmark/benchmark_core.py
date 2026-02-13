# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import time
from collections.abc import Sequence
from datetime import datetime

from . import backends
from .backends import get_default_output_filename
from .interfaces import MeasurementDataRecorder
from .measurements import DictMetadata, FloatMetadata, IntMetadata, Measurement, MetadataBase, StringMetadata, TestPhase
from .recorders import CPUInfoRecorder, GPUInfoRecorder, MemoryInfoRecorder, VersionInfoRecorder

logger = logging.getLogger(__name__)

# Valid measurement and metadata class names (to support both isaaclab and isaacsim types)
_MEASUREMENT_CLASS_NAMES = {
    "Measurement",
    "SingleMeasurement",
    "StatisticalMeasurement",
    "BooleanMeasurement",
    "DictMeasurement",
    "ListMeasurement",
}
_METADATA_CLASS_NAMES = {"MetadataBase", "StringMetadata", "IntMetadata", "FloatMetadata", "DictMetadata"}


def _is_measurement_type(obj: object) -> bool:
    """Check if object is a measurement type by class name (supports isaacsim types)."""
    return type(obj).__name__ in _MEASUREMENT_CLASS_NAMES


def _is_metadata_type(obj: object) -> bool:
    """Check if object is a metadata type by class name (supports isaacsim types)."""
    return type(obj).__name__ in _METADATA_CLASS_NAMES


class BaseIsaacLabBenchmark:
    """Base benchmark class for IsaacLab's benchmarks."""

    def __init__(
        self,
        benchmark_name: str,
        backend_type: str,
        output_path: str,
        use_recorders: bool = True,
        output_prefix: str | None = None,
        workflow_metadata: dict | None = None,
        frametime_recorders: bool = False,
    ):
        """Initialize common benchmark state and recorders.

        Args:
            benchmark_name: Name of benchmark to use in outputs.
            backend_type: Type of backend used to collect and print metrics.
            output_path: Path to output directory.
            use_recorders: Whether to use recorders to collect metrics. Defaults to True.
            output_filename: Filename to use for the output file, defaults to None.
            workflow_metadata: Metadata describing benchmark, defaults to None.
            frametime_recorders: Whether to use frametime recorders to collect metrics. Defaults to True.
        """
        self.benchmark_name = benchmark_name

        # Resolve output path
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except Exception as e:
                raise ValueError(f"Could not create output directory {output_path}: {e}")
        self.output_path = output_path
        if output_prefix is None:
            output_prefix = "benchmark"
            logger.warning("No output prefix provided, using default prefix: benchmark")
        self.output_prefix = get_default_output_filename(output_prefix)

        # Get metrics backend
        logger.info("Using metrics backend = %s", backend_type)
        self._metrics = backends.MetricsBackend.get_instance(instance_type=backend_type)
        self._phases: dict[str, TestPhase] = {}

        # Generate workflow-level metadata
        workflow_name = StringMetadata(name="workflow_name", data=self.benchmark_name)
        timestamp = StringMetadata(name="timestamp", data=datetime.now().isoformat())
        self.add_measurement("benchmark_info", metadata=workflow_name)
        self.add_measurement("benchmark_info", metadata=timestamp)
        if workflow_metadata:
            if "metadata" in workflow_metadata:
                self.add_measurement("benchmark_info", metadata=self._metadata_from_dict(workflow_metadata))
            else:
                logger.warning(
                    "workflow_metadata provided, but missing expected 'metadata' entry. Metadata will not be read."
                )

        # Whether to use recorders to collect metrics.
        self._use_recorders = use_recorders
        self._use_frametime_recorders = frametime_recorders

        # Initialize frametime recorders dict (always, even when not using recorders)
        self._frametime_recorders: dict[str, MeasurementDataRecorder] = {}

        if self._use_recorders:
            # Recorders that need to be updated manually since they don't depend on the kit timeline.
            self._manual_recorders: dict[str, MeasurementDataRecorder] = {
                "CPUInfo": CPUInfoRecorder(),
                "GPUInfo": GPUInfoRecorder(),
                "MemoryInfo": MemoryInfoRecorder(),
                "VersionInfo": VersionInfoRecorder(),
            }

            # If we're using Kit, then we can use IsaacSim's benchmark services to peak into the frametimes.
            if self._use_frametime_recorders:
                try:
                    # Enable the benchmark services extension first
                    from isaacsim.core.utils.extensions import enable_extension

                    enable_extension("isaacsim.benchmark.services")

                    from isaacsim.benchmark.services.datarecorders.app_frametime import AppFrametimeRecorder
                    from isaacsim.benchmark.services.datarecorders.gpu_frametime import GPUFrametimeRecorder
                    from isaacsim.benchmark.services.datarecorders.physics_frametime import PhysicsFrametimeRecorder
                    from isaacsim.benchmark.services.datarecorders.render_frametime import RenderFrametimeRecorder

                    self._frametime_recorders["PhysicsFrametime"] = PhysicsFrametimeRecorder()
                    self._frametime_recorders["RenderFrametime"] = RenderFrametimeRecorder()
                    self._frametime_recorders["AppFrametime"] = AppFrametimeRecorder()
                    self._frametime_recorders["GPUFrametime"] = GPUFrametimeRecorder()
                except ImportError as e:
                    logger.warning(
                        f"Could not import kit recorders: {e}. Kit related measurements will not be available."
                    )

                # Start collecting frametime recorders.
                for recorder in self._frametime_recorders.values():
                    recorder.start_collecting()

        # Set the start time of the benchmark.
        logger.info("Starting")
        self.benchmark_start_time = time.time()

    @property
    def output_file_path(self) -> str:
        """Get the full path to the output file."""
        return os.path.join(self.output_path, f"{self.output_prefix}.json")

    def _metadata_from_dict(self, metadata_dict: dict) -> list[MetadataBase]:
        """Convert a dictionary with metadata lists into a list of MetadataBase objects.

        Example:
        .. code-block:: python
            metadata = self._metadata_from_dict({"metadata": [{"name": "gpu", "data": "A10"}]})

        Args:
            metadata_dict: A dictionary with metadata lists.

        Returns:
            A list of MetadataBase objects.
        """
        metadata: list[MetadataBase] = []
        metadata_mapping = {str: StringMetadata, int: IntMetadata, float: FloatMetadata, dict: DictMetadata}
        for meas in metadata_dict["metadata"]:
            if "data" in meas:
                metadata_type = metadata_mapping.get(type(meas["data"]))
                if metadata_type:
                    curr_meta = metadata_type(name=meas["name"], data=meas["data"])
                    metadata.append(curr_meta)
        return metadata

    def update_manual_recorders(self) -> None:
        """Update manual recorders that don't depend on the kit timeline."""

        if not self._use_recorders:
            logger.warning("Recorders are not enabled. Skipping update of manual recorders.")
            return

        for recorder in self._manual_recorders.values():
            recorder.update()

    def add_measurement(
        self,
        phase_name: str,
        measurement: Measurement | Sequence[Measurement] | None = None,
        metadata: MetadataBase | Sequence[MetadataBase] | None = None,
    ) -> None:
        """Add a measurement to the benchmark.

        Args:
            phase_name: The name of the phase to add the measurement to.
            measurement: The measurement to add.
            metadata: The metadata to add.
        """
        if phase_name not in self._phases:
            self._phases[phase_name] = TestPhase(phase_name=phase_name)
            # Add required phase metadata for backends
            phase_metadata = StringMetadata(name="phase", data=phase_name)
            workflow_metadata = StringMetadata(name="workflow_name", data=self.benchmark_name)
            self._phases[phase_name].metadata.extend([phase_metadata, workflow_metadata])

        if measurement:
            if isinstance(measurement, Sequence):
                # Check that all the elements are of type Measurement
                for m in measurement:
                    if not _is_measurement_type(m):
                        raise ValueError(f"Measurement element {m} is not of type Measurement")
                self._phases[phase_name].measurements.extend(measurement)
            else:
                # Check that the element is of type Measurement
                if not _is_measurement_type(measurement):
                    raise ValueError(f"Measurement element {measurement} is not of type Measurement")
                self._phases[phase_name].measurements.append(measurement)
        if metadata:
            if isinstance(metadata, Sequence):
                # Check that all the elements are of type MetadataBase
                for m in metadata:
                    if not _is_metadata_type(m):
                        raise ValueError(f"Metadata element {m} is not of type MetadataBase")
                self._phases[phase_name].metadata.extend(metadata)
            else:
                # Check that the element is of type MetadataBase
                if not _is_metadata_type(metadata):
                    raise ValueError(f"Metadata element {metadata} is not of type MetadataBase")
                self._phases[phase_name].metadata.append(metadata)

    def _finalize_impl(self) -> None:
        # Stop collecting frametime recorders.
        for recorder in self._frametime_recorders.values():
            recorder.stop_collecting()

        # Add measurements and metadata from recorders to the phases.
        if self._use_recorders:
            for recorder_name, measurement_data in self._manual_recorders.items():
                data = measurement_data.get_data()
                # Add measurements to runtime phase if present
                if data.measurements:
                    self.add_measurement("runtime", measurement=data.measurements)
                # Add metadata to appropriate phase (even if no measurements)
                if data.metadata:
                    if recorder_name == "VersionInfo":
                        self.add_measurement("version_info", metadata=data.metadata)
                    else:
                        self.add_measurement("hardware_info", metadata=data.metadata)
            for recorder_name, measurement_data in self._frametime_recorders.items():
                data = measurement_data.get_data()
                # Add measurements to runtime phase if present
                if data.measurements:
                    self.add_measurement("frametime", measurement=data.measurements)

        # Check that there are phases to write.
        if not self._phases:
            logger.warning("No phases collected.No metrics will be written.")
            return

        # Add the phases to the metrics backend.
        for phase in self._phases.values():
            self._metrics.add_metrics(phase)

        self._metrics.finalize(self.output_path, self.output_prefix)
        self._manual_recorders = None
        self._frametime_recorders = None
