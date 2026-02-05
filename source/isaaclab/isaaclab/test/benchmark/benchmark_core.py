import logging
import time
import os
from collections.abc import Sequence
from datetime import datetime

from . import backends
from .interfaces import MeasurementDataRecorder
from .recorders import CPUInfoRecorder, GPUInfoRecorder, MemoryInfoRecorder, VersionInfoRecorder
from .measurements import MetadataBase, StringMetadata, DictMetadata, IntMetadata, FloatMetadata, Measurement, TestPhase


logger = logging.getLogger(__name__)


def get_default_output_filename(prefix: str = "benchmark") -> str:
    """Generate default output filename with current date and time.

    Args:
        prefix: Prefix for the filename (e.g., "articulation_benchmark").

    Returns:
        Filename string with timestamp.
    """
    datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{datetime_str}.json"


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
    ):
        """Initialize common benchmark state and recorders.

        Args:
            benchmark_name: Name of benchmark to use in outputs.
            backend_type: Type of backend used to collect and print metrics.
            output_path: Path to output directory.
            use_recorders: Whether to use recorders to collect metrics. Defaults to True.
            output_filename: Filename to use for the output file, defaults to None.
            workflow_metadata: Metadata describing benchmark, defaults to None.
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
            logger.warning(f"No output prefix provided, using default prefix: benchmark")
        self.output_prefix = get_default_output_filename(output_prefix)
        self.output_file_path = os.path.join(self.output_path, self.output_prefix)

        # Get metrics backend
        logger.info("Using metrics backend = %s", backend_type)
        self._metrics = backends.MetricsBackend.get_instance(instance_type=backend_type)
        self._phases: dict[str, TestPhase] = {}

        # Generate workflow-level metadata
        workflow_name = StringMetadata(name="workflow_name", data=self.benchmark_name)
        timestamp = StringMetadata(name="timestamp", data=datetime.now().isoformat())
        self.add_measurement("benchmark_info", metadata = workflow_name)
        self.add_measurement("benchmark_info", metadata = timestamp)
        if workflow_metadata:
            if "metadata" in workflow_metadata:
                self.add_measurement("benchmark_info", metadata = self._metadata_from_dict(workflow_metadata))
            else:
                logger.warning(
                    "workflow_metadata provided, but missing expected 'metadata' entry. Metadata will not be read."
                )

        # Whether to use recorders to collect metrics.
        self._use_recorders = use_recorders
        if self._use_recorders:
            # Recorders that need to be updated manually since they don't depend on the kit timeline.
            self._manual_recorders: dict[str, MeasurementDataRecorder] = {
                "CPUInfo": CPUInfoRecorder(),
                "GPUInfo": GPUInfoRecorder(),
                "MemoryInfo": MemoryInfoRecorder(),
                "VersionInfo": VersionInfoRecorder(),
            }

            # If we're using Kit, then we can use IsaacSim's benchmark services to peak into the frametimes.
            self._automatic_recorders: dict[str, MeasurementDataRecorder] = {}
            try:
                from isaacsim.benchmark.services.datarecorders import physics_frametime
                from isaacsim.benchmark.services.datarecorders import render_frametime
                from isaacsim.benchmark.services.datarecorders import app_frametime
                self._automatic_recorders["PhysicsFrametime"] = physics_frametime()
                self._automatic_recorders["RenderFrametime"] = render_frametime()
                self._automatic_recorders["AppFrametime"] = app_frametime()
            except ImportError:
                logger.warning("Could not import kit recorders. Kit related measurements will not be available.")

        # Set the start time of the benchmark.
        logger.info("Starting")
        self.benchmark_start_time = time.time()

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


    def add_measurement(self,
        phase_name: str,
        measurement: Measurement | Sequence[Measurement] | None = None,
        metadata: MetadataBase | Sequence[MetadataBase] | None = None
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
                    if not isinstance(m, Measurement):
                        raise ValueError(f"Measurement element {m} is not of type Measurement")
                self._phases[phase_name].measurements.extend(measurement)
            else:
                # Check that the element is of type Measurement
                if not isinstance(measurement, Measurement):
                    raise ValueError(f"Measurement element {measurement} is not of type Measurement")
                self._phases[phase_name].measurements.append(measurement)
        if metadata:
            if isinstance(metadata, Sequence):
                # Check that all the elements are of type MetadataBase
                for m in metadata:
                    if not isinstance(m, MetadataBase):
                        raise ValueError(f"Metadata element {m} is not of type MetadataBase")
                self._phases[phase_name].metadata.extend(metadata)
            else:
                # Check that the element is of type MetadataBase
                if not isinstance(metadata, MetadataBase):
                    raise ValueError(f"Metadata element {metadata} is not of type MetadataBase")
                self._phases[phase_name].metadata.append(metadata)


    def _finalize_impl(self) -> None:
        # Add measurements and metadata from recorders to the phases.
        if self._use_recorders:
            recorders: dict[str, MeasurementDataRecorder] = {**self._manual_recorders, **self._automatic_recorders}
            for recorder_name, measurement_data in recorders.items():
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

        # Check that there are phases to write.
        if not self._phases:
            logger.warning(
                "No phases collected."
                "No metrics will be written."
            )
            return

        # Add the phases to the metrics backend.
        for phase in self._phases.values():
            self._metrics.add_metrics(phase)

        self._metrics.finalize(self.output_path)
        self._manual_recorders = None
        self._automatic_recorders = None