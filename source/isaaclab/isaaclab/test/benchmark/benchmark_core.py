# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import os
import time
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING

from isaaclab.test.benchmark import formatters
from isaaclab.test.benchmark.formatters import get_default_output_filename
from isaaclab.test.benchmark.interfaces import MeasurementDataRecorder
from isaaclab.test.benchmark.measurements import (
    DictMetadata,
    FloatMetadata,
    IntMetadata,
    ListMeasurement,
    Measurement,
    MetadataBase,
    SingleMeasurement,
    StringMetadata,
    TestPhase,
)
from isaaclab.test.benchmark.recorders import CPUInfoRecorder, GPUInfoRecorder, MemoryInfoRecorder, VersionInfoRecorder

if TYPE_CHECKING:
    from isaaclab.test.benchmark.schema import (
        LearningCurve,
        MeanStd,
        PlayBundle,
        Runtime,
        RuntimeBundle,
        StartupBundle,
        TrainingBundle,
    )

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


def _stat_measurements(name: str, stats: "MeanStd", unit: str, scale: float = 1.0) -> list[Measurement]:
    """Convert a schema aggregate to flat scalar measurements."""
    measurements: list[Measurement] = [
        SingleMeasurement(name=f"Mean {name}", value=stats.mean * scale, unit=unit),
        SingleMeasurement(name=f"Std {name}", value=stats.std * scale, unit=unit),
    ]
    if stats.peak is not None:
        measurements.append(SingleMeasurement(name=f"Max {name}", value=stats.peak * scale, unit=unit))
    return measurements


def _runtime_measurements(runtime: "Runtime") -> dict[str, list[Measurement]]:
    """Convert schema runtime metrics to startup and runtime phases."""
    startup_fields = (
        ("app_launch", "App Launch Time"),
        ("python_imports", "Python Imports Time"),
        ("task_config", "Task Creation and Start Time"),
        ("env_creation", "Scene Creation Time"),
        ("first_step", "Simulation Start Time"),
    )
    startup = [
        SingleMeasurement(name=label, value=value * 1000.0, unit="ms")
        for field, label in startup_fields
        if (value := getattr(runtime.startup_time_s, field)) is not None
    ]
    if startup:
        startup.append(
            SingleMeasurement(
                name="Total Start Time (Launch to Train)",
                value=sum(float(measurement.value) for measurement in startup),
                unit="ms",
            )
        )

    runtime_metrics: list[Measurement] = [
        SingleMeasurement(name="Iterations Completed", value=runtime.iterations_completed, unit="count"),
        SingleMeasurement(name="Total Wall Time", value=runtime.total_wall_time_s, unit="s"),
        SingleMeasurement(name="Steps per Iteration", value=runtime.steps_per_iteration, unit="frames"),
    ]
    runtime_metrics.extend(_stat_measurements("Iteration Time", runtime.iteration_time_s, "ms", 1000.0))
    runtime_metrics.extend(_stat_measurements("Collection FPS", runtime.collection_fps, "FPS"))
    runtime_metrics.extend(_stat_measurements("Total FPS", runtime.total_fps, "FPS"))
    runtime_metrics.extend(_stat_measurements("Iterations per Second", runtime.iterations_per_s, "iterations/s"))
    return {"startup": startup, "runtime": runtime_metrics}


def _curve_measurements(label: str, curve: "LearningCurve", ema_alpha: float) -> list[Measurement]:
    """Convert one training curve to scalar and optional series measurements."""
    measurements: list[Measurement] = [
        SingleMeasurement(name=f"Last {label}", value=curve.final_raw, unit="float"),
        SingleMeasurement(name=f"EMA {ema_alpha:g} {label}", value=curve.final_ema, unit="float"),
    ]
    if curve.series_per_iter is not None:
        plural = "Rewards" if label == "Reward" else "Episode Lengths"
        measurements.append(ListMeasurement(name=plural, value=curve.series_per_iter))
        if curve.series_per_iter:
            measurements.append(SingleMeasurement(name=f"Max {plural}", value=max(curve.series_per_iter), unit="float"))
    return measurements


def _measurements_from_bundle(
    bundle: "RuntimeBundle | TrainingBundle | StartupBundle | PlayBundle",
) -> dict[str, list[Measurement]]:
    """Project a typed bundle into flat phases for non-schema formatters."""
    from isaaclab.test.benchmark.schema import PlayBundle, StartupBundle, TrainingBundle

    if isinstance(bundle, StartupBundle):
        projected: dict[str, list[Measurement]] = {}
        for phase_name, phase in bundle.phases.items():
            measurements: list[Measurement] = [
                SingleMeasurement(name="Wall Clock Time", value=phase.total_time_s, unit="s")
            ]
            for function in phase.top_functions:
                measurements.extend(
                    [
                        SingleMeasurement(name=f"{function.name} Own Time", value=function.own_time_s, unit="s"),
                        SingleMeasurement(name=f"{function.name} Cumulative Time", value=function.cum_time_s, unit="s"),
                        SingleMeasurement(name=f"{function.name} Calls", value=function.calls, unit="count"),
                    ]
                )
            projected[phase_name] = measurements
        return projected

    projected = _runtime_measurements(bundle.runtime)
    if isinstance(bundle, TrainingBundle):
        train = _curve_measurements("Reward", bundle.learning.reward, bundle.learning.ema_alpha)
        train.extend(_curve_measurements("Episode Length", bundle.learning.ep_length, bundle.learning.ema_alpha))
        if bundle.success_rate is not None:
            train.append(SingleMeasurement(name="success_rate", value=bundle.success_rate, unit="float"))
        projected["train"] = train
    elif isinstance(bundle, PlayBundle):
        play: list[Measurement] = []
        if bundle.reward is not None:
            play.extend(_stat_measurements("Reward", bundle.reward, "float"))
        if bundle.ep_length is not None:
            play.extend(_stat_measurements("Episode Length", bundle.ep_length, "steps"))
        if bundle.success_rate is not None:
            play.append(SingleMeasurement(name="success_rate", value=bundle.success_rate, unit="float"))
        if play:
            projected["play"] = play
    return projected


class BaseIsaacLabBenchmark:
    """Base benchmark class for IsaacLab's benchmarks."""

    def __init__(
        self,
        benchmark_name: str,
        formatter_type: str | list[str] | None = None,
        output_path: str | None = None,
        use_recorders: bool = True,
        output_prefix: str | None = None,
        workflow_metadata: dict | None = None,
        frametime_recorders: bool = False,
        backend_type: str | list[str] | None = None,
    ):
        """Initialize common benchmark state and recorders.

        Args:
            benchmark_name: Name of benchmark to use in outputs.
            formatter_type: Formatter(s) used to collect and print metrics. Accepts a single
                type name, a list of type names, or a comma-separated string (e.g.
                ``"schema,omniperf"``); each selected formatter writes its own output file.
            output_path: Path to output directory.
            use_recorders: Whether to use recorders to collect metrics. Defaults to True.
            output_prefix: Prefix used to generate the output filename. Defaults to ``None``.
            workflow_metadata: Metadata describing benchmark, defaults to None.
            frametime_recorders: Whether to use frametime recorders to collect metrics. Defaults to ``False``.
            backend_type: Alias for :paramref:`formatter_type`.
        """
        if formatter_type is None:
            formatter_type = backend_type or "omniperf"
        elif backend_type is not None and backend_type != formatter_type:
            raise ValueError("Specify either formatter_type or backend_type, not both.")
        if output_path is None:
            raise ValueError("output_path must be provided.")

        self.benchmark_name = benchmark_name

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

        if isinstance(formatter_type, str):
            formatter_type = [t.strip() for t in formatter_type.split(",") if t.strip()] or ["omniperf"]
        formatter_type = list(dict.fromkeys(formatter_type))
        logger.info("Using metrics formatters = %s", formatter_type)
        self._metrics = [(t, formatters.MetricsFormatter.get_instance(instance_type=t)) for t in formatter_type]
        self._bundle = None
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

            # "Kit-full" means Isaac Sim (Kit) runtime is available, so benchmark services can
            # provide frametime recorders. "Kit-less" means those services are absent; we should
            # gracefully skip or fall back while still allowing mixed modes (e.g., kit-full physics
            # with kit-less rendering).
            # If we're using Kit, then we can use IsaacSim's benchmark services to peek into the frametimes.
            if self._use_frametime_recorders:
                try:
                    # Enable the benchmark services extension first
                    from isaacsim.core.experimental.utils.app import enable_extension

                    enable_extension("isaacsim.benchmark.services")

                    added_any = False

                    # Try individual recorders first so we can collect partial metrics when only some are available.
                    try:
                        from isaacsim.benchmark.services.datarecorders.physics_frametime import PhysicsFrametimeRecorder

                        self._frametime_recorders["PhysicsFrametime"] = PhysicsFrametimeRecorder()
                        added_any = True
                    except (ImportError, Exception) as e:
                        logger.debug(f"Physics frametime recorder unavailable: {e}")

                    try:
                        from isaacsim.benchmark.services.datarecorders.render_frametime import RenderFrametimeRecorder

                        self._frametime_recorders["RenderFrametime"] = RenderFrametimeRecorder()
                        added_any = True
                    except (ImportError, Exception) as e:
                        logger.debug(f"Render frametime recorder unavailable: {e}")

                    try:
                        from isaacsim.benchmark.services.datarecorders.app_frametime import AppFrametimeRecorder

                        self._frametime_recorders["AppFrametime"] = AppFrametimeRecorder()
                        added_any = True
                    except (ImportError, Exception) as e:
                        logger.debug(f"App frametime recorder unavailable: {e}")

                    try:
                        from isaacsim.benchmark.services.datarecorders.gpu_frametime import GPUFrametimeRecorder

                        self._frametime_recorders["GPUFrametime"] = GPUFrametimeRecorder()
                        added_any = True
                    except (ImportError, Exception) as e:
                        logger.debug(f"GPU frametime recorder unavailable: {e}")

                    if not added_any:
                        # Fallback for Isaac Sim packaging that bundles frametime recorders in a single module.
                        try:
                            from isaacsim.benchmark.services.datarecorders.interface import InputContext
                            from isaacsim.benchmark.services.recorders import IsaacFrameTimeRecorder

                            context = InputContext(phase="frametime")
                            self._frametime_recorders["IsaacFrameTime"] = IsaacFrameTimeRecorder(
                                context=context, gpu_frametime=False
                            )
                        except ImportError as e:
                            logger.warning(
                                "Could not import bundled frametime recorder: "
                                f"{e}. Frametime measurements will not be available."
                            )
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

    def attach_bundle(self, bundle: "RuntimeBundle | TrainingBundle | StartupBundle | PlayBundle | None") -> None:
        """Attach a typed bundle for schema serialization and flat-formatter projection.

        Args:
            bundle: Runtime, training, startup, or play benchmark bundle.
        """
        self._bundle = bundle
        if bundle is not None:
            for phase_name, measurements in _measurements_from_bundle(bundle).items():
                self.add_measurement(phase_name, measurement=measurements)

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
            # Add required phase metadata for formatters
            phase_metadata = StringMetadata(name="phase", data=phase_name)
            workflow_metadata = StringMetadata(name="workflow_name", data=self.benchmark_name)
            self._phases[phase_name].metadata.extend([phase_metadata, workflow_metadata])

        if measurement:
            if isinstance(measurement, Sequence):
                for m in measurement:
                    if not _is_measurement_type(m):
                        raise ValueError(f"Measurement element {m} is not of type Measurement")
                self._phases[phase_name].measurements.extend(measurement)
            else:
                if not _is_measurement_type(measurement):
                    raise ValueError(f"Measurement element {measurement} is not of type Measurement")
                self._phases[phase_name].measurements.append(measurement)
        if metadata:
            if isinstance(metadata, Sequence):
                for m in metadata:
                    if not _is_metadata_type(m):
                        raise ValueError(f"Metadata element {m} is not of type MetadataBase")
                self._phases[phase_name].metadata.extend(metadata)
            else:
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

        if not self._phases:
            logger.warning("No phases collected.No metrics will be written.")
            return

        # Add the phases to each metrics formatter and write its output file. When more than one
        # formatter is selected, suffix the filename with the formatter key so they don't collide on
        # the shared ".json" extension.
        multi = len(self._metrics) > 1
        for formatter_key, metrics in self._metrics:
            for phase in self._phases.values():
                metrics.add_metrics(phase)
            filename = f"{self.output_prefix}_{formatter_key}" if multi else self.output_prefix
            metrics.finalize(self.output_path, filename, bundle=self._bundle)
        self._manual_recorders = None
        self._frametime_recorders = None
