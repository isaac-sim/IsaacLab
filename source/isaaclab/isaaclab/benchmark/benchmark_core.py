# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import logging
import os
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.utils import has_kit

from . import formatters
from .formatters import get_default_output_filename
from .interfaces import MeasurementDataRecorder
from .measurements import ListMeasurement, Measurement, MetadataBase, SingleMeasurement, StringMetadata, TestPhase
from .recorders import CPUInfoRecorder, GPUInfoRecorder, MemoryInfoRecorder, VersionInfoRecorder

if TYPE_CHECKING:
    from .schema import (
        LearningCurve,
        MeanStd,
        PlayBundle,
        Runtime,
        RuntimeBundle,
        StartupBundle,
        TrainingBundle,
    )

logger = logging.getLogger(__name__)

# Measurement and metadata types are matched by class name so Isaac Sim's equivalents are accepted too.
_MEASUREMENT_CLASS_NAMES = {
    "Measurement",
    "SingleMeasurement",
    "StatisticalMeasurement",
    "BooleanMeasurement",
    "DictMeasurement",
    "ListMeasurement",
}
_METADATA_CLASS_NAMES = {"MetadataBase", "StringMetadata", "IntMetadata", "FloatMetadata", "DictMetadata"}

# Isaac Sim frametime recorders, tried individually so partial availability still yields metrics.
_FRAMETIME_RECORDERS = (
    ("PhysicsFrametime", "isaacsim.benchmark.services.datarecorders.physics_frametime", "PhysicsFrametimeRecorder"),
    ("RenderFrametime", "isaacsim.benchmark.services.datarecorders.render_frametime", "RenderFrametimeRecorder"),
    ("AppFrametime", "isaacsim.benchmark.services.datarecorders.app_frametime", "AppFrametimeRecorder"),
    ("GPUFrametime", "isaacsim.benchmark.services.datarecorders.gpu_frametime", "GPUFrametimeRecorder"),
)


def _extend_validated(target: list, items: object, class_names: set[str], label: str) -> None:
    """Append one or more items to ``target`` after checking their class names."""
    items = list(items) if isinstance(items, Sequence) else [items]
    for item in items:
        if type(item).__name__ not in class_names:
            raise ValueError(f"{label} element {item} is not of type {label}")
    target.extend(items)


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

    timing = runtime.environment_step_timing
    serialized_diagnostic = timing is not None and timing.measurement_mode == "serialized_synchronized"
    metric_prefix = "Serialized Diagnostic " if serialized_diagnostic else ""
    runtime_metrics: list[Measurement] = [
        SingleMeasurement(name="Iterations Completed", value=runtime.iterations_completed, unit="count"),
        SingleMeasurement(name=f"{metric_prefix}Total Wall Time", value=runtime.total_wall_time_s, unit="s"),
        SingleMeasurement(name="Steps per Iteration", value=runtime.steps_per_iteration, unit="frames"),
    ]
    runtime_metrics.extend(_stat_measurements(f"{metric_prefix}Iteration Time", runtime.iteration_time_s, "ms", 1000.0))
    runtime_metrics.extend(_stat_measurements(f"{metric_prefix}Collection FPS", runtime.collection_fps, "FPS"))
    runtime_metrics.extend(_stat_measurements(f"{metric_prefix}Total FPS", runtime.total_fps, "FPS"))
    if timing is not None:
        step_rate_label = (
            "Environment Step Host-Return FPS"
            if timing.measurement_mode == "host_return"
            else "Serialized Synchronized Environment Step FPS"
        )
        runtime_metrics.extend(_stat_measurements(step_rate_label, timing.environment_step_fps, "FPS"))
        if timing.simulation_step_time_s is not None:
            assert timing.outside_simulation_step_time_s is not None
            assert timing.outside_simulation_step_fraction is not None
            runtime_metrics.extend(
                _stat_measurements(
                    "Synchronized Simulation Time per Environment Step",
                    timing.simulation_step_time_s,
                    "ms",
                    1000.0,
                )
            )
            runtime_metrics.extend(
                _stat_measurements(
                    "Outside Simulation Time per Environment Step",
                    timing.outside_simulation_step_time_s,
                    "ms",
                    1000.0,
                )
            )
            runtime_metrics.append(
                SingleMeasurement(
                    name="Outside Simulation Step Fraction",
                    value=timing.outside_simulation_step_fraction,
                    unit="ratio",
                )
            )
    runtime_metrics.extend(
        _stat_measurements(f"{metric_prefix}Iterations per Second", runtime.iterations_per_s, "iterations/s")
    )
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
    from .schema import PlayBundle, StartupBundle, TrainingBundle

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

        try:
            # ``exist_ok`` also covers concurrent ranks of a multi-GPU benchmark racing to
            # create the shared output directory.
            os.makedirs(output_path, exist_ok=True)
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

        self.add_measurement("benchmark_info", metadata=StringMetadata(name="workflow_name", data=benchmark_name))
        self.add_measurement(
            "benchmark_info", metadata=StringMetadata(name="timestamp", data=datetime.now().isoformat())
        )
        if workflow_metadata:
            if "metadata" in workflow_metadata:
                self.add_measurement("benchmark_info", metadata=TestPhase.metadata_from_dict(workflow_metadata))
            else:
                logger.warning(
                    "workflow_metadata provided, but missing expected 'metadata' entry. Metadata will not be read."
                )

        self._use_recorders = use_recorders
        self._use_frametime_recorders = frametime_recorders
        self._frametime_recorders: dict[str, MeasurementDataRecorder] = {}

        if use_recorders:
            # Recorders sampled explicitly, since they do not depend on the Kit timeline.
            self._manual_recorders: dict[str, MeasurementDataRecorder] = {
                "CPUInfo": CPUInfoRecorder(),
                "GPUInfo": GPUInfoRecorder(),
                "MemoryInfo": MemoryInfoRecorder(),
                "VersionInfo": VersionInfoRecorder(),
            }
            # Frametime recorders come from Isaac Sim's benchmark services, so they are optional
            # and only attempted while Kit is running.
            if frametime_recorders and not has_kit():
                logger.warning("Kit is not running. Kit related measurements will not be available.")
            elif frametime_recorders:
                self._start_frametime_recorders()

        logger.info("Starting")
        self.benchmark_start_time = time.time()

    def _start_frametime_recorders(self) -> None:
        """Create and start the Kit frametime recorders that are importable in this runtime."""
        try:
            from isaaclab.sim.utils import enable_extension

            enable_extension("isaacsim.benchmark.services")
            for key, module_name, class_name in _FRAMETIME_RECORDERS:
                try:
                    module = importlib.import_module(module_name)
                    self._frametime_recorders[key] = getattr(module, class_name)()
                except Exception as e:
                    logger.debug(f"{key} recorder unavailable: {e}")
            if not self._frametime_recorders:
                # Older Isaac Sim packaging bundles every frametime recorder in one module.
                try:
                    from isaacsim.benchmark.services.datarecorders.interface import InputContext
                    from isaacsim.benchmark.services.recorders import IsaacFrameTimeRecorder

                    self._frametime_recorders["IsaacFrameTime"] = IsaacFrameTimeRecorder(
                        context=InputContext(phase="frametime"), gpu_frametime=False
                    )
                except ImportError as e:
                    logger.warning(
                        f"Could not import bundled frametime recorder: {e}."
                        " Frametime measurements will not be available."
                    )
        # Kit may stop after the availability check above; the non-Kit recorders remain usable.
        except (ImportError, RuntimeError) as e:
            logger.warning(
                f"Could not initialize Kit frametime recorders: {e}. Kit related measurements will not be available."
            )

        for recorder in self._frametime_recorders.values():
            recorder.start_collecting()

    @property
    def output_file_path(self) -> str:
        """Get the full path to the output file."""
        return os.path.join(self.output_path, f"{self.output_prefix}.json")

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
            # Formatters read the phase and workflow names back from the phase metadata.
            self._phases[phase_name] = TestPhase(
                phase_name=phase_name,
                metadata=[
                    StringMetadata(name="phase", data=phase_name),
                    StringMetadata(name="workflow_name", data=self.benchmark_name),
                ],
            )
        phase = self._phases[phase_name]
        if measurement:
            _extend_validated(phase.measurements, measurement, _MEASUREMENT_CLASS_NAMES, "Measurement")
        if metadata:
            _extend_validated(phase.metadata, metadata, _METADATA_CLASS_NAMES, "MetadataBase")

    def finalize(self) -> tuple[Path, ...]:
        """Finalize metric collection and write selected formatter outputs.

        Returns:
            Files written by the selected formatters.
        """
        return self._finalize_impl()

    def _finalize_impl(self) -> tuple[Path, ...]:
        if self._bundle is None and any(key == "schema" for key, _ in self._metrics):
            raise RuntimeError("The schema formatter requires an attached benchmark bundle.")

        for recorder in self._frametime_recorders.values():
            recorder.stop_collecting()

        if self._use_recorders:
            for recorder_name, recorder in self._manual_recorders.items():
                data = recorder.get_data()
                if data.measurements:
                    self.add_measurement("runtime", measurement=data.measurements)
                if data.metadata:
                    phase_name = "version_info" if recorder_name == "VersionInfo" else "hardware_info"
                    self.add_measurement(phase_name, metadata=data.metadata)
            for recorder in self._frametime_recorders.values():
                data = recorder.get_data()
                if data.measurements:
                    self.add_measurement("frametime", measurement=data.measurements)

        if not self._phases:
            logger.warning("No phases collected. No metrics will be written.")
            return ()

        # Add the phases to each metrics formatter and write its output file. When more than one
        # formatter is selected, suffix the filename with the formatter key so they don't collide on
        # the shared ".json" extension.
        multi = len(self._metrics) > 1
        output_paths: list[Path] = []
        for formatter_key, metrics in self._metrics:
            for phase in self._phases.values():
                metrics.add_metrics(phase)
            filename = f"{self.output_prefix}_{formatter_key}" if multi else self.output_prefix
            metrics.finalize(self.output_path, filename, bundle=self._bundle)
            if formatter_key == "osmo" and len(self._phases) > 1:
                output_paths.extend(
                    Path(self.output_path) / f"{filename}_{phase_name}.json" for phase_name in self._phases
                )
            else:
                output_paths.append(Path(self.output_path) / f"{filename}.json")
        self._manual_recorders = None
        self._frametime_recorders = None
        return tuple(output_paths)
