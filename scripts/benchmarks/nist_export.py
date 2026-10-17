# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Export saved AlphaNIST observations through the existing OmniPerf formatter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def build_artifact_manifest(run: Path) -> dict:
    """Hash current input files and record missing or unreadable inputs."""
    files = {}
    for name in (
        "invocation.json", "execution.json", "process.json", "throughput.json",
        "latency.json", "memory.json", "outcomes.json", "outcomes-native.json",
        "outcomes-recovered.json", "events.jsonl", "outcomes.jsonl", "nvml.jsonl",
        "output.log", "stdout.log",
    ):
        record = {"path": name, "status": "missing", "sha256": None, "bytes": None}
        try:
            digest, size = hashlib.sha256(), 0
            with (run / name).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            record.update(status="present", sha256=digest.hexdigest(), bytes=size)
        except FileNotFoundError:
            pass
        except OSError as error:
            record.update(status="error", error={"type": type(error).__name__, "message": str(error)})
        files[name] = record
    return {
        "scope": "Current run inputs directly in the run directory; not recursive.",
        "mode_reports": {
            "files": ["throughput.json", "latency.json"],
            "note": "Throughput and latency are separate modes; one of these reports is normally missing.",
        },
        "not_inputs": {
            "previous-invocations/": "Earlier invocations; not current input.",
            "benchmark_nist_omniperf.json": "Output.",
            "benchmark_result.json": "Output.",
            "export.log": "Written after export.",
            "export-status.json": "Written after export.",
        },
        "files": files,
    }


def _read_report(path: Path):
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Nonfinite JSON number: {value}")
        return result

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    try:
        report = json.loads(path.read_text(), parse_float=finite_float, parse_constant=reject_constant)
        if not isinstance(report, dict):
            raise ValueError("Expected a report object")
        return report, {"status": "loaded"}
    except FileNotFoundError:
        return None, {"status": "missing"}
    except (OSError, ValueError) as error:
        return None, {"status": "error", "type": type(error).__name__, "message": str(error)}


def export_run(input_dir: Path, output_path: Path) -> Path:
    """Format existing evidence without measuring the exporter process."""
    from isaaclab.benchmark.benchmark_core import BaseIsaacLabBenchmark
    from isaaclab.benchmark.measurements import DictMetadata, SingleMeasurement

    input_dir, output_path = Path(input_dir), Path(output_path)
    reports, report_status = {}, {}
    for name in ("invocation", "execution", "throughput", "latency", "memory", "outcomes"):
        reports[name], report_status[name] = _read_report(input_dir / f"{name}.json")
    invocation, execution = reports["invocation"] or {}, reports["execution"] or {}
    mode = invocation.get("mode") or execution.get("mode")
    if mode is None:
        available = [name for name in ("throughput", "latency") if reports[name] is not None]
        mode = available[0] if len(available) == 1 else None
    measurement = reports.get(mode) or {}
    outcomes, memory = reports["outcomes"] or {}, reports["memory"] or {}

    benchmark = BaseIsaacLabBenchmark(
        benchmark_name="AlphaNIST", formatter_type="omniperf", output_path=str(output_path),
        output_prefix="benchmark_nist_omniperf", use_recorders=False,
    )

    def metadata(phase, name, value):
        benchmark.add_measurement(phase, metadata=DictMetadata(name=name, data=value))

    def metrics(phase, values, units):
        metadata(phase, "metrics", values)
        metadata(phase, "metric_units", units)
        for name, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                benchmark.add_measurement(
                    phase, measurement=SingleMeasurement(name=name, value=value, unit=units[name]),
                )

    configuration = measurement.get("configuration") or outcomes.get("configuration") or {}
    metadata("benchmark_info", "nist", {
        "schema": "nist-omniperf-v1", "mode": mode,
        "timestamp_scope": "The formatter timestamp is export time, not measurement time.",
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_invocation_id": measurement.get("invocation_id") or outcomes.get("measurement_invocation_id"),
        "memory_observer_invocation_id": (configuration.get("memory_observation") or {}).get("invocation_id"),
        "invocation": reports["invocation"], "execution": reports["execution"],
        "input_reports": report_status, "artifacts": build_artifact_manifest(input_dir),
    })
    metadata("measurement", "native_report", {key: value for key, value in measurement.items() if key != "samples"})
    elapsed = measurement.get("elapsed_seconds") or {}
    metrics("startup", {"native_startup_seconds": elapsed.get("startup")}, {"native_startup_seconds": "s"})
    metadata("startup", "scope", {"description": measurement.get("startup_scope")})
    values = {"measurement_elapsed_seconds": elapsed.get("measurement")}
    units = {"measurement_elapsed_seconds": "s"}
    if mode == "throughput":
        values.update(
            completed_environment_steps=measurement.get("environment_steps"),
            environment_steps_per_second=measurement.get("environment_steps_per_second"),
        )
        units.update(completed_environment_steps="environment steps", environment_steps_per_second="environment steps/s")
    elif mode == "latency":
        statistics = (measurement.get("statistics") or {}).get("all") or {}
        completed = statistics.get("completed_ms") or {}
        for percentile in (50, 95, 99):
            key = f"batched_env_step_p{percentile}_ms"
            values[key], units[key] = completed.get(f"p{percentile}"), "ms"
        values["completed_latency_samples"], units["completed_latency_samples"] = completed.get("n"), "samples"
    metrics("measurement", values, units)

    metadata("memory", "native_report", memory)
    metrics("memory", {"observed_process_peak_bytes": memory.get("observed_process_peak_bytes")},
            {"observed_process_peak_bytes": "bytes"})
    for phase in memory.get("phases", []):
        name = f"memory_{phase['name']}"
        metadata(name, "observation", phase)
        metrics(name, {"observed_peak_bytes": phase.get("observed_peak_bytes")}, {"observed_peak_bytes": "bytes"})

    metadata("assembly", "native_report", outcomes)
    aggregate = outcomes.get("aggregate") or {}
    names = (
        "ended_episodes", "successful_assemblies", "native_failed_episodes", "timed_out_episodes",
        "unfinished_started_episodes", "active_episodes", "success_fraction_of_ended",
    )
    metrics("assembly", {name: aggregate.get(name) for name in names},
            {name: "fraction" if name == "success_fraction_of_ended" else "episodes" for name in names})
    generated, = benchmark.finalize()
    destination = output_path / "benchmark_nist_omniperf.json"
    Path(generated).replace(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()
    print(export_run(args.input_dir, args.output_path))
