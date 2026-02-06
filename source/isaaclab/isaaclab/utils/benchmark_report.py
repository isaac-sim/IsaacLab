"""
Benchmark reporting utilities.

This module provides a lightweight, dependency-minimal way to:
- collect a small set of run metrics (optionally grouped by phases)
- print a human-readable "Summary Report" to stdout at end-of-run
- optionally emit a machine-readable JSON artifact

It is intentionally independent of Kit / Isaac Sim extensions so it can be used
across different physics backends (Newton, oxphysx, etc.) and execution modes.

Integration
-----------
To integrate into a benchmark script (e.g., scripts/benchmarks/benchmark_non_rl.py):

    from isaaclab.utils import BenchmarkReporter

    # At script start (after args parsed)
    reporter = BenchmarkReporter(
        task=args_cli.task,
        profile="benchmark_non_rl",
        output_dir="/tmp",
        identifiers={"num_envs": args_cli.num_envs, "seed": args_cli.seed},
    )
    reporter.start_phase("startup")

    # ... environment creation ...

    reporter.end_phase("startup")
    reporter.start_phase("runtime")

    # ... benchmark loop ...

    reporter.end_phase("runtime")
    reporter.record("runtime", "Mean FPS", mean_fps, unit="FPS")
    reporter.record_series("runtime", "Step times", step_times_ms, unit="ms", compute_stats=True)

    # At end (before env.close())
    reporter.finalize(print_stdout=True, write_json=True)

Remaining work before this can be tested in kitless Isaac Lab runs
------------------------------------------------------------------
The module itself (BenchmarkReporter) is complete and standalone. The integration
work below connects it to the existing benchmark scripts so they work without Kit.

Step 1 — Update ``scripts/benchmarks/utils.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Currently hard-imports Isaac Sim types that don't exist in kitless mode::

    from isaacsim.benchmark.services import BaseIsaacBenchmark                # line 12
    from isaacsim.benchmark.services.metrics.measurements import (            # line 13
        DictMeasurement, ListMeasurement, SingleMeasurement,
    )

Option A (recommended): Make the log_* helpers accept a ``BenchmarkReporter``
*or* ``BaseIsaacBenchmark`` via duck typing / ``Union`` type hint, and wrap the
Isaac Sim imports in a try/except so the file can be imported in kitless mode.

Option B: Create a parallel ``scripts/benchmarks/utils_kitless.py`` that
re-implements the same log_* functions using ``BenchmarkReporter``.

Step 2 — Update (or duplicate) the benchmark scripts for kitless mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Each of the three benchmark scripts currently does this at the top::

    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("isaacsim.benchmark.services")
    from isaacsim.benchmark.services import BaseIsaacBenchmark

These imports will fail in a kitless environment. We need to either:
- Add a try/except fallback that uses ``BenchmarkReporter`` instead, or
- Create kitless-specific benchmark scripts that import ``BenchmarkReporter``
  directly.

Files to modify or duplicate:
  - ``scripts/benchmarks/benchmark_non_rl.py``      (lines 58-62)
  - ``scripts/benchmarks/benchmark_rlgames.py``      (lines 57-61)
  - ``scripts/benchmarks/benchmark_rsl_rl.py``       (lines 88-91)

Step 3 — Add psutil for CPU / USS memory (canonical schema compliance)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The design doc requires CPU usage and USS memory. These need ``psutil``::

    import psutil
    proc = psutil.Process()
    self.process_info["cpu_percent"] = proc.cpu_percent(interval=0.1)
    mem_info = proc.memory_full_info()
    self.process_info["uss_mb"] = mem_info.uss / (1024 * 1024)

Add ``psutil`` to the project's dependencies (e.g., setup.cfg / pyproject.toml)
and uncomment / replace the TODO block in ``finalize()``.

Step 4 — Validate with a real kitless run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run one of the benchmark scripts in kitless mode (no Kit, no Isaac Sim extensions)
and confirm:
  1. The terminal prints a "Summary Report" ASCII table
  2. A ``benchmark_metrics_<run_id>.json`` file is written to ``--output-path``
  3. The JSON contains all canonical schema fields (identifiers, execution,
     performance, system/hardware, schema version)

Note: The canonical schema contract is already enforced by ``to_dict()`` which
always emits ``schema_version``, ``task``, ``profile``, ``identifiers``,
``system``, ``process``, and ``phases``. Downstream consumers can rely on
these fields being present in every ``benchmark_metrics_*.json`` artifact.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _safe_stdev(values: list[float], mean: float) -> float | None:
    if len(values) < 2:
        return None
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance**0.5


def _maybe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _read_proc_meminfo_mb() -> dict[str, float] | None:
    """Best-effort memory info from /proc/meminfo (Linux)."""
    try:
        info_kb: dict[str, float] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                # Example: "MemTotal:       65843092 kB"
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    key = parts[0].rstrip(":")
                    info_kb[key] = float(parts[1])
        # convert common keys to MB
        out: dict[str, float] = {}
        for key in ("MemTotal", "MemAvailable", "MemFree"):
            if key in info_kb:
                out[f"{key}_mb"] = info_kb[key] / 1024.0
        return out or None
    except Exception:
        return None


def _nvidia_smi_query() -> dict[str, str] | None:
    """Best-effort NVIDIA GPU info via nvidia-smi (if present)."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        if not out:
            return None
        # Use first GPU line
        first = out.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) < 3:
            return None
        return {
            "gpu_name": parts[0],
            "gpu_driver_version": parts[1],
            "gpu_memory_total_mb": parts[2],
        }
    except Exception:
        return None


def collect_system_info() -> dict[str, Any]:
    """Collect a minimal system metadata dictionary (best-effort)."""
    uname = platform.uname()
    info: dict[str, Any] = {
        "os": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "cpu": {
            "logical_cores": os.cpu_count(),
        },
    }
    mem = _read_proc_meminfo_mb()
    if mem:
        info["memory"] = mem

    # Prefer torch (when available) for CUDA version + device name; otherwise try nvidia-smi.
    torch_info: dict[str, Any] = {}
    try:
        import torch  # noqa: PLC0415

        torch_info["torch_version"] = torch.__version__
        torch_info["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        if torch.cuda.is_available():
            torch_info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    if torch_info:
        info["gpu"] = torch_info

    smi = _nvidia_smi_query()
    if smi:
        info.setdefault("gpu", {}).update({k: v for k, v in smi.items() if v is not None})
    return info


@dataclass
class Metric:
    name: str
    value: Any
    unit: str | None = None
    kind: Literal["single", "series", "dict"] = "single"
    # For series metrics we optionally compute and store basic stats.
    stats: dict[str, float] | None = None


@dataclass
class Phase:
    name: str
    metrics: list[Metric] = field(default_factory=list)
    started_at_unix_s: float | None = None
    ended_at_unix_s: float | None = None

    @property
    def duration_s(self) -> float | None:
        if self.started_at_unix_s is None or self.ended_at_unix_s is None:
            return None
        return self.ended_at_unix_s - self.started_at_unix_s


class BenchmarkReporter:
    """
    A small in-process reporting utility for benchmarks.

    Typical usage:

        reporter = BenchmarkReporter(task=task, profile="benchmark_non_rl", output_dir=out)
        reporter.start_phase("startup")
        ...
        reporter.end_phase("startup")
        reporter.start_phase("runtime")
        reporter.record_series("runtime", "Environment step time", step_times_ms, unit="ms", compute_stats=True)
        reporter.record("runtime", "Mean FPS", mean_fps, unit="FPS")
        reporter.finalize(print_stdout=True, write_json=True)
    """

    def __init__(
        self,
        *,
        task: str | None,
        profile: str | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        run_id: str | None = None,
        identifiers: dict[str, Any] | None = None,
    ) -> None:
        self.task = task
        self.profile = profile
        self.output_dir = Path(output_dir) if output_dir else None
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.created_at_utc = _utc_now_iso()
        self.identifiers: dict[str, Any] = identifiers.copy() if identifiers else {}

        self._phases: dict[str, Phase] = {}
        self._report_width = 60

        self.system_info = collect_system_info()
        self.process_info = {
            "pid": os.getpid(),
        }

        # Optional torch GPU peak stats.
        self._torch_cuda_peak_enabled = False
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self._torch_cuda_peak_enabled = True
        except Exception:
            pass

    # -----------------
    # Phase management
    # -----------------
    def start_phase(self, name: str) -> None:
        phase = self._phases.get(name)
        if phase is None:
            phase = Phase(name=name)
            self._phases[name] = phase
        phase.started_at_unix_s = time.time()

    def end_phase(self, name: str) -> None:
        phase = self._phases.get(name)
        if phase is None:
            phase = Phase(name=name)
            self._phases[name] = phase
        phase.ended_at_unix_s = time.time()

    # -----------------
    # Metric recording
    # -----------------
    def record(self, phase: str, name: str, value: Any, *, unit: str | None = None) -> None:
        self._ensure_phase(phase).metrics.append(Metric(name=name, value=value, unit=unit, kind="single"))

    def record_dict(self, phase: str, name: str, value: dict[str, Any], *, unit: str | None = None) -> None:
        self._ensure_phase(phase).metrics.append(Metric(name=name, value=value, unit=unit, kind="dict"))

    def record_series(
        self,
        phase: str,
        name: str,
        values: list[Any],
        *,
        unit: str | None = None,
        compute_stats: bool = True,
    ) -> None:
        stats: dict[str, float] | None = None
        if compute_stats:
            floats = [v for v in (_maybe_float(x) for x in values) if v is not None]
            if floats:
                mean_val = float(_safe_mean(floats) or 0.0)
                stdev_val = _safe_stdev(floats, mean_val)
                stats = {
                    "min": float(min(floats)),
                    "max": float(max(floats)),
                    "mean": mean_val,
                }
                if stdev_val is not None:
                    stats["stdev"] = float(stdev_val)
        self._ensure_phase(phase).metrics.append(Metric(name=name, value=values, unit=unit, kind="series", stats=stats))

    def set_identifier(self, key: str, value: Any) -> None:
        self.identifiers[key] = value

    # -------------------------------------------------------------------
    # Compatibility aliases (match Isaac Sim BaseIsaacBenchmark naming)
    # -------------------------------------------------------------------
    def store_custom_measurement(self, phase: str, name: str, value: Any, *, unit: str | None = None) -> None:
        """Alias for ``record()`` — matches the Isaac Sim ``BaseIsaacBenchmark`` API name."""
        self.record(phase, name, value, unit=unit)

    def stop(self, *, print_stdout: bool = True, write_json: bool = True) -> dict[str, Any]:
        """Alias for ``finalize()`` — matches the Isaac Sim ``BaseIsaacBenchmark.stop()`` convention."""
        return self.finalize(print_stdout=print_stdout, write_json=write_json)

    # -------------
    # Finalization
    # -------------
    def finalize(self, *, print_stdout: bool = True, write_json: bool = True) -> dict[str, Any]:
        # Capture end-of-run process usage.
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is KB on Linux, bytes on macOS. We assume Linux in this workspace.
        max_rss_mb = float(usage.ru_maxrss) / 1024.0
        self.process_info["max_rss_mb"] = max_rss_mb

        # TODO(future): Add psutil-based metrics for canonical schema compliance:
        #   import psutil
        #   proc = psutil.Process()
        #   self.process_info["cpu_percent"] = proc.cpu_percent(interval=0.1)
        #   mem_info = proc.memory_full_info()
        #   self.process_info["uss_mb"] = mem_info.uss / (1024 * 1024)

        if self._torch_cuda_peak_enabled:
            try:
                import torch  # noqa: PLC0415

                self.process_info["cuda_max_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated()) / (
                    1024.0 * 1024.0
                )
                self.process_info["cuda_max_memory_reserved_mb"] = float(torch.cuda.max_memory_reserved()) / (
                    1024.0 * 1024.0
                )
            except Exception:
                pass

        payload = self.to_dict()
        if print_stdout:
            self.print_summary()
        if write_json and self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.output_dir / f"benchmark_metrics_{self.run_id}.json"
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return payload

    def to_dict(self) -> dict[str, Any]:
        phases = {k: asdict(v) | {"duration_s": v.duration_s} for k, v in self._phases.items()}
        return {
            "schema_version": "benchmark_report.v1",
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "task": self.task,
            "profile": self.profile,
            "identifiers": self.identifiers,
            "system": self.system_info,
            "process": self.process_info,
            "phases": phases,
        }

    # ----------------
    # Pretty printing
    # ----------------
    def print_summary(self) -> None:
        print(self._separator())
        print(self._center_line("Summary Report"))
        print(self._separator())
        self._print_kv("run_id", self.run_id)
        self._print_kv("created_at_utc", self.created_at_utc)
        if self.profile:
            self._print_kv("profile", self.profile)
        if self.task:
            self._print_kv("task", self.task)
        for k, v in self.identifiers.items():
            self._print_kv(k, v)
        # system highlights
        gpu = self.system_info.get("gpu") or {}
        os_info = self.system_info.get("os") or {}
        self._print_kv("os", f"{os_info.get('system')} {os_info.get('release')}".strip())
        if gpu.get("gpu_name"):
            self._print_kv("gpu_name", gpu.get("gpu_name"))
        if gpu.get("gpu_driver_version"):
            self._print_kv("gpu_driver_version", gpu.get("gpu_driver_version"))
        if gpu.get("cuda_version"):
            self._print_kv("cuda_version", gpu.get("cuda_version"))

        for phase_name, phase in self._phases.items():
            print(self._separator())
            print(self._left_line(f"Phase: {phase_name}"))
            if phase.duration_s is not None:
                self._print_kv("duration_s", f"{phase.duration_s:.3f}")
            for m in phase.metrics:
                self._print_metric(m)
        print(self._separator())

    # ----------------
    # Internal helpers
    # ----------------
    def _ensure_phase(self, name: str) -> Phase:
        if name not in self._phases:
            self._phases[name] = Phase(name=name)
        return self._phases[name]

    def _separator(self) -> str:
        return "|" + ("-" * (self._report_width - 2)) + "|"

    def _center_line(self, text: str) -> str:
        return f"| {text:^{self._report_width - 4}} |"

    def _left_line(self, text: str) -> str:
        if len(text) > self._report_width - 4:
            text = text[: self._report_width - 7] + "..."
        return f"| {text:<{self._report_width - 4}} |"

    def _print_kv(self, k: str, v: Any) -> None:
        text = f"{k}: {v}"
        print(self._left_line(text))

    def _print_metric(self, m: Metric) -> None:
        if m.kind == "single":
            suffix = f" {m.unit}" if m.unit else ""
            self._print_kv(m.name, f"{m.value}{suffix}")
            return
        if m.kind == "series":
            # Print stats, keep raw series only in JSON.
            if m.stats:
                suffix = f" {m.unit}" if m.unit else ""
                self._print_kv(f"Min {m.name}", f"{m.stats['min']}{suffix}")
                self._print_kv(f"Max {m.name}", f"{m.stats['max']}{suffix}")
                self._print_kv(f"Mean {m.name}", f"{m.stats['mean']}{suffix}")
                if "stdev" in m.stats:
                    self._print_kv(f"Stdev {m.name}", f"{m.stats['stdev']}{suffix}")
            else:
                self._print_kv(m.name, f"(series, len={len(m.value)})")
            return
        # dict metric
        if isinstance(m.value, dict):
            self._print_kv(m.name, f"(dict, keys={len(m.value)})")
        else:
            self._print_kv(m.name, "(dict)")

