# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Workstation capability check for XR teleoperation.

Measures the host against the recommended teleop workstation spec documented in
``docs/source/how-to/cloudxr_teleoperation.rst`` (45 FPS streaming at 120 Hz
physics) and reports which requirements are unmet.

The check is **advisory**: it never blocks a teleop session.  Its result is
logged to the terminal and pushed to the connected XR client as a
``system_notice`` message so the operator sees the warning in the headset
rather than silently running at a fraction of the target frame rate.

Thresholds are numeric proxies rather than CPU/GPU model allowlists, so
equivalent hardware passes.  CPU checks are weighted most heavily: Pink IK and
CPU-side PhysX are dominated by single-thread throughput, so a workstation can
carry a top-tier GPU and still teleoperate badly.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import platform
import subprocess
import time

import numpy as np

logger = logging.getLogger(__name__)

DOC_URL = "https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html"
"""Link included in the client notice so the operator can read the full spec."""

# -- CPU ---------------------------------------------------------------------

CPU_REFERENCE_SCORE = 1.0
"""Single-thread score of the reference workstation CPU.

Measured with :func:`measure_cpu_single_thread_score` on an AMD Ryzen
Threadripper 7960X (24 cores), the CPU named in the recommended spec.  Scores
are normalized against this machine, so the reference is 1.0 by construction
and a host scoring 0.7 is 70% as fast per thread.

See :data:`_CPU_REFERENCE_NS_PER_ITER` for the underlying raw measurement.
"""

CPU_SCORE_MIN = 0.70
"""Minimum acceptable fraction of :data:`CPU_REFERENCE_SCORE`.

Set well below 1.0 because the reference is a recommendation, not a floor:
mid-range parts with strong single-thread performance teleoperate acceptably.
"""

CPU_CLOCK_MIN_GHZ = 4.0
"""Minimum CPU boost clock [GHz]. Diagnostic only -- see module docstring."""

CPU_PHYSICAL_CORES_MIN = 8
"""Minimum physical core count.

Deliberately a low floor rather than a match for the reference CPU's 24 cores.
Pink IK is single-threaded and CPU PhysX scales across only a handful of
threads, so an 8-core high-IPC part outperforms a many-core low-clock part for
teleop.  A threshold near the reference core count would fail exactly the
machines that perform well.
"""

CPU_GOVERNOR_EXPECTED = "performance"
"""Expected Linux cpufreq governor.

Ubuntu workstations default to ``powersave``, which measurably increases Pink
IK solve latency.  This is the single most actionable item in the check.
"""

CPU_GOVERNOR_FIX = "sudo cpupower frequency-set -g performance"
"""Command that resolves a non-``performance`` governor."""

# -- GPU ---------------------------------------------------------------------

GPU_VRAM_MIN_GB = 24.0
"""Minimum GPU memory [GB]."""

GPU_COMPUTE_CAPABILITY_MIN = (8, 9)
"""Minimum CUDA compute capability (Ada Lovelace).

The recommended GPUs -- RTX PRO 6000 and RTX 5090 -- are Blackwell (12.0); an
RTX 4090 is Ada (8.9).  Anything older lacks the encode throughput for
comfortable 45 FPS stereo streaming.
"""

DRIVER_VERSION_MIN = 580
"""Minimum NVIDIA driver major version."""

# -- System ------------------------------------------------------------------

RAM_MIN_GIB = 60.0
"""Minimum system memory [GiB].

The spec says 64 GB, which is ~59.6 GiB once firmware reservations are
subtracted from what the OS reports, so the threshold is expressed in GiB and
set just below to avoid failing genuine 64 GB machines.
"""

SUPPORTED_MACHINES = ("x86_64", "AMD64")
"""Supported CPU architectures. ARM support is not yet available for teleop."""

# Raw reference measurement backing CPU_REFERENCE_SCORE [ns per kernel
# iteration].  Measured 2026-07-29 on an AMD Ryzen Threadripper 7960X (24
# cores), the CPU named in the recommended spec; run-to-run spread was 0.2%.
# Recorded under the ``powersave`` governor, which the kernel's sustained load
# boosts out of, so a ``performance``-governor host scores slightly above 1.0.
# Update this together with the kernel in _measure_cpu_ns_per_iter if the
# kernel ever changes -- the two are only meaningful as a pair.
_CPU_REFERENCE_NS_PER_ITER = 10320.0
_CPU_BENCH_ELEMENTS = 8192
_CPU_BENCH_VARIANCE_MAX = 0.20
_CPU_BENCH_RUNS = 3


@dataclasses.dataclass(frozen=True)
class SystemCheckItem:
    """One measured requirement.

    Attributes:
        name: Human-readable requirement name, e.g. ``"CPU single-thread"``.
        passed: ``True`` when the measurement meets the threshold.  A skipped
            item (see :attr:`skipped`) is always ``True`` so an unavailable
            probe never produces a false warning.
        actual: The measured value, formatted for display.
        required: The threshold, formatted for display.
        detail: Optional actionable hint, e.g. a command that fixes the item.
        skipped: ``True`` when the probe was unavailable and no verdict could
            be reached.
    """

    name: str
    passed: bool
    actual: str
    required: str
    detail: str = ""
    skipped: bool = False


@dataclasses.dataclass(frozen=True)
class SystemCheckResult:
    """Aggregate result of :func:`check_system_requirements`."""

    items: tuple[SystemCheckItem, ...] = ()

    @property
    def passed(self) -> bool:
        """``True`` when every non-skipped item met its threshold."""
        return all(item.passed for item in self.items)

    @property
    def failures(self) -> tuple[SystemCheckItem, ...]:
        """The items that did not meet their threshold."""
        return tuple(item for item in self.items if not item.passed)

    def format_table(self) -> str:
        """Render all items as an aligned table for terminal output.

        Returns:
            A multi-line string with one row per checked requirement.
        """
        if not self.items:
            return "Teleop workstation check: no requirements could be measured."

        name_width = max(len(item.name) for item in self.items)
        actual_width = max(len(item.actual) for item in self.items)

        header = (
            "Teleop workstation check: PASSED"
            if self.passed
            else f"Teleop workstation check: {len(self.failures)} requirement(s) below the recommended spec"
        )
        lines = [header, f"See {DOC_URL}", ""]
        for item in self.items:
            status = "skip" if item.skipped else ("ok" if item.passed else "WARN")
            name = f"{item.name:<{name_width}}"
            actual = f"{item.actual:<{actual_width}}"
            lines.append(f"  [{status:>4}] {name}  {actual}  (need {item.required})")
            if item.detail and not item.passed:
                lines.append(f"         -> {item.detail}")
        return "\n".join(lines)

    def to_message(self) -> dict:
        """Build the ``system_notice`` envelope sent to the XR client.

        Only failing items are included so the in-headset banner stays
        readable.

        Returns:
            A JSON-serializable dict matching the teleop message-channel
            envelope convention (``{"type": ..., "message": ...}``).
        """
        return {
            "type": "system_notice",
            "message": {
                "level": "warning",
                "title": "Workstation below recommended spec",
                "summary": "Teleoperation may run below the 45 FPS target.",
                "items": [
                    {
                        "name": item.name,
                        "actual": item.actual,
                        "required": item.required,
                        "detail": item.detail,
                    }
                    for item in self.failures
                ],
                "doc_url": DOC_URL,
            },
        }


def measure_cpu_single_thread_score() -> float | None:
    """Measure single-thread CPU throughput relative to the reference machine.

    Runs a fixed-work, single-threaded floating-point kernel and normalizes the
    per-iteration time against :data:`_CPU_REFERENCE_NS_PER_ITER`.  This
    measures IPC and achieved clock together, which is what actually predicts
    Pink IK solve latency -- a core count or a nameplate boost frequency does
    not.

    The kernel is a chain of elementwise floating-point ufuncs over an
    L2-resident array, with no allocation in the hot loop.  Elementwise ufuncs
    are never multithreaded by NumPy, so the measurement stays single-threaded
    regardless of the BLAS build -- unlike a matrix product, which would be
    parallelized and would measure the wrong thing.  The array is sized so real
    arithmetic dominates interpreter dispatch overhead.  The best of
    :data:`_CPU_BENCH_RUNS` runs is used to blunt scheduler noise.

    Returns:
        The score, where ``1.0`` matches the reference CPU, or ``None`` when
        the reading is untrustworthy (run-to-run variance above
        :data:`_CPU_BENCH_VARIANCE_MAX`, indicating a busy machine) or the
        reference has not been calibrated.
    """
    if _CPU_REFERENCE_NS_PER_ITER <= 0.0:
        return None

    per_iter = _measure_cpu_ns_per_iter()
    if per_iter is None:
        return None
    return _CPU_REFERENCE_NS_PER_ITER / per_iter


def _measure_cpu_ns_per_iter() -> float | None:
    """Time the benchmark kernel, returning nanoseconds per iteration.

    Returns:
        The best per-iteration time [ns] across :data:`_CPU_BENCH_RUNS` runs,
        or ``None`` when run-to-run variance exceeds
        :data:`_CPU_BENCH_VARIANCE_MAX`.
    """
    iterations = 1500
    rng = np.random.default_rng(seed=0)
    a = rng.random(_CPU_BENCH_ELEMENTS)
    b = rng.random(_CPU_BENCH_ELEMENTS)
    out = np.empty(_CPU_BENCH_ELEMENTS, dtype=np.float64)

    # Warm-up: fault in the arrays and settle the clocks so the first timed run
    # is not penalized relative to the rest, which would trip the variance guard.
    for _ in range(200):
        np.multiply(a, b, out=out)
        np.add(out, b, out=out)
        np.sqrt(out, out=out)

    timings = []
    for _ in range(_CPU_BENCH_RUNS):
        start = time.perf_counter()
        for _ in range(iterations):
            np.multiply(a, b, out=out)
            np.add(out, b, out=out)
            np.sqrt(out, out=out)
        timings.append((time.perf_counter() - start) * 1e9 / iterations)

    best = min(timings)
    if best <= 0.0:
        return None
    # A busy machine yields a wide spread; refuse to judge rather than warn wrongly.
    if (max(timings) - best) / best > _CPU_BENCH_VARIANCE_MAX:
        logger.debug(f"CPU benchmark variance too high to judge: {timings}")
        return None
    return best


def _read_cpu_governor() -> str | None:
    """Read the cpufreq governor of CPU 0.

    Returns:
        The governor name, or ``None`` on non-Linux hosts or when cpufreq is
        unavailable (common in containers and VMs).
    """
    path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return handle.read().strip()


def _read_driver_version() -> str | None:
    """Read the installed NVIDIA driver version.

    Tries NVML first and falls back to parsing ``nvidia-smi`` output, matching
    the probe strategy used by the benchmark recorders.

    Returns:
        The driver version string, or ``None`` when it cannot be determined.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            version = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
        return version.decode() if isinstance(version, bytes) else str(version)
    except Exception:
        logger.debug("NVML driver query failed; falling back to nvidia-smi", exc_info=True)

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        logger.debug("nvidia-smi driver query failed", exc_info=True)
        return None
    version = result.stdout.strip().splitlines()
    return version[0].strip() if version else None


def _check_cpu() -> list[SystemCheckItem]:
    """Build the CPU requirement items."""
    import psutil

    items: list[SystemCheckItem] = []

    score = measure_cpu_single_thread_score()
    if score is None:
        items.append(
            SystemCheckItem(
                name="CPU single-thread",
                passed=True,
                actual="not measured",
                required=f">= {CPU_SCORE_MIN:.2f} of reference",
                skipped=True,
            )
        )
    else:
        items.append(
            SystemCheckItem(
                name="CPU single-thread",
                passed=score >= CPU_SCORE_MIN,
                actual=f"{score:.2f} of reference",
                required=f">= {CPU_SCORE_MIN:.2f} of reference",
                detail=("Pink IK and CPU physics are single-thread bound; expect reduced teleop frame rate."),
            )
        )

    governor = _read_cpu_governor()
    if governor is None:
        items.append(
            SystemCheckItem(
                name="CPU governor",
                passed=True,
                actual="unavailable",
                required=CPU_GOVERNOR_EXPECTED,
                skipped=True,
            )
        )
    else:
        items.append(
            SystemCheckItem(
                name="CPU governor",
                passed=governor == CPU_GOVERNOR_EXPECTED,
                actual=governor,
                required=CPU_GOVERNOR_EXPECTED,
                detail=CPU_GOVERNOR_FIX,
            )
        )

    freq = psutil.cpu_freq()
    if freq is None or not freq.max:
        items.append(
            SystemCheckItem(
                name="CPU boost clock",
                passed=True,
                actual="unavailable",
                required=f">= {CPU_CLOCK_MIN_GHZ:.1f} GHz",
                skipped=True,
            )
        )
    else:
        clock_ghz = freq.max / 1000.0
        items.append(
            SystemCheckItem(
                name="CPU boost clock",
                passed=clock_ghz >= CPU_CLOCK_MIN_GHZ,
                actual=f"{clock_ghz:.1f} GHz",
                required=f">= {CPU_CLOCK_MIN_GHZ:.1f} GHz",
            )
        )

    cores = psutil.cpu_count(logical=False)
    if cores is None:
        items.append(
            SystemCheckItem(
                name="CPU physical cores",
                passed=True,
                actual="unavailable",
                required=f">= {CPU_PHYSICAL_CORES_MIN}",
                skipped=True,
            )
        )
    else:
        items.append(
            SystemCheckItem(
                name="CPU physical cores",
                passed=cores >= CPU_PHYSICAL_CORES_MIN,
                actual=str(cores),
                required=f">= {CPU_PHYSICAL_CORES_MIN}",
            )
        )

    return items


def _resolve_cuda_index(device: str | int | None) -> int:
    """Resolve *device* to a CUDA ordinal to probe.

    Args:
        device: A torch device string (``"cuda:1"``), a CUDA ordinal, or
            ``None``.  ``None``, a bare ``"cuda"``, and non-CUDA strings such as
            ``"cpu"`` all fall back to the process's current CUDA device.  A
            CPU simulation device still warrants probing a GPU, because CloudXR
            encodes on one regardless.

    Returns:
        The CUDA ordinal to measure.  Falls back to ``0`` when the requested
        ordinal does not exist, so an out-of-range device string cannot crash
        the check.
    """
    import torch

    index: int | None = None
    if isinstance(device, int):
        index = device
    elif isinstance(device, str) and device.startswith("cuda"):
        _, _, ordinal = device.partition(":")
        if ordinal.isdigit():
            index = int(ordinal)

    if index is None:
        index = torch.cuda.current_device()
    if not 0 <= index < torch.cuda.device_count():
        logger.debug(f"CUDA device {device!r} is out of range; probing device 0 instead")
        index = 0
    return index


def _check_gpu(device: str | int | None = None) -> list[SystemCheckItem]:
    """Build the GPU requirement items.

    Args:
        device: The device teleoperation will run on.  Probing a fixed ordinal
            would measure the wrong adapter on a multi-GPU workstation whose
            simulation runs on, say, ``cuda:1``, producing a false warning or a
            false pass.  See :func:`_resolve_cuda_index` for how this resolves.
    """
    import torch

    if not torch.cuda.is_available():
        return [
            SystemCheckItem(
                name="NVIDIA GPU",
                passed=False,
                actual="not detected",
                required="required",
                detail="Teleoperation requires an NVIDIA GPU for CloudXR encoding.",
            )
        ]

    index = _resolve_cuda_index(device)

    items: list[SystemCheckItem] = []
    props = torch.cuda.get_device_properties(index)

    vram_gb = props.total_memory / 1e9
    items.append(
        SystemCheckItem(
            name="GPU memory",
            passed=vram_gb >= GPU_VRAM_MIN_GB,
            # Name the ordinal so a multi-GPU host shows which adapter was measured.
            actual=f"{vram_gb:.0f} GB (cuda:{index} {props.name})",
            required=f">= {GPU_VRAM_MIN_GB:.0f} GB",
        )
    )

    capability = (props.major, props.minor)
    items.append(
        SystemCheckItem(
            name="GPU architecture",
            passed=capability >= GPU_COMPUTE_CAPABILITY_MIN,
            actual=f"compute {props.major}.{props.minor}",
            required=f">= compute {GPU_COMPUTE_CAPABILITY_MIN[0]}.{GPU_COMPUTE_CAPABILITY_MIN[1]}",
            detail="Pre-Ada GPUs lack the encode throughput for 45 FPS stereo streaming.",
        )
    )

    driver = _read_driver_version()
    if driver is None:
        items.append(
            SystemCheckItem(
                name="NVIDIA driver",
                passed=True,
                actual="unavailable",
                required=f">= {DRIVER_VERSION_MIN}",
                skipped=True,
            )
        )
    else:
        try:
            major = int(driver.split(".")[0])
        except ValueError:
            major = None
        items.append(
            SystemCheckItem(
                name="NVIDIA driver",
                passed=True if major is None else major >= DRIVER_VERSION_MIN,
                actual=driver,
                required=f">= {DRIVER_VERSION_MIN}",
                skipped=major is None,
            )
        )

    return items


def _check_system() -> list[SystemCheckItem]:
    """Build the memory and platform requirement items."""
    import psutil

    items: list[SystemCheckItem] = []

    ram_gib = psutil.virtual_memory().total / 2**30
    items.append(
        SystemCheckItem(
            name="System memory",
            passed=ram_gib >= RAM_MIN_GIB,
            actual=f"{ram_gib:.0f} GiB",
            required=f">= {RAM_MIN_GIB:.0f} GiB",
        )
    )

    machine = platform.machine()
    items.append(
        SystemCheckItem(
            name="CPU architecture",
            passed=machine in SUPPORTED_MACHINES,
            actual=machine,
            required=" or ".join(SUPPORTED_MACHINES),
            detail="ARM support for teleoperation is not yet available.",
        )
    )

    return items


def check_system_requirements(device: str | int | None = None) -> SystemCheckResult:
    """Measure this workstation against the recommended teleop spec.

    Each probe group is isolated: a failure inside one (a missing optional
    dependency, an unreadable sysfs path) drops only that group rather than
    aborting the check, because this must never prevent a teleop session from
    starting.

    Args:
        device: The device teleoperation will run on, e.g. ``"cuda:1"`` or a
            CUDA ordinal.  On a multi-GPU workstation this selects which adapter
            the GPU checks measure; passing the wrong one (or leaving it
            unset on a host whose simulation is not on the current device)
            yields a false warning or a false pass.  Defaults to the process's
            current CUDA device.

    Returns:
        A :class:`SystemCheckResult` holding one
        :class:`SystemCheckItem` per measured requirement.
    """
    items: list[SystemCheckItem] = []
    for probe in (_check_cpu, lambda: _check_gpu(device), _check_system):
        try:
            items.extend(probe())
        except Exception:
            logger.debug("A teleop capability probe failed; skipping it", exc_info=True)
    return SystemCheckResult(items=tuple(items))
