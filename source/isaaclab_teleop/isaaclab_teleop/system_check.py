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
import json
import logging
import os
import platform
import subprocess
import time

import numpy as np

logger = logging.getLogger(__name__)

DOC_URL = "https://isaac-sim.github.io/IsaacLab/develop/source/how-to/cloudxr_teleoperation.html"
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

CPU_SCORE_MIN = 0.80
"""Minimum acceptable fraction of :data:`CPU_REFERENCE_SCORE`.

Set below 1.0 because the reference is a recommendation, not a floor: parts
somewhat slower per thread than the reference still teleoperate acceptably.
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

Ubuntu workstations default to ``powersave``, which costs per-frame ramp-up
latency in the bursty workload teleop actually generates.  This is the most
actionable item in the check when it fires.

It is only reported as unmet when the single-thread score falls below
:data:`CPU_SCORE_MIN`, or could not be measured.  The governor is a proxy for
delivered throughput, so a machine that already clears the bar is fast enough
whatever the governor says, and flagging it there would be noise.
"""

CPU_GOVERNOR_FIX = "sudo cpupower frequency-set -g performance"
"""Command that resolves a non-``performance`` governor."""

# -- GPU ---------------------------------------------------------------------

GPU_VRAM_MIN_GB = 24.0
"""Minimum GPU memory [GB]."""

GPU_ARCH_MIN = 89
"""Minimum CUDA architecture (Ada Lovelace, compute 8.9).

Expressed the way Warp reports :attr:`warp.Device.arch` -- compute capability
without the decimal point, so 8.9 is ``89`` and 12.0 is ``120``.

The recommended GPUs -- RTX PRO 6000 and RTX 5090 -- are Blackwell (120); an
RTX 4090 is Ada (89).  Anything older lacks the encode throughput for
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
# iteration].  Measured on an AMD Ryzen Threadripper 7960X (24 cores), the CPU
# named in the recommended spec, and re-verified 2026-07-30 under the
# ``performance`` governor that :ref:`install-isaac-teleop` tells operators to
# set.  Fresh-process runs land in 10250-10390 ns under both governors, so this
# is a central value rather than a best case: the reference machine scores
# ~1.00, not a flattering 1.05.
#
# The governor does not measurably move this number.  The benchmark's warm-up
# drives the clocks to boost under sustained load whatever the governor says,
# so the powersave/performance difference is lost in run-to-run noise.  That is
# exactly why the governor is checked separately rather than left to this
# score: teleop drives the CPU in per-frame bursts, where a ``powersave`` host
# pays ramp-up latency that a sustained-throughput benchmark never sees.
#
# Take a median across fresh processes when recalibrating, not the best of a
# single burst -- within one process the spread is ~0.4%, but across processes
# it is several percent, and a single lucky burst sets the bar too high for
# every other machine.
#
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

    def to_message(self, max_bytes: int | None = None) -> dict:
        """Build the ``system_notice`` envelope sent to the XR client.

        Only failing items are included so the in-headset banner stays
        readable.

        When *max_bytes* is given the notice is budgeted down to fit, because
        the teleop message channel drops anything larger and the worst machines
        -- the ones this check exists for -- fail the most requirements and so
        produce the largest payloads.  Silently losing the warning exactly when
        it matters most is the failure mode being avoided here.

        Degradation is ordered least-useful first: the per-item ``detail`` hints
        go before any requirement is dropped, and if items must still be
        dropped the summary says how many, so the operator knows the list is
        partial and can read the full table in the terminal.

        Args:
            max_bytes: Budget for the UTF-8 encoded JSON, or ``None`` for no
                limit.

        Returns:
            A JSON-serializable dict matching the teleop message-channel
            envelope convention (``{"type": ..., "message": ...}``).
        """
        summary = "Teleoperation may run below the 45 FPS target."
        items = [
            {
                "name": item.name,
                "actual": item.actual,
                "required": item.required,
                "detail": item.detail,
            }
            for item in self.failures
        ]

        def build(entries: list[dict], note: str) -> dict:
            return {
                "type": "system_notice",
                "message": {
                    "level": "warning",
                    "title": "Workstation below recommended spec",
                    "summary": summary + note,
                    "items": entries,
                    "doc_url": DOC_URL,
                },
            }

        message = build(items, "")
        if max_bytes is None or _encoded_size(message) <= max_bytes:
            return message

        # 1. Drop the remediation hints, the largest per-item field.
        stripped = [{**entry, "detail": ""} for entry in items]
        message = build(stripped, "")
        if _encoded_size(message) <= max_bytes:
            return message

        # 2. Still too large: drop requirements from the end until it fits.
        for kept in range(len(stripped) - 1, -1, -1):
            omitted = len(stripped) - kept
            message = build(stripped[:kept], f" ({omitted} more not shown; see the terminal.)")
            if _encoded_size(message) <= max_bytes:
                return message

        # Nothing but the envelope fits; send it so the operator still sees
        # that something is wrong.
        return message


def _encoded_size(message: dict) -> int:
    """Size of *message* as it goes on the wire [bytes]."""
    return len(json.dumps(message).encode())


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


def _format_arch(arch: int) -> str:
    """Render a Warp CUDA architecture as a compute capability.

    Args:
        arch: Compute capability without the decimal point, e.g. ``89``.

    Returns:
        The dotted form, e.g. ``"8.9"``.
    """
    return f"{arch // 10}.{arch % 10}"


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

    # The governor only matters as a proxy for delivered throughput. When the
    # benchmark already clears the bar, warning about it is noise: the machine
    # is demonstrably fast enough whatever the governor is set to. Only gate on
    # it when the score falls short, or when it could not be measured at all.
    score_meets_bar = score is not None and score >= CPU_SCORE_MIN
    governor_required = f"{CPU_GOVERNOR_EXPECTED} (or single-thread >= {CPU_SCORE_MIN:.2f})"

    governor = _read_cpu_governor()
    if governor is None:
        items.append(
            SystemCheckItem(
                name="CPU governor",
                passed=True,
                actual="unavailable",
                required=governor_required,
                skipped=True,
            )
        )
    else:
        items.append(
            SystemCheckItem(
                name="CPU governor",
                passed=score_meets_bar or governor == CPU_GOVERNOR_EXPECTED,
                actual=governor,
                required=governor_required,
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


def _resolve_gpu_device(device: str | int | None):
    """Resolve *device* to the Warp CUDA device to probe.

    Args:
        device: A device string (``"cuda:1"``), a CUDA ordinal, or ``None``.
            ``None``, a bare ``"cuda"``, an unknown ordinal, and non-CUDA
            values such as ``"cpu"`` all fall back to Warp's default CUDA
            device.  A CPU simulation device still warrants probing a GPU,
            because CloudXR encodes on one regardless.

    Returns:
        The ``warp.Device`` to measure, or ``None`` when the host has no CUDA
        device at all.
    """
    import warp as wp

    if not wp.is_cuda_available():
        return None

    if device is not None:
        alias = f"cuda:{device}" if isinstance(device, int) else str(device)
        try:
            resolved = wp.get_device(alias)
        except Exception:
            # Unknown alias (e.g. an ordinal this host does not have).
            logger.debug(f"Device {alias!r} is not available; probing the default CUDA device", exc_info=True)
        else:
            if resolved.is_cuda:
                return resolved

    try:
        default = wp.get_device()
        if default.is_cuda:
            return default
    except Exception:
        logger.debug("Could not resolve the default Warp device", exc_info=True)

    cuda_devices = wp.get_cuda_devices()
    return cuda_devices[0] if cuda_devices else None


def _check_gpu(device: str | int | None = None) -> list[SystemCheckItem]:
    """Build the GPU requirement items.

    Args:
        device: The device teleoperation will run on.  Probing a fixed ordinal
            would measure the wrong adapter on a multi-GPU workstation whose
            simulation runs on, say, ``cuda:1``, producing a false warning or a
            false pass.  See :func:`_resolve_gpu_device` for how this resolves.
    """
    gpu = _resolve_gpu_device(device)
    if gpu is None:
        return [
            SystemCheckItem(
                name="NVIDIA GPU",
                passed=False,
                actual="not detected",
                required="required",
                detail="Teleoperation requires an NVIDIA GPU for CloudXR encoding.",
            )
        ]

    items: list[SystemCheckItem] = []

    vram_gb = gpu.total_memory / 1e9
    items.append(
        SystemCheckItem(
            name="GPU memory",
            passed=vram_gb >= GPU_VRAM_MIN_GB,
            # Name the device so a multi-GPU host shows which adapter was measured.
            actual=f"{vram_gb:.0f} GB ({gpu.alias} {gpu.name})",
            required=f">= {GPU_VRAM_MIN_GB:.0f} GB",
        )
    )

    items.append(
        SystemCheckItem(
            name="GPU architecture",
            passed=gpu.arch >= GPU_ARCH_MIN,
            actual=f"compute {_format_arch(gpu.arch)}",
            required=f">= compute {_format_arch(GPU_ARCH_MIN)}",
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
