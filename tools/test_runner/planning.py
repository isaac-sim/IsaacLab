# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""What to run: the runner configuration, the work unit, and the planner.

:class:`RunnerConfig` centralizes every knob the runner used to read ad hoc from
the environment or hardcode as a module constant. :class:`Unit` is the work item
(a file plus the device mask to run it under). :class:`Planner` turns a file into
its units, splitting a ``device_isolated`` file into one single-device unit per
device when the run spans more than one.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Per-file fresh-process retry counts for tests that can enter stale render states.
# A default so the runner has no hardcoded file list; override via RunnerConfig.
_DEFAULT_PROCESS_FAILURE_RETRIES = {
    "test_visualizer_integration_physx.py": 4,
    "test_visualizer_integration_newton.py": 4,
    "test_visualizer_tiled_integration_physx.py": 4,
    "test_visualizer_tiled_integration_newton.py": 4,
}


@dataclass
class RunnerConfig:
    """All runner behavior as data, so nothing is hardcoded mid-pipeline.

    Build it once with :meth:`from_env` at the pytest entry point and pass it to
    every stage. Fields group into: run scope (mask, queue, isaacsim_ci),
    collection (filters/sharding), per-unit execution (timeouts, deadlines,
    retries, cold-cache), and reporting (paths).

    Attributes:
        workspace_root: Repository root; pytest ``--config-file`` and the
            ``tools/`` plugin dir derive from it.
        runtime_mask: Device mask for the run (e.g. ``"110"`` cpu+cuda:0,
            ``"0001"`` a shard). Concrete — never the open-ended ``"X"`` form.
        queue_path: Shared work-queue root for mgpu work-stealing; empty when the
            runner iterates its own file list.
        sim_device: Kit boot device; names the per-shard work-queue subdir and
            labels rows in the report.
        isaacsim_ci: Whether to pass ``-m isaacsim_ci`` to each subprocess.
        filter_pattern: Substring a test path must contain to run; empty = all.
        exclude_pattern: Comma-separated substrings; a path matching any is skipped.
        include_files: Basenames to restrict the run to; empty = no restriction.
        quarantined_only: Run only the quarantined set.
        curobo_only: Run only the cuRobo set.
        shard_index: This file-shard's index, or ``None`` for no file sharding.
        shard_count: Total file shards, or ``None``.
        default_timeout: Per-unit hard timeout [s] when a file has no override.
        per_file_timeouts: Basename -> hard timeout [s] overrides.
        cold_cache_buffer: Extra timeout [s] for the first camera-enabled test
            (shader compile on a cold cache).
        cold_cache_marker: Source substring that marks a camera-enabled test.
        startup_deadline: Seconds to reach AppLauncher init / pytest collection
            before declaring a startup hang.
        startup_hang_retries: Startup-hang retries before giving up.
        timeout_retries: Hard-timeout retries before giving up.
        shutdown_grace_period: Seconds to wait for clean exit after the report
            file appears before killing a shutdown-hung process.
        process_failure_retries: Basename -> extra fresh-process retries on test
            failure (for tests with stale-state flakiness).
        report_dir: Directory for per-unit and aggregate JUnit XML.
        result_file: Aggregate report filename within :attr:`report_dir`.
    """

    workspace_root: str
    # -- run scope --
    runtime_mask: str = "110"
    queue_path: str = ""
    sim_device: str = "cuda:0"
    isaacsim_ci: bool = False
    # -- collection --
    filter_pattern: str = ""
    exclude_pattern: str = ""
    include_files: frozenset[str] = frozenset()
    quarantined_only: bool = False
    curobo_only: bool = False
    shard_index: int | None = None
    shard_count: int | None = None
    # -- per-unit execution --
    default_timeout: int = 600
    per_file_timeouts: dict = field(default_factory=dict)
    cold_cache_buffer: int = 700
    cold_cache_marker: str = "enable_cameras=True"
    startup_deadline: int = 120
    startup_hang_retries: int = 2
    timeout_retries: int = 0
    shutdown_grace_period: int = 30
    process_failure_retries: dict = field(default_factory=lambda: dict(_DEFAULT_PROCESS_FAILURE_RETRIES))
    # -- reporting --
    report_dir: str = "tests"
    result_file: str = "full_report.xml"

    @property
    def config_file(self) -> str:
        """Path passed to pytest's ``--config-file`` for each subprocess."""
        return os.path.join(self.workspace_root, "pyproject.toml")

    @property
    def tools_dir(self) -> str:
        """Directory added to a subprocess PYTHONPATH so it can import the selector."""
        return os.path.join(self.workspace_root, "tools")

    @classmethod
    def from_env(cls, workspace_root: str) -> RunnerConfig:
        """Build a config from the environment and ``test_settings``.

        This is the single place environment variables are read; the rest of the
        runner takes the resulting config.

        Args:
            workspace_root: Repository root.

        Returns:
            A fully-populated config for this run.
        """
        import test_settings  # tools/ module; imported lazily so unit tests need no env

        shard_index = os.environ.get("TEST_SHARD_INDEX", "")
        shard_count = os.environ.get("TEST_SHARD_COUNT", "")
        raw_includes = os.environ.get("TEST_INCLUDE_FILES", "").split(",")
        include = {os.path.basename(f.strip()) for f in raw_includes if f.strip()}
        return cls(
            workspace_root=workspace_root,
            # An unset mask means single-GPU CI: cpu + cuda:0, matching test_devices()'s default.
            runtime_mask=os.environ.get("ISAACLAB_TEST_DEVICES") or "110",
            queue_path=os.environ.get("ISAACLAB_TEST_QUEUE", ""),
            # Kit boot device; names the per-shard work-queue subdir and labels the report.
            sim_device=os.environ.get("ISAACLAB_SIM_DEVICE") or "cuda:0",
            isaacsim_ci=os.environ.get("ISAACSIM_CI_SHORT", "false") == "true",
            filter_pattern=os.environ.get("TEST_FILTER_PATTERN", ""),
            exclude_pattern=os.environ.get("TEST_EXCLUDE_PATTERN", ""),
            include_files=frozenset(include),
            quarantined_only=os.environ.get("TEST_QUARANTINED_ONLY", "false") == "true",
            curobo_only=os.environ.get("TEST_CUROBO_ONLY", "false") == "true",
            shard_index=int(shard_index) if shard_index else None,
            shard_count=int(shard_count) if shard_count else None,
            default_timeout=test_settings.DEFAULT_TIMEOUT,
            per_file_timeouts=dict(test_settings.PER_TEST_TIMEOUTS),
            result_file=os.environ.get("TEST_RESULT_FILE", "full_report.xml"),
        )


@dataclass(frozen=True)
class Unit:
    """One pytest subprocess to run.

    Attributes:
        file: Test file to run.
        mask: Device mask set as ``ISAACLAB_TEST_DEVICES`` for the subprocess;
            ``test_devices()`` reads it to select the device variants.
        split: Whether this is one of several units from splitting a
            ``device_isolated`` file. The executor uses it only to give split
            units distinct report filenames; a lone unit keeps the suffix-free
            name the mgpu aggregator expects.
    """

    file: str
    mask: str
    split: bool = False


class Planner:
    """Turns a test file into the units to run, given the run's device mask.

    A ``device_isolated`` file (a backend with a process-global device lock, e.g.
    ovphysx) splits into one single-device unit per device, but only when the run
    spans more than one device — an mgpu shard's single-device mask yields one
    unit either way.
    """

    _ISOLATED_MARK_RE = re.compile(r"^\s*pytestmark\b.*\bdevice_isolated\b", re.MULTILINE)
    """Match a module-level ``pytestmark`` mentioning ``device_isolated`` (single or list form)."""

    def __init__(self, runtime_mask: str):
        if "X" in runtime_mask:
            raise ValueError(f"runtime mask {runtime_mask!r} must be concrete; 'X' is a scope-only construct")
        self._mask = runtime_mask

    @staticmethod
    def _single_bit_mask(index: int, width: int) -> str:
        """Return a width-``width`` mask with only ``index`` set, e.g. ``(1, 3) -> "010"``."""
        return "".join("1" if pos == index else "0" for pos in range(width))

    def is_isolated(self, source: str) -> bool:
        """Whether a test file's source declares the ``device_isolated`` marker."""
        return bool(self._ISOLATED_MARK_RE.search(source))

    def plan(self, test_file: str, source: str) -> list[Unit]:
        """Plan the units for one file.

        Args:
            test_file: Path to the test file.
            source: The file's source text (the caller already reads it for the
                cold-cache check), inspected for the isolation marker.

        Returns:
            One unit at the run's mask, or — for an isolated file on a
            multi-device run — one single-device unit per active device.
        """
        set_bits = [pos for pos, char in enumerate(self._mask) if char == "1"]
        if self.is_isolated(source) and len(set_bits) > 1:
            return [Unit(test_file, self._single_bit_mask(pos, len(self._mask)), split=True) for pos in set_bits]
        return [Unit(test_file, self._mask)]
