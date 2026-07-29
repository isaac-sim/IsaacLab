# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Background monitoring thread for continuous benchmark recording during blocking operations."""

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .benchmark_core import BaseIsaacLabBenchmark


class BenchmarkMonitor:
    """Background thread that periodically updates benchmark recorders.

    This utility enables continuous system resource monitoring during blocking
    RL training loops (RSL-RL, RL-Games) where `update_manual_recorders()` would
    otherwise only be called once after training completes.

    Usage:
        with BenchmarkMonitor(benchmark, interval=1.0):
            runner.learn(...)  # Blocking training call
    """

    def __init__(self, benchmark: "BaseIsaacLabBenchmark", interval: float = 1.0):
        """Initialize the benchmark monitor.

        Args:
            benchmark: The benchmark instance to monitor.
            interval: Time between recorder updates in seconds. Defaults to 1.0.
        """
        self._benchmark = benchmark
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._exception: Exception | None = None

    def _monitor_loop(self) -> None:
        """Background loop that updates recorders at the specified interval."""
        while not self._stop_event.is_set():
            try:
                self._benchmark.update_manual_recorders()
            except Exception as e:
                self._exception = e
                # Log but don't crash - monitoring failure shouldn't stop training
                print(f"[BenchmarkMonitor] Warning: update failed: {e}")

            # Wait for interval or stop signal
            self._stop_event.wait(timeout=self._interval)

    def start(self) -> None:
        """Start the monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # Already running

        self._stop_event.clear()
        self._exception = None
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the monitoring thread and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)
            self._thread = None

    def __enter__(self) -> "BenchmarkMonitor":
        """Start monitoring when entering context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop monitoring when exiting context."""
        self.stop()
