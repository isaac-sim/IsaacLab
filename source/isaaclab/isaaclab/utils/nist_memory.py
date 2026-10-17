# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional native phase, allocator and asset-client observations.

This module adds no device completion calls. The demo supplies the existing
measurement boundary completion state, and the external sampler covers process
launch, imports and non-Torch GPU allocations.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def _error(error: BaseException) -> dict:
    return {"type": type(error).__name__, "message": str(error)}


def _source(value) -> str | None:
    """Keep the asset address without URL credentials or signed query values."""
    if not isinstance(value, (str, os.PathLike)):
        return None
    value = str(value)
    try:
        parts = urlsplit(value)
        if not parts.scheme:
            return value
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parts.port is not None:
            host += f":{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except (TypeError, ValueError):
        return None


class TorchAllocatorProbe:
    """Read selected-device Torch allocator counters without initializing CUDA."""

    def __init__(self, cuda, device: str, *, clock=time.monotonic_ns):
        self.cuda = cuda
        self.device = str(device)
        self.clock = clock
        self.peak_scope = {"kind": "before_first_phase_observation"}

    def _ready(self) -> bool:
        return self.device.split(":", 1)[0] == "cuda" and self.cuda.is_initialized()

    def begin_phase(self) -> dict:
        reset_started = self.clock()
        try:
            if self._ready():
                self.cuda.reset_peak_memory_stats(self.device)
                self.peak_scope = {
                    "kind": "since_phase_start_peak_reset",
                    "reset_started_ns": reset_started,
                    "reset_finished_ns": self.clock(),
                    "allocator_state_retained": True,
                }
            else:
                self.peak_scope = {
                    "kind": "native_allocator_initialization_after_phase_hook_if_cuda_initializes",
                    "earliest_possible_ns": reset_started,
                    "exact_initialization_timestamp_observed": False,
                }
        except Exception as error:
            self.peak_scope = {"kind": "unreset_allocator_history", "reset_error": _error(error)}
        return self.snapshot()

    def snapshot(self) -> dict:
        result = {
            "status": "unavailable", "device": self.device, "gpu_uuid": None,
            "allocator_backend": None, "allocated_bytes": None, "reserved_bytes": None,
            "max_allocated_bytes": None, "max_reserved_bytes": None,
            "query_start_ns": self.clock(), "peak_scope": dict(self.peak_scope),
        }
        try:
            if self.device.split(":", 1)[0] != "cuda":
                result["status"] = "not_cuda"
            elif not self.cuda.is_initialized():
                result["status"] = "not_initialized"
            else:
                result.update(
                    status="ok",
                    allocated_bytes=int(self.cuda.memory_allocated(self.device)),
                    reserved_bytes=int(self.cuda.memory_reserved(self.device)),
                    max_allocated_bytes=int(self.cuda.max_memory_allocated(self.device)),
                    max_reserved_bytes=int(self.cuda.max_memory_reserved(self.device)),
                )
                try:
                    result["allocator_backend"] = self.cuda.memory.get_allocator_backend()
                except Exception as error:
                    result["allocator_backend_error"] = _error(error)
                try:
                    value = getattr(self.cuda.get_device_properties(self.device), "uuid", None)
                    if isinstance(value, bytes) and len(value) == 16:
                        value = uuid.UUID(bytes=value)
                    elif value is not None:
                        value = uuid.UUID(str(value).removeprefix("GPU-"))
                    result["gpu_uuid"] = "GPU-" + str(value) if value is not None else None
                except Exception as error:
                    result["gpu_uuid_error"] = _error(error)
        except Exception as error:
            result["status"] = "error"
            result["error"] = _error(error)
        result["query_end_ns"] = self.clock()
        return result


class MemoryRecorder:
    """Append and flush native events while preserving workload exceptions."""

    def __init__(self, path, probe: TorchAllocatorProbe, *, clock=time.monotonic_ns, stream=None, main_entered_ns=None):
        self.path = str(path)
        self.probe = probe
        self.clock = clock
        self.pid = os.getpid()
        self.invocation_id = str(uuid.uuid4())
        self.sequence = 0
        self.phase = "startup"
        self.closed = False
        self.errors: list[dict] = []
        self.workload_failures = 0
        self.episode_ids: dict[int, int] = {}
        self._restorations = []
        self._lock = threading.RLock()
        self._stream = stream
        if stream is None:
            try:
                destination = Path(path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                self._stream = destination.open("a", encoding="utf-8")
            except Exception as error:
                self._observation_error("open_events", error)
        self.emit(
            "hook_start", main_entered_ns=main_entered_ns,
            scope="Native hook after argument parsing; process launch and imports precede this hook.",
            allocator_scope=(
                "Torch allocator counters only; initial resident allocations remain visible, "
                "but pre-hook historical peaks are not represented as phase peaks."
            ),
            timing_effect="Phase reads/JSON flushes and reset-event writes add host work; no GPU synchronization is added.",
            cache_environment={key: os.environ.get(key) for key in (
                "TMPDIR", "XDG_CACHE_HOME", "WARP_CACHE_PATH", "CUDA_CACHE_PATH", "UV_CACHE_DIR",
            )},
        )
        self.emit("phase_start", torch=self.probe.begin_phase(), boundary_completion="not_drained", reason="hook_installed",
                  boundary_time_scope="After peak-reset/current-counter observation, before native startup work.")

    def _observation_error(self, operation: str, error: BaseException) -> None:
        self.errors.append({"operation": operation, **_error(error)})
        try:
            print(f"NIST memory observer {operation}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        except Exception:
            pass

    def emit(self, kind: str, **fields) -> None:
        with self._lock:
            self.sequence += 1
            event = {
                "schema": "nist-native-memory-events-v1", "kind": kind,
                "phase": self.phase, "monotonic_ns": self.clock(), "pid": self.pid,
                "invocation_id": self.invocation_id, "sequence": self.sequence, **fields,
            }
            if self._stream is not None:
                try:
                    self._stream.write(json.dumps(event, allow_nan=False) + "\n")
                    self._stream.flush()
                except Exception as error:
                    self._observation_error("write_event", error)

    def transition(self, phase: str, *, boundary_completion: str, reason: str, phase_complete: bool = False) -> None:
        if phase == self.phase:
            return
        boundary_ns = self.clock()
        self.emit("phase_end", monotonic_ns=boundary_ns, torch=self.probe.snapshot(),
                  boundary_completion=boundary_completion, phase_complete=phase_complete, reason=reason,
                  boundary_time_scope="Before counter observation; completion state describes preceding native work.")
        self.phase = phase
        self.emit("phase_start", torch=self.probe.begin_phase(), boundary_completion=boundary_completion, reason=reason,
                  boundary_time_scope="After peak-reset/current-counter observation, before next native phase work.")

    def record_reset(self, board_id: int, global_control_step: int, assembled_parts: int, complete_board: bool, native_episode_steps: int) -> None:
        episode_id = self.episode_ids.get(board_id, 0)
        self.emit(
            "reset", board_id=board_id, episode_id=episode_id, global_control_step=global_control_step,
            assembled_parts=assembled_parts, complete_board=complete_board, native_episode_steps=native_episode_steps,
            observation_scope="Native terminal bookkeeping after env.step and its automatic reset; not reset-start timing.",
        )
        self.episode_ids[board_id] = episode_id + 1

    def failure(self, stage: str, error: BaseException) -> None:
        self.workload_failures += 1
        self.emit("failure", stage=stage, error=_error(error))

    def install_asset_observer(self, assets) -> None:
        """Observe lazy OmniClient access and restore original callables on close."""
        original_get = assets._get_omni_client
        wrapped_clients: set[int] = set()

        @functools.wraps(original_get)
        def get_client(*args, **kwargs):
            client = original_get(*args, **kwargs)
            with self._lock:
                if id(client) not in wrapped_clients:
                    for operation in ("copy", "read_file", "stat"):
                        try:
                            original = getattr(client, operation)
                            wrapped = self._asset_wrapper(client, operation, original)
                            setattr(client, operation, wrapped)
                            self._restorations.append((client, operation, original, wrapped))
                        except Exception as error:
                            self._observation_error("install_asset_" + operation, error)
                    wrapped_clients.add(id(client))
            return client

        assets._get_omni_client = get_client
        self._restorations.append((assets, "_get_omni_client", original_get, get_client))
        self.emit("asset_observer_installed", operations=["copy", "read_file", "stat"],
                  scope="Asset-client operation time, including any client/cache/filesystem/network work.")

    def _asset_wrapper(self, client, operation, original):
        @functools.wraps(original)
        def observed(*args, **kwargs):
            phase_at_start = self.phase
            start_ns = self.clock()
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                end_ns = self.clock()
                self.emit("asset_io", operation=operation, start_ns=start_ns, end_ns=end_ns,
                          elapsed_ns=end_ns-start_ns, phase_at_start=phase_at_start,
                          status="exception", error=_error(error), source=_source(args[0] if args else kwargs.get("url", kwargs.get("src_url"))))
                raise
            end_ns = self.clock()
            try:
                code = result[0] if isinstance(result, tuple) else result
                ok = code == client.Result.OK
                byte_count, byte_scope = None, None
                if ok and operation == "read_file" and isinstance(result, tuple) and len(result) > 2:
                    try:
                        byte_count = memoryview(result[2]).nbytes
                        byte_scope = "returned content bytes; not network-transfer bytes"
                    except (TypeError, ValueError):
                        pass
                elif ok and operation == "copy":
                    destination = args[1] if len(args) > 1 else kwargs.get("dst_url")
                    if isinstance(destination, str) and "://" not in destination:
                        try:
                            byte_count = Path(destination).stat().st_size
                            byte_scope = "destination file bytes; not network-transfer bytes"
                        except OSError:
                            pass
                self.emit(
                    "asset_io", operation=operation, start_ns=start_ns, end_ns=end_ns,
                    elapsed_ns=end_ns-start_ns, phase_at_start=phase_at_start,
                    status="ok" if ok else "returned_non_ok", result_code=str(code),
                    source=_source(args[0] if args else kwargs.get("url", kwargs.get("src_url"))),
                    byte_count=byte_count, byte_count_scope=byte_scope,
                    timing_scope="asset-client operation time; not pure network time",
                )
            except Exception as error:
                self._observation_error("asset_result_observation", error)
            return result

        return observed

    def close(self) -> None:
        if self.closed:
            return
        for target, name, original, wrapped in reversed(self._restorations):
            try:
                if getattr(target, name) is wrapped:
                    setattr(target, name, original)
            except Exception as error:
                self._observation_error("restore_asset_observer", error)
        boundary_ns = self.clock()
        self.emit("phase_end", monotonic_ns=boundary_ns, torch=self.probe.snapshot(),
                  boundary_completion="not_drained", phase_complete=self.phase == "shutdown", reason="hook_close")
        self.emit("hook_close", status="complete" if not self.errors else "observation_errors", observation_errors=self.errors,
                  workload_status="failed" if self.workload_failures else "no_failure_observed", workload_failures=self.workload_failures)
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception as error:
                self._observation_error("close_events", error)
        self.closed = True
