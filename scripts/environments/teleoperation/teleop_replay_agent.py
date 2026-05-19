# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CI/automation entry point for replaying captured Isaac Teleop sessions.

This is the non-interactive counterpart to ``teleop_se3_agent.py``. It builds
a teleop environment, attaches an :class:`~isaaclab_teleop.IsaacTeleopDevice`
configured in :class:`isacteleop.teleop_session_manager.SessionMode.REPLAY`,
and pumps the simulation loop until the recorded operator presses STOP (or
``--max_replay_duration_s`` elapses, or Kit is closed). The user-journey
teleop script remains ``teleop_se3_agent.py``.

Inputs:
    ``--replay_file`` is an MCAP capture produced by Isaac Teleop's
    ``McapRecordingConfig`` path (typically written by ``record_demos.py
    --mcap_record_path``). The recorder lays down per-tracker flatbuffer
    messages (head / hands / controllers) plus the ``_teleop_control``
    ``MessageChannelTracker`` that captured the operator's START / STOP /
    RESET gestures. TeleopCore's
    :class:`~isacteleop.deviceio_session.ReplaySession` re-emits all of
    them on the same monotonic-time cadence they were recorded on, so
    :func:`~isaaclab_teleop.poll_control_events` returns the same edges
    here that ``record_demos.py``'s loop saw at recording time.

Gating:
    The env-step loop mirrors ``teleop_se3_agent.py``: each iteration
    calls :meth:`IsaacTeleopDevice.advance` and
    :func:`~isaaclab_teleop.poll_control_events`, gates ``env.step()`` on
    ``ctrl.is_active`` (so pre-START operator setup frames are rendered
    without stepping), and handles mid-demo ``ctrl.should_reset`` events
    by running the full sim/env/teleop reset cycle.

End-of-replay termination:
    Four distinct signals end the current run by breaking out of the
    inner loop. ``_run_single_replay`` returns its populated
    :class:`_RunStats`, and the outer batch driver in :func:`main`
    moves on to the next replay (or returns from ``main`` when the
    batch is done; ``__main__`` then calls ``simulation_app.close()``).
    The signals are:

    1. **The recorded operator STOP**, replayed via the
       ``_teleop_control`` ``MessageChannelTracker`` at the same
       recording-frame index it was captured at -- ``ctrl.is_active``
       transitions True->False on that frame.
    2. **The env's success condition firing for
       ``--num_success_steps`` consecutive steps**, the natural end of
       a ``record_demos.py``-style capture. Post-success MCAP frames
       are operator wind-down (releases, idle drift) that we have no
       use for in replay, so we skip the ``_handle_reset`` cycle the
       live agent would do.
    3. **A task-specific failure term** (``terminated`` or ``truncated``
       from ``env.step``) -- the recorded trajectory did not reproduce;
       the operator has no agency to recover during replay.
    4. **Wall-clock ``--max_replay_duration_s`` safety cap**, for
       recordings that produce neither a STOP, a success, nor a failure
       within the configured window.

    With ``--num_replays N > 1`` each run is independently terminated by
    one of the four signals above; the agent then rebuilds the
    :class:`IsaacTeleopDevice` (reopening the MCAP at frame 0) and runs
    again. The USD stage is loaded only once.

Stats output:
    Every iteration where ``env.step()`` actually ran contributes one
    CPU frame-time sample (``time.perf_counter()`` delta in ms). Pre-
    START render-only frames, warmup ticks, and post-quit render-only
    spin-down are excluded so the resulting numbers reflect the
    steady-state replay workload rather than agent bookkeeping
    overhead. Per-run samples are summarised into mean / p50 / p90 /
    p95 / p99 / min / max / stddev (under ``cpu_frame_time_ms``) plus
    derived FPS metrics (under ``fps``). The two blocks measure the
    same ``env.step`` event and stay self-consistent: ``fps.mean``
    equals ``1000 / cpu_frame_time_ms.mean`` (harmonic mean of FPS
    = total frames / total step time). Kit's HUD displays the render
    rate, which is this FPS multiplied by ``decimation /
    render_interval`` (Kit pumps multiple frames per ``env.step``);
    derive that from the env config if you need it.

    Each active iteration also emits one ``GpuStatsProvider.sample()``
    call. The default :class:`NvmlGpuStatsProvider` snapshots GPU
    utilization (%) and used memory (MB) via ``pynvml``, summarised
    under ``gpu_stats`` with the same percentile shape as
    ``cpu_frame_time_ms``. It soft-fails when ``nvidia-ml-py`` is
    missing or the driver is unreachable (``gpu_stats.available =
    false`` + reason). Renderer-specific providers (Kit viewport
    telemetry, Newton, ...) can be slotted in by implementing the
    :class:`GpuStatsProvider` Protocol.

    A multi-run batch aggregates by taking the mean-of-means,
    mean-of-p90s, etc. across runs.

    A one-line-per-run stdout summary is always printed at the end of
    the batch. Pass ``--stats_output_file <path>`` to additionally
    persist the report as JSON. Schema (schema_version 1)::

        {
          "schema_version": 1,
          "task": "Isaac-PickPlace-GR1T2-Abs-v0",
          "replay_file": "/tmp/pickplace_gr1t2.mcap",
          "num_replays": 5,
          "outcomes": {"success": 4, "failure": 1, "incomplete": 0, "timeout": 0},
          "success_rate": 0.8,
          "runs": [
            {
              "run_index": 0,
              "outcome": "success",
              "active_iterations": 322,
              "active_duration_s": 21.503,
              "success_step_count": 1,
              "cpu_frame_time_ms": {
                "mean": ..., "p50": ..., "p90": ..., "p95": ..., "p99": ...,
                "min": ..., "max": ..., "stddev": ..., "n": ...
              },
              "fps": {"mean": ..., "min_instantaneous": ..., "max_instantaneous": ...},
              "gpu_stats": {
                "backend": "nvml", "available": True,
                "device_index": 0, "device_name": ..., "memory_total_mb": ...,
                "utilization_percent": {<cpu_frame_time_ms shape>},
                "memory_used_mb": {<cpu_frame_time_ms shape>}
              }
            }
          ],
          "aggregate": {
            "cpu_frame_time_ms": {"mean_of_means": ..., "mean_of_p90s": ...,
                                  "mean_of_p99s": ..., "min_overall": ...,
                                  "max_overall": ...},
            "fps": {"mean_of_means": ...}
          }
        }

Exit codes:
    The process exits with a status code that CI can branch on. With
    ``--num_replays N`` the worst-of-N outcome wins (precedence
    ``timeout > failure > incomplete > success``):

    * ``0`` -- every run reproduced the recording (``success_term``
      fired on each run).
    * ``1`` -- one or more runs terminated/truncated mid-trajectory,
      or finished without any explicit terminator firing.
    * ``2`` -- one or more runs hit ``--max_replay_duration_s``.

Warmup:
    Before stepping the env, the agent waits deterministically for Kit
    to finish loading the USD stage by polling
    ``omni.usd.UsdContext.get_stage_loading_status()`` until no assets
    are pending (bounded by ``--max_stage_load_wait_s`` as a safety net).
    It then pumps a fixed number of additional renderer-settle frames so
    shaders / articulation views finish warming up before any action
    lands. ``--replay_start_delay_s`` is available as an optional
    wall-clock buffer on top of the deterministic wait for hardware
    that needs more grace. During warmup the agent does not call
    :meth:`IsaacTeleopDevice.advance`, so ``ReplaySession.update()``
    does not advance through the MCAP.

XR-active replay:
    Pass ``--cloudxr_env <shorthand-or-path>`` (and optionally
    ``--no-auto_launch_cloudxr``) to auto-spawn the CloudXR runtime and
    engage Kit's XR pipeline during replay. ``--cloudxr_env`` mirrors
    the flag on ``record_demos.py`` and accepts the same ``cloudxrjs``
    / ``avp`` shorthands. This is required (not optional) for two
    distinct reasons:

    A. **Performance parity with live teleop.** A pure-replay run (no
       XR, no CloudXR) skips the entire Kit XR rendering pipeline,
       so frame timings, render load, GPU/CPU contention, and any
       XR-side bottlenecks do not appear -- a captured trajectory
       that replayed at 90Hz under those conditions could easily run
       at 30Hz once XR is actually active. For perf regression or
       benchmarking the replay loop must reproduce the same Kit
       configuration the original recording ran under.

    B. **Correct ``world_T_anchor`` for playback.** The recorded
       tracker stream (head / hands / controllers) lives in
       OpenXR-local space; the world-frame poses the env consumes
       come from ``world_T_anchor @ oxr_pose``. With XR active,
       :class:`~isaaclab_teleop.XrAnchorManager` resolves
       ``world_T_anchor`` through ``XrAnchorSynchronizer`` (the same
       path used at record time), so the live anchor semantics --
       including any dynamic-anchor following of a prim and runtime
       recentering -- are reproduced. Without XR active, the manager
       falls back to the static :class:`~isaaclab_teleop.XrCfg`
       values, which only happen to match record-time semantics when
       the anchor never moved.

    The full incantation also needs ``AppLauncher``'s ``--xr`` flag
    plus a few Kit-side carb settings to flip the AR profile and load
    the teleop XR bridge (the replay path skips both for the
    headless-CI default; we have not yet promoted them to a single
    ``--xr_active`` knob)::

        ./isaaclab.sh -p teleop_replay_agent.py \\
            --task <task> --replay_file <X.mcap> \\
            --xr --device cuda:0 \\
            --cloudxr_env cloudxrjs \\
            --kit_args="--/xr/profile/ar/enabled=true \\
                        --enable isaacsim.kit.xr.teleop.bridge \\
                        --/persistent/xr/openxr/disableInputBindings=true"

    The headset is purely a viewer / anchor source -- the recorded MCAP
    remains the sole source of action; live controller input from the
    spectator's headset does not displace the replayed trajectory.

    Multi-run note: ``--num_replays > 1`` IS supported in XR-active
    mode. The CloudXR runtime is launched once at the agent (batch)
    scope and shared across runs (``_maybe_launch_cloudxr``); each
    per-run :class:`~isaaclab_teleop.IsaacTeleopDevice` is constructed
    with ``auto_launch_cloudxr=False`` so the per-run lifecycle does
    not stop the runtime on teardown. Only the per-run
    ``TeleopSession`` is torn down between replays; Kit's OpenXR
    instance/session stay alive.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=(
        "Replay a captured Isaac Teleop MCAP session against an Isaac Lab environment. "
        "CI/automation entry point; for interactive teleoperation see teleop_se3_agent.py."
    )
)
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--replay_file",
    type=str,
    required=True,
    help="Absolute path to the Isaac Teleop MCAP capture to replay.",
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=1,
    help=(
        "Number of consecutive steps the task success term must hold before declaring success and"
        " resetting the env. Mirrors the equivalent flag in record_demos.py."
    ),
)
parser.add_argument(
    "--max_replay_duration_s",
    type=float,
    default=600.0,
    help=(
        "Maximum wall-clock seconds to keep a single replay running before ending it with the"
        " ``timeout`` outcome, measured from the end of the warmup window. Safety net for"
        " malformed MCAPs that omit the operator's STOP gesture -- with a clean recording the"
        " agent ends the run on the replayed STOP edge well before this cap. Applies per run when"
        " ``--num_replays > 1``. Default is 600s (10 min)."
    ),
)
parser.add_argument(
    "--num_replays",
    type=int,
    default=1,
    help=(
        "Number of times to replay the MCAP back-to-back. Each replay rebuilds the IsaacTeleopDevice"
        " (re-opens the MCAP at frame 0) and resets the env in place without reloading Kit; the"
        " CloudXR runtime and Kit's OpenXR session stay alive across runs so subsequent replays"
        " start ~instantly. Per-run and aggregated success/failure rates are reported in the stats"
        " summary; the exit code reflects the worst outcome across runs. Default 1."
    ),
)
parser.add_argument(
    "--stats_output_file",
    type=str,
    default=None,
    help=(
        "Optional path to write a JSON stats report (CPU frame time, FPS, outcome) to after the"
        " run(s) complete. When omitted only a stdout summary is printed. Schema is documented"
        " in the 'Stats output' section of the script's module docstring."
    ),
)
parser.add_argument(
    "--replay_start_delay_s",
    type=float,
    default=0.0,
    help=(
        "Optional wall-clock buffer added on top of the deterministic stage-load wait."
        " The agent always blocks until omni.usd reports no assets pending and then renders a"
        " fixed number of settle frames before consuming MCAP frames; this flag inserts an"
        " additional render-only window after that if the deterministic check is not enough"
        " for a given hardware/asset combination. Default is 0s -- bump it if you still see"
        " a race after the deterministic wait."
    ),
)
parser.add_argument(
    "--max_stage_load_wait_s",
    type=float,
    default=300.0,
    help=(
        "Safety cap on how long to wait for omni.usd to finish loading the stage before"
        " proceeding anyway. Hit only when something is misconfigured (missing asset, slow"
        " Nucleus, etc.); a warning is logged and replay continues. Default is 300s."
    ),
)
parser.add_argument(
    "--cloudxr_env",
    type=str,
    default=None,
    help=(
        "Path to a CloudXR ``.env`` file, or a shorthand: 'cloudxrjs' (Quest/Pico) or 'avp'"
        " (Apple Vision Pro). Default is None -- CloudXR is not launched. Pair with"
        " AppLauncher's ``--xr`` and Kit-side AR-profile settings for spectate-on-headset"
        " replay; see the script docstring for the full command."
    ),
)
parser.add_argument(
    "--auto_launch_cloudxr",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Auto-launch the CloudXR runtime when ``--cloudxr_env`` is set. Use"
        " ``--no-auto_launch_cloudxr`` to skip the launch (e.g. when running the"
        " runtime externally). Ignored when ``--cloudxr_env`` is omitted."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import gymnasium as gym
import torch
from isaaclab_teleop import IsaacTeleopDevice, create_isaac_teleop_device, poll_control_events

from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.envs import ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

logger = logging.getLogger(__name__)

_CLOUDXR_ENV_SHORTHANDS: dict[str, str] = {}


# ----------------------------------------------------------------------
# Perf stats: per-run collection + multi-run reporting
# ----------------------------------------------------------------------

_STATS_SCHEMA_VERSION = 1
"""Bump when the JSON shape produced by :func:`_build_report` changes in a
non-additive way (renamed / removed keys). Additive changes (new optional
keys) do not require a bump."""


@dataclass
class _RunStats:
    """Per-replay performance + outcome record.

    ``active_frame_times_ms`` is sampled only on iterations where
    ``env.step()`` actually ran (post-START, pre-terminator). Pre-START
    render-only frames, warmup ticks, and post-quit render-only spin-down
    are intentionally excluded so the resulting stats reflect the steady-
    state replay workload rather than the agent's bookkeeping overhead.
    """

    outcome: str = "incomplete"  # "success" | "failure" | "incomplete" | "timeout"
    active_frame_times_ms: list[float] = field(default_factory=list)
    active_duration_s: float = 0.0
    success_step_count: int = 0  # final consecutive-success counter at terminator
    # Filled in at run end from ``GpuStatsProvider.summary()``.
    # Defaults to ``{}`` so a run that never reached the active loop
    # produces a missing-but-not-None ``"gpu_stats"`` slot.
    gpu_stats: dict = field(default_factory=dict)

    def to_dict(self, run_index: int) -> dict:
        cpu_stats = _compute_frame_stats(self.active_frame_times_ms)
        fps_stats = _compute_fps_stats(self.active_frame_times_ms)
        return {
            "run_index": run_index,
            "outcome": self.outcome,
            "active_iterations": len(self.active_frame_times_ms),
            "active_duration_s": round(self.active_duration_s, 6),
            "success_step_count": self.success_step_count,
            "cpu_frame_time_ms": cpu_stats,
            "fps": fps_stats,
            "gpu_stats": self.gpu_stats,
        }


def _compute_frame_stats(samples_ms: list[float]) -> dict:
    """Compute summary stats for a list of per-frame CPU times (in ms).

    Uses :func:`statistics.quantiles` with ``n=100, method="inclusive"`` so
    the result is a stable in-process measurement with no numpy dependency.
    The 99 quantile cut-points returned by ``quantiles`` are interpreted as
    p1..p99; we sample p50 / p90 / p95 / p99 plus mean / min / max /
    stddev / n. Handles empty (``n=0``, all numeric fields ``None``) and
    single-sample (``n=1``, all numeric fields equal to the single sample,
    ``stddev=0.0``) inputs without raising.
    """
    n = len(samples_ms)
    if n == 0:
        return {
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "stddev": None,
            "n": 0,
        }
    sample_min = min(samples_ms)
    sample_max = max(samples_ms)
    sample_mean = statistics.fmean(samples_ms)
    if n == 1:
        return {
            "mean": sample_mean,
            "p50": samples_ms[0],
            "p90": samples_ms[0],
            "p95": samples_ms[0],
            "p99": samples_ms[0],
            "min": sample_min,
            "max": sample_max,
            "stddev": 0.0,
            "n": 1,
        }
    # quantiles(n=100) returns 99 cut points; cuts[i] is the (i+1)-th percentile.
    cuts = statistics.quantiles(samples_ms, n=100, method="inclusive")
    return {
        "mean": sample_mean,
        "p50": cuts[49],
        "p90": cuts[89],
        "p95": cuts[94],
        "p99": cuts[98],
        "min": sample_min,
        "max": sample_max,
        "stddev": statistics.stdev(samples_ms),
        "n": n,
    }


def _compute_fps_stats(samples_ms: list[float]) -> dict:
    """Compute env.step-throughput FPS stats from per-step CPU times.

    All three fields are derived from the same ``samples_ms`` series
    that feeds ``cpu_frame_time_ms`` and stay self-consistent with it:
    ``mean == 1000 / cpu_frame_time_ms.mean`` (harmonic mean of FPS =
    total frames / total step time). The harmonic mean is what
    Devdeep's "use harmonic mean for FPS and it will agree with the
    arithmetic mean of frame time" prescription expects -- it avoids
    the upward bias of arithmetic-mean-of-instantaneous-FPS
    (dominated by the fastest frames) and the downward bias of
    ``n / active_duration_s`` (dragged down by inter-step
    bookkeeping).

    Note that this is the ``env.step`` rate, not Kit's render rate:
    Kit pumps ``cfg.decimation / cfg.sim.render_interval`` frames per
    ``env.step`` call, so the HUD shows a higher number than what is
    reported here. Compute the render rate as
    ``fps.mean * decimation / render_interval`` from the env config
    if needed.

    ``min_instantaneous`` and ``max_instantaneous`` are derived from
    the slowest / fastest individual step respectively.
    """
    n = len(samples_ms)
    if n == 0:
        return {"mean": None, "min_instantaneous": None, "max_instantaneous": None}
    sample_mean_ms = statistics.fmean(samples_ms)
    sample_max_ms = max(samples_ms)
    sample_min_ms = min(samples_ms)
    return {
        "mean": 1000.0 / sample_mean_ms if sample_mean_ms > 0 else None,
        "min_instantaneous": 1000.0 / sample_max_ms if sample_max_ms > 0 else None,
        "max_instantaneous": 1000.0 / sample_min_ms if sample_min_ms > 0 else None,
    }


# =============================================================================
# GPU statistics
# =============================================================================
#
# ``GpuStatsProvider`` is the modularity seam. The agent constructs a
# provider at the start of each run, calls :meth:`sample` once per
# active iteration, and consumes :meth:`summary` at run end to embed
# the resulting dict under ``"gpu_stats"`` in the per-run report.
#
# ``NvmlGpuStatsProvider`` (default) is renderer-agnostic: it queries
# NVML directly via ``pynvml`` and works wherever an NVIDIA driver is
# installed -- no Kit dependency, no CUDA context needed. If you swap
# the renderer out (e.g. move to a non-Kit visualization), this
# provider still works.
#
# To add a renderer-specific provider in the future (Kit viewport
# telemetry, Newton, etc.), define a class with the same
# ``sample`` / ``summary`` signature and instantiate it inside
# ``_run_single_replay`` in place of ``NvmlGpuStatsProvider``.


@runtime_checkable
class GpuStatsProvider(Protocol):
    """Per-run GPU telemetry source.

    Renderer-agnostic interface for sampling GPU state during a
    replay. Implementations are expected to be cheap on
    :meth:`sample` (<<1 ms; the agent calls it once per active
    iteration in the hot path) and to return a JSON-serializable
    dict from :meth:`summary` matching the shape documented on the
    concrete impl.
    """

    def sample(self) -> None:
        """Snapshot current GPU state. Called once per active iteration."""
        ...

    def summary(self) -> dict:
        """Return the aggregated stats fragment for the run, embedded
        as the ``"gpu_stats"`` value in the run's report dict."""
        ...


class NvmlGpuStatsProvider:
    """NVML-backed :class:`GpuStatsProvider`.

    Snapshots GPU utilization (%) and used memory (MB) for one
    device per :meth:`sample` call via ``pynvml``. Per-call cost is
    <100 us so per-frame sampling is fine at any realistic frame
    rate. Soft-fails when ``pynvml`` is missing or initialization
    fails (no NVIDIA driver, etc.); :meth:`sample` is then a no-op
    and :meth:`summary` reports the failure reason.

    Args:
        device_index: NVML device index (typically 0 for the
            workstation's primary GPU). Defaults to 0.

    Summary shape on success::

        {
          "backend": "nvml",
          "available": True,
          "device_index": 0,
          "device_name": "NVIDIA GeForce RTX 4090",
          "memory_total_mb": 24564.0,
          "utilization_percent": {<frame_stats shape>},
          "memory_used_mb": {<frame_stats shape>}
        }

    On failure the ``"backend"`` / ``"available"`` fields are still
    present plus a ``"reason"`` string.
    """

    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._available = False
        self._reason: str | None = None
        self._device_name: str | None = None
        self._memory_total_mb: float | None = None
        self._util_samples: list[float] = []
        self._mem_used_samples_mb: list[float] = []
        try:
            import pynvml

            self._pynvml = pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._memory_total_mb = pynvml.nvmlDeviceGetMemoryInfo(self._handle).total / (1024 * 1024)
            # nvmlDeviceGetName returns bytes on older bindings and str on newer ones; coerce both.
            name = pynvml.nvmlDeviceGetName(self._handle)
            self._device_name = name.decode("utf-8") if isinstance(name, bytes) else str(name)
            self._available = True
        except ImportError:
            self._reason = "pynvml not installed (`pip install nvidia-ml-py` to enable)"
        except Exception as exc:
            self._reason = f"NVML init failed: {exc}"

    def sample(self) -> None:
        if not self._available:
            return
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._util_samples.append(float(util.gpu))
            self._mem_used_samples_mb.append(mem.used / (1024 * 1024))
        except Exception:
            # Transient NVML query failures shouldn't kill the replay loop;
            # missing samples are reflected in the final ``n`` count.
            pass

    def summary(self) -> dict:
        if not self._available:
            return {"backend": "nvml", "available": False, "reason": self._reason}
        return {
            "backend": "nvml",
            "available": True,
            "device_index": self._device_index,
            "device_name": self._device_name,
            "memory_total_mb": round(self._memory_total_mb, 1) if self._memory_total_mb is not None else None,
            "utilization_percent": _compute_frame_stats(self._util_samples),
            "memory_used_mb": _compute_frame_stats(self._mem_used_samples_mb),
        }


def _build_report(args, all_runs: list[_RunStats]) -> dict:
    """Build the structured JSON report dict from a list of completed runs."""
    outcomes_count = {"success": 0, "failure": 0, "incomplete": 0, "timeout": 0}
    for r in all_runs:
        outcomes_count[r.outcome] = outcomes_count.get(r.outcome, 0) + 1

    total = max(len(all_runs), 1)
    success_rate = outcomes_count.get("success", 0) / total

    run_dicts = [r.to_dict(i) for i, r in enumerate(all_runs)]

    return {
        "schema_version": _STATS_SCHEMA_VERSION,
        "task": args.task,
        "replay_file": args.replay_file,
        "num_replays": len(all_runs),
        "outcomes": outcomes_count,
        "success_rate": success_rate,
        "runs": run_dicts,
        "aggregate": _aggregate_runs(run_dicts),
    }


def _aggregate_runs(run_dicts: list[dict]) -> dict:
    """Aggregate per-run CPU / FPS stats across a multi-run batch.

    Returns ``mean_of_means``, ``p90_of_p90s``, etc. so reviewers can scan
    a one-line summary without recomputing from individual runs. Runs that
    produced no active frames (e.g. the recording never reached START)
    contribute no samples and are skipped for that field; if no run had
    samples the aggregate value is ``None``.
    """

    def _gather(key_path: list[str]) -> list[float]:
        out: list[float] = []
        for run in run_dicts:
            value = run
            for key in key_path:
                if value is None:
                    break
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, (int, float)):
                out.append(float(value))
        return out

    def _mean_or_none(values: list[float]) -> float | None:
        return statistics.fmean(values) if values else None

    def _min_or_none(values: list[float]) -> float | None:
        return min(values) if values else None

    def _max_or_none(values: list[float]) -> float | None:
        return max(values) if values else None

    return {
        "cpu_frame_time_ms": {
            "mean_of_means": _mean_or_none(_gather(["cpu_frame_time_ms", "mean"])),
            "mean_of_p90s": _mean_or_none(_gather(["cpu_frame_time_ms", "p90"])),
            "mean_of_p99s": _mean_or_none(_gather(["cpu_frame_time_ms", "p99"])),
            "min_overall": _min_or_none(_gather(["cpu_frame_time_ms", "min"])),
            "max_overall": _max_or_none(_gather(["cpu_frame_time_ms", "max"])),
        },
        "fps": {
            "mean_of_means": _mean_or_none(_gather(["fps", "mean"])),
        },
        "gpu_stats": {
            "utilization_percent": {
                "mean_of_means": _mean_or_none(_gather(["gpu_stats", "utilization_percent", "mean"])),
                "mean_of_p90s": _mean_or_none(_gather(["gpu_stats", "utilization_percent", "p90"])),
                "max_overall": _max_or_none(_gather(["gpu_stats", "utilization_percent", "max"])),
            },
            "memory_used_mb": {
                "mean_of_means": _mean_or_none(_gather(["gpu_stats", "memory_used_mb", "mean"])),
                "mean_of_p90s": _mean_or_none(_gather(["gpu_stats", "memory_used_mb", "p90"])),
                "max_overall": _max_or_none(_gather(["gpu_stats", "memory_used_mb", "max"])),
            },
        },
    }


def _print_stdout_summary(report: dict) -> None:
    """Print a one-line-per-run summary plus an aggregate line to stdout."""
    runs = report.get("runs", [])
    total = report.get("num_replays", len(runs))

    def _fmt(value: float | None, suffix: str = "") -> str:
        return f"{value:.2f}{suffix}" if isinstance(value, (int, float)) else "n/a"

    def _gpu_segment(gpu_stats: dict) -> str:
        """Render a compact GPU summary suffix, or empty when unavailable."""
        if not isinstance(gpu_stats, dict) or not gpu_stats.get("available"):
            return ""
        util = gpu_stats.get("utilization_percent") or {}
        mem = gpu_stats.get("memory_used_mb") or {}
        return f" | gpu={_fmt(util.get('mean'), '%')} mem={_fmt(mem.get('max'), 'MB')}"

    print("--- Replay stats ---")
    for run in runs:
        idx = run["run_index"] + 1
        cpu = run["cpu_frame_time_ms"]
        fps = run["fps"]
        print(
            f"Replay {idx}/{total}: outcome={run['outcome']}"
            f" | frames={run['active_iterations']}"
            f" | active={run['active_duration_s']:.2f}s"
            f" | mean={_fmt(cpu['mean'], 'ms')}"
            f" p90={_fmt(cpu['p90'], 'ms')}"
            f" p99={_fmt(cpu['p99'], 'ms')}"
            f" | mean_fps={_fmt(fps['mean'])}"
            f"{_gpu_segment(run.get('gpu_stats', {}))}"
        )

    succ = report["outcomes"].get("success", 0)
    agg = report["aggregate"]
    agg_gpu = agg.get("gpu_stats", {})
    agg_util = agg_gpu.get("utilization_percent", {}) if isinstance(agg_gpu, dict) else {}
    agg_mem = agg_gpu.get("memory_used_mb", {}) if isinstance(agg_gpu, dict) else {}
    print(
        f"Aggregate: success_rate={succ}/{total} ({report['success_rate']:.2f})"
        f" | mean_fps={_fmt(agg['fps']['mean_of_means'])}"
        f" | mean_p90={_fmt(agg['cpu_frame_time_ms']['mean_of_p90s'], 'ms')}"
        f" | mean_gpu={_fmt(agg_util.get('mean_of_means'), '%')}"
        f" | max_mem={_fmt(agg_mem.get('max_overall'), 'MB')}"
    )


def _write_json_report(path: str, report: dict) -> None:
    """Persist the report to ``path`` as a UTF-8 JSON file (pretty-printed)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=False)
        fh.write("\n")
    print(f"Stats report written to {path}")


def _exit_code_for_outcomes(all_runs: list[_RunStats]) -> int:
    """Map the worst-of-N replay outcome to a CI-friendly exit code.

    Precedence: ``timeout`` > ``failure`` > ``incomplete`` > ``success``.
    Any single bad run fails the whole batch; timeouts get their own code
    so CI can distinguish a perf cliff from a broken trajectory.
    """
    if not all_runs:
        return 1
    outcomes = {r.outcome for r in all_runs}
    if outcomes <= {"success"}:
        return 0
    if "timeout" in outcomes:
        return 2
    return 1  # any "failure" or "incomplete"


def _resolve_cloudxr_env(value: str | None) -> str | None:
    """Resolve ``--cloudxr_env`` shorthands to absolute ``.env`` file paths.

    Mirrors :func:`scripts.tools.record_demos._resolve_cloudxr_env` so the same
    short names (``"cloudxrjs"``, ``"avp"``) behave identically on the
    recording and replay sides. Accepts ``"none"`` / empty / ``None`` to mean
    "no CloudXR" and otherwise returns the value unchanged.
    """
    if value is None or value.strip() == "" or value.lower() == "none":
        return None
    if not _CLOUDXR_ENV_SHORTHANDS:
        from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV

        _CLOUDXR_ENV_SHORTHANDS["cloudxrjs"] = CLOUDXR_JS_ENV
        _CLOUDXR_ENV_SHORTHANDS["avp"] = CLOUDXR_AVP_ENV
    return _CLOUDXR_ENV_SHORTHANDS.get(value.lower(), value)


def _maybe_launch_cloudxr(cloudxr_env_path: str | None, auto_launch: bool):
    """Launch a CloudXR runtime owned at the agent (batch) scope.

    The CloudXR runtime is process-scoped, not session-scoped: tearing it
    down between teleop sessions in the same Kit process severs Kit's
    OpenXR runtime IPC, so the next session's ``xrCreateInstance`` fails
    with ``XR_ERROR_RUNTIME_UNAVAILABLE`` and the XR pipeline hangs. To
    support multi-run XR replay we hoist CloudXR ownership out of the
    per-run :class:`~isaaclab_teleop.session_lifecycle.TeleopSessionLifecycle`
    (which would otherwise stop it on every device teardown) into the
    agent. Each replay's ``IsaacTeleopDevice`` is constructed with
    ``auto_launch_cloudxr=False`` so the lifecycle leaves the runtime
    alone; the agent terminates the launcher in its ``finally`` block.

    Mirrors the gating in :meth:`TeleopSessionLifecycle._ensure_cloudxr_runtime`
    (``--cloudxr_env`` set, ``--auto_launch_cloudxr`` enabled,
    ``ISAACLAB_CXR_SKIP_AUTOLAUNCH=1`` env var not set) so behavior parity
    is preserved.

    Args:
        cloudxr_env_path: Resolved CloudXR ``.env`` file path, or ``None``
            to skip launching.
        auto_launch: Whether to honor the request (mirrors
            ``--auto_launch_cloudxr``).

    Returns:
        The launched ``CloudXRLauncher`` instance, or ``None`` when
        nothing should be launched. Caller is responsible for calling
        ``.stop()`` at the end of the batch.
    """
    if cloudxr_env_path is None or not auto_launch:
        return None

    if os.environ.get("ISAACLAB_CXR_SKIP_AUTOLAUNCH", "").strip() == "1":
        logger.info("CloudXR auto-launch skipped (ISAACLAB_CXR_SKIP_AUTOLAUNCH=1)")
        return None

    from isaacteleop.cloudxr import CloudXRLauncher

    launcher = CloudXRLauncher(
        install_dir=str(Path.home() / ".cloudxr"),
        env_config=cloudxr_env_path,
        accept_eula=False,
    )
    logger.info("CloudXR runtime launched (process-scoped, shared across replays)")
    return launcher


def _prepare_env_cfg(task: str, num_envs: int, device: str) -> tuple[ManagerBasedRLEnvCfg, object | None]:
    """Build and tweak an env config suitable for non-interactive replay.

    Mirrors the env-config mutations performed by ``record_demos.py``'s
    :func:`create_environment_config`:

    * The ``success`` term is extracted and cleared from the env config so the
      script can drive success detection (and the matching reset cycle)
      explicitly via :func:`_process_success_condition`, gated by
      ``--num_success_steps``. This matches record_demos.py's pattern of
      manually counting consecutive success steps before resetting.
    * The ``time_out`` term is cleared for the same reason it is cleared in
      :file:`scripts/tools/record_demos.py` and
      :file:`scripts/imitation_learning/robomimic/play.py`: a recorded
      trajectory often exceeds ``episode_length_s`` (pick-place is 20s by
      default; a successful operator demo can easily run 25-30s). With the
      term active, the env auto-truncates partway through the MCAP, resets
      to the default pose, and the remainder of the recorded actions get
      retargeted against the freshly-reset robot -- which manifests as
      "robot moves correctly for a bit, then snaps back / acts wrong."
      The recorder itself did not run with ``time_out`` enabled, so
      reproducing record-time semantics requires clearing it here too.
    * Other failure terms (e.g. ``object_dropping``, ``object_too_far``)
      are left active. ``env.step`` then auto-invokes ``_reset_idx`` for any
      env whose termination fires; the main loop detects this via the
      returned ``terminated``/``truncated`` tensors and completes the reset
      cycle (sim reinit + teleop device reset) so Pink IK starts the next
      attempt with fresh articulation views.

    Returns:
        Tuple ``(env_cfg, success_term)``. ``success_term`` is ``None`` when
        the env doesn't define a ``success`` termination term.
    """
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    env_cfg.env_name = task.split(":")[-1]
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "teleop_replay_agent only supports ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )
    success_term: object | None = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        logger.warning(
            "No success termination term was found in the environment;"
            " success-driven resets will not fire during replay."
        )
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    env_cfg = remove_camera_configs(env_cfg)
    env_cfg.sim.render.antialiasing_mode = "DLSS"
    return env_cfg, success_term


def _handle_reset(env: gym.Env, teleop_interface: IsaacTeleopDevice) -> None:
    """Run the full env+teleop reset cycle used by ``record_demos.py``.

    Mirrors :func:`scripts.tools.record_demos.handle_reset` (sans the
    instruction-display update, which the headless replay agent doesn't
    own). ``env.sim.reset()`` does the hard physics reinit that keeps Pink
    IK seeded against fresh articulation views; see the initial-reset note
    in :func:`main`. ``env.recorder_manager.reset()`` is a no-op when no
    recorders are configured (the default for this script), but kept for
    parity with record_demos.py so future recorder additions don't have to
    re-derive the call sequence.
    """
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    teleop_interface.reset()


def _process_success_condition(
    env: gym.Env,
    success_term: object | None,
    success_step_count: int,
    num_success_steps: int,
) -> tuple[int, bool]:
    """Track consecutive success steps and decide whether to reset.

    Mirrors :func:`scripts.tools.record_demos.process_success_condition`
    minus the recorder-export side effects, which this script does not own.

    Returns:
        Tuple ``(updated_success_step_count, reset_due_to_success)``.
    """
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= num_success_steps:
            print(f"Success condition met after {success_step_count} consecutive steps; resetting env.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


_RENDERER_SETTLE_FRAMES: int = 30
"""Number of additional render frames pumped after the USD stage finishes loading.

Kit's stage-load status flips to ``count_loading == 0`` as soon as every referenced asset
has been resolved, but the renderer pipeline (shader compilation, articulation-view
binding, material warm-up) typically needs a few more event-loop ticks to converge. Thirty
frames at the default Kit render cadence is ~0.5 s on most machines and is deterministic
per-machine -- unlike a wall-clock delay it does not have to be tuned for hardware.
"""


def _wait_for_stage_load(simulation_app, max_wait_s: float) -> None:
    """Block until the USD stage finishes resolving every referenced asset.

    Polls :meth:`omni.usd.UsdContext.get_stage_loading_status`. The third element of
    the returned tuple is the count of assets Kit still has pending; when it reaches
    zero the stage is fully streamed in and the renderer pipeline is ready to draw
    against it. After the count reaches zero this function pumps an additional
    :data:`_RENDERER_SETTLE_FRAMES` ``simulation_app.update()`` calls so shaders,
    materials, and articulation views finish warming up before the caller begins
    consuming MCAP frames or stepping the env.

    Args:
        simulation_app: The :class:`isaaclab.app.SimulationApp` instance whose
            event loop to pump while waiting.
        max_wait_s: Upper bound on how long to spin on a non-zero loading count
            before warning and returning. Acts as a safety net for misconfigured
            scenes (missing assets, slow Nucleus); a successful run typically
            completes well within this bound.

    The function is best-effort: when ``omni.usd`` is unavailable (e.g. when
    running outside a Kit context) it returns immediately so callers do not
    need a separate code path.
    """
    try:
        import omni.usd
    except (ImportError, ModuleNotFoundError):
        logger.warning("omni.usd not available; skipping deterministic stage-load wait")
        return

    print("Waiting for USD stage to finish loading...")
    start_s = time.monotonic()
    last_progress_log_s = start_s
    while simulation_app.is_running():
        context = omni.usd.get_context()
        if context is None:
            break
        # get_stage_loading_status -> (message, count_loaded, count_loading)
        _, _, count_loading = context.get_stage_loading_status()
        if count_loading == 0:
            break
        elapsed_s = time.monotonic() - start_s
        if elapsed_s >= max_wait_s:
            logger.warning(
                "Stage still reports %d assets pending after %.1fs; proceeding anyway. Replay may race the renderer.",
                count_loading,
                max_wait_s,
            )
            break
        if time.monotonic() - last_progress_log_s >= 5.0:
            print(f"  stage loading: {count_loading} assets pending (elapsed {elapsed_s:.1f}s)")
            last_progress_log_s = time.monotonic()
        simulation_app.update()

    elapsed_s = time.monotonic() - start_s
    print(f"Stage load complete after {elapsed_s:.1f}s; settling renderer for {_RENDERER_SETTLE_FRAMES} frames...")
    for _ in range(_RENDERER_SETTLE_FRAMES):
        if not simulation_app.is_running():
            return
        simulation_app.update()


def _run_single_replay(
    env: gym.Env,
    isaac_teleop_cfg,
    success_term: object | None,
    run_index: int,
    total_runs: int,
) -> _RunStats:
    """Run a single replay pass against ``env`` and return the per-run stats.

    Builds a fresh :class:`IsaacTeleopDevice` so each call re-opens the MCAP
    reader at frame 0 -- exiting and re-entering the device's context manager
    tears down the previous ``TeleopSession`` and constructs a new one. The
    USD stage is left untouched; the caller (``main``) is responsible for
    building / closing ``env`` once across the full multi-run batch.

    Per-frame sampling is restricted to iterations where ``env.step()``
    actually ran (post-START, pre-terminator). Pre-START render-only
    frames, warmup ticks, and reset cycles do not contribute to the
    returned stats so the numbers reflect the steady-state replay
    workload rather than agent bookkeeping overhead.

    Args:
        env: The (already built) Isaac Lab environment, shared across runs.
        isaac_teleop_cfg: The :class:`IsaacTeleopCfg` extracted from
            ``env_cfg.isaac_teleop``.
        success_term: The original ``success`` termination term (or ``None``);
            forwarded to :func:`_process_success_condition` each frame.
        run_index: Zero-indexed run number within the multi-run batch.
            Used to gate one-time work (the deterministic stage-load wait
            only runs on ``run_index == 0``).
        total_runs: Total number of runs in the batch; used only for log
            framing ("Replay 1/5: ...").
    """
    stats = _RunStats()

    # Default NVML-backed provider samples GPU utilization + used
    # memory once per active iteration. ``NvmlGpuStatsProvider``
    # soft-fails (the summary dict carries ``available: False`` and a
    # reason) when ``nvidia-ml-py`` is missing or the driver is
    # unreachable, so a missing GPU never blocks the replay. To swap
    # in a renderer-specific provider (Kit viewport telemetry, Newton,
    # ...) implement the :class:`GpuStatsProvider` Protocol and
    # construct it here in place of :class:`NvmlGpuStatsProvider`.
    gpu_stats_provider: GpuStatsProvider = NvmlGpuStatsProvider()
    if run_index == 0 and not getattr(gpu_stats_provider, "_available", False):
        print(f"[GPU stats] disabled: {getattr(gpu_stats_provider, '_reason', 'unknown reason')}")

    # CloudXR is owned by the agent (see ``_maybe_launch_cloudxr`` in
    # ``main``), so the per-run lifecycle must not try to launch -- or
    # stop -- it. ``cloudxr_env_file=None`` + ``auto_launch_cloudxr=False``
    # short-circuits ``TeleopSessionLifecycle._ensure_cloudxr_runtime`` so
    # the runtime survives across replays and Kit's OpenXR session keeps
    # its IPC connection.
    teleop_interface = create_isaac_teleop_device(
        isaac_teleop_cfg,
        sim_device=args_cli.device,
        callbacks={},
        cloudxr_env_file=None,
        auto_launch_cloudxr=False,
        mcap_replay_path=args_cli.replay_file,
    )
    if run_index == 0:
        print(f"Using teleop device: {teleop_interface}")

    with teleop_interface:
        # Mirror the reset sequence used by ``record_demos.py``: ``sim.reset()``
        # does a hard physics reinit (re-binds articulation views, plays the
        # timeline) that ``env.reset()`` alone does not perform. Pink IK reads
        # ``data.joint_pos.torch`` every step to seed Pinocchio's configuration
        # and to compute ``target = curr + delta``; if the articulation view is
        # stale, every IK call produces zero-delta arm targets while the
        # hand-finger path (which bypasses IK) keeps tracking. See PR #5507.
        env.sim.reset()
        env.reset()
        teleop_interface.reset()

        # Deterministic warmup is only required on the first run -- once the
        # stage has been streamed in and the renderer settled, subsequent
        # runs share the same stage and only need the per-run ``env.sim.reset``
        # above. Skipping it on later runs saves the renderer-settle frames
        # at the cost of doing nothing measurable (the wait returns
        # immediately on a fully-loaded stage anyway).
        if run_index == 0:
            _wait_for_stage_load(simulation_app, args_cli.max_stage_load_wait_s)

            # Optional extra wall-clock buffer on top of the deterministic
            # wait. Useful as an escape hatch when the deterministic check
            # is not enough (e.g. very slow shader compilation paths).
            if args_cli.replay_start_delay_s > 0:
                print(
                    f"Additional warmup buffer: rendering for {args_cli.replay_start_delay_s:.1f}s"
                    " before consuming MCAP frames."
                )
                buffer_start_s = time.monotonic()
                while simulation_app.is_running() and time.monotonic() - buffer_start_s < args_cli.replay_start_delay_s:
                    env.sim.render()

        print(
            f"Replay {run_index + 1}/{total_runs} started; replaying MCAP from {args_cli.replay_file}"
            f" (max_replay_duration_s={args_cli.max_replay_duration_s:.1f})."
        )
        teleop_active = False
        teleop_was_active = False  # only terminate on STOP after a real START
        success_step_count = 0
        replay_start_s = time.monotonic()
        # First time we run env.step on this replay; used to bound the
        # active duration that drives mean-FPS. Stays None until a sample
        # is recorded so render-only / pre-START frames don't widen the
        # window.
        active_start_s: float | None = None
        # End-of-active-window timestamp. Updated each sampled iteration
        # so the active duration is always "first active iter -> last
        # active iter" even when a terminator fires mid-loop and the
        # subsequent renders are excluded.
        last_active_end_s: float | None = None

        while simulation_app.is_running():
            try:
                with torch.inference_mode():
                    # Wall-clock safety cap. Only hit when the recording
                    # never reaches a natural terminator -- e.g. it omits
                    # an operator STOP AND the env's success/failure
                    # terms never fire within the configured window. A
                    # clean ``record_demos.py``-style capture exits on
                    # the success edge well before this triggers.
                    elapsed_s = time.monotonic() - replay_start_s
                    if elapsed_s >= args_cli.max_replay_duration_s:
                        print(f"Replay reached max_replay_duration_s={args_cli.max_replay_duration_s:.1f}; ending run.")
                        stats.outcome = "timeout"
                        break

                    action = teleop_interface.advance()
                    ctrl = poll_control_events(teleop_interface)

                    # Track active state from the replayed _teleop_control
                    # channel. ``ctrl.is_active`` follows the same shape
                    # that ``record_demos.py`` and ``teleop_se3_agent.py``
                    # consume; None means "no transition this frame."
                    prev_active = teleop_active
                    if ctrl.is_active is not None:
                        teleop_active = ctrl.is_active
                    if teleop_active:
                        teleop_was_active = True

                    # End-of-run on the first STOP edge after a real
                    # START -- the operator pressed Stop during
                    # recording, and ``ReplayMessageChannelTrackerImpl``
                    # surfaces that payload at the same recording-frame
                    # index it was captured at. Per-frame tracker EOF on
                    # its own does NOT trigger this branch:
                    # :class:`TeleopMessageProcessor` keeps emitting
                    # valid False booleans for KILL / RUN_TOGGLE / RESET
                    # after the message-channel MCAP exhausts, so the
                    # state manager stays in its last state and
                    # ``ctrl.is_active`` does not flip. Recordings
                    # without an operator STOP are terminated instead by
                    # the success / failure / wall-clock terminators
                    # below.
                    if prev_active and not teleop_active and teleop_was_active:
                        print("Replay end observed (STOP edge); ending run.")
                        break

                    if ctrl.should_reset:
                        _handle_reset(env, teleop_interface)
                        success_step_count = 0
                        continue

                    # Gate stepping on the active state (mirrors
                    # teleop_se3_agent.py:309-328). Pre-START operator
                    # setup frames render only; the recorded START flips
                    # us into the stepping branch.
                    if action is None or not teleop_active:
                        env.sim.render()
                        continue

                    # Sample CPU frame time across the env.step call only
                    # (the active-frame window the stats report covers).
                    iter_start_s = time.perf_counter()
                    if active_start_s is None:
                        active_start_s = iter_start_s
                    actions = action.repeat(env.num_envs, 1)
                    _, _, terminated, truncated, _ = env.step(actions)
                    iter_end_s = time.perf_counter()
                    stats.active_frame_times_ms.append((iter_end_s - iter_start_s) * 1000.0)
                    last_active_end_s = iter_end_s

                    # Snapshot GPU state right after env.step so the
                    # sample reflects the active workload (post-render
                    # for that frame). Provider is cheap (~50 us for
                    # NVML, no-op when disabled).
                    gpu_stats_provider.sample()

                    # Failure path: ``env.step`` already invoked
                    # ``_reset_idx`` for any env whose task-specific
                    # failure term fired (``time_out`` was cleared by
                    # ``_prepare_env_cfg``; ``success`` is handled
                    # below).
                    #
                    # Replay-specific behavior: a failure mid-trajectory
                    # means the recorded demo did not reproduce -- the
                    # operator has no agency to recover here, so the
                    # rest of the MCAP would just feed retargeted
                    # actions to a freshly-reset env, which is not
                    # meaningful replay. End the run with a failure
                    # outcome so the batch's exit code reflects it.
                    if bool(terminated.any().item()) or bool(truncated.any().item()):
                        print("Replay failure: env terminated/truncated mid-trajectory; ending run.")
                        stats.outcome = "failure"
                        break

                    # Success path: ``success_term`` was cleared from the
                    # env cfg so ``env.step`` does not auto-reset on it.
                    # ``_process_success_condition`` consults the original
                    # success term and reports when it has held for
                    # ``--num_success_steps`` consecutive steps.
                    #
                    # Replay-specific behavior: success is the natural
                    # end-of-replay for ``record_demos.py``-style single
                    # episode captures, so end the run here instead of
                    # invoking ``_handle_reset`` like the live agent
                    # does. The alternative -- resetting and continuing
                    # into the post-success MCAP tail -- would just
                    # replay operator wind-down frames (controller
                    # releases, idle motion before they hit Stop on
                    # recording), which is not meaningful demo data and
                    # quickly exhausts the per-frame tracker streams
                    # anyway.
                    success_step_count, reset_on_success = _process_success_condition(
                        env, success_term, success_step_count, args_cli.num_success_steps
                    )
                    if reset_on_success:
                        print("Recorded demo succeeded; ending run.")
                        stats.outcome = "success"
                        break
            except Exception:
                # ``logger.exception`` preserves the full traceback; bare
                # ``logger.error`` would only log the message. Classify as
                # ``failure`` so the per-run outcome and the batch exit
                # code reflect that the recorded trajectory did not
                # complete -- staying at the default ``incomplete`` would
                # silently mask a crash mid-replay in CI reports.
                logger.exception("Error during simulation step")
                stats.outcome = "failure"
                break

    # Stamp the active window duration. Falls back to 0.0 when no env
    # step ever ran (recording never reached START, or terminated before
    # the first active frame).
    if active_start_s is not None and last_active_end_s is not None:
        stats.active_duration_s = last_active_end_s - active_start_s
    stats.success_step_count = success_step_count
    stats.gpu_stats = gpu_stats_provider.summary()
    return stats


def main() -> int:
    """Replay a captured Isaac Teleop session against an Isaac Lab environment.

    Builds the env once, then loops :func:`_run_single_replay` for
    ``--num_replays`` iterations. Each iteration builds a fresh
    :class:`IsaacTeleopDevice` (so the MCAP reader reopens at frame 0) and
    resets the env in place; the USD stage stays loaded between runs so
    multi-run batches start essentially instantly.

    Per-replay control flow (see :func:`_run_single_replay` for details):
        * Pre-loop warmup: ``_wait_for_stage_load`` polls
          ``omni.usd.UsdContext.get_stage_loading_status`` until Kit
          reports zero pending assets, then renders a fixed number of
          settle frames (only on ``run_index == 0``). An optional
          ``--replay_start_delay_s`` buffer can be appended for hardware
          that needs more grace. ``advance()`` is not called during
          warmup so ``ReplaySession.update`` does not consume MCAP
          frames yet.
        * Main loop: :meth:`IsaacTeleopDevice.advance` returns an action
          tensor derived from the MCAP-replayed tracker stream and
          :func:`poll_control_events` returns the START / STOP / RESET
          edges replayed from the ``_teleop_control`` channel. The env
          steps only when ``ctrl.is_active`` is True, mirroring
          ``teleop_se3_agent.py`` and ``record_demos.py`` exactly --
          pre-START operator-setup frames render only.
        * End-of-replay terminators (any of these breaks the inner
          loop and sets ``_RunStats.outcome`` accordingly; the function
          returns and the outer batch driver moves to the next replay):
            1. Replayed STOP edge from ``_teleop_control`` -- the
               operator pressed Stop during recording. Does not
               overwrite ``outcome``: if success fired earlier in this
               run, the outcome stays ``"success"``; otherwise it stays
               ``"incomplete"``, since stopping without reaching
               success is not a successful reproduction.
            2. Success condition met for ``--num_success_steps``
               consecutive steps -- the natural end of a
               ``record_demos.py`` single-episode capture. Sets
               outcome ``"success"``. ``_handle_reset`` is intentionally
               skipped here because the post-success MCAP tail is
               operator wind-down, not demo data.
            3. ``env.step`` ``terminated`` / ``truncated`` -- a task-
               specific failure term fired during the recorded
               trajectory. Sets outcome ``"failure"``.
            4. Wall-clock ``--max_replay_duration_s`` safety cap. Sets
               outcome ``"timeout"``.

        Kit itself is left running between replays so a fresh
        :class:`IsaacTeleopDevice` can be constructed without reloading
        the USD stage; ``__main__`` calls ``simulation_app.close()``
        after the whole batch finishes.

    Stats output:
        Each iteration where ``env.step()`` ran contributes one CPU
        frame-time sample (``perf_counter`` delta in ms). At the end of
        the run the samples are summarised into mean / p50 / p90 / p95 /
        p99 / min / max / stddev + a mean / instantaneous-min / max FPS
        triple, then serialised into the ``runs[]`` array of the report
        dict. ``--stats_output_file`` controls whether the dict is
        persisted to disk; a one-line-per-run stdout summary is always
        printed. See the module docstring for the full JSON schema.

    Resource cleanup is wrapped in a ``try/finally`` so that ``env.close()``
    always runs, even when device construction or any subsequent setup
    raises -- otherwise the USD stage would leak across CI runs.

    Returns:
        The host process exit code, mapped from the worst-of-N outcome
        across the multi-run batch: ``0`` if every run's
        ``success_term`` fired, ``2`` if any run hit
        ``--max_replay_duration_s``, otherwise ``1`` (any failure or
        incomplete run).
    """
    env: gym.Env | None = None
    cloudxr_launcher = None
    all_runs: list[_RunStats] = []

    if args_cli.num_replays < 1:
        raise ValueError(f"--num_replays must be >= 1; got {args_cli.num_replays}")

    try:
        # CloudXR launch is hoisted to the agent (batch scope) so it
        # survives across per-run device teardown; per-run lifecycles
        # are explicitly told not to launch / stop it (see the
        # ``cloudxr_env_file=None, auto_launch_cloudxr=False`` call in
        # ``_run_single_replay``). This is what lets ``--num_replays > 1``
        # work in XR-active mode without losing Kit's OpenXR runtime
        # IPC between runs.
        cloudxr_launcher = _maybe_launch_cloudxr(
            _resolve_cloudxr_env(args_cli.cloudxr_env), args_cli.auto_launch_cloudxr
        )

        env_cfg, success_term = _prepare_env_cfg(args_cli.task, args_cli.num_envs, args_cli.device)

        if not hasattr(env_cfg, "isaac_teleop") or env_cfg.isaac_teleop is None:
            raise ValueError(
                f"Task '{args_cli.task}' does not configure an IsaacTeleop pipeline. "
                "MCAP replay requires env_cfg.isaac_teleop to be set."
            )

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

        for run_idx in range(args_cli.num_replays):
            run_stats = _run_single_replay(
                env=env,
                isaac_teleop_cfg=env_cfg.isaac_teleop,
                success_term=success_term,
                run_index=run_idx,
                total_runs=args_cli.num_replays,
            )
            all_runs.append(run_stats)
            print(f"Replay {run_idx + 1}/{args_cli.num_replays} outcome: {run_stats.outcome}")
            if not simulation_app.is_running():
                # Kit was closed externally mid-batch; stop the outer loop
                # rather than spawning a fresh device against a dead app.
                break
    finally:
        if env is not None:
            env.close()
            print("Environment closed")
        if cloudxr_launcher is not None:
            try:
                cloudxr_launcher.stop()
                logger.info("CloudXR runtime stopped (end of batch)")
            except Exception:
                logger.exception("Failed to stop CloudXR launcher cleanly")

    report = _build_report(args_cli, all_runs)
    _print_stdout_summary(report)
    if args_cli.stats_output_file is not None:
        _write_json_report(args_cli.stats_output_file, report)
    return _exit_code_for_outcomes(all_runs)


if __name__ == "__main__":
    exit_code = main()
    simulation_app.update()
    simulation_app.close()
    sys.exit(exit_code)
