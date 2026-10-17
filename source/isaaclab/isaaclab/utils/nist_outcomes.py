# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Native assembly outcomes linked to one benchmark invocation.

This module consumes CPU scalars supplied by the native loop. It never imports a
GPU runtime or waits for device work. Counts are current native assembly state,
not cumulative assembly work or independent physical-correctness verification.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import uuid
from pathlib import Path


def _error(error):
    return {"type": type(error).__name__, "message": str(error)}


def _phase(step, warmup, mode):
    return "ordinary" if mode == "ordinary_demo" else "warmup" if step <= warmup else "measurement"


def _overlap(start, end, warmup, mode):
    return {
        "observed_control_steps": end - start,
        "warmup_control_steps": max(0, min(end, warmup) - start) if mode != "ordinary_demo" else 0,
        "measured_control_steps": max(0, end - max(start, warmup)) if mode != "ordinary_demo" else 0,
    }


def _reasons(success, failed, terminated, timed_out):
    reasons = []
    if success:
        reasons.append("native_success")
    if failed:
        reasons.append("native_assembly_failed")
    if terminated and not success and not failed:
        reasons.append("other_termination")
    if timed_out:
        reasons.append("time_limit")
    return reasons


def _counts(episodes):
    ended = len(episodes)
    successes = sum(int(row["complete_board"]) for row in episodes)
    return {
        "ended_episodes": ended,
        "successful_assemblies": successes,
        "native_failed_episodes": sum(int(row["assembly_failed"]) for row in episodes),
        "terminated_episodes": sum(int(row["terminated"]) for row in episodes),
        "timed_out_episodes": sum(int(row["timed_out"]) for row in episodes),
        "success_fraction_of_ended": successes / ended if ended else None,
    }


def _summarize(episodes, active):
    result = _counts(episodes)
    result.update(
        active_episodes=len(active),
        unfinished_started_episodes=sum(int(row["observed_control_steps"] > 0) for row in active),
        active_zero_observed_step_episodes=sum(int(row["observed_control_steps"] == 0) for row in active),
        by_completion_phase={phase: _counts([row for row in episodes if row["completion_phase"] == phase])
                             for phase in ("warmup", "measurement", "ordinary")},
    )
    return result


def _make_report(context, configuration, episodes, active, cutoff, status, errors, error=None):
    report = {
        "schema": "nist-native-outcomes-v1",
        "invocation_id": context["invocation_id"],
        "measurement_invocation_id": context.get("measurement_invocation_id"),
        "mode": context["mode"],
        "boards": context["boards"],
        "parts_per_board": context["parts_per_board"],
        "requested_warmup_control_steps": context["requested_warmup_control_steps"],
        "requested_measured_control_steps": context["requested_measured_control_steps"],
        "status": status,
        "measurement_status": "complete" if cutoff["collection_complete"] else "incomplete",
        "configuration": configuration,
        "cutoff": cutoff,
        "completed_episodes": episodes,
        "active_episodes_at_cutoff": active,
        "aggregate": _summarize(episodes, active),
        "per_board": [{"board_id": board,
                       **_summarize([row for row in episodes if row["board_id"] == board],
                                    [row for row in active if row["board_id"] == board])}
                      for board in range(context["boards"])],
        "observation_errors": errors,
        "semantics": {
            "outcome_source": "Native task predicate before automatic reset; not independent physical correctness.",
            "assembled_parts": "Instantaneous aligned parts out of ten; the count can decrease.",
            "simulated_seconds": "Native episode control steps multiplied by configured control_dt_seconds; not wall time.",
            "ending_flags": "terminated, timed_out and native assembly_failed are independent flags and can overlap.",
            "assembly_failed": "Native aggregate of invalid geometry or nonfinite robot joint position/velocity; causes are not separately evaluated.",
            "success_fraction_of_ended": "Successful assemblies divided by ended episodes in the stated subset; active episodes are excluded, not classified as failures.",
            "completion_phase": "Phase containing the terminal control step; warmup/measured overlap is reported separately.",
            "active_episode": "Episode known at the last fully observed control step; an immediate post-reset episode has zero observed steps and unknown progress.",
            "progress": "Last observed pre-reset state only, with its observation step. Missing or stale observations are not inferred as current.",
            "observer_effect": "Host bookkeeping and flushed terminal JSONL events are inside throughput timing. The warmup-boundary CPU snapshot is included in the measured wall interval because the existing timer has already started it; the final-boundary snapshot is after the measured interval. Snapshot transfers can wait for their data; no new per-step completion callback is added.",
        },
    }
    if error is not None:
        report["error"] = error
    return copy.deepcopy(report)


class OutcomeRecorder:
    """Preserve terminal batches and explicit cutoff state without GPU operations."""

    def __init__(self, path, report_path, boards, warmup_steps, measured_steps, mode, *,
                 invocation_id=None, measurement_invocation_id=None, clock=time.monotonic_ns, stream=None):
        self.path = str(path)
        self.report_path = str(report_path)
        self.clock = clock
        self.pid = os.getpid()
        self.sequence = 0
        self.invocation_id = invocation_id or str(uuid.uuid4())
        self.context = {
            "invocation_id": self.invocation_id,
            "measurement_invocation_id": measurement_invocation_id,
            "boards": boards, "parts_per_board": 10, "mode": mode,
            "requested_warmup_control_steps": warmup_steps,
            "requested_measured_control_steps": measured_steps,
        }
        self.configuration = {}
        self.episodes = []
        self.episode_ids = [0] * boards
        self.episode_starts = [0] * boards
        self.progress = [None] * boards
        self.returned_step = 0
        self.observed_step = 0
        self.errors = []
        self.cutoff = None
        self.closed = False
        self._stream = stream
        if stream is None:
            try:
                destination = Path(path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                self._stream = destination.open("w", encoding="utf-8")
            except Exception as error:
                self._observation_error("open_events", error)
        self.emit("run_context", **self.context)
        self.publish(self.report())

    def _observation_error(self, operation, error):
        self.errors.append({"operation": operation, **_error(error)})
        try:
            print(f"NIST outcome observer {operation}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        except Exception:
            pass

    def emit(self, kind, **fields):
        self.sequence += 1
        event = {"schema": "nist-native-outcome-events-v1", "kind": kind,
                 "monotonic_ns": self.clock(), "pid": self.pid,
                 "invocation_id": self.invocation_id, "sequence": self.sequence, **fields}
        if self._stream is not None:
            try:
                self._stream.write(json.dumps(event, allow_nan=False) + "\n")
                self._stream.flush()
            except Exception as error:
                self._observation_error("write_event", error)

    def configure(self, configuration):
        self.configuration = copy.deepcopy(configuration)
        self.emit("configuration", configuration=self.configuration)

    def step_returned(self, global_control_step):
        """Keep returned work distinct if CPU telemetry extraction then fails."""
        self.returned_step = int(global_control_step)

    def observe_step(self, global_control_step, terminal_records, *, progress=None):
        """Atomically account for every terminal board from one returned step.

        terminal_records contain CPU board_id, assembled_parts, complete_board,
        native_episode_steps, terminated, timed_out, and assembly_failed values.
        Optional progress is the already-read full vector snapshot in latency mode.
        """
        end = int(global_control_step)
        dt = self.configuration.get("control_dt_seconds")
        warmup = self.context["requested_warmup_control_steps"]
        mode = self.context["mode"]
        episodes = []
        for source in terminal_records:
            board = int(source["board_id"])
            row = {"board_id": board, "episode_id": self.episode_ids[board],
                   "completion_phase": _phase(end, warmup, mode),
                   "start_control_step": self.episode_starts[board], "end_control_step": end,
                   **_overlap(self.episode_starts[board], end, warmup, mode),
                   "assembled_parts": int(source["assembled_parts"]),
                   "complete_board": bool(source["complete_board"]),
                   "native_episode_steps": int(source["native_episode_steps"]),
                   "terminated": bool(source["terminated"]), "timed_out": bool(source["timed_out"]),
                   "assembly_failed": bool(source["assembly_failed"])}
            row["simulated_seconds"] = row["native_episode_steps"] * dt if dt is not None else None
            row["termination_reasons"] = _reasons(row["complete_board"], row["assembly_failed"], row["terminated"], row["timed_out"])
            episodes.append(row)
        next_progress = None
        if progress is not None:
            next_starts = list(self.episode_starts)
            for row in episodes:
                next_starts[row["board_id"]] = end
            next_progress = self._progress_rows(end, progress, next_starts)
        # Conversion above finishes before changing any board's episode identity.
        for row in episodes:
            board = row["board_id"]
            self.episode_ids[board] += 1
            self.episode_starts[board] = end
            self.progress[board] = None
        self.episodes.extend(episodes)
        self.observed_step = end
        if next_progress is not None:
            self.progress = next_progress
        if episodes:
            self.emit("episode_end_batch", global_control_step=end, episodes=episodes)

    def _progress_rows(self, step, snapshot, episode_starts):
        dt = self.configuration.get("control_dt_seconds")
        rows = []
        for board in range(self.context["boards"]):
            if episode_starts[board] == step:
                rows.append(None)
                continue
            native_steps = int(snapshot["episode_steps"][board])
            rows.append({
                "global_control_step": step,
                "phase": _phase(step, self.context["requested_warmup_control_steps"], self.context["mode"]),
                "assembled_parts": int(snapshot["assembled_parts"][board]),
                "native_episode_steps": native_steps,
                "simulated_seconds": native_steps * dt if dt is not None else None,
                "complete_board": bool(snapshot["assembly_success"][board]),
                "terminated": bool(snapshot["terminated"][board]),
                "timed_out": bool(snapshot["timed_out"][board]),
                "assembly_failed": bool(snapshot["assembly_failed"][board]),
            })
        return rows

    def snapshot(self, step, snapshot, *, reason):
        self.progress = self._progress_rows(int(step), snapshot, self.episode_starts)
        self.emit("progress_snapshot", global_control_step=int(step), reason=reason,
                  active_episodes=self.active_episodes(final_state_known=self.returned_step == self.observed_step))

    def active_episodes(self, *, final_state_known):
        dt = self.configuration.get("control_dt_seconds")
        warmup = self.context["requested_warmup_control_steps"]
        mode = self.context["mode"]
        return [
            {"board_id": board, "episode_id": self.episode_ids[board],
             "episode_identity_at_cutoff": "known" if final_state_known else "unknown_after_last_observation",
             "start_control_step": start, "known_through_control_step": self.observed_step,
             **_overlap(start, self.observed_step, warmup, mode),
             "observed_simulated_seconds": (self.observed_step - start) * dt if dt is not None else None,
             "returned_control_steps_since_start": self.returned_step - start if final_state_known else None,
             "last_observed_progress": copy.deepcopy(self.progress[board]),
             "progress_at_cutoff": "unknown" if self.progress[board] is None else
                  "current" if final_state_known and self.progress[board]["global_control_step"] == self.returned_step else "last_known"}
            for board, start in enumerate(self.episode_starts)
        ]

    def capture_cutoff(self, *, collection_complete, reason, final_state_known=None):
        if final_state_known is None:
            final_state_known = self.returned_step == self.observed_step
        self.cutoff = {"returned_control_step": self.returned_step,
                       "observed_control_step": self.observed_step,
                       "unobserved_returned_control_steps": self.returned_step - self.observed_step,
                       "collection_complete": bool(collection_complete),
                       "final_state_known": bool(final_state_known), "reason": reason}
        self.emit("cutoff", **self.cutoff, active_episodes=self.active_episodes(final_state_known=final_state_known))

    def report(self, *, error=None):
        cutoff = self.cutoff or {"returned_control_step": self.returned_step,
                                "observed_control_step": self.observed_step,
                                "unobserved_returned_control_steps": self.returned_step - self.observed_step,
                                "collection_complete": False, "final_state_known": False,
                                "reason": "collection_not_finalized"}
        return _make_report(self.context, self.configuration, self.episodes,
                            self.active_episodes(final_state_known=cutoff["final_state_known"]), cutoff,
                            "failed" if error is not None else "complete" if cutoff["collection_complete"] else "incomplete",
                            self.errors, _error(error) if error is not None else None)

    def publish(self, report):
        destination = Path(self.report_path)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(destination)
        except Exception as error:
            self._observation_error("write_report", error)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def close(self, *, error=None):
        if self.closed:
            return
        if self.cutoff is None:
            self.capture_cutoff(collection_complete=False, reason="exception_or_startup_stop",
                                final_state_known=False)
        report = self.report(error=error)
        self.emit("run_end", status=report["status"], error=report.get("error"), observation_errors=list(self.errors))
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception as close_error:
                self._observation_error("close_events", close_error)
        self.publish(self.report(error=error))
        self.closed = True


def recover_outcomes(path, execution=None):
    """Recover flushed observations after an interrupted process, with unknown tail.

    A recovered final cutoff retains its explicit completeness. Without a cutoff,
    board identities/progress are known only through the last flushed batch or
    snapshot; later work and ending state are unknown, rather than zero/failed.
    """
    events, errors = [], []
    try:
        stream = Path(path).open(encoding="utf-8")
    except OSError as error:
        return {"schema": "nist-native-outcomes-v1", "status": "unavailable",
                "observation_errors": [{"operation": "recover_open", **_error(error)}],
                "recovery": {"events_path": str(path), "execution": execution, "reason": "no_readable_journal"}}
    with stream:
        for line_number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("Event is not a JSON object.")
                events.append(event)
            except (ValueError, TypeError) as error:
                errors.append({"operation": "recover_event", "line": line_number, **_error(error)})
    contexts = [row for row in events if row.get("kind") == "run_context"]
    if not contexts:
        return {"schema": "nist-native-outcomes-v1", "status": "unavailable", "observation_errors": errors,
                "recovery": {"events_path": str(path), "execution": execution, "reason": "no_run_context"}}
    context = contexts[0]
    configuration, episodes, active = {}, [], []
    starts = [0] * context["boards"]
    ids = [0] * context["boards"]
    progress = [None] * context["boards"]
    observed = 0
    cutoff = None
    end_event = None
    for event in events:
        if event.get("invocation_id") != context["invocation_id"]:
            errors.append({"operation": "recover_event", "message": "Event invocation differs from run context."})
            continue
        kind = event.get("kind")
        if kind == "configuration":
            configuration = event["configuration"]
        elif kind == "episode_end_batch":
            observed = event["global_control_step"]
            for row in event["episodes"]:
                episodes.append(row)
                board = row["board_id"]
                starts[board] = row["end_control_step"]
                ids[board] = row["episode_id"] + 1
                progress[board] = None
        elif kind == "progress_snapshot":
            observed = event["global_control_step"]
            for row in event["active_episodes"]:
                progress[row["board_id"]] = row["last_observed_progress"]
        elif kind == "cutoff":
            cutoff = {key: value for key, value in event.items() if key in (
                "returned_control_step", "observed_control_step", "unobserved_returned_control_steps",
                "collection_complete", "final_state_known", "reason")}
            active = event["active_episodes"]
        elif kind == "run_end":
            end_event = event
    if cutoff is None:
        cutoff = {"returned_control_step": None, "observed_control_step": observed,
                  "unobserved_returned_control_steps": None, "collection_complete": False,
                  "final_state_known": False, "reason": "recovered_without_final_cutoff"}
        dt = configuration.get("control_dt_seconds")
        active = [{"board_id": board, "episode_id": ids[board],
                   "episode_identity_at_cutoff": "unknown_after_last_observation",
                   "start_control_step": start, "known_through_control_step": observed,
                   **_overlap(start, observed, context["requested_warmup_control_steps"], context["mode"]),
                   "observed_simulated_seconds": (observed - start) * dt if dt is not None else None,
                   "returned_control_steps_since_start": None, "last_observed_progress": progress[board],
                   "progress_at_cutoff": "last_known" if progress[board] is not None else "unknown"}
                  for board, start in enumerate(starts)]
    status = end_event["status"] if end_event else "incomplete"
    invocation_error = end_event.get("error") if end_event else None
    errors.extend(end_event.get("observation_errors", []) if end_event else [])
    report = _make_report(context, configuration, episodes, active, cutoff, status, errors, invocation_error)
    report["recovery"] = {"events_path": str(path), "execution": execution,
                          "run_end_observed": end_event is not None, "event_count": len(events)}
    return report
