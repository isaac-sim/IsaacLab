#!/usr/bin/env python3

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coordinate terminal status across the OSMO tasks in one distributed run."""

from __future__ import annotations

import argparse
import contextlib
import json
import socket
import time
from pathlib import Path


def _read_message(connection: socket.socket, maximum_bytes: int = 4096) -> dict[str, object]:
    """Read one newline-terminated JSON status message."""
    payload = bytearray()
    while len(payload) < maximum_bytes:
        chunk = connection.recv(min(1024, maximum_bytes - len(payload)))
        if not chunk:
            break
        payload.extend(chunk)
        if b"\n" in chunk:
            break
    if b"\n" not in payload:
        raise ValueError("status message is not newline terminated")
    message = json.loads(payload.split(b"\n", 1)[0])
    if not isinstance(message, dict):
        raise ValueError("status message must be a JSON object")
    return message


def _serve(args: argparse.Namespace) -> int:
    """Release ready nodes together, then collect every terminal status."""
    startup_deadline = time.monotonic() + args.startup_timeout_seconds
    training_deadline: float | None = None
    completion_deadline: float | None = None
    ready_nodes: set[int] = set()
    statuses: dict[int, int] = {}
    barrier_status = 0
    peer_failed = False
    with socket.create_server((args.bind, args.port), reuse_port=False) as server:
        server.settimeout(1.0)
        print(
            f"ddp_barrier_listening bind={args.bind} port={args.port} expected_nodes={args.expected_nodes}",
            flush=True,
        )
        while len(statuses) < args.expected_nodes:
            now = time.monotonic()
            all_ready = len(ready_nodes) == args.expected_nodes
            if not all_ready and now >= startup_deadline:
                missing = sorted(set(range(args.expected_nodes)) - ready_nodes)
                print(f"ddp_barrier_startup_timeout missing_nodes={missing}", flush=True)
                barrier_status = 2
                break
            if training_deadline is not None and now >= training_deadline:
                missing = sorted(set(range(args.expected_nodes)) - statuses.keys())
                print(f"ddp_barrier_training_timeout missing_nodes={missing}", flush=True)
                barrier_status = 2
                break
            if completion_deadline is not None and now >= completion_deadline:
                missing = sorted(set(range(args.expected_nodes)) - statuses.keys())
                print(f"ddp_barrier_completion_timeout missing_nodes={missing}", flush=True)
                barrier_status = 2
                break
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            with connection:
                connection.settimeout(10.0)
                try:
                    message = _read_message(connection)
                    workflow_id = str(message["workflow_id"])
                    node_rank = int(message["node_rank"])
                    phase = str(message["phase"])
                    if workflow_id != args.workflow_id:
                        raise ValueError("workflow ID does not match the barrier")
                    if not 0 <= node_rank < args.expected_nodes:
                        raise ValueError(f"node rank {node_rank} is outside the distributed world")
                    if phase == "probe":
                        response = {"accepted": True}
                    elif phase == "ready":
                        ready_nodes.add(node_rank)
                        if len(ready_nodes) == args.expected_nodes and training_deadline is None:
                            training_deadline = time.monotonic() + args.training_timeout_seconds
                            print("ddp_barrier_ready all_nodes_ready=true", flush=True)
                        response = {
                            "accepted": True,
                            "released": len(ready_nodes) == args.expected_nodes,
                            "failed": any(status != 0 for status in statuses.values()),
                        }
                    elif phase == "complete":
                        status = int(message["status"])
                        previous = statuses.get(node_rank)
                        if previous is not None and previous != status:
                            raise ValueError(f"node rank {node_rank} reported conflicting statuses")
                        statuses[node_rank] = status
                        if status != 0:
                            peer_failed = True
                        elif completion_deadline is None:
                            completion_deadline = time.monotonic() + args.completion_grace_seconds
                        response = {"accepted": True}
                        print(
                            f"ddp_barrier_status node_rank={node_rank} status={status} "
                            f"received={len(statuses)}/{args.expected_nodes}",
                            flush=True,
                        )
                    else:
                        raise ValueError(f"unsupported barrier phase {phase!r}")
                    connection.sendall(json.dumps(response, separators=(",", ":")).encode() + b"\n")
                    if peer_failed:
                        break
                except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as error:
                    with contextlib.suppress(OSError):
                        connection.sendall(json.dumps({"accepted": False, "error": str(error)}).encode() + b"\n")
                    print(f"ddp_barrier_rejected error={error}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as status_file:
        status_file.write("node_rank\texit_code\n")
        for node_rank, status in sorted(statuses.items()):
            status_file.write(f"{node_rank}\t{status}\n")
    if barrier_status != 0:
        return barrier_status
    failed = {node_rank: status for node_rank, status in statuses.items() if status != 0}
    if failed:
        print(f"ddp_barrier_failed statuses={failed}", flush=True)
        return 1
    print("ddp_barrier_complete all_nodes_succeeded=true", flush=True)
    return 0


def _report(args: argparse.Namespace) -> int:
    """Retry a probe, readiness arrival, or terminal-status report."""
    deadline = time.monotonic() + args.timeout_seconds
    payload = (
        json.dumps(
            {
                "workflow_id": args.workflow_id,
                "node_rank": args.node_rank,
                "phase": args.phase,
                **({"status": args.status} if args.phase == "complete" else {}),
            },
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    last_error: OSError | ValueError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((args.host, args.port), timeout=10.0) as connection:
                connection.sendall(payload)
                response = _read_message(connection)
            if response.get("accepted") is True:
                if args.phase == "ready" and response.get("failed") is True:
                    print("ddp_barrier_ready_aborted peer_failed=true", flush=True)
                    return 4
                if args.phase == "ready" and response.get("released") is not True:
                    time.sleep(args.retry_seconds)
                    continue
                print(
                    f"ddp_barrier_reported phase={args.phase} node_rank={args.node_rank} "
                    f"status={args.status} host={args.host}",
                    flush=True,
                )
                return 0
            last_error = ValueError(str(response.get("error", "barrier rejected the status")))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            last_error = error
        time.sleep(args.retry_seconds)
    print(f"ddp_barrier_report_failed node_rank={args.node_rank} error={last_error}", flush=True)
    return 3


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="wait for all node statuses")
    serve.add_argument("--bind", default="0.0.0.0")
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--expected-nodes", type=int, required=True)
    serve.add_argument("--workflow-id", required=True)
    serve.add_argument("--startup-timeout-seconds", type=float, default=3600.0)
    serve.add_argument("--training-timeout-seconds", type=float, default=518400.0)
    serve.add_argument("--completion-grace-seconds", type=float, default=600.0)
    serve.add_argument("--output", type=Path, required=True)
    serve.set_defaults(handler=_serve)

    report = subparsers.add_parser("report", help="report one node's terminal status")
    report.add_argument("--host", required=True)
    report.add_argument("--port", type=int, required=True)
    report.add_argument("--node-rank", type=int, required=True)
    report.add_argument("--phase", choices=("probe", "ready", "complete"), required=True)
    report.add_argument("--status", type=int, default=0)
    report.add_argument("--workflow-id", required=True)
    report.add_argument("--timeout-seconds", type=float, default=300.0)
    report.add_argument("--retry-seconds", type=float, default=2.0)
    report.set_defaults(handler=_report)
    return parser


def main() -> int:
    """Run the selected barrier operation."""
    args = _parser().parse_args()
    if args.port < 1 or args.port > 65535:
        raise SystemExit("port must be between 1 and 65535")
    if getattr(args, "expected_nodes", 1) < 1:
        raise SystemExit("expected-nodes must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
