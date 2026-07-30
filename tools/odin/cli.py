# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command-line entry point for OdinV2.

Subcommands:

- ``build-image`` — pin a git ref into a benchmark image.
- ``dispatch`` — plan rows, render workflows, submit, poll, fetch.
- ``status`` — summarise a dispatch from ``dispatch.json``.
- ``fetch`` — re-pull results for a dispatch's completed rows.
- ``discover`` — generate the task list from the Gym registry.
- ``harvest`` — derive per-task metadata from a completed dispatch's bundles.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import sys
from collections import Counter
from pathlib import Path

from tools.odin.client import OsmoClient
from tools.odin.config_file import OdinConfig, OdinConfigError, load_odin_config
from tools.odin.discover import DiscoveryError, discover_tasks, expand_rows, filter_rows, write_task_list
from tools.odin.harvest import DEFAULT_TIMEOUT_HEADROOM, HarvestError, harvest_dispatch, write_task_metadata
from tools.odin.image import DEFAULT_CUDA_IMAGE, ImageBuildError, build_image
from tools.odin.plan import PlanError, PlannedRow, apply_metadata, chunk_rows, load_task_rows, plan_rows
from tools.odin.poller import PollError, poll_until_terminal
from tools.odin.results import dispatch_output_uri, fetch_results, validate_bundle
from tools.odin.state import (
    SCHEMA_VERSION,
    DispatchState,
    FailureInfo,
    JobEntry,
    read_dispatch_state,
    reset_in_flight_to_pending,
    write_dispatch_state,
)
from tools.odin.workflow import osmo_safe_task_name, render_workflow_yaml

__all__ = ["main"]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_seeds(text: str) -> list[int]:
    """Parse a comma-separated seed list."""
    return [int(part) for part in text.split(",") if part.strip()]


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _allocate_dispatch_id(runs_root: Path) -> str:
    """Return a fresh ``YYYYMMDD-HHMMSS`` id, disambiguated on collision."""
    base = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    candidate, suffix = base, 1
    while (runs_root / candidate).exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _side_row_key(row_key: str, side: str) -> str:
    """Disambiguate a row key by A/B side so both sides coexist in one dispatch."""
    return row_key if side == "a" else f"{row_key}_{side}"


def _sided(row: PlannedRow, side: str) -> PlannedRow:
    """Return *row* re-keyed for one A/B side."""
    row_key = _side_row_key(row.row_key, side)
    return dataclasses.replace(row, row_key=row_key, osmo_task_name=osmo_safe_task_name(row_key))


def _job_from_row(row: PlannedRow, *, side: str, image_ref: str) -> JobEntry:
    """Build a :class:`JobEntry` for one already-sided row."""
    return JobEntry(
        row_key=row.row_key,
        task_id=row.task_id,
        rl_library=row.rl_library,
        physics=row.physics,
        renderer=row.renderer,
        presets=row.presets,
        seed=row.seed,
        num_envs=row.num_envs,
        max_iterations=row.max_iterations,
        timeout_s=row.timeout_s,
        osmo_task_name=row.osmo_task_name,
        image_ref=image_ref,
        side=side,
    )


def command_build_image(args: argparse.Namespace) -> int:
    """Build, and optionally push, a commit-pinned benchmark image."""
    cfg = load_odin_config(args.config)
    tag = build_image(
        spec=cfg.image,
        ref=args.ref,
        repo_root=_REPO_ROOT,
        context_dir=args.context_dir,
        cuda_image=args.cuda_image,
        push=args.push,
        dry_run=args.dry_run,
    )
    print(tag)
    return 0


def _resolve_dispatch_id(runs_root: Path, target: str) -> str:
    """Resolve a dispatch id, accepting ``LATEST`` for the most recent one.

    Args:
        runs_root: Directory holding dispatches.
        target: A dispatch id, or ``LATEST``.

    Returns:
        The resolved dispatch id.

    Raises:
        PlanError: If *target* is ``LATEST`` and no dispatch exists.
    """
    if target != "LATEST":
        return target
    candidates = sorted(path.name for path in runs_root.iterdir() if (path / "dispatch.json").exists())
    if not candidates:
        raise PlanError(f"no dispatch found under {runs_root}")
    return candidates[-1]


def command_dispatch(args: argparse.Namespace) -> int:
    """Plan rows, render workflows, and (unless dry-running) submit and poll."""
    cfg = load_odin_config(args.config)
    if args.resume is not None:
        return _resume_dispatch(args=args, cfg=cfg)
    chunk_size = args.chunk_size if args.chunk_size is not None else cfg.chunk_size
    # Validated before anything is written: chunking runs after dispatch.json
    # exists, so failing there would strand an orphan dispatch that LATEST
    # resolves to.
    if chunk_size < 1:
        raise PlanError(f"chunk_size must be >= 1, got {chunk_size}")

    # --image and --seeds are only optional on the resume path, which re-attaches
    # to work that was already planned and submitted.
    if args.image is None:
        print("[odin] --image is required unless resuming", file=sys.stderr)
        return 1
    if args.seeds is None and args.retry_failed is None:
        print("[odin] --seeds is required unless retrying (retries reuse the parent's seeds)", file=sys.stderr)
        return 1

    # Each element pairs a task-row list with the seeds it expands across. A
    # fresh dispatch has one; a retry has one per seed, because a row that
    # passed on seed 42 must not be re-run because seed 43 failed.
    plan_groups: list[tuple[list[dict], list[int]]]
    parent_dispatch_id: str | None = None
    if args.retry_failed is not None:
        parent_dispatch_id = _resolve_dispatch_id(args.runs_root, args.retry_failed)
        rows_by_seed = _failed_rows_from(args.runs_root / parent_dispatch_id)
        if not rows_by_seed:
            print(f"[odin] {parent_dispatch_id} has no failed rows to retry", file=sys.stderr)
            return 1
        plan_groups = [(task_rows, [seed]) for seed, task_rows in sorted(rows_by_seed.items())]
        seeds = sorted(rows_by_seed)
        retried = sum(len(task_rows) for task_rows, _ in plan_groups)
        print(f"[odin] retrying {retried} failed row(s) from {parent_dispatch_id}")
    else:
        if not args.tasks_yaml.exists():
            print(
                f"[odin] no task list at {args.tasks_yaml}. It is generated, not shipped: "
                "run `odin discover` first, or pass --tasks_yaml with a hand-written list.",
                file=sys.stderr,
            )
            return 1
        task_rows = load_task_rows(args.tasks_yaml)
        seeds = args.seeds
        if args.metadata_yaml is not None:
            task_rows = apply_metadata(task_rows, load_task_rows(args.metadata_yaml))
        plan_groups = [(task_rows, seeds)]

    if args.play or args.keep_checkpoints:
        overrides: dict = {}
        if args.play:
            overrides["play"] = True
        if args.keep_checkpoints:
            overrides["keep_checkpoint"] = True
        if args.video_length:
            overrides["video_length"] = args.video_length
        plan_groups = [([dict(row, **overrides) for row in task_rows], s) for task_rows, s in plan_groups]

    rows = [
        row
        for task_rows, group_seeds in plan_groups
        for row in plan_rows(task_rows=task_rows, seeds=group_seeds, include=args.include)
    ]
    if not rows:
        print("[odin] no rows matched; nothing to dispatch", file=sys.stderr)
        return 1

    sides: list[tuple[str, str]] = [("a", args.image)]
    if args.image_b is not None:
        sides.append(("b", args.image_b))

    dispatch_id = _allocate_dispatch_id(args.runs_root)
    dispatch_dir = args.runs_root / dispatch_id
    dispatch_dir.mkdir(parents=True, exist_ok=True)

    sided_rows: dict[str, list[PlannedRow]] = {side: [_sided(row, side) for row in rows] for side, _ in sides}
    state = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id=dispatch_id,
        started_at=_utc_now_iso(),
        ended_at=None,
        seeds=list(seeds),
        images={side: image for side, image in sides},
        jobs=[_job_from_row(row, side=side, image_ref=image) for side, image in sides for row in sided_rows[side]],
        parent_dispatch_id=parent_dispatch_id,
    )
    write_dispatch_state(dispatch_dir, state)

    rendered: list[tuple[Path, str]] = []
    for side, image in sides:
        for chunk_index, chunk in chunk_rows(sided_rows[side], chunk_size):
            yaml_text = render_workflow_yaml(
                dispatch_id=dispatch_id,
                chunk_index=chunk_index,
                rows=chunk,
                cfg=cfg,
                image_ref=image,
                output_uri=dispatch_output_uri(cfg.results_uri, dispatch_id),
            )
            path = dispatch_dir / f"workflow.{side}{chunk_index}.yaml"
            path.write_text(yaml_text)
            rendered.append((path, side))

    print(f"[odin] dispatch {dispatch_id}: {len(state.jobs)} rows, {len(rendered)} workflow(s) -> {dispatch_dir}")
    if args.dry_run:
        print("[odin] dry run: nothing submitted")
        return 0

    return _submit_and_poll(args=args, cfg=cfg, state=state, dispatch_dir=dispatch_dir, rendered=rendered)


def _failed_rows_from(dispatch_dir: Path) -> dict[int, list[dict]]:
    """Return a dispatch's failed rows as task rows, grouped by seed.

    Failure is a property of ``(identity, seed)``, not of an identity: a row
    that passed on seed 42 must not be re-run because seed 43 failed. Grouping
    by seed keeps each identity paired with the seeds it actually failed on.

    Args:
        dispatch_dir: Directory of the dispatch to retry.

    Returns:
        Seed to task rows that failed on it, each suitable for
        :func:`plan_rows` with that single seed.

    Raises:
        PlanError: If the dispatch cannot be read, or is an A/B dispatch.
    """
    state = read_dispatch_state(dispatch_dir)
    if state is None:
        raise PlanError(f"no dispatch.json under {dispatch_dir}")

    # A retry plans one row per identity and dispatches it against --image, so
    # a side-B failure would silently be re-run on side A's image.
    sides = sorted({job.side for job in state.jobs})
    if len(sides) > 1:
        raise PlanError(
            f"{dispatch_dir.name} is an A/B dispatch (sides {sides}); a retry would run both sides against one "
            "image. Re-dispatch each side on its own instead."
        )

    grouped: dict[int, list[dict]] = {}
    seen: set[tuple] = set()
    for job in state.jobs:
        if job.status != "failed":
            continue
        identity = (job.seed, job.task_id, job.rl_library, job.physics, job.renderer, job.presets)
        if identity in seen:
            continue
        seen.add(identity)
        row: dict = {"task_id": job.task_id, "rl_library": job.rl_library}
        if job.physics is not None:
            row["physics"] = job.physics
        if job.renderer is not None:
            row["renderer"] = job.renderer
        if job.presets:
            row["presets"] = list(job.presets)
        if job.num_envs is not None:
            row["num_envs"] = job.num_envs
        if job.max_iterations is not None:
            row["max_iterations"] = job.max_iterations
        if job.timeout_s is not None:
            row["timeout_s"] = job.timeout_s
        grouped.setdefault(job.seed, []).append(row)

    return grouped


def _resume_dispatch(*, args: argparse.Namespace, cfg: OdinConfig) -> int:
    """Re-attach to a dispatch whose workflows are already submitted.

    A crashed controller leaves rows stuck in ``running`` while OSMO carries on,
    so resuming flips those back to ``pending`` and re-enters the poll loop.
    Nothing is re-submitted: the workflow ids are already recorded, and
    submitting again would duplicate every task.

    Args:
        args: Parsed ``dispatch`` arguments.
        cfg: Validated ``odin.yaml`` contents.

    Returns:
        Process exit code.
    """
    dispatch_id = _resolve_dispatch_id(args.runs_root, args.resume)
    dispatch_dir = args.runs_root / dispatch_id
    state = read_dispatch_state(dispatch_dir)
    if state is None:
        print(f"[odin] no dispatch.json under {dispatch_dir}", file=sys.stderr)
        return 1
    if not state.osmo_workflow_ids:
        print(
            f"[odin] {dispatch_id} has no submitted workflows; it never got past planning. "
            "Start a fresh dispatch instead.",
            file=sys.stderr,
        )
        return 1

    in_flight = sum(1 for job in state.jobs if job.status == "running")
    reset_in_flight_to_pending(state)
    write_dispatch_state(dispatch_dir, state)
    print(
        f"[odin] resuming {dispatch_id}: {in_flight} in-flight row(s) reset, {len(state.osmo_workflow_ids)} workflow(s)"
    )

    return _poll_to_completion(
        client=OsmoClient(profile=cfg.osmo_profile),
        cfg=cfg,
        state=state,
        dispatch_dir=dispatch_dir,
        poll_interval_s=args.poll_interval,
    )


def _submit_and_poll(
    *,
    args: argparse.Namespace,
    cfg: OdinConfig,
    state: DispatchState,
    dispatch_dir: Path,
    rendered: list[tuple[Path, str]],
) -> int:
    """Submit every rendered workflow, then poll to completion and fetch results."""
    client = OsmoClient(profile=cfg.osmo_profile)

    # Preflight: OSMO uploads results itself via each task's outputs block, so a
    # read-only bucket would not surface until every task had already run.
    if not args.skip_preflight and not client.data_check(cfg.results_uri):
        print(
            f"[odin] {cfg.results_uri} is not writable; results would be lost. "
            "Fix the bucket mode or pass --skip_preflight to override.",
            file=sys.stderr,
        )
        return 1

    # Validate every chunk before submitting any of them, so a schema error
    # cannot leave a dispatch half-submitted.
    if not args.skip_preflight:
        for path, _side in rendered:
            client.validate(path)

    for path, _side in rendered:
        workflow_id = client.submit(
            path,
            pool=args.pool or cfg.pool,
            priority=args.priority or cfg.priority,
        )
        state.osmo_workflow_ids.append(workflow_id)
        print(f"[odin] submitted {path.name} -> {workflow_id}")
        write_dispatch_state(dispatch_dir, state)

    return _poll_to_completion(
        client=client, cfg=cfg, state=state, dispatch_dir=dispatch_dir, poll_interval_s=args.poll_interval
    )


def _poll_to_completion(
    *,
    client: OsmoClient,
    cfg: OdinConfig,
    state: DispatchState,
    dispatch_dir: Path,
    poll_interval_s: float,
) -> int:
    """Poll submitted workflows to completion, fetching each row as it lands.

    Shared by a fresh dispatch and by ``--resume``: once workflows are
    submitted, recovering a crashed controller is exactly this loop again.
    """

    def on_completed(job: JobEntry) -> None:
        row_dir = fetch_results(
            client=client,
            base_uri=cfg.results_uri,
            dispatch_id=state.dispatch_id,
            row_key=job.row_key,
            dest_dir=dispatch_dir,
        )
        if validate_bundle(row_dir):
            return
        # OSMO reported COMPLETED but nothing parseable came back. Route through
        # pending because completed -> failed is not a legal direct transition.
        job.transition_to("pending")
        job.transition_to(
            "failed",
            failure=FailureInfo(kind="malformed_bundle", message=f"no readable schema bundle under {row_dir}"),
        )

    poll_until_terminal(
        client=client,
        state=state,
        dispatch_dir=dispatch_dir,
        on_task_completed=on_completed,
        poll_interval_s=poll_interval_s,
    )

    state.ended_at = _utc_now_iso()
    write_dispatch_state(dispatch_dir, state)
    _print_summary(state)
    return 1 if any(job.status == "failed" for job in state.jobs) else 0


def _print_summary(state: DispatchState) -> None:
    """Print status counts, plus failure-kind counts when anything failed."""
    counts = Counter(job.status for job in state.jobs)
    print(f"[odin] {state.dispatch_id}: " + ", ".join(f"{status}={count}" for status, count in sorted(counts.items())))
    kinds = Counter(job.failure.kind for job in state.jobs if job.failure is not None)
    if kinds:
        print("[odin] failures: " + ", ".join(f"{kind}={count}" for kind, count in sorted(kinds.items())))


def command_status(args: argparse.Namespace) -> int:
    """Summarise a dispatch from its ``dispatch.json``."""
    state = read_dispatch_state(args.runs_root / args.dispatch_id)
    if state is None:
        print(f"[odin] no dispatch.json under {args.runs_root / args.dispatch_id}", file=sys.stderr)
        return 1
    _print_summary(state)
    return 0


def command_fetch(args: argparse.Namespace) -> int:
    """Re-pull results for every completed row of a dispatch."""
    dispatch_dir = args.runs_root / args.dispatch_id
    state = read_dispatch_state(dispatch_dir)
    if state is None:
        print(f"[odin] no dispatch.json under {dispatch_dir}", file=sys.stderr)
        return 1
    cfg = load_odin_config(args.config)
    client = OsmoClient(profile=cfg.osmo_profile)

    valid = 0
    completed = [job for job in state.jobs if job.status == "completed"]
    for job in completed:
        row_dir = fetch_results(
            client=client,
            base_uri=cfg.results_uri,
            dispatch_id=state.dispatch_id,
            row_key=job.row_key,
            dest_dir=dispatch_dir,
        )
        valid += 1 if validate_bundle(row_dir) else 0
    print(f"[odin] fetched {len(completed)} row(s), {valid} with a readable bundle")
    return 0 if valid == len(completed) else 1


def command_discover(args: argparse.Namespace) -> int:
    """Generate the dispatch task list by walking the Gym registry."""
    tasks = discover_tasks()
    rows = filter_rows(
        expand_rows(tasks),
        include=args.include,
        exclude=args.exclude,
        libraries=args.library,
        physics=args.physics,
        renderers=args.renderer,
        presets=args.presets,
        scope=args.scope,
        max_rows=args.max_rows,
    )
    if not rows:
        print("[odin] discovery produced no rows; check the filters", file=sys.stderr)
        return 1
    write_task_list(
        args.output,
        rows,
        meta={
            "task_count": len({row["task_id"] for row in rows}),
            "row_count": len(rows),
        },
    )
    scopes = Counter(row.get("scope", "core") for row in rows)
    print(
        f"[odin] discovered {len({r['task_id'] for r in rows})} task(s) -> {len(rows)} row(s) "
        f"({', '.join(f'{k}={v}' for k, v in sorted(scopes.items()))}) -> {args.output}"
    )
    return 0


def command_harvest(args: argparse.Namespace) -> int:
    """Derive per-task metadata from a completed dispatch's bundles."""
    dispatch_dir = args.runs_root / args.dispatch_id
    if not dispatch_dir.is_dir():
        print(f"[odin] no dispatch under {dispatch_dir}", file=sys.stderr)
        return 1
    entries = harvest_dispatch(dispatch_dir, timeout_headroom=args.timeout_headroom)
    if not entries:
        print(f"[odin] no completed bundles under {dispatch_dir}", file=sys.stderr)
        return 1
    write_task_metadata(args.output, entries)
    print(f"[odin] harvested {len(entries)} task/library/physics combination(s) -> {args.output}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="odin", description="Dispatch Isaac Lab benchmarks to OSMO.")
    sub = parser.add_subparsers(dest="command")

    build = sub.add_parser("build-image", help="Pin a git ref into a benchmark image.")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--ref", type=str, default="HEAD")
    build.add_argument("--cuda_image", type=str, default=DEFAULT_CUDA_IMAGE)
    build.add_argument("--context_dir", type=Path, default=Path("./.odin-build"))
    build.add_argument("--push", action="store_true")
    build.add_argument("--dry_run", action="store_true")

    dispatch = sub.add_parser("dispatch", help="Plan, submit, and poll a dispatch.")
    dispatch.add_argument("--config", type=Path, required=True)
    dispatch.add_argument("--tasks_yaml", type=Path, default=Path(__file__).resolve().parent / "config" / "tasks.yaml")
    dispatch.add_argument("--image", type=str, default=None, help="Digest-pinned image for side A.")
    dispatch.add_argument("--image_b", type=str, default=None, help="Digest-pinned image for side B; enables A/B mode.")
    dispatch.add_argument("--seeds", type=_parse_seeds, default=None)
    dispatch.add_argument("--include", type=str, default=None, help="Glob filter over task_id.")
    dispatch.add_argument(
        "--metadata_yaml",
        type=Path,
        default=None,
        help="Harvested task_metadata.yaml to overlay sizing from. Seed-list values win.",
    )
    dispatch.add_argument("--pool", type=str, default=None)
    dispatch.add_argument("--priority", choices=["HIGH", "NORMAL", "LOW"], default=None)
    dispatch.add_argument("--chunk_size", type=int, default=None)
    dispatch.add_argument("--poll_interval", type=float, default=15.0)
    dispatch.add_argument("--runs_root", type=Path, default=Path("./odin_runs"))
    dispatch.add_argument(
        "--play",
        action="store_true",
        help="Chain a play rollout after training in the same task, recording a video.",
    )
    dispatch.add_argument("--video_length", type=int, default=None, help="Frames to record during play.")
    dispatch.add_argument(
        "--keep_checkpoints",
        action="store_true",
        help="Copy the trained checkpoint into the uploaded output; OSMO does not collect logs/.",
    )
    dispatch.add_argument("--dry_run", action="store_true")
    dispatch.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="DISPATCH_ID",
        help="Re-attach to an already-submitted dispatch and keep polling. Accepts LATEST.",
    )
    dispatch.add_argument(
        "--retry_failed",
        type=str,
        default=None,
        metavar="DISPATCH_ID",
        help="Start a new dispatch containing only the failed rows of another. Accepts LATEST.",
    )
    dispatch.add_argument(
        "--skip_preflight",
        action="store_true",
        help="Submit without checking that results_uri is writable.",
    )

    status = sub.add_parser("status", help="Summarise a dispatch.")
    status.add_argument("dispatch_id", type=str)
    status.add_argument("--runs_root", type=Path, default=Path("./odin_runs"))

    fetch = sub.add_parser("fetch", help="Re-pull results for a dispatch.")
    fetch.add_argument("dispatch_id", type=str)
    fetch.add_argument("--config", type=Path, required=True)
    fetch.add_argument("--runs_root", type=Path, default=Path("./odin_runs"))

    discover = sub.add_parser("discover", help="Generate the task list from the Gym registry.")
    discover.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "config" / "tasks.yaml")
    discover.add_argument("--include", type=str, default=None, help="Glob a task_id must match.")
    discover.add_argument("--exclude", type=str, default=None, help="Glob a task_id must not match.")
    discover.add_argument("--library", action="append", default=None, help="Restrict the library axis; repeatable.")
    discover.add_argument("--physics", action="append", default=None, help="Restrict the physics axis; repeatable.")
    discover.add_argument("--renderer", action="append", default=None, help="Restrict the renderer axis; repeatable.")
    discover.add_argument(
        "--presets", action="append", default=None, help="Restrict the domain-preset axis; repeatable."
    )
    discover.add_argument("--scope", choices=["core", "contrib", "all"], default="all")
    discover.add_argument("--max_rows", type=int, default=None, help="Deterministic head of the sorted order.")

    harvest = sub.add_parser("harvest", help="Derive per-task metadata from a completed dispatch.")
    harvest.add_argument("dispatch_id", type=str)
    harvest.add_argument("--runs_root", type=Path, default=Path("./odin_runs"))
    harvest.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "task_metadata.yaml",
    )
    harvest.add_argument(
        "--timeout_headroom",
        type=float,
        default=DEFAULT_TIMEOUT_HEADROOM,
        help="Multiplier applied to the slowest observed run when deriving timeout_s.",
    )

    return parser


_COMMANDS = {
    "build-image": command_build_image,
    "dispatch": command_dispatch,
    "status": command_status,
    "fetch": command_fetch,
    "discover": command_discover,
    "harvest": command_harvest,
}


def main(argv: list[str] | None = None) -> int:
    """Run the OdinV2 CLI.

    Args:
        argv: Command-line arguments; uses ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command is None:
        parser.print_help()
        return 2
    try:
        return _COMMANDS[args.command](args)
    except (OdinConfigError, PlanError, ImageBuildError, DiscoveryError, HarvestError, PollError) as exc:
        print(f"[odin] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
