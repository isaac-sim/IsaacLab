# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train, collect, review, and publish pretrained Isaac Lab checkpoints.

The core-task workflow selects RSL-RL when it is registered, falls back to
RL-Games, and uses SKRL MAPPO for multi-agent environments. Backend-aware
checkpoints are collected into one subdirectory per RL library:

.. code-block:: text

    logs/pretrained_checkpoints/
    ├── rl_games/
    ├── rsl_rl/
    └── skrl/

Each checkpoint is named
``<task_name>_<physics_backend>_<render_backend>_<rl_library><extension>``.
State-only tasks use ``none`` as the render backend because their policies do
not depend on rendering. This workflow targets core tasks only; other
registered tasks do not receive published checkpoints from this matrix. The
core matrix excludes Newton Kamino presets.

Examples:

.. code-block:: shell

    # List the preferred core-task training matrix.
    uv run python scripts/tools/train_and_publish_checkpoints.py \
        --list --all --core

    # Smoke-test every supported core-task backend combination.
    uv run python scripts/tools/train_and_publish_checkpoints.py \
        --smoke --all --core

    # Run full training and collect checkpoints.
    uv run python scripts/tools/train_and_publish_checkpoints.py \
        --train --all --core

    # Resume collection after training jobs have completed.
    uv run python scripts/tools/train_and_publish_checkpoints.py \
        --collect --all --core

    # Select legacy jobs with workflow and task wildcards.
    rl_games:Isaac-Humanoid-*        # Wildcard for any Humanoid version
    rsl_rl:Isaac-Ant-*               # Wildcard for any Ant environment
    *:IsaacContrib-Velocity-Flat-Spot    # Wildcard for any workflow, specific task

Legacy ``workflow:task`` job selectors remain supported when ``--core`` is
omitted.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import posixpath
import shutil
import subprocess
import sys
from dataclasses import dataclass

# Prefer this checkout over editable installs that may point at another worktree.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SOURCE_ROOT = os.path.join(_REPO_ROOT, "source")
_REPO_SOURCE_PATHS = sorted(
    entry.path for entry in os.scandir(_SOURCE_ROOT) if entry.is_dir() and os.path.isdir(entry.path)
)
for source_path in reversed(_REPO_SOURCE_PATHS):
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

import gymnasium as gym

from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils import Checkpoint
from isaaclab.utils.assets import NUCLEUS_ASSET_ROOT_DIR

from isaaclab_rl.entrypoints.common import resolve_checkpoint_selector
from isaaclab_rl.utils.pretrained_checkpoint import WORKFLOWS, CheckpointBundle

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.preset_cli import enumerate_task_presets
from isaaclab_tasks.utils.preset_target import PresetTarget

_TRAINING_COMPLETE_FILENAME = ".pretrained_checkpoint_training_complete"
_REVIEW_FILENAME = "pretrained_checkpoint_review.json"


def _preset_args(physics_selector: str | None, render_selector: str | None) -> list[str]:
    """Return the typed preset selectors as command-line overrides."""
    args = []
    if physics_selector is not None:
        args.append(f"physics={physics_selector}")
    if render_selector is not None:
        args.append(f"renderer={render_selector}")
    return args


@dataclass(frozen=True)
class CheckpointJob(CheckpointBundle):
    """One workflow, task, physics, and renderer training combination, and its local training runs."""

    physics_selector: str | None = None
    render_selector: str | None = None
    agent: str | None = None
    algorithm: str | None = None

    @classmethod
    def from_task(
        cls,
        task_name: str,
        workflow: str,
        physics_selector: str | None = None,
        render_selector: str | None = None,
        agent: str | None = None,
        algorithm: str | None = None,
    ) -> CheckpointJob:
        """Build the job for a task's preset selection from the config those presets produce.

        The backend names and the declared checkpoints are read from that config, so the published
        filename and file set describe what the presets actually train.
        """
        env_cfg, _ = resolve_task_config(
            task_name, None, overrides=tuple(_preset_args(physics_selector, render_selector))
        )
        return cls.from_env_cfg(
            workflow,
            task_name,
            env_cfg,
            physics_selector=physics_selector,
            render_selector=render_selector,
            agent=agent,
            algorithm=algorithm,
        )

    """
    Identity.
    """

    @property
    def job_id(self) -> str:
        """Return the stable human-readable job identifier."""
        if self.is_legacy:
            return f"{self.workflow}:{self.task_name}"
        return f"{self.workflow}:{self.task_name}:{self.physics_backend}:{self.render_backend}"

    @property
    def preset_args(self) -> list[str]:
        """Return typed preset selectors for this job."""
        return _preset_args(self.physics_selector, self.render_selector)

    """
    Training runs.
    """

    @property
    def log_root(self) -> str:
        """The absolute directory holding this variant's training runs."""
        return os.path.abspath(os.path.join("logs", self.workflow, self.stem))

    def trained_path(self, checkpoint: Checkpoint | None = None) -> str | None:
        """Return the policy a training run wrote, or a declared checkpoint beside it, or ``None``.

        The policy is selected like ``--checkpoint best``: the newest run carrying a run manifest,
        the library's preferred file when present, else the last checkpoint. A declared checkpoint
        is the newest match of its glob in that same run.
        """
        if checkpoint is None:
            try:
                return resolve_checkpoint_selector(
                    self.log_root,
                    "best",
                    library=self.workflow,
                    task=self.task_name,
                    **self.library.selector_args(self.stem),
                )
            except ValueError:
                return None
        run_path = self.latest_run
        return None if run_path is None else checkpoint.find_in(run_path)

    @property
    def latest_run(self) -> str | None:
        """The run directory holding the selected policy, or ``None`` before a run wrote one."""
        policy_path = self.trained_path()
        if policy_path is None:
            return None
        run_path = os.path.dirname(policy_path)
        for _ in posixpath.dirname(self.library.policy_glob).split("/"):
            if _:
                run_path = os.path.dirname(run_path)
        return run_path

    @property
    def has_run(self) -> bool:
        """Whether a training run was started for this variant."""
        return os.path.exists(self.log_root)

    @property
    def has_finished(self) -> bool:
        """Whether a training run wrote a policy checkpoint."""
        return self.trained_path() is not None

    @property
    def is_trained(self) -> bool:
        """Whether this job's training finished and left a checkpoint.

        A core job must also carry the marker its training subprocess writes, so a run killed
        after writing a checkpoint is retried. Legacy jobs predate the marker.
        """
        if self.is_legacy:
            return self.has_finished
        run_path = self.latest_run
        return run_path is not None and os.path.isfile(os.path.join(run_path, _TRAINING_COMPLETE_FILENAME))

    def mark_trained(self) -> None:
        """Record that the latest training subprocess exited successfully."""
        run_path = self.latest_run
        if run_path is None:
            raise RuntimeError(f"Unable to determine the latest run for {self.job_id}")
        with open(os.path.join(run_path, _TRAINING_COMPLETE_FILENAME), "w", encoding="utf-8") as marker_file:
            marker_file.write(f"{self.job_id}\n")

    @property
    def review_path(self) -> str | None:
        """The review JSON path in the selected run, or ``None`` before a run wrote a policy."""
        run_path = self.latest_run
        return None if run_path is None else os.path.join(run_path, _REVIEW_FILENAME)

    @property
    def review(self) -> dict | None:
        """The review JSON data of the selected run, or ``None`` if it was not reviewed."""
        review_path = self.review_path
        if review_path is None or not os.path.exists(review_path):
            return None
        with open(review_path) as f:
            return json.load(f)

    """
    Commands.
    """

    def _command(self, action: str) -> list[str]:
        command = ["uv", "run", "isaaclab", action, "--rl_library", self.workflow, "--task", self.task_name]
        if self.agent is not None:
            command.extend(["--agent", self.agent])
        if self.algorithm is not None:
            command.extend(["--algorithm", self.algorithm])
        return command

    def train_command(self, args: argparse.Namespace, smoke: bool = False) -> list[str]:
        """Build the unified training command for this job."""
        command = self._command("train")
        experiment_name = f"{self.stem}_smoke" if smoke else self.stem
        if self.library.experiment_variable is not None:
            command.append(f"{self.library.experiment_variable}={experiment_name}")
        if smoke:
            command.extend(["--max_iterations", "1", "--num_envs", str(args.num_envs or 4)])
        else:
            if args.max_iterations is not None:
                command.extend(["--max_iterations", str(args.max_iterations)])
            if args.num_envs is not None:
                command.extend(["--num_envs", str(args.num_envs)])
        command.extend(self.preset_args)
        return command

    def play_command(self, args: argparse.Namespace, checkpoint_path: str) -> list[str]:
        """Build the unified playback command for this job."""
        command = self._command("play") + ["--checkpoint", checkpoint_path]
        if args.num_envs is not None:
            command.extend(["--num_envs", str(args.num_envs)])
        command.extend(self.preset_args)
        return command


def _create_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Train and manage backend-aware pretrained checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "jobs",
        nargs="*",
        help=(
            "Job patterns. Legacy jobs use workflow:task. Core matrix patterns "
            "also match workflow:task:physics:renderer."
        ),
    )
    parser.add_argument("-t", "--train", action="store_true", help="Run full training and collect checkpoints.")
    parser.add_argument("--smoke", action="store_true", help="Run one-iteration backend smoke training.")
    parser.add_argument("--collect", action="store_true", help="Collect completed checkpoints without training.")
    parser.add_argument("-p", "--publish_checkpoint", action="store_true", help="Publish collected checkpoints.")
    parser.add_argument("-r", "--review", action="store_true", help="Interactively review checkpoints.")
    parser.add_argument("-l", "--list", action="store_true", help="List matching checkpoint jobs.")
    parser.add_argument("-f", "--force", action="store_true", help="Repeat completed training jobs.")
    parser.add_argument("-a", "--all", action="store_true", help="Match all jobs in the selected scope.")
    parser.add_argument(
        "--core",
        action="store_true",
        help="Use core tasks, preferred RL libraries, and the supported backend matrix.",
    )
    parser.add_argument(
        "--physics_backends",
        default="physx,newtonmjwarp",
        help="Comma-separated normalized physics backends for --core (default: physx,newtonmjwarp).",
    )
    parser.add_argument(
        "--render_backends",
        default="rtx,newton",
        help="Comma-separated normalized camera render backends for --core (default: rtx,newton).",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("logs", "pretrained_checkpoints"),
        help="Structured checkpoint output directory.",
    )
    parser.add_argument(
        "--publish_root",
        default=None,
        help="Writable Nucleus root ending in PretrainedCheckpoints; defaults to the configured asset root.",
    )
    parser.add_argument(
        "-E",
        "--exclude",
        action="append",
        default=[],
        help="Exclude jobs matching a wildcard pattern.",
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Override the training environment count.")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Override full-training iterations. Omit to use each agent config's full schedule.",
    )
    parser.add_argument("--force_review", action="store_true", help="Replace an existing review.")
    parser.add_argument("--force_publish", action="store_true", help="Publish without an accepted review.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop after the first failed job.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    return parser


def _parse_backend_list(value: str, supported: set[str], option: str) -> list[str]:
    """Parse and validate a comma-separated backend option."""
    backends = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(backends) - supported
    if unknown:
        raise ValueError(f"{option} contains unsupported values: {sorted(unknown)}")
    return backends


def _is_core_task(task_spec: gym.EnvSpec) -> bool:
    """Return whether a Gym task is registered from ``isaaclab_tasks.core``."""
    env_cfg_entry_point = task_spec.kwargs.get("env_cfg_entry_point")
    return (
        isinstance(env_cfg_entry_point, str)
        and env_cfg_entry_point.startswith("isaaclab_tasks.core.")
        and not task_spec.kwargs.get("deprecated")
    )


def _select_workflow(task_spec: gym.EnvSpec, env_cfg) -> tuple[str, str | None, str | None]:
    """Select the preferred RL workflow and optional agent arguments."""
    if isinstance(env_cfg, DirectMARLEnvCfg):
        if "skrl_mappo_cfg_entry_point" not in task_spec.kwargs:
            raise ValueError(f"Multi-agent task {task_spec.id!r} does not register an SKRL MAPPO config")
        return "skrl", "skrl_mappo_cfg_entry_point", "MAPPO"
    if "rsl_rl_cfg_entry_point" in task_spec.kwargs:
        return "rsl_rl", None, None
    if "rl_games_cfg_entry_point" in task_spec.kwargs:
        return "rl_games", None, None
    raise ValueError(f"Task {task_spec.id!r} has neither an RSL-RL nor an RL-Games config")


def _select_physics_variants(
    variants: list[str],
    default_backend: str | None,
    requested_backends: list[str],
) -> list[tuple[str, str | None]]:
    """Return normalized physics backends and their task preset selectors."""
    selections = []
    for backend in requested_backends:
        selector = None
        if variants:
            if backend == "physx" and "isaacsim_physx" in variants:
                selector = "isaacsim_physx"
            elif backend == "newtonmjwarp":
                selector = next(
                    (
                        candidate
                        for candidate in ("newton_mjwarp", "newton_mjwarp_vbd", "newton_mjwarp_vbd_proxy")
                        if candidate in variants
                    ),
                    None,
                )
            if selector is None:
                continue
        elif backend != default_backend:
            continue
        selections.append((backend, selector))
    return selections


def _select_render_variants(
    variants: list[str],
    requested_backends: list[str],
) -> list[tuple[str, str | None]]:
    """Return normalized render backends and their task preset selectors."""
    if not variants:
        return [("none", None)]

    selections = []
    for backend in requested_backends:
        selector = None
        if backend == "rtx":
            selector = next(
                (candidate for candidate in ("isaacsim_rtx", "rtx", "ovrtx") if candidate in variants), None
            )
        elif backend == "newton" and "newton_renderer" in variants:
            selector = "newton_renderer"
        if selector is not None:
            selections.append((backend, selector))
    return selections


def _build_core_jobs(args: argparse.Namespace) -> list[CheckpointJob]:
    """Build the supported preferred-workflow matrix for core tasks."""
    physics_backends = _parse_backend_list(
        args.physics_backends,
        {"newtonmjwarp", "physx"},
        "--physics_backends",
    )
    render_backends = _parse_backend_list(
        args.render_backends,
        {"newton", "rtx"},
        "--render_backends",
    )
    jobs = []
    for task_spec in sorted(gym.registry.values(), key=lambda spec: spec.id):
        if not _is_core_task(task_spec):
            continue

        preset_map = enumerate_task_presets(task_spec.id) or {}
        physics_variants = preset_map.get(PresetTarget.PHYSICS, [])
        render_variants = preset_map.get(PresetTarget.RENDERER, [])
        env_cfg, _ = resolve_task_config(task_spec.id, None)
        workflow, agent, algorithm = _select_workflow(task_spec, env_cfg)
        default_physics = None
        if not physics_variants:
            default_physics, _ = CheckpointBundle.backend_names(env_cfg)

        physics_selections = _select_physics_variants(physics_variants, default_physics, physics_backends)
        render_selections = _select_render_variants(render_variants, render_backends)
        for _physics_backend, physics_selector in physics_selections:
            for _render_backend, render_selector in render_selections:
                jobs.append(
                    CheckpointJob.from_task(task_spec.id, workflow, physics_selector, render_selector, agent, algorithm)
                )
    return jobs


def _build_legacy_jobs() -> list[CheckpointJob]:
    """Build every legacy workflow/task pair registered with Gym."""
    jobs = []
    for workflow in WORKFLOWS:
        for task_spec in sorted(gym.registry.values(), key=lambda spec: spec.id):
            if workflow + "_cfg_entry_point" in task_spec.kwargs and not task_spec.kwargs.get("deprecated"):
                jobs.append(CheckpointJob(workflow=workflow, task_name=task_spec.id))
    return jobs


def _filter_jobs(jobs: list[CheckpointJob], args: argparse.Namespace) -> list[CheckpointJob]:
    """Apply positional include patterns and exclusion patterns."""
    include_patterns = ["*"] if args.all else args.jobs
    selected = []
    for job in jobs:
        if not any(fnmatch.fnmatch(job.job_id, pattern) for pattern in include_patterns):
            continue
        if any(fnmatch.fnmatch(job.job_id, pattern) for pattern in args.exclude):
            continue
        selected.append(job)
    return selected


def _run_command(command: list[str], dry_run: bool) -> int:
    """Print and run a subprocess command."""
    print("Running:", " ".join(command), flush=True)
    if dry_run:
        return 0
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    python_paths = _REPO_SOURCE_PATHS + ([existing_pythonpath] if existing_pythonpath else [])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.pop("CONDA_PREFIX", None)
    # Keep the active environment ahead of Isaac Sim's bundled site-packages.
    # This matters for packages such as Newton, where the project environment
    # can carry the source-compatible revision while Isaac Sim bundles an older
    # release.
    env["VIRTUAL_ENV"] = sys.prefix
    return subprocess.run(command, check=False, cwd=_REPO_ROOT, env=env).returncode


def train_job(job: CheckpointJob, args: argparse.Namespace, smoke: bool = False) -> bool:
    """Train or smoke-test one checkpoint job."""
    if not smoke and not args.force and job.is_trained:
        print(f"Skipping completed training job {job.job_id}")
        return True

    result = _run_command(job.train_command(args, smoke), args.dry_run)
    if result != 0:
        print(f"Training failed for {job.job_id} with exit code {result}", file=sys.stderr)
        return False
    if smoke or args.dry_run:
        return True
    if not job.has_finished:
        print(f"Training did not produce a checkpoint for {job.job_id}", file=sys.stderr)
        return False
    job.mark_trained()
    return True


def collect_pretrained_checkpoint(job: CheckpointJob, output_dir: str, dry_run: bool = False) -> str | None:
    """Copy the last or best checkpoint into the structured output directory."""
    destination = job.collected_path(output_dir)
    if dry_run:
        print(f"Would collect the completed checkpoint -> {destination}")
        return destination

    source_path = job.trained_path()
    if source_path is None or not os.path.isfile(source_path):
        print(f"No completed checkpoint to collect for {job.job_id}")
        return None

    print(f"Collecting {source_path} -> {destination}")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy2(source_path, destination)
    for checkpoint in job.companions:
        companion_source = job.trained_path(checkpoint)
        if companion_source is None:
            print(f"No {checkpoint.name} checkpoint matched {checkpoint.run_glob!r} for {job.job_id}")
            continue
        companion_destination = job.collected_path(output_dir, checkpoint)
        print(f"Collecting {companion_source} -> {companion_destination}")
        shutil.copy2(companion_source, companion_destination)
    return destination


def review_pretrained_checkpoint(job: CheckpointJob, args: argparse.Namespace) -> bool:
    """Play and interactively review one checkpoint."""
    if not job.has_run:
        print(f"Skipping review of {job.job_id}; it has not been trained")
        return False
    checkpoint_path = job.trained_path()
    if checkpoint_path is None:
        print(f"Skipping review of {job.job_id}; training is incomplete")
        return False

    review = job.review
    if not args.force_review and review and review.get("reviewed"):
        print(f"Review already complete for {job.job_id}")
        return True

    if _run_command(job.play_command(args, checkpoint_path), args.dry_run) != 0:
        return False
    if args.dry_run:
        return True

    answer = input(f"Accept checkpoint {job.job_id}? yes, no, or undetermined (y/n/u) [u]: ").strip().lower()
    answer_map = {"y": "accepted", "n": "rejected", "u": "undetermined"}
    result = answer_map.get(answer, "undetermined")
    notes = input("Review notes (optional): ").strip()
    review_data = {"reviewed": True, "result": result}
    if notes:
        review_data["notes"] = notes

    with open(job.review_path, "w", encoding="utf-8") as review_file:
        json.dump(review_data, review_file, indent=4)
    return True


def publish_pretrained_checkpoint(job: CheckpointJob, args: argparse.Namespace) -> bool:
    """Publish an accepted checkpoint to the configured Nucleus asset root."""
    if args.publish_root is None and not NUCLEUS_ASSET_ROOT_DIR:
        raise RuntimeError("A pretrained-checkpoint Nucleus asset root is not configured")
    local_path = job.collected_path(args.output_dir)
    if not os.path.isfile(local_path):
        print(f"Skipping publish of {job.job_id}; no collected checkpoint was found")
        return False

    if not args.force_publish:
        review = job.review
        if not review or review.get("result") != "accepted":
            print(f"Skipping publish of {job.job_id}; it does not have an accepted review")
            return False

    uploads = [(local_path, job.published_path(root=args.publish_root))]
    for checkpoint in job.companions:
        local_companion = job.collected_path(args.output_dir, checkpoint)
        if not os.path.isfile(local_companion):
            # a component that declares a checkpoint needs it to play, so publishing the policy
            # alone would advertise a bundle that fails on load
            print(
                f"Not publishing {job.job_id}; its {checkpoint.name} checkpoint was not collected."
                " Collect the job again, or drop the declaration if the task no longer writes it.",
                file=sys.stderr,
            )
            return False
        uploads.append((local_companion, job.published_path(checkpoint, root=args.publish_root)))
    for source, destination in uploads:
        print(f"Publishing {source} -> {destination}")
    if args.dry_run:
        return True

    import omni.client
    from omni.client._omniclient import CopyBehavior

    for source, destination in uploads:
        result = omni.client.copy_file(source, destination, CopyBehavior.OVERWRITE)
        if result != omni.client.Result.OK:
            print(f"Publishing {source} failed for {job.job_id}: {result}", file=sys.stderr)
            return False
    return True


def _summary_row(job: CheckpointJob, output_dir: str) -> list[str | bool]:
    """Return one CSV summary row."""
    review = job.review
    return [
        job.workflow,
        job.task_name,
        job.physics_backend or "",
        job.render_backend or "",
        job.physics_selector or "",
        job.render_selector or "",
        job.has_run,
        job.is_trained,
        os.path.isfile(job.collected_path(output_dir)),
        (review or {}).get("result", ""),
    ]


def main(argv: list[str] | None = None) -> int:
    """Run checkpoint management actions."""
    parser = _create_parser()
    args = parser.parse_args(argv)
    if not args.all and not args.jobs:
        parser.error("provide one or more job patterns, or pass --all")
    if not any((args.train, args.smoke, args.collect, args.publish_checkpoint, args.review, args.list)):
        parser.error(
            "select at least one action: --train, --smoke, --collect, --publish_checkpoint, --review, or --list"
        )
    if args.train and args.smoke:
        parser.error("--train and --smoke are separate actions")

    jobs = _build_core_jobs(args) if args.core else _build_legacy_jobs()
    jobs = _filter_jobs(jobs, args)
    if not jobs:
        print("No jobs matched the requested scope and patterns.", file=sys.stderr)
        return 1
    if (args.train or args.collect) and not args.dry_run:
        for workflow in sorted({job.workflow for job in jobs}):
            os.makedirs(os.path.join(args.output_dir, workflow), exist_ok=True)

    if args.list:
        writer = csv.writer(sys.stdout, quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            [
                "Workflow",
                "Task",
                "Physics",
                "Renderer",
                "Physics selector",
                "Renderer selector",
                "Ran",
                "Finished",
                "Collected",
                "Review",
            ]
        )
        writer.writerows(_summary_row(job, args.output_dir) for job in jobs)
        return 0

    failed_jobs = []
    for job in jobs:
        success = True
        if args.smoke:
            success = train_job(job, args, smoke=True)
        if args.train:
            success = train_job(job, args)
            if success:
                success = collect_pretrained_checkpoint(job, args.output_dir, args.dry_run) is not None
        elif args.collect:
            success = collect_pretrained_checkpoint(job, args.output_dir, args.dry_run) is not None
        if success and args.review:
            success = review_pretrained_checkpoint(job, args)
        if success and args.publish_checkpoint:
            success = publish_pretrained_checkpoint(job, args)

        if not success:
            failed_jobs.append(job.job_id)
            if args.fail_fast:
                break

    if failed_jobs:
        print("Failed jobs:", file=sys.stderr)
        for job_id in failed_jobs:
            print(f"  {job_id}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
