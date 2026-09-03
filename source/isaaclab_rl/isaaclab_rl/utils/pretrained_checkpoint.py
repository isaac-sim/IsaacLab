# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for handling various pre-trained checkpoint tasks"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import posixpath
import re
from collections.abc import Callable

import gymnasium as gym

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.renderers import RendererCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, NUCLEUS_ASSET_ROOT_DIR, retrieve_file_path

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

PRETRAINED_CHECKPOINT_PATH = ISAACLAB_NUCLEUS_DIR + "/PretrainedCheckpoints"
"""URL for where we store all the pre-trained checkpoints"""

WORKFLOWS = ["rl_games", "rsl_rl", "sb3", "skrl"]
"""The supported workflows for pre-trained checkpoints"""

WORKFLOW_TRAINER = {w: "scripts/reinforcement_learning/train.py" for w in WORKFLOWS}
"""A dict mapping workflow to their training program path.

All workflows share the unified training entrypoint; pass ``--rl_library <workflow>`` when invoking it.
"""

WORKFLOW_PLAYER = {w: "scripts/reinforcement_learning/play.py" for w in WORKFLOWS}
"""A dict mapping workflow to their play program path.

All workflows share the unified playback entrypoint; pass ``--rl_library <workflow>`` when invoking it.
"""

WORKFLOW_PRETRAINED_CHECKPOINT_FILENAMES = {
    "rl_games": "checkpoint.pth",
    "rsl_rl": "checkpoint.pt",
    "sb3": "checkpoint.zip",
    "skrl": "checkpoint.pt",
}
"""Legacy filename for checkpoints used by the different workflows."""

WORKFLOW_PRETRAINED_CHECKPOINT_EXTENSIONS = {
    "rl_games": ".pth",
    "rsl_rl": ".pt",
    "sb3": ".zip",
    "skrl": ".pt",
}
"""The checkpoint filename extension used by each workflow."""

WORKFLOW_EXPERIMENT_NAME_VARIABLE = {
    "rl_games": "agent.params.config.name",
    "rsl_rl": "agent.experiment_name",
    "sb3": None,
    "skrl": "agent.agent.experiment.directory",
}
"""Maps workflow to the agent variable name that determines the logging directory logs/{workflow}/{variable}"""

PRETRAINED_CHECKPOINT_DEFAULT_VARIANT = "default"
"""Variant identifier for the checkpoint whose artifact name has no variant suffix."""

_EnvCfg = ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg


@dataclasses.dataclass(frozen=True)
class PretrainedCheckpointCfg:
    """Declaration for one pretrained policy contract.

    Physics and rendering backends remain independent dimensions. The task's
    :attr:`PretrainedCheckpointSetCfg.variant_resolver` derives ``variant``
    from the final resolved environment configuration.

    Attributes:
        workflow: RL workflow that owns the checkpoint.
        variant: Stable policy variant. ``"default"`` preserves the artifact
            filename without a variant suffix.
        training_presets: Presets used to reproduce this variant for training.
        agent: Non-default agent configuration entry point, if required.
        algorithm: Non-default algorithm passed to the training CLI, if required.
        smoke_num_envs: Environment count for smoke training, if task-specific.
        physics_backends: Supported normalized physics backends, or ``None`` for all.
        render_backends: Supported normalized render backends, or ``None`` for all.
    """

    workflow: str
    variant: str = PRETRAINED_CHECKPOINT_DEFAULT_VARIANT
    training_presets: tuple[str, ...] = ()
    agent: str | None = None
    algorithm: str | None = None
    smoke_num_envs: int | None = None
    physics_backends: tuple[str, ...] | None = None
    render_backends: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the declaration at construction time."""
        if self.workflow not in WORKFLOWS:
            raise ValueError(f"Unsupported workflow: {self.workflow!r}")
        if not isinstance(self.variant, str) or re.fullmatch(r"[A-Za-z0-9_-]+", self.variant) is None:
            raise ValueError(f"Invalid checkpoint variant: {self.variant!r}")
        if self.smoke_num_envs is not None and self.smoke_num_envs < 1:
            raise ValueError("smoke_num_envs must be positive")

    def supports_backends(self, physics_backend: str, render_backend: str) -> bool:
        """Return whether this checkpoint declaration supports the backends."""
        return (self.physics_backends is None or physics_backend in self.physics_backends) and (
            self.render_backends is None or render_backend in self.render_backends
        )

    def artifact_task_name(self, task_name: str) -> str:
        """Return the task component used in this checkpoint's artifact name."""
        if self.variant == PRETRAINED_CHECKPOINT_DEFAULT_VARIANT:
            return task_name
        return f"{task_name}_{self.variant}"


@dataclasses.dataclass(frozen=True)
class PretrainedCheckpointSetCfg:
    """Pretrained policies declared for a task with multiple policy contracts.

    The resolver inspects the final environment configuration, after presets
    and scalar overrides have been applied. A lookup must match the resulting
    variant exactly, which prevents an undeclared policy contract from falling
    back to a shape-incompatible default checkpoint.

    Attributes:
        variant_resolver: Function mapping a resolved environment configuration
            to a declared policy variant, or ``None`` when no published policy
            is compatible.
        checkpoints: Published policy contracts and their selectors.
    """

    variant_resolver: Callable[[_EnvCfg], str | None]
    checkpoints: tuple[PretrainedCheckpointCfg, ...]

    def __post_init__(self) -> None:
        """Validate the variant resolver."""
        if not callable(self.variant_resolver):
            raise TypeError("variant_resolver must be callable")


def get_pretrained_checkpoint_set_cfg(task_name: str) -> PretrainedCheckpointSetCfg | None:
    """Load a task's optional pretrained-checkpoint declaration.

    Args:
        task_name: Registered Gym task name.

    Returns:
        The declaration, or ``None`` when the task uses the legacy implicit
        default checkpoint.

    Raises:
        TypeError: If the registered entry point returns the wrong type.
        ValueError: If a checkpoint references an unregistered agent configuration.
    """
    task_name = task_name.split(":")[-1]
    task_spec = gym.spec(task_name)
    if "pretrained_checkpoint_cfg_entry_point" not in task_spec.kwargs:
        return None
    cfg = load_cfg_from_registry(task_name, "pretrained_checkpoint_cfg_entry_point")
    if not isinstance(cfg, PretrainedCheckpointSetCfg):
        raise TypeError(f"Expected PretrainedCheckpointSetCfg for {task_name!r}, received {type(cfg).__name__}")
    for checkpoint in cfg.checkpoints:
        agent = checkpoint.agent or f"{checkpoint.workflow}_cfg_entry_point"
        if agent not in task_spec.kwargs:
            raise ValueError(f"Checkpoint for {task_name!r} references unregistered agent {agent!r}")
    return cfg


def select_pretrained_checkpoint(
    workflow: str,
    task_name: str,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent: str | None = None,
) -> PretrainedCheckpointCfg | None:
    """Select the checkpoint declaration matching a resolved task configuration.

    Tasks without an explicit checkpoint set retain the legacy implicit default.
    Declared tasks fail closed when the resolved policy variant, agent, or
    backends do not match an entry.

    Args:
        workflow: RL workflow name.
        task_name: Registered Gym task name.
        env_cfg: Resolved environment configuration.
        agent: Agent configuration entry point selected by the CLI.

    Returns:
        The matching checkpoint declaration, or ``None`` if no compatible
        checkpoint is declared.

    Raises:
        ValueError: If multiple declarations match.
    """
    checkpoint_set = get_pretrained_checkpoint_set_cfg(task_name)
    if checkpoint_set is None:
        return PretrainedCheckpointCfg(workflow=workflow)
    backend_names = get_pretrained_checkpoint_backend_names(env_cfg)
    return _select_declared_pretrained_checkpoint(workflow, task_name, env_cfg, agent, checkpoint_set, backend_names)


def _select_declared_pretrained_checkpoint(
    workflow: str,
    task_name: str,
    env_cfg: _EnvCfg,
    agent: str | None,
    checkpoint_set: PretrainedCheckpointSetCfg,
    backend_names: tuple[str, str],
) -> PretrainedCheckpointCfg | None:
    """Select a checkpoint using declarations and normalized backends already loaded by the caller."""
    active_variant = checkpoint_set.variant_resolver(env_cfg)
    if active_variant is None:
        return None
    physics_backend, render_backend = backend_names
    default_agent = f"{workflow}_cfg_entry_point"
    selected_agent = None if agent in (None, default_agent) else agent
    matches = [
        checkpoint
        for checkpoint in checkpoint_set.checkpoints
        if checkpoint.workflow == workflow
        and checkpoint.agent == selected_agent
        and checkpoint.supports_backends(physics_backend, render_backend)
        and checkpoint.variant == active_variant
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple pretrained checkpoints match {task_name!r}, {workflow!r}, variant {active_variant!r}"
        )
    return matches[0] if matches else None


def has_pretrained_checkpoints_asset_root_dir() -> bool:
    """Returns True if and only if the asset root directory is configured in the app kit file."""
    return bool(NUCLEUS_ASSET_ROOT_DIR)


def get_pretrained_checkpoint_filename(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
) -> str:
    """Return the published checkpoint filename.

    Backend-aware checkpoints use
    ``<task_name>_<physics_backend>_<render_backend>_<rl_library><extension>``.
    Omitting both backend names returns the legacy workflow-specific filename.

    Args:
        workflow: RL workflow name.
        task_name: Registered task name.
        physics_backend: Physics backend name, such as ``"physx"`` or
            ``"newtonmjwarp"``.
        render_backend: Render backend name, such as ``"rtx"``, ``"newton"``, or ``"none"``.

    Returns:
        The checkpoint filename.

    Raises:
        ValueError: If the workflow or backend arguments are invalid.
    """
    if workflow not in WORKFLOW_PRETRAINED_CHECKPOINT_EXTENSIONS:
        raise ValueError(f"Unsupported workflow: {workflow!r}")
    if physics_backend is None and render_backend is None:
        return WORKFLOW_PRETRAINED_CHECKPOINT_FILENAMES[workflow]
    if physics_backend is None or render_backend is None:
        raise ValueError("physics_backend and render_backend must be provided together")
    if physics_backend not in {"newtonmjwarp", "physx"}:
        raise ValueError(f"Unsupported physics backend: {physics_backend!r}")
    if render_backend not in {"newton", "none", "rtx"}:
        raise ValueError(f"Unsupported render backend: {render_backend!r}")
    return (
        f"{task_name}_{physics_backend}_{render_backend}_{workflow}"
        f"{WORKFLOW_PRETRAINED_CHECKPOINT_EXTENSIONS[workflow]}"
    )


def get_pretrained_checkpoint_backend_names(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
) -> tuple[str, str]:
    """Return normalized physics and render backend names for an environment config.

    Args:
        env_cfg: Resolved environment configuration.

    Returns:
        A ``(physics_backend, render_backend)`` tuple suitable for
        :func:`get_pretrained_checkpoint_filename`.

    Raises:
        ValueError: If a backend cannot be identified or multiple renderer
            backends are active.
    """
    sim_cfg = getattr(env_cfg, "sim", None)
    physics_cfg = getattr(sim_cfg, "physics", None)
    physics_backend = _get_physics_backend_name(physics_cfg)

    renderer_types = {cfg.renderer_type for cfg in _find_renderer_cfgs(env_cfg)}
    render_backends = {_normalize_render_backend_name(name) for name in renderer_types}
    if not render_backends:
        render_backend = "none"
    elif len(render_backends) == 1:
        render_backend = render_backends.pop()
    else:
        raise ValueError(f"Multiple renderer backends are active: {sorted(render_backends)}")
    return physics_backend, render_backend


def get_log_root_path(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
) -> str:
    """Return the absolute log root for a workflow, task, and backend combination."""
    experiment_name = _get_pretrained_checkpoint_stem(workflow, task_name, physics_backend, render_backend)
    return os.path.abspath(os.path.join("logs", workflow, experiment_name))


def get_latest_job_run_path(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
) -> str | None:
    """Return the local log path of the most recent matching run."""
    log_root_path = get_log_root_path(workflow, task_name, physics_backend, render_backend)
    return _get_latest_file_or_directory(log_root_path)


def get_pretrained_checkpoint_path(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
) -> str | None:
    """Return the trained checkpoint path from the latest local run."""
    path = get_latest_job_run_path(workflow, task_name, physics_backend, render_backend)
    if not path:
        return None

    checkpoint_stem = _get_pretrained_checkpoint_stem(workflow, task_name, physics_backend, render_backend)
    if workflow == "rl_games":
        preferred_path = os.path.join(path, "nn", f"{checkpoint_stem}.pth")
        if os.path.isfile(preferred_path):
            return preferred_path
        return _get_latest_file_or_directory(os.path.join(path, "nn"), "*.pth")
    elif workflow == "rsl_rl":
        return _get_latest_file_or_directory(path, "*.pt")
    elif workflow == "sb3":
        return os.path.join(path, "model.zip")
    elif workflow == "skrl":
        preferred_path = os.path.join(path, "checkpoints", "best_agent.pt")
        if os.path.isfile(preferred_path):
            return preferred_path
        return _get_latest_file_or_directory(os.path.join(path, "checkpoints"), "*.pt")
    else:
        raise ValueError(f"Unsupported workflow: {workflow!r}")


def get_pretrained_checkpoint_publish_path(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
) -> str:
    """Return the path where a checkpoint is published."""
    filename = get_pretrained_checkpoint_filename(workflow, task_name, physics_backend, render_backend)
    if physics_backend is None:
        return posixpath.join(PRETRAINED_CHECKPOINT_PATH, workflow, task_name, filename)
    return posixpath.join(PRETRAINED_CHECKPOINT_PATH, workflow, filename)


def get_published_pretrained_checkpoint_path(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
) -> str:
    """Return the path from which a published checkpoint is fetched."""
    filename = get_pretrained_checkpoint_filename(workflow, task_name, physics_backend, render_backend)
    path_parts = [ISAACLAB_NUCLEUS_DIR, "PretrainedCheckpoints", workflow]
    if physics_backend is None:
        path_parts.append(task_name)
    return posixpath.join(*path_parts, filename)


def get_published_pretrained_checkpoint(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
) -> str | None:
    """Gets the path for the pre-trained checkpoint.

    If the checkpoint is not cached locally then the file is downloaded.
    The cached path is then returned.

    Args:
        workflow: The workflow.
        task_name: The task name.
        physics_backend: Physics backend name. Omit with :paramref:`render_backend`
            to use the legacy checkpoint layout.
        render_backend: Render backend name. Omit with :paramref:`physics_backend`
            to use the legacy checkpoint layout.

    Returns:
        The path.
    """
    filename = get_pretrained_checkpoint_filename(workflow, task_name, physics_backend, render_backend)
    ov_path = get_published_pretrained_checkpoint_path(workflow, task_name, physics_backend, render_backend)
    download_dir = os.path.join(".pretrained_checkpoints", workflow)
    if physics_backend is None:
        download_dir = os.path.join(download_dir, task_name)
    resume_path = os.path.join(download_dir, filename)

    if not os.path.exists(resume_path):
        print(f"Fetching pre-trained checkpoint : {ov_path}")
        try:
            resume_path = retrieve_file_path(ov_path, download_dir)
        except Exception:
            print("A pre-trained checkpoint is currently unavailable for this task.")
            return None
    else:
        print("Using pre-fetched pre-trained checkpoint")
    return resume_path


def get_published_pretrained_checkpoint_for_env(
    workflow: str,
    task_name: str,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent: str | None = None,
) -> str | None:
    """Fetch the published checkpoint compatible with a resolved environment.

    Args:
        workflow: RL workflow name.
        task_name: Registered Gym task name.
        env_cfg: Resolved environment configuration.
        agent: Agent configuration entry point selected by the CLI.

    Returns:
        The local checkpoint path, or ``None`` when no compatible checkpoint
        is declared or published.
    """
    checkpoint_set = get_pretrained_checkpoint_set_cfg(task_name)
    backend_names = get_pretrained_checkpoint_backend_names(env_cfg)
    if checkpoint_set is None:
        checkpoint = PretrainedCheckpointCfg(workflow=workflow)
    else:
        checkpoint = _select_declared_pretrained_checkpoint(
            workflow, task_name, env_cfg, agent, checkpoint_set, backend_names
        )
    if checkpoint is None:
        print("A pre-trained checkpoint is currently unavailable for this task configuration.")
        return None
    return get_published_pretrained_checkpoint(workflow, checkpoint.artifact_task_name(task_name), *backend_names)


def has_pretrained_checkpoint_job_run(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
) -> bool:
    """Return whether an experiment exists for the workflow, task, and backends."""
    return os.path.exists(get_log_root_path(workflow, task_name, physics_backend, render_backend))


def has_pretrained_checkpoint_job_finished(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
) -> bool:
    """Return whether an experiment has a checkpoint result."""
    local_path = get_pretrained_checkpoint_path(workflow, task_name, physics_backend, render_backend)
    return local_path is not None and os.path.exists(local_path)


def get_pretrained_checkpoint_review_path(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
) -> str | None:
    """Return the review JSON path for a workflow, task, and backends."""
    run_path = get_latest_job_run_path(workflow, task_name, physics_backend, render_backend)
    if not run_path:
        return None
    return os.path.join(run_path, "pretrained_checkpoint_review.json")


def get_pretrained_checkpoint_review(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
) -> dict | None:
    """Return the review JSON data for a workflow, task, and backends."""
    review_path = get_pretrained_checkpoint_review_path(workflow, task_name, physics_backend, render_backend)
    if not review_path:
        return None

    if os.path.exists(review_path):
        with open(review_path) as f:
            return json.load(f)

    return None


def _get_physics_backend_name(physics_cfg: PhysicsCfg | None) -> str:
    """Return the normalized physics backend name for a resolved physics config."""
    if physics_cfg is None:
        return "physx"
    type_path = f"{type(physics_cfg).__module__}.{type(physics_cfg).__name__}".lower()
    if "newton" in type_path:
        solver_cfg = getattr(physics_cfg, "solver_cfg", None)
        solver_type_path = f"{type(solver_cfg).__module__}.{type(solver_cfg).__name__}".lower()
        if "mjwarp" in solver_type_path:
            return "newtonmjwarp"
        raise ValueError(f"Unsupported Newton solver for pretrained checkpoints: {type(solver_cfg).__name__}")
    if "physx" in type_path:
        return "physx"
    raise ValueError(f"Unable to identify physics backend from {type(physics_cfg).__name__}")


def _normalize_render_backend_name(renderer_type: str) -> str:
    """Return the normalized render backend name for a renderer type identifier."""
    if renderer_type == "newton_warp":
        return "newton"
    if renderer_type in {"auto_rtx", "default", "isaac_rtx", "ovrtx", "rtx"}:
        return "rtx"
    raise ValueError(f"Unable to identify render backend from renderer type {renderer_type!r}")


def _find_renderer_cfgs(value, visited: set[int] | None = None) -> list[RendererCfg]:
    """Find renderer configs nested in a resolved environment config."""
    if visited is None:
        visited = set()
    value_id = id(value)
    if value_id in visited:
        return []
    visited.add(value_id)

    if isinstance(value, RendererCfg):
        return [value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        configs = []
        for field in dataclasses.fields(value):
            configs.extend(_find_renderer_cfgs(getattr(value, field.name), visited))
        return configs
    if isinstance(value, dict):
        configs = []
        for item in value.values():
            configs.extend(_find_renderer_cfgs(item, visited))
        return configs
    if isinstance(value, (list, tuple)):
        configs = []
        for item in value:
            configs.extend(_find_renderer_cfgs(item, visited))
        return configs
    return []


def _get_pretrained_checkpoint_stem(
    workflow: str,
    task_name: str,
    physics_backend: str | None,
    render_backend: str | None,
) -> str:
    """Return the checkpoint filename without its workflow extension."""
    if physics_backend is None and render_backend is None:
        return task_name
    filename = get_pretrained_checkpoint_filename(workflow, task_name, physics_backend, render_backend)
    return filename.removesuffix(WORKFLOW_PRETRAINED_CHECKPOINT_EXTENSIONS[workflow])


def _get_latest_file_or_directory(path: str, pattern: str = "*") -> str | None:
    """Returns the path to the most recently modified file or directory at a path matching an optional pattern"""
    g = glob.glob(f"{path}/{pattern}")
    if len(g):
        return max(g, key=os.path.getmtime)
    return None
