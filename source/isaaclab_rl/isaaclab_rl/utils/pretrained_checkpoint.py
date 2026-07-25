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

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.renderers import RendererCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, NUCLEUS_ASSET_ROOT_DIR, retrieve_file_path

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: F401

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
    ``<task_name>_<physics_backend>_<render_backend><extension>``. Omitting
    both backend names returns the legacy workflow-specific filename.

    Args:
        workflow: RL workflow name.
        task_name: Registered task name.
        physics_backend: Physics backend name, such as ``"physx"`` or ``"newton"``.
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
    if physics_backend not in {"newton", "physx"}:
        raise ValueError(f"Unsupported physics backend: {physics_backend!r}")
    if render_backend not in {"newton", "none", "rtx"}:
        raise ValueError(f"Unsupported render backend: {render_backend!r}")
    return f"{task_name}_{physics_backend}_{render_backend}{WORKFLOW_PRETRAINED_CHECKPOINT_EXTENSIONS[workflow]}"


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
        return os.path.join(PRETRAINED_CHECKPOINT_PATH, workflow, task_name, filename)
    return os.path.join(PRETRAINED_CHECKPOINT_PATH, workflow, filename)


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
    return os.path.join(*path_parts, filename)


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
        return "newton"
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
    filename = get_pretrained_checkpoint_filename(workflow, task_name, physics_backend, render_backend)
    return filename.removesuffix(WORKFLOW_PRETRAINED_CHECKPOINT_EXTENSIONS[workflow])


def _get_latest_file_or_directory(path: str, pattern: str = "*") -> str | None:
    """Returns the path to the most recently modified file or directory at a path matching an optional pattern"""
    g = glob.glob(f"{path}/{pattern}")
    if len(g):
        return max(g, key=os.path.getmtime)
    return None
