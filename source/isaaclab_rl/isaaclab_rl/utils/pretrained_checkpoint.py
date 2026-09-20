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
import sys
from collections.abc import Sequence

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
    *,
    preset_names: Sequence[str] = (),
) -> str:
    """Return the published checkpoint filename.

    Backend-aware checkpoints use
    ``<task_name>[_<preset_names>]_<physics_backend>_<render_backend>_<rl_library><extension>``.
    Omitting both backend names returns the legacy workflow-specific filename.

    Args:
        workflow: RL workflow name.
        task_name: Registered task name.
        physics_backend: Physics backend name, such as ``"physx"``,
            ``"newtonmjwarp"``, or ``"newtonmjwarpvbdproxy"`` for a coupled solver.
        render_backend: Render backend name, such as ``"rtx"``, ``"newton"``, or ``"none"``.
        preset_names: Non-default domain presets that affect policy compatibility.

    Returns:
        The checkpoint filename.

    Raises:
        ValueError: If the workflow or backend arguments are invalid.
    """
    if workflow not in WORKFLOW_PRETRAINED_CHECKPOINT_EXTENSIONS:
        raise ValueError(f"Unsupported workflow: {workflow!r}")
    preset_names = _normalize_pretrained_checkpoint_preset_names(preset_names)
    if physics_backend is None and render_backend is None:
        if preset_names:
            raise ValueError("preset_names require backend-aware checkpoint naming")
        return WORKFLOW_PRETRAINED_CHECKPOINT_FILENAMES[workflow]
    if physics_backend is None or render_backend is None:
        raise ValueError("physics_backend and render_backend must be provided together")
    if not physics_backend:
        raise ValueError(f"Unsupported physics backend: {physics_backend!r}")
    if render_backend not in {"newton", "none", "rtx"}:
        raise ValueError(f"Unsupported render backend: {render_backend!r}")
    preset_suffix = "" if not preset_names else f"_{'_'.join(preset_names)}"
    return (
        f"{task_name}{preset_suffix}_{physics_backend}_{render_backend}_{workflow}"
        f"{WORKFLOW_PRETRAINED_CHECKPOINT_EXTENSIONS[workflow]}"
    )


def get_pretrained_checkpoint_preset_names(task_name: str, overrides: Sequence[str] | None = None) -> tuple[str, ...]:
    """Return non-default domain presets selected for a checkpoint.

    Preset aliases whose value is the preset's ``default`` are omitted so existing
    unsuffixed checkpoints remain compatible. Typed physics and renderer selectors
    are also omitted because they already have dedicated checkpoint fields.

    Args:
        task_name: Registered task name.
        overrides: Hydra-style overrides. Defaults to :data:`sys.argv`.

    Returns:
        Selected non-default domain preset names in canonical order.
    """
    from isaaclab_tasks.utils.hydra import collect_presets
    from isaaclab_tasks.utils.preset_target import PresetTarget

    selected_names = []
    for override in sys.argv[1:] if overrides is None else overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        if key.lstrip("-") == PresetTarget.DOMAIN.value:
            selected_names.extend(name.strip() for name in value.split(",") if name.strip())

    if not selected_names:
        return ()

    preset_fields = collect_presets(load_cfg_from_registry(task_name, "env_cfg_entry_point"))
    typed_targets = tuple(target for target in PresetTarget if target.base_classes)
    typed_names = {
        name
        for fields in preset_fields.values()
        for name, value in fields.items()
        if any(target.matches(value) for target in typed_targets)
    }
    non_default_domain_names = {
        name
        for fields in preset_fields.values()
        for name, value in fields.items()
        if name != "default" and name not in typed_names and not _preset_value_matches_default(value, fields)
    }
    return _normalize_pretrained_checkpoint_preset_names(
        name for name in selected_names if name in non_default_domain_names
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
    *,
    preset_names: Sequence[str] = (),
) -> str:
    """Return the absolute log root for a workflow, task, and backend combination."""
    experiment_name = _get_pretrained_checkpoint_stem(
        workflow, task_name, physics_backend, render_backend, preset_names=preset_names
    )
    return os.path.abspath(os.path.join("logs", workflow, experiment_name))


def get_latest_job_run_path(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
    *,
    preset_names: Sequence[str] = (),
) -> str | None:
    """Return the local log path of the most recent matching run."""
    log_root_path = get_log_root_path(workflow, task_name, physics_backend, render_backend, preset_names=preset_names)
    return _get_latest_file_or_directory(log_root_path)


def get_pretrained_checkpoint_path(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
    *,
    preset_names: Sequence[str] = (),
) -> str | None:
    """Return the trained checkpoint path from the latest local run."""
    path = get_latest_job_run_path(workflow, task_name, physics_backend, render_backend, preset_names=preset_names)
    if not path:
        return None

    checkpoint_stem = _get_pretrained_checkpoint_stem(
        workflow, task_name, physics_backend, render_backend, preset_names=preset_names
    )
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
    *,
    preset_names: Sequence[str] = (),
) -> str:
    """Return the path where a checkpoint is published."""
    filename = get_pretrained_checkpoint_filename(
        workflow, task_name, physics_backend, render_backend, preset_names=preset_names
    )
    if physics_backend is None:
        return posixpath.join(PRETRAINED_CHECKPOINT_PATH, workflow, task_name, filename)
    return posixpath.join(PRETRAINED_CHECKPOINT_PATH, workflow, filename)


def get_published_pretrained_checkpoint_path(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
    *,
    preset_names: Sequence[str] = (),
) -> str:
    """Return the path from which a published checkpoint is fetched."""
    filename = get_pretrained_checkpoint_filename(
        workflow, task_name, physics_backend, render_backend, preset_names=preset_names
    )
    path_parts = [ISAACLAB_NUCLEUS_DIR, "PretrainedCheckpoints", workflow]
    if physics_backend is None:
        path_parts.append(task_name)
    return posixpath.join(*path_parts, filename)


def get_published_pretrained_checkpoint(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
    *,
    preset_names: Sequence[str] | None = None,
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
        preset_names: Non-default domain presets that affect policy compatibility.
            For backend-aware checkpoints, defaults to resolving ``presets=`` selectors
            from :data:`sys.argv`. Legacy checkpoints do not use preset-qualified names.

    Returns:
        The path, or None when the asset server does not report a checkpoint for this task
        and backend combination. That covers both a checkpoint that was never published and
        a server that could not be reached, which ``omni.client`` does not distinguish, so a
        transient outage is not evidence that a checkpoint does not exist. The reason is
        printed before returning.

    Raises:
        RuntimeError: If the checkpoint is published but could not be downloaded, for
            instance because the local cache directory is not writable. The originating
            error is chained as the cause.
    """
    if preset_names is None and physics_backend is not None and render_backend is not None:
        preset_names = get_pretrained_checkpoint_preset_names(task_name)
    elif preset_names is None:
        preset_names = ()
    filename = get_pretrained_checkpoint_filename(
        workflow, task_name, physics_backend, render_backend, preset_names=preset_names
    )
    ov_path = get_published_pretrained_checkpoint_path(
        workflow, task_name, physics_backend, render_backend, preset_names=preset_names
    )
    download_dir = os.path.join(".pretrained_checkpoints", workflow)
    if physics_backend is None:
        download_dir = os.path.join(download_dir, task_name)
    resume_path = os.path.join(download_dir, filename)

    if not os.path.exists(resume_path):
        print(f"Fetching pre-trained checkpoint : {ov_path}")
        try:
            resume_path = retrieve_file_path(ov_path, download_dir)
        except FileNotFoundError:
            # the asset server reports a checkpoint that was never published and a server it
            # cannot reach the same way, so both are covered by the same message
            backends = (
                ""
                if physics_backend is None
                else f" with the '{physics_backend}' physics and '{render_backend}' render backends"
            )
            print(
                "A pre-trained checkpoint is currently unavailable for this task.\n"
                f"  The asset server does not provide '{ov_path}'.\n"
                f"  Either no checkpoint is published for task '{task_name}'{backends}, or the asset"
                " server could not be reached.\n"
                "  Train the task, or pass --checkpoint <path> to use a checkpoint of your own."
            )
            return None
        except Exception as exc:
            # the checkpoint exists on the server, so this is a local failure the user has to fix;
            # reporting it as an unavailable checkpoint would send them looking in the wrong place
            hint = ""
            if isinstance(exc, OSError):
                hint = (
                    " Check that the cache directory is writable and that the disk is not full;"
                    " a directory left behind by a container run is owned by root."
                )
            raise RuntimeError(
                f"Failed to download the pre-trained checkpoint '{ov_path}' into"
                f" '{os.path.abspath(download_dir)}': {type(exc).__name__}: {exc}.{hint}"
            ) from exc
    else:
        print("Using pre-fetched pre-trained checkpoint")
    return resume_path


def has_pretrained_checkpoint_job_run(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
    *,
    preset_names: Sequence[str] = (),
) -> bool:
    """Return whether an experiment exists for the workflow, task, and backends."""
    return os.path.exists(
        get_log_root_path(workflow, task_name, physics_backend, render_backend, preset_names=preset_names)
    )


def has_pretrained_checkpoint_job_finished(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
    *,
    preset_names: Sequence[str] = (),
) -> bool:
    """Return whether an experiment has a checkpoint result."""
    local_path = get_pretrained_checkpoint_path(
        workflow, task_name, physics_backend, render_backend, preset_names=preset_names
    )
    return local_path is not None and os.path.exists(local_path)


def get_pretrained_checkpoint_review_path(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
    *,
    preset_names: Sequence[str] = (),
) -> str | None:
    """Return the review JSON path for a workflow, task, and backends."""
    run_path = get_latest_job_run_path(workflow, task_name, physics_backend, render_backend, preset_names=preset_names)
    if not run_path:
        return None
    return os.path.join(run_path, "pretrained_checkpoint_review.json")


def get_pretrained_checkpoint_review(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
    *,
    preset_names: Sequence[str] = (),
) -> dict | None:
    """Return the review JSON data for a workflow, task, and backends."""
    review_path = get_pretrained_checkpoint_review_path(
        workflow, task_name, physics_backend, render_backend, preset_names=preset_names
    )
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
        solver_name = _get_newton_solver_name(solver_cfg)
        if solver_name is None:
            raise ValueError(f"Unsupported Newton solver for pretrained checkpoints: {type(solver_cfg).__name__}")
        return f"newton{solver_name}"
    if "physx" in type_path:
        return "physx"
    raise ValueError(f"Unable to identify physics backend from {type(physics_cfg).__name__}")


def _get_newton_solver_name(solver_cfg) -> str | None:
    """Return the checkpoint name of a Newton solver config, or ``None`` when unpublished.

    A coupled solver is named by its entry solvers in order followed by its coupling
    scheme, so a proxy coupler over MJWarp and VBD entries gives ``mjwarpvbdproxy``.
    """
    if solver_cfg is None:
        return None
    class_name = type(solver_cfg).__name__
    entries = getattr(solver_cfg, "entries", None)
    if entries is None:
        return "mjwarp" if "mjwarp" in class_name.lower() else None
    families = (type(entry.solver_cfg).__name__.removesuffix("SolverCfg").lower() for entry in entries)
    return "".join(families) + class_name.removeprefix("Coupler").removesuffix("Cfg").lower()


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
    *,
    preset_names: Sequence[str] = (),
) -> str:
    """Return the checkpoint filename without its workflow extension."""
    if physics_backend is None and render_backend is None:
        return task_name
    filename = get_pretrained_checkpoint_filename(
        workflow, task_name, physics_backend, render_backend, preset_names=preset_names
    )
    return filename.removesuffix(WORKFLOW_PRETRAINED_CHECKPOINT_EXTENSIONS[workflow])


def _normalize_pretrained_checkpoint_preset_names(preset_names: Sequence[str]) -> tuple[str, ...]:
    """Return unique, validated checkpoint preset names in canonical order."""
    normalized = tuple(sorted(set(preset_names)))
    invalid = [name for name in normalized if not name or not name.replace("_", "").isalnum()]
    if invalid:
        raise ValueError(f"Invalid checkpoint preset names: {invalid}")
    return normalized


def _preset_value_matches_default(value, fields: dict) -> bool:
    """Return whether a preset value is structurally equivalent to its default."""
    default = fields.get("default")
    try:
        return bool(value == default)
    except (RuntimeError, TypeError, ValueError):
        return value is default


def _get_latest_file_or_directory(path: str, pattern: str = "*") -> str | None:
    """Returns the path to the most recently modified file or directory at a path matching an optional pattern"""
    g = glob.glob(f"{path}/{pattern}")
    if len(g):
        return max(g, key=os.path.getmtime)
    return None
