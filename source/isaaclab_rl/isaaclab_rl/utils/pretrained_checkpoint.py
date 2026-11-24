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
from isaaclab.visualizers import VisualizerCfg

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

WORKFLOW_CHECKPOINT_VARIANTS = {"rsl_rl": ("sac",)}
"""Maps workflow to the variants of non-default agents whose checkpoints are published next to the default ones."""


def has_pretrained_checkpoints_asset_root_dir() -> bool:
    """Returns True if and only if the asset root directory is configured in the app kit file."""
    return bool(NUCLEUS_ASSET_ROOT_DIR)


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
    from isaaclab_tasks.utils.hydra import PresetTarget, collect_presets

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
        :class:`PretrainedCheckpointId`.

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


@dataclasses.dataclass(frozen=True)
class PretrainedCheckpointId:
    """Identity of a pretrained checkpoint and the paths derived from it.

    Backend-aware checkpoints are named
    ``<task_name>[_<preset_names>]_<physics_backend>_<render_backend>_<rl_library>[_<variant>]<extension>``.
    Omitting both backend names selects the legacy workflow-specific layout.

    Raises:
        ValueError: If the workflow, backends, presets or variant are invalid.
    """

    workflow: str
    """RL workflow name."""

    task_name: str
    """Registered task name."""

    physics_backend: str | None = None
    """Physics backend name, such as ``"physx"``, ``"newtonmjwarp"``, or ``"newtonmjwarpvbdproxy"`` for a coupled
    solver."""

    render_backend: str | None = None
    """Render backend name, such as ``"rtx"``, ``"newton"``, or ``"none"``."""

    preset_names: tuple[str, ...] = ()
    """Non-default domain presets that affect policy compatibility."""

    variant: str | None = None
    """Filename suffix of a checkpoint trained with a non-default agent, such as ``"sac"``.
    See :data:`WORKFLOW_CHECKPOINT_VARIANTS`."""

    def __post_init__(self):
        if self.workflow not in WORKFLOW_PRETRAINED_CHECKPOINT_EXTENSIONS:
            raise ValueError(f"Unsupported workflow: {self.workflow!r}")
        object.__setattr__(self, "preset_names", _normalize_pretrained_checkpoint_preset_names(self.preset_names))
        if self.is_legacy:
            if self.preset_names or self.variant:
                raise ValueError("preset_names and variant require backend-aware checkpoint naming")
            return
        if self.physics_backend is None or self.render_backend is None:
            raise ValueError("physics_backend and render_backend must be provided together")
        if not self.physics_backend:
            raise ValueError(f"Unsupported physics backend: {self.physics_backend!r}")
        if self.render_backend not in {"newton", "none", "rtx"}:
            raise ValueError(f"Unsupported render backend: {self.render_backend!r}")
        if self.variant is not None and self.variant not in WORKFLOW_CHECKPOINT_VARIANTS.get(self.workflow, ()):
            raise ValueError(f"Unsupported {self.workflow} checkpoint variant: {self.variant!r}")

    @property
    def is_legacy(self) -> bool:
        """Whether the checkpoint uses the legacy layout without backend names."""
        return self.physics_backend is None and self.render_backend is None

    @property
    def filename(self) -> str:
        """Published checkpoint filename."""
        if self.is_legacy:
            return WORKFLOW_PRETRAINED_CHECKPOINT_FILENAMES[self.workflow]
        return self.stem + WORKFLOW_PRETRAINED_CHECKPOINT_EXTENSIONS[self.workflow]

    @property
    def stem(self) -> str:
        """Checkpoint filename without its extension, which also names the training experiment."""
        if self.is_legacy:
            return self.task_name
        parts = [self.task_name, *self.preset_names, self.physics_backend, self.render_backend, self.workflow]
        if self.variant is not None:
            parts.append(self.variant)
        return "_".join(parts)

    @property
    def remote_path(self) -> str:
        """Asset-server path from which the checkpoint is fetched and to which it is published."""
        path_parts = [ISAACLAB_NUCLEUS_DIR, "PretrainedCheckpoints", self.workflow]
        if self.is_legacy:
            path_parts.append(self.task_name)
        return posixpath.join(*path_parts, self.filename)

    @property
    def log_root_path(self) -> str:
        """Absolute log root of the training runs."""
        return os.path.abspath(os.path.join("logs", self.workflow, self.stem))

    def latest_run_path(self) -> str | None:
        """Return the local log path of the most recent training run."""
        return _get_latest_file_or_directory(self.log_root_path)

    def checkpoint_path(self) -> str | None:
        """Return the trained checkpoint path from the latest local run."""
        path = self.latest_run_path()
        if not path:
            return None
        if self.workflow == "rl_games":
            preferred_path = os.path.join(path, "nn", f"{self.stem}.pth")
            if os.path.isfile(preferred_path):
                return preferred_path
            return _get_latest_file_or_directory(os.path.join(path, "nn"), "*.pth")
        if self.workflow == "rsl_rl":
            return _get_latest_file_or_directory(path, "*.pt")
        if self.workflow == "sb3":
            return os.path.join(path, "model.zip")
        preferred_path = os.path.join(path, "checkpoints", "best_agent.pt")
        if os.path.isfile(preferred_path):
            return preferred_path
        return _get_latest_file_or_directory(os.path.join(path, "checkpoints"), "*.pt")

    def has_run(self) -> bool:
        """Return whether a training experiment exists."""
        return os.path.exists(self.log_root_path)

    def has_finished(self) -> bool:
        """Return whether the latest training run produced a checkpoint."""
        local_path = self.checkpoint_path()
        return local_path is not None and os.path.exists(local_path)

    def review_path(self) -> str | None:
        """Return the review JSON path of the latest training run."""
        run_path = self.latest_run_path()
        return os.path.join(run_path, "pretrained_checkpoint_review.json") if run_path else None

    def review(self) -> dict | None:
        """Return the review JSON data of the latest training run."""
        review_path = self.review_path()
        if not review_path or not os.path.exists(review_path):
            return None
        with open(review_path) as f:
            return json.load(f)


def get_published_pretrained_checkpoint(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
    *,
    preset_names: Sequence[str] | None = None,
    variant: str | None = None,
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
        variant: Filename suffix of a checkpoint trained with a non-default agent, such as ``"sac"``.

    Returns:
        The path, or None when the asset server does not report a checkpoint for this task
        and backend combination. That covers both a checkpoint that was never published and
        a server that could not be reached, which the asset helpers do not distinguish, so a
        transient outage is not evidence that a checkpoint does not exist. The reason is
        printed before returning.

    Raises:
        RuntimeError: If the checkpoint is published but could not be downloaded, for
            instance because the local cache directory is not writable. The originating
            error is chained as the cause.
    """
    if preset_names is None and physics_backend is not None and render_backend is not None:
        preset_names = get_pretrained_checkpoint_preset_names(task_name)
    checkpoint = PretrainedCheckpointId(
        workflow, task_name, physics_backend, render_backend, tuple(preset_names or ()), variant
    )
    filename = checkpoint.filename
    ov_path = checkpoint.remote_path
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
    # OVPhysX configs match too: they run the same PhysX engine, so they share the PhysX checkpoints
    if "physx" in type_path:
        return "physx"
    raise ValueError(f"Unable to identify physics backend from {type(physics_cfg).__name__}")


def _get_newton_solver_name(solver_cfg) -> str | None:
    """Return the checkpoint name of a Newton solver config, or ``None`` without one.

    A solver is generally named by its config class without ``SolverCfg``, so ``KaminoPADMMSolverCfg``
    gives ``kaminopadmm``. A coupled solver is named by its entry solvers in order followed by
    its coupling scheme, so a proxy coupler over MJWarp and VBD entries gives ``mjwarpvbdproxy``.
    """
    if solver_cfg is None:
        return None
    class_name = type(solver_cfg).__name__
    if class_name == "KaminoSCCSolverCfg":
        # Keep existing checkpoint paths after the Kamino solver config rename.
        return "kaminodvi"
    entries = getattr(solver_cfg, "entries", None)
    if entries is None:
        return class_name.removesuffix("SolverCfg").lower()
    families = (type(entry.solver_cfg).__name__.removesuffix("SolverCfg").lower() for entry in entries)
    return "".join(families) + class_name.removeprefix("Coupler").removesuffix("Cfg").lower()


def _normalize_render_backend_name(renderer_type: str) -> str:
    """Return the normalized render backend name for a renderer type identifier."""
    if renderer_type in {"newton_warp", "newton_optix"}:
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
    # a visualizer's streaming-camera renderer does not produce policy observations
    if isinstance(value, VisualizerCfg):
        return []
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
