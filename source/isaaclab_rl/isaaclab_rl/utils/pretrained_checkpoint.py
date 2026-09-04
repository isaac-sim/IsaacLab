# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for handling various pre-trained checkpoint tasks"""

from __future__ import annotations

import json
import os
import posixpath
from dataclasses import dataclass

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.renderers import RendererCfg
from isaaclab.utils import Checkpoint
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.configclass import find_cfgs
from isaaclab.utils.io import latest_file

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: F401

PRETRAINED_CHECKPOINT_PATH = ISAACLAB_NUCLEUS_DIR + "/PretrainedCheckpoints"
"""URL for where we store all the pre-trained checkpoints"""

WORKFLOW_POLICY: dict[str, Checkpoint] = {
    "rl_games": Checkpoint(name="rl_games", run_glob="nn/*.pth"),
    "rsl_rl": Checkpoint(name="rsl_rl", run_glob="*.pt"),
    "sb3": Checkpoint(name="sb3", run_glob="model.zip"),
    "skrl": Checkpoint(name="skrl", run_glob="checkpoints/*.pt"),
}
"""The policy checkpoint each supported workflow writes into its run directory."""

WORKFLOWS = tuple(WORKFLOW_POLICY)
"""The supported workflows for pre-trained checkpoints"""

WORKFLOW_EXPERIMENT_NAME_VARIABLE = {
    "rl_games": "agent.params.config.name",
    "rsl_rl": "agent.experiment_name",
    "sb3": None,
    "skrl": "agent.agent.experiment.directory",
}
"""Maps workflow to the agent variable name that determines the logging directory logs/{workflow}/{variable}"""

RENDER_BACKENDS = ("newton", "none", "rtx")
"""Render backend names a published checkpoint can be trained under."""

_PREFERRED_POLICY_FILE = {"rl_games": "nn/{stem}.pth", "skrl": "checkpoints/best_agent.pt"}
"""Run files loaded before falling back to the newest match of the workflow's policy glob."""


@dataclass(frozen=True)
class CheckpointBundle:
    """The published files of one trained task variant: a policy and the checkpoints declared beside it.

    A variant is one workflow, task, physics backend and render backend. Backend-aware bundles are
    published flat under ``<root>/<workflow>/`` as ``<task>_<physics>_<render>_<workflow><ext>``, with
    a declared checkpoint at ``<stem>_<name><ext>`` beside the policy. Omitting both backends selects
    the legacy layout ``<root>/<workflow>/<task>/checkpoint<ext>``.
    """

    workflow: str
    task_name: str
    physics_backend: str | None = None
    render_backend: str | None = None
    checkpoints: tuple[Checkpoint, ...] = ()
    """Run artifacts the task's components declare, published beside the policy."""

    def __post_init__(self) -> None:
        if self.workflow not in WORKFLOW_POLICY:
            raise ValueError(f"Unsupported workflow: {self.workflow!r}")
        if self.is_legacy:
            return
        if self.physics_backend is None or self.render_backend is None:
            raise ValueError("physics_backend and render_backend must be provided together")
        # Coupled solvers name themselves from their entries, so the set is open-ended.
        if not self.physics_backend:
            raise ValueError(f"Unsupported physics backend: {self.physics_backend!r}")
        if self.render_backend not in RENDER_BACKENDS:
            raise ValueError(f"Unsupported render backend: {self.render_backend!r}")

    @classmethod
    def from_env_cfg(
        cls,
        workflow: str,
        task_name: str,
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        physics_backend: str | None = None,
        render_backend: str | None = None,
        **fields,
    ) -> CheckpointBundle:
        """Build the bundle a resolved environment config trains.

        The backends are read from the config unless both are given. The checkpoints are every run
        artifact a component declares anywhere in the config, so a task lists nothing itself.

        Args:
            workflow: RL workflow name.
            task_name: Registered task name.
            env_cfg: Resolved environment configuration.
            physics_backend: Physics backend name. Omit with :paramref:`render_backend` to read both
                from :paramref:`env_cfg`.
            render_backend: Render backend name. Omit with :paramref:`physics_backend` to read both
                from :paramref:`env_cfg`.
            **fields: Further fields of a subclass.
        """
        if physics_backend is None and render_backend is None:
            physics_backend, render_backend = cls.backend_names(env_cfg)
        # a declaration can be reachable through several config paths; the name is the identity
        unique: dict[str, Checkpoint] = {}
        for checkpoint in find_cfgs(env_cfg, Checkpoint):
            if checkpoint.is_run_artifact:
                unique.setdefault(checkpoint.name, checkpoint)
        return cls(workflow, task_name, physics_backend, render_backend, tuple(unique.values()), **fields)

    @staticmethod
    def backend_names(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> tuple[str, str]:
        """Return the normalized ``(physics_backend, render_backend)`` names an environment config uses.

        Raises:
            ValueError: If a backend cannot be identified or multiple renderer backends are active.
        """
        physics_backend = _get_physics_backend_name(getattr(getattr(env_cfg, "sim", None), "physics", None))
        renderer_types = {cfg.renderer_type for cfg in find_cfgs(env_cfg, RendererCfg)}
        render_backends = {_normalize_render_backend_name(name) for name in renderer_types}
        if not render_backends:
            render_backend = "none"
        elif len(render_backends) == 1:
            render_backend = render_backends.pop()
        else:
            raise ValueError(f"Multiple renderer backends are active: {sorted(render_backends)}")
        return physics_backend, render_backend

    """
    Identity.
    """

    @property
    def is_legacy(self) -> bool:
        """Whether the bundle uses the legacy per-task layout without backend names."""
        return self.physics_backend is None and self.render_backend is None

    @property
    def policy(self) -> Checkpoint:
        """The policy checkpoint the workflow writes."""
        return WORKFLOW_POLICY[self.workflow]

    @property
    def stem(self) -> str:
        """The published filename without extension. Training runs log under this experiment name."""
        if self.is_legacy:
            return self.task_name
        return f"{self.task_name}_{self.physics_backend}_{self.render_backend}_{self.workflow}"

    def filename(self, checkpoint: Checkpoint | None = None) -> str:
        """Return the published filename of the policy, or of a declared checkpoint beside it."""
        stem = "checkpoint" if self.is_legacy else self.stem
        if checkpoint is None:
            return f"{stem}{self.policy.extension}"
        return f"{stem}_{checkpoint.name}{checkpoint.extension}"

    def _relative_path(self, checkpoint: Checkpoint | None) -> tuple[str, ...]:
        """Return the path of a file in this bundle below a publish or collect root."""
        if self.is_legacy:
            return self.workflow, self.task_name, self.filename(checkpoint)
        return self.workflow, self.filename(checkpoint)

    """
    Published files.
    """

    def published_path(self, checkpoint: Checkpoint | None = None, root: str | None = None) -> str:
        """Return the remote path of a file in this bundle.

        Args:
            checkpoint: A declared checkpoint, or ``None`` for the policy.
            root: The publish root. Defaults to :data:`PRETRAINED_CHECKPOINT_PATH`.
        """
        root = PRETRAINED_CHECKPOINT_PATH if root is None else root.rstrip("/")
        return posixpath.join(root, *self._relative_path(checkpoint))

    @property
    def cache_dir(self) -> str:
        """Where fetched files land. Play treats it as the run log directory, so each bundle gets its own."""
        return os.path.join(".pretrained_checkpoints", self.workflow, self.stem)

    def fetch(self) -> str | None:
        """Download the policy and every declared checkpoint into :attr:`cache_dir`.

        :func:`~isaaclab.utils.assets.retrieve_file_path` skips an up-to-date local copy itself.

        Returns:
            The local policy path, or ``None`` if no policy is published for this bundle.
        """
        remote_path = self.published_path()
        print(f"Fetching pre-trained checkpoint : {remote_path}")
        local_path = _fetch(remote_path, self.cache_dir)
        if local_path is None:
            print("A pre-trained checkpoint is currently unavailable for this task.")
            return None
        for checkpoint in self.checkpoints:
            _fetch(self.published_path(checkpoint), self.cache_dir)
        return local_path

    def collected_path(self, output_dir: str, checkpoint: Checkpoint | None = None) -> str:
        """Return where the collect step copies a file, mirroring the published layout under ``output_dir``."""
        return os.path.abspath(os.path.join(output_dir, *self._relative_path(checkpoint)))

    """
    Local training runs.
    """

    @property
    def log_root(self) -> str:
        """The absolute directory holding this variant's training runs."""
        return os.path.abspath(os.path.join("logs", self.workflow, self.stem))

    @property
    def latest_run(self) -> str | None:
        """The most recent training run directory, or ``None`` before the first run."""
        return latest_file(self.log_root)

    def trained_path(self, checkpoint: Checkpoint | None = None) -> str | None:
        """Return the file the latest training run wrote for the policy, or for a declared checkpoint.

        The policy prefers the workflow's best or final file when it exists, then the newest match of
        its glob. Returns ``None`` when there is no run or no matching file.
        """
        run_path = self.latest_run
        if run_path is None:
            return None
        if checkpoint is None:
            preferred = _PREFERRED_POLICY_FILE.get(self.workflow)
            if preferred is not None:
                preferred_path = os.path.join(run_path, preferred.format(stem=self.stem))
                if os.path.isfile(preferred_path):
                    return preferred_path
            checkpoint = self.policy
        return checkpoint.find_in(run_path)

    @property
    def has_run(self) -> bool:
        """Whether a training run was started for this variant."""
        return os.path.exists(self.log_root)

    @property
    def has_finished(self) -> bool:
        """Whether the latest training run wrote a policy checkpoint."""
        return self.trained_path() is not None

    """
    Review.
    """

    @property
    def review_path(self) -> str | None:
        """The review JSON path in the latest run, or ``None`` before the first run."""
        run_path = self.latest_run
        return None if run_path is None else os.path.join(run_path, "pretrained_checkpoint_review.json")

    @property
    def review(self) -> dict | None:
        """The review JSON data of the latest run, or ``None`` if it was not reviewed."""
        review_path = self.review_path
        if review_path is None or not os.path.exists(review_path):
            return None
        with open(review_path) as f:
            return json.load(f)


def get_published_pretrained_checkpoint(
    workflow: str,
    task_name: str,
    physics_backend: str | None = None,
    render_backend: str | None = None,
    *,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg | None = None,
) -> str | None:
    """Gets the path for the pre-trained checkpoint.

    If the checkpoint is not cached locally then the file is downloaded, together with every
    checkpoint the task's components declare, into one directory a component reading its log
    directory finds. The cached path is then returned.

    Args:
        workflow: The workflow.
        task_name: The task name.
        physics_backend: Physics backend name. Omit with :paramref:`render_backend`
            to use the legacy checkpoint layout, or to derive both from :paramref:`env_cfg`.
        render_backend: Render backend name. Omit with :paramref:`physics_backend`
            to use the legacy checkpoint layout, or to derive both from :paramref:`env_cfg`.
        env_cfg: Resolved environment configuration. Supplies the backends when they are not
            given, and the checkpoints its components declare.

    Returns:
        The path.
    """
    if env_cfg is None:
        bundle = CheckpointBundle(workflow, task_name, physics_backend, render_backend)
    else:
        bundle = CheckpointBundle.from_env_cfg(workflow, task_name, env_cfg, physics_backend, render_backend)
    return bundle.fetch()


def _fetch(remote_path: str, download_dir: str) -> str | None:
    """Download a published file into the cache, or report why it could not be."""
    try:
        return retrieve_file_path(remote_path, download_dir)
    except Exception as error:  # noqa: BLE001
        print(f"[WARNING]: Could not fetch {remote_path}: {error}")
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
