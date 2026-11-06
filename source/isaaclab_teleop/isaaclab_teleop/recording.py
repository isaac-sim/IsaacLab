# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demonstration export through the task's existing recorder and episode lifecycle."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import gymnasium as gym
import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import ActionTermCfg, DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import clone, configclass

from .input_cfg import TeleopInputCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DemoRecordingCfg:
    """Playback export options; the task retains its observations and episode rules."""

    dataset_file: str
    num_demos: int = 0
    export_mode: Literal["succeeded", "all"] = "succeeded"


@configclass
class _TeleopActiveRecorderCfg(RecorderTermCfg):
    """Identify the action term that owns the shared Capture input sample."""

    class_type: type[RecorderTerm] | str = "{DIR}.recording:_TeleopActiveRecorder"
    action_term_name: str = ""


class _TeleopActiveRecorder(RecorderTerm):
    """Label physical transitions selected from Capture, including policy fallback."""

    cfg: _TeleopActiveRecorderCfg

    def record_pre_step(self) -> tuple[str, torch.Tensor]:
        action_input = self._env.action_manager.get_input(self.cfg.action_term_name)
        if action_input is None:
            raise RuntimeError("Capture provenance requires an activated task action input.")
        sample = action_input.sample
        active = sample is not None and sample.active and not sample.reset and sample.actions is not None
        return "metadata/teleop_active", torch.full((self.num_envs, 1), active, dtype=torch.bool, device=self.device)


class _DemoPolicyObservationsRecorder(RecorderTerm):
    """Keep named dataset observations without changing the policy's input layout."""

    def record_pre_step(self) -> tuple[str, dict]:
        observations = self._env.obs_buf["policy"]
        return "obs", observations if isinstance(observations, dict) else {"policy": observations}


def prepare_demo_recording(
    env_cfg: ManagerBasedRLEnvCfg,
    cfg: DemoRecordingCfg,
    task_name: str,
    env_ids: Sequence[int],
) -> DemoRecordingCfg:
    """Configure export before launch without changing task or policy behavior.

    Existing recorder terms take precedence over the task's registered demonstration
    recorder and the standard action/state recorder. The returned options contain
    the absolute output path. Existing files are rejected before any launch work.
    """
    if env_cfg.scene.num_envs != 1 or list(env_ids) != [0]:
        raise ValueError("Demonstration recording requires --num_envs 1 and Capture controlling row 0.")
    if cfg.num_demos < 0:
        raise ValueError("num_demos must be non-negative; zero records without a count limit.")
    if cfg.export_mode not in ("succeeded", "all"):
        raise ValueError("Demonstration export_mode must be 'succeeded' or 'all'.")
    if not cfg.dataset_file.strip():
        raise ValueError("A non-empty dataset_file is required for demonstration recording.")
    output = Path(cfg.dataset_file).expanduser()
    if not output.suffix:
        output = output.with_suffix(".hdf5")
    if output.suffix != ".hdf5":
        raise ValueError("Demonstration dataset_file must use the .hdf5 extension.")
    if output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite demonstration dataset symlink: {output}")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Demonstration dataset already exists: {output}")

    terminations = env_cfg.terminations
    termination_terms = terminations if isinstance(terminations, dict) else vars(terminations) if terminations else {}
    if cfg.export_mode == "succeeded" and termination_terms.get("success") is None:
        raise ValueError("Successful-demo export requires a task 'success' termination; select export_mode='all'.")

    actions = env_cfg.actions if isinstance(env_cfg.actions, dict) else vars(env_cfg.actions)
    owners = [
        name
        for name, term in actions.items()
        if isinstance(term, ActionTermCfg) and isinstance(term.input, TeleopInputCfg) and term.input.source is not None
    ]
    if len(owners) != 1:
        raise ValueError("Demonstration recording requires exactly one Capture source-owning action input.")

    recorder_cfg = env_cfg.recorders
    recorder_terms = recorder_cfg if isinstance(recorder_cfg, dict) else vars(recorder_cfg) if recorder_cfg else {}
    if not any(isinstance(term, RecorderTermCfg) for term in recorder_terms.values()):
        registered = gym.spec(task_name.split(":")[-1]).kwargs.get("demo_recorder_cfg_entry_point")
        if registered is None:
            recorder_cfg = ActionStateRecorderManagerCfg()
            recorder_cfg.record_pre_step_flat_policy_observations.class_type = _DemoPolicyObservationsRecorder
        else:
            from isaaclab_tasks.utils import load_cfg_from_registry

            recorder_cfg = load_cfg_from_registry(task_name, "demo_recorder_cfg_entry_point")
    if not isinstance(recorder_cfg, RecorderManagerBaseCfg):
        raise TypeError("Demonstration recorders must use RecorderManagerBaseCfg.")
    recorder_cfg = clone(recorder_cfg)
    if not any(isinstance(term, RecorderTermCfg) for term in vars(recorder_cfg).values()):
        raise ValueError("The task's demonstration recorder has no active recorder terms.")
    if getattr(recorder_cfg, "record_teleop_active", None) is not None:
        raise ValueError("The recorder term name 'record_teleop_active' is reserved for Capture provenance.")
    recorder_cfg.record_teleop_active = _TeleopActiveRecorderCfg(action_term_name=owners[0])
    recorder_cfg.dataset_export_dir_path = str(output.parent)
    recorder_cfg.dataset_filename = output.name
    recorder_cfg.dataset_export_mode = (
        DatasetExportMode.EXPORT_SUCCEEDED_ONLY if cfg.export_mode == "succeeded" else DatasetExportMode.EXPORT_ALL
    )
    recorder_cfg.export_in_record_pre_reset = True
    recorder_cfg.export_in_close = False
    env_cfg.env_name = task_name.split(":")[-1]
    env_cfg.recorders = recorder_cfg
    return replace(cfg, dataset_file=str(output))


class DemoRecording:
    """Track physical episodes and completion while the environment owns HDF5 export."""

    def __init__(self, cfg: DemoRecordingCfg) -> None:
        self._cfg = cfg
        self._episode_steps = 0
        self._exported = 0
        self._reset_export_setting: bool | None = None

    @property
    def complete(self) -> bool:
        """Whether the requested number of exported demonstrations has been reached."""
        return self._cfg.num_demos > 0 and self._exported >= self._cfg.num_demos

    def before_explicit_reset(self, env: ManagerBasedRLEnv) -> None:
        """Suppress empty episodes; pair with ``after_explicit_reset`` in a finally block."""
        recorder = env.recorder_manager
        self._reset_export_setting = recorder.cfg.export_in_record_pre_reset
        if self._episode_steps == 0:
            recorder.reset()
            recorder.cfg.export_in_record_pre_reset = False
        self._episode_steps = 0

    def after_explicit_reset(self, env: ManagerBasedRLEnv) -> None:
        """Restore normal automatic exports and count a completed manual-reset episode."""
        if self._reset_export_setting is not None:
            env.recorder_manager.cfg.export_in_record_pre_reset = self._reset_export_setting
            self._reset_export_setting = None
        self._report_progress(env)

    def after_step(self, env: ManagerBasedRLEnv, done: torch.Tensor) -> None:
        """Observe one real environment step, after any automatic episode export/reset."""
        self._episode_steps = 0 if bool(done.any()) else self._episode_steps + 1
        self._report_progress(env)

    def _report_progress(self, env: ManagerBasedRLEnv) -> None:
        count = env.recorder_manager.exported_successful_episode_count
        if self._cfg.export_mode == "all":
            count += env.recorder_manager.exported_failed_episode_count
        if count != self._exported:
            self._exported = count
            logger.info("Recorded %d demonstrations to %s", count, self._cfg.dataset_file)
