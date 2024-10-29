# Copyright (c) 2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.managers.manager_term_cfg import RecorderTermCfg
from omni.isaac.lab.managers.recorder_manager import RecorderTerm

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv


class InitialStateRecorder(RecorderTerm):
    """Recorder term that records the initial state of the environment after reset."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

    def record_post_reset(self, env_ids: Sequence[int] | None):
        return "initial_state", self._env.scene.get_state(is_relative=True)


class PostStepStatesRecorder(RecorderTerm):
    """Recorder term that records the state of the environment at the end of each step."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

    def record_post_step(self):
        return "states", self._env.scene.get_state(is_relative=True)


class PreStepActionsRecorder(RecorderTerm):
    """Recorder term that records the actions in the beginning of each step."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

    def record_pre_step(self):
        return "actions", self._env.action_manager.action


class PreStepFlatPolicyObservationsRecorder(RecorderTerm):
    """Recorder term that records the policy group observations in each step."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

    def record_pre_step(self):
        return "obs", self._env.obs_buf["policy"]


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

    def record_pre_step(self):
        return "obs/subtask_term_signals", self._env.obs_buf["subtask_terms"]
