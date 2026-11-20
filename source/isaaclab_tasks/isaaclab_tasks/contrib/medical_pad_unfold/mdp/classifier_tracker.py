# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frame-level classifier judgement of a generated chunk, shared by the reward and the terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ClassifierTracker:
    """Score each generated chunk frame by frame and find the frame at which the episode succeeds or fails.

    The world model delivers a chunk of frames at once, so the task's judgement is made inside the chunk: a
    classifier scores every frame, and an episode succeeds (fails) at the first frame that completes
    ``streak_frames`` consecutive frames of the success (failure) class. The reward counts the progress up to and
    including that frame and nothing after it, since the frames after it belong to an episode that has already ended.
    The terminations report the outcome at the end of the chunk, where the environment can reset.

    The reward and the terminations read one tracker, which scores a chunk once per environment step. Streaks and the
    previous score persist across chunks and start again with each episode.

    Attributes:
        reward: Progress of the chunk that completed in this step, up to the terminal frame, ``(num_envs,)``.
        success: Environments whose episode succeeded in the chunk that completed in this step.
        failure: Environments whose episode failed in that chunk.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        checkpoint: str,
        config_file: str,
        success_class: int,
        failure_class: int,
        streak_frames: int,
    ):
        from ..classifier import MedicalPadClassifier

        self.classifier = MedicalPadClassifier(checkpoint, config_file, torch.device(env.device))
        self.success_class, self.failure_class, self.streak_frames = success_class, failure_class, streak_frames
        n, device = env.num_envs, env.device
        self.previous = torch.zeros(n, device=device)
        self.success_streak = torch.zeros(n, dtype=torch.long, device=device)
        self.failure_streak = torch.zeros(n, dtype=torch.long, device=device)
        self.reward = torch.zeros(n, device=device)
        self.success = torch.zeros(n, dtype=torch.bool, device=device)
        self.failure = torch.zeros(n, dtype=torch.bool, device=device)
        self._step = -1
        self._length = torch.zeros(n, dtype=torch.long, device=device)

    def update(self, env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> None:
        """Judge the chunk that completed in this environment step, once per step."""
        if env.common_step_counter == self._step:
            return
        self._step = env.common_step_counter
        # A shorter episode counter means the environment was reset since the last step: start its judgement again.
        restarted = env.episode_length_buf < self._length
        self._length = env.episode_length_buf.clone()
        for state in (self.previous, self.success_streak, self.failure_streak):
            state.masked_fill_(restarted, 0)
        self.reward.zero_()
        self.success.zero_()
        self.failure.zero_()

        data = env.scene[sensor_cfg.name].data
        # Every environment gets a new chunk on the same step, and only then, so the host flag picks the step without
        # reading the camera's mask on the device.
        if not env.sim.backend_manager.new_frames:
            return
        scores, probabilities = self.classifier.score_frames(data.frame_history["rgb"].torch)
        scores = scores.to(env.device)
        labels = probabilities.argmax(-1).to(env.device)
        count, frames = scores.shape
        success, failure = self.success_streak, self.failure_streak
        terminal = torch.full((count,), frames, dtype=torch.long, device=env.device)
        won = torch.zeros(count, dtype=torch.bool, device=env.device)
        lost = torch.zeros_like(won)
        for t in range(frames):
            success = torch.where(labels[:, t] == self.success_class, success + 1, 0)
            failure = torch.where(labels[:, t] == self.failure_class, failure + 1, 0)
            first = (terminal == frames) & ((success >= self.streak_frames) | (failure >= self.streak_frames))
            terminal = torch.where(first, t, terminal)
            won |= first & (success >= self.streak_frames)
            lost |= first & ~(success >= self.streak_frames)
        # Frames up to and including the terminal one are valid; without a terminal frame every frame is.
        valid = torch.arange(frames, device=env.device)[None] <= terminal[:, None]
        progress = torch.cat([self.previous[:, None], scores], dim=1).diff(dim=1)
        self.reward.copy_((progress * valid).sum(dim=1))
        last = terminal.clamp(max=frames - 1)
        self.previous.copy_(scores.gather(1, last[:, None])[:, 0])
        self.success_streak.copy_(success)
        self.failure_streak.copy_(failure)
        self.success.copy_(won)
        self.failure.copy_(lost)


def get_tracker(env: ManagerBasedRLEnv, classifier: dict, sensor_cfg: SceneEntityCfg) -> ClassifierTracker:
    """Return the environment's tracker, built on first use, updated for this step."""
    tracker = getattr(env, "_medical_pad_tracker", None)
    if tracker is None:
        tracker = env._medical_pad_tracker = ClassifierTracker(env, **classifier)
    tracker.update(env, sensor_cfg)
    return tracker
