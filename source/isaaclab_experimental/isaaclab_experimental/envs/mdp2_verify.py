# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Debug check that a task's captured steps compute what its uncaptured steps compute.

A capturable Python hook or stage runs once, at capture, and its launches replay. One that reads a value on the host or
keeps Python-side state gives replayed steps the value it had at capture, without an error. :func:`verify_capture`
finds it by stepping the task twice, uncaptured then captured, from the same seed and with the same actions, and
comparing every step.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab_experimental import mdp2

from .mdp2_env_cfg import MDP2EnvCfg, Stage

if TYPE_CHECKING:
    from .mdp2_env import MDP2Env


def verify_capture(env_type: type[MDP2Env], cfg: MDP2EnvCfg, num_steps: int) -> None:
    """Step a task ``num_steps`` times uncaptured, then captured, and raise where a step differs.

    Both runs use a copy of the config with at most 8 worlds (unless it splits them between tasks), the same seed, and
    seeded random actions in ``[-1, 1]``. Steps are compared on the rewards, the terminated and truncated flags, the
    observations, and every quantity and term state, which are named in the error in schedule order, with the Python
    hooks and stages that run inside the graphs, where the cause usually is. The worlds start at staggered episode
    lengths, so that, with a time limit, every world times out and resets during the checked steps, also in the
    captured ones. Contact-rich physics is not reproducible
    run to run on GPUs, so the runs use the backend's reproducible mode where it has one (Newton's
    ``deterministic_mode``) and compare bitwise; otherwise they compare to a relative 1e-3 and should be short.
    """
    # One seed for both runs, also when the config asks for a random one.
    seed = 0 if cfg.seed in (None, -1) else cfg.seed
    cfg = dataclasses.replace(cfg, capture=False, verify_capture=0, seed=seed)
    if cfg.task_worlds is None:
        # A shallow copy keeps the entities that were added to the scene config as attributes.
        cfg.scene = copy.copy(cfg.scene)
        cfg.scene.num_envs = min(cfg.scene.num_envs, 8)
    reproducible = _with_reproducible_physics(cfg)
    runs = [_rollout(env_type, cfg, num_steps), _rollout(env_type, dataclasses.replace(cfg, capture=True), num_steps)]
    (eager, _), (captured, suspects) = runs
    for step, (expected, actual) in enumerate(zip(eager, captured)):
        for name, value in expected.items():
            # NaN marks values such as an absent heading target, so it matches NaN.
            tolerance = {"rtol": 0.0, "atol": 0.0} if reproducible else {"rtol": 1e-3, "atol": 1e-5}
            if not torch.allclose(value.double(), actual[name].double(), equal_nan=True, **tolerance):
                raise RuntimeError(
                    f"Captured step {step} differs from the uncaptured step at '{name}'. A capturable Python hook or"
                    " stage may read a value on the host or keep Python-side state, which a CUDA graph freezes at"
                    f" capture. Capturable Python code of this task: {suspects or 'none'}."
                )


def _with_reproducible_physics(cfg: MDP2EnvCfg) -> bool:
    """Make the config's physics reproducible run to run, where the backend offers it. Returns whether it does."""
    physics = cfg.sim.physics
    if type(physics).__module__.split(".")[0] != "isaaclab_newton":
        return False
    if physics.solver_cfg.class_type.max_deterministic_mode < wp.DeterministicMode.RUN_TO_RUN:
        return False
    cfg.sim = dataclasses.replace(cfg.sim, physics=dataclasses.replace(physics, deterministic_mode="run_to_run"))
    return True


def _rollout(env_type: type[MDP2Env], cfg: MDP2EnvCfg, num_steps: int) -> tuple[list[dict], list[str]]:
    """Reset and step a copy of the task, and return each step's results by name, and its capturable Python code."""
    env = env_type(cfg)
    try:
        generator = torch.Generator(device=env.device).manual_seed(0)
        env.reset()
        if env.mdp.max_episode_length is not None:
            # World k times out at step k + 1, cycling over the checked steps after the first, uncaptured, one.
            worlds = torch.arange(env.num_envs, device=env.device)
            remaining = 1 + worlds % max(num_steps - 1, 1)
            lengths = (env.mdp.max_episode_length - remaining).clamp(min=0).to(torch.int32)
            env.state.episode_length.torch.copy_(lengths)
        named = {}
        for op in env.mdp.term_ops:
            if op.phase is mdp2.Phase.QUANTITIES:
                named[str(op.writes[0])] = env.state.arrays[op.writes[0]].torch
            for name in op.state:
                selector = mdp2.term_state(op.use, name)
                named[str(selector)] = env.state.arrays[selector].torch
        steps = []
        for _ in range(num_steps):
            actions = 2.0 * torch.rand(env.num_envs, env.mdp.action_dim, device=env.device, generator=generator) - 1.0
            env.step(actions)
            transition = env.transition
            results = {
                "rewards": transition.rewards.torch,
                "terminated": transition.terminated.torch,
                "truncated": transition.truncated.torch,
                **{f"observations.{group}": values.torch for group, values in transition.observations.items()},
                **named,
            }
            steps.append({name: value.clone() for name, value in results.items()})
        # Python hooks that run inside the graphs; fused wp.func hooks cannot read the host.
        suspects = [op.name for op in env.mdp.term_ops if op.capturable and not isinstance(op.compute, wp.Function)]
        stages = [item for item in (*cfg.step, *cfg.reset_step) if isinstance(item, Stage)]
        suspects += [f"stage {stage.fn.__name__}" for stage in stages if stage.capturable]
        return steps, suspects
    finally:
        env.close()
