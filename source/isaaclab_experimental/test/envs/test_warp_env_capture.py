# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Eager and CUDA-graph execution of the Warp environments must produce the same rollout.

Each task runs the same seeded action sequence twice, once with ``ISAACLAB_WARP_CAPTURE=0`` and
once with stage capture on. For manager-based tasks, a hard :meth:`SimulationContext.reset` halfway
through rebuilds the Newton model and reallocates every simulation buffer; recorded graphs still
point at the freed buffers, so the captured rollout only stays equal when the environment records
its stages again.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())))

import isaaclab_tasks_experimental  # noqa: F401
import pytest
import torch
from isaaclab_experimental.envs.frontend import WarpFrontend
from isaaclab_experimental.utils.warp_graph_cache import CAPTURE_ENV_VAR

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

_NUM_ENVS = 32
_STEPS = 80
_REBIND_STEP = 40


def _rollout(task_id: str, capture: bool, rebind: bool, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run the seeded action sequence and return the rollout with the recorded stages."""
    monkeypatch.setenv(CAPTURE_ENV_VAR, "1" if capture else "0")
    env_cfg, _ = resolve_task_config(task_id, "", overrides=("physics=newton_mjwarp",))
    env_cfg.seed = 7
    env_cfg.scene.num_envs = _NUM_ENVS
    # contacts are only reproducible between runs with deterministic contact ordering
    env_cfg.sim.physics.deterministic_mode = "run_to_run"
    env_cfg.sim.physics.solver_cfg.disable_sensors = True
    sim_utils.create_new_stage()
    env = WarpFrontend.build_env(env_cfg, task_id).unwrapped
    rollout = {"obs": [], "reward": [], "terminated": [], "truncated": []}
    try:
        env.reset()
        generator = torch.Generator().manual_seed(123)
        # a per-env bias drives environments to terminate at different steps
        bias = torch.rand((_NUM_ENVS, env.action_space.shape[-1]), generator=generator) * 2.0 - 1.0
        for step in range(_STEPS):
            if rebind and step == _REBIND_STEP:
                rollout["stages_before_rebind"] = env._warp_graph_cache.captured_stages
                env.sim.reset()
                rollout["stages_at_rebind"] = env._warp_graph_cache.captured_stages
                env.reset()
            noise = torch.rand((_NUM_ENVS, bias.shape[1]), generator=generator) * 2.0 - 1.0
            obs, reward, terminated, truncated, _ = env.step((bias + 0.5 * noise).to(env.device))
            rollout["obs"].append(obs["policy"].clone())
            rollout["reward"].append(reward.clone())
            rollout["terminated"].append(terminated.clone())
            rollout["truncated"].append(truncated.clone())
        rollout["stages_at_end"] = env._warp_graph_cache.captured_stages
    finally:
        env.close()
    for key in ("obs", "reward", "terminated", "truncated"):
        rollout[key] = torch.stack(rollout[key])
    return rollout


# Direct tasks keep the simulation arrays they bind in ``__init__``, so they do not survive a rebind.
@pytest.mark.parametrize(
    ("task_id", "rebind"),
    [("Isaac-Cartpole", True), ("Isaac-Cartpole-Direct", False)],
    ids=["manager-cartpole", "direct-cartpole"],
)
def test_captured_rollout_matches_eager(task_id: str, rebind: bool, monkeypatch: pytest.MonkeyPatch):
    eager = _rollout(task_id, capture=False, rebind=rebind, monkeypatch=monkeypatch)
    captured = _rollout(task_id, capture=True, rebind=rebind, monkeypatch=monkeypatch)

    assert eager["stages_at_end"] == ()
    assert captured["stages_at_end"], "no stage was recorded; the comparison proves nothing"
    if rebind:
        assert captured["stages_at_rebind"] == (), "the rebind must drop graphs that read freed buffers"
        assert captured["stages_at_end"] == captured["stages_before_rebind"]
    for window in (slice(0, _REBIND_STEP), slice(_REBIND_STEP, _STEPS)):
        dones = captured["terminated"][window] | captured["truncated"][window]
        assert dones.any(), "each half must reset environments so the reset stages replay"
    for key in ("obs", "reward", "terminated", "truncated"):
        assert torch.equal(captured[key], eager[key]), f"{key} differs between eager and captured execution"
