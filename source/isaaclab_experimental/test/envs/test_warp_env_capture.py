# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Eager and CUDA-graph execution of the Warp environments must produce the same rollout.

Each task runs the same seeded action sequence twice, once with capture disabled
(:attr:`CapturedStage.enabled` False) and once with stage capture on. For manager-based tasks, a hard
:meth:`SimulationContext.reset` halfway through rebuilds the Newton model and reallocates every simulation
buffer; recorded graphs still point at the freed buffers, so the captured rollout only stays equal when the
stages record again. A variant marks one term of every manager as not capturable: its stage must record the
other terms and run that term eagerly.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())))

import isaaclab_tasks_experimental  # noqa: F401
import pytest
import torch
from isaaclab_experimental.envs.frontend import WarpFrontend
from isaaclab_experimental.utils import CapturedStage
from isaaclab_experimental.utils.warp import WarpCapturable

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

_NUM_ENVS = 32
_MANAGERS = ("action_manager", "observation_manager", "event_manager", "reward_manager", "termination_manager")
_STEPS = 80
_REBIND_STEP = 40

# One function term per staged manager of the manager-based Cartpole.
_CARTPOLE_EAGER_TERMS = (
    "observations.policy.joint_pos_rel",
    "rewards.pole_pos",
    "terminations.cart_out_of_bounds",
    "events.reset_cart_position",
)


def _recorded_stages(env) -> tuple[str, ...]:
    """``Owner.method`` of every stage currently backed by a recorded graph, sorted."""
    owners = [env, *(getattr(env, name) for name in _MANAGERS if hasattr(env, name))]
    return tuple(
        sorted(
            f"{type(owner).__name__}.{method.__name__}"
            for owner in owners
            for method, stage in owner.__dict__.get("_captured_stages", {}).items()
            if stage.num_graphs
        )
    )


def _stale_stages(env) -> list[str]:
    """Stages whose graphs were recorded before the latest physics rebind."""
    owners = [env, *(getattr(env, name) for name in _MANAGERS if hasattr(env, name))]
    return [
        method.__name__
        for owner in owners
        for method, stage in owner.__dict__.get("_captured_stages", {}).items()
        if stage.num_graphs and stage._generation != CapturedStage.generation
    ]


def _rollout(
    task_id: str, capture: bool, rebind: bool, monkeypatch: pytest.MonkeyPatch, eager_terms: tuple[str, ...] = ()
) -> dict:
    """Run the seeded action sequence and return the rollout with the recorded stages."""
    env_cfg, _ = resolve_task_config(task_id, "", overrides=("physics=newton_mjwarp",))
    env_cfg.seed = 7
    env_cfg.scene.num_envs = _NUM_ENVS
    # contacts are only reproducible between runs with deterministic contact ordering
    env_cfg.sim.physics.deterministic_mode = "run_to_run"
    env_cfg.sim.physics.solver_cfg.disable_sensors = True
    WarpFrontend.adapt_cfg(env_cfg)
    for path in eager_terms:
        term_cfg = env_cfg
        for name in path.split("."):
            term_cfg = getattr(term_cfg, name)
        # the guard raises if the term is ever recorded into a graph
        term_cfg.func = WarpCapturable(False, reason="marked eager by the test")(term_cfg.func)
    sim_utils.create_new_stage()
    env = WarpFrontend.build_env(env_cfg, task_id).unwrapped
    # the environment enables capture when it is built
    monkeypatch.setattr(CapturedStage, "enabled", capture)
    rollout = {"obs": [], "reward": [], "terminated": [], "truncated": []}
    try:
        env.reset()
        generator = torch.Generator().manual_seed(123)
        # a per-env bias drives environments to terminate at different steps
        bias = torch.rand((_NUM_ENVS, env.action_space.shape[-1]), generator=generator) * 2.0 - 1.0
        for step in range(_STEPS):
            if rebind and step == _REBIND_STEP:
                rollout["stages_before_rebind"] = _recorded_stages(env)
                generation = CapturedStage.generation
                env.sim.reset()
                rollout["rebind_advanced_generation"] = CapturedStage.generation > generation
                env.reset()
            noise = torch.rand((_NUM_ENVS, bias.shape[1]), generator=generator) * 2.0 - 1.0
            obs, reward, terminated, truncated, _ = env.step((bias + 0.5 * noise).to(env.device))
            rollout["obs"].append(obs["policy"].clone())
            rollout["reward"].append(reward.clone())
            rollout["terminated"].append(terminated.clone())
            rollout["truncated"].append(truncated.clone())
        rollout["stages_at_end"] = _recorded_stages(env)
        rollout["stale_stages"] = _stale_stages(env)
    finally:
        env.close()
    for key in ("obs", "reward", "terminated", "truncated"):
        rollout[key] = torch.stack(rollout[key])
    return rollout


# Direct tasks keep the simulation arrays they bind in ``__init__``, so they do not survive a rebind.
@pytest.mark.parametrize(
    ("task_id", "rebind", "eager_terms"),
    [
        ("Isaac-Cartpole", True, ()),
        ("Isaac-Cartpole", False, _CARTPOLE_EAGER_TERMS),
        ("Isaac-Cartpole-Direct", False, ()),
    ],
    ids=["manager-cartpole", "manager-cartpole-eager-terms", "direct-cartpole"],
)
def test_captured_rollout_matches_eager(
    task_id: str, rebind: bool, eager_terms: tuple[str, ...], monkeypatch: pytest.MonkeyPatch
):
    eager = _rollout(task_id, capture=False, rebind=rebind, monkeypatch=monkeypatch, eager_terms=eager_terms)
    captured = _rollout(task_id, capture=True, rebind=rebind, monkeypatch=monkeypatch, eager_terms=eager_terms)

    assert eager["stages_at_end"] == ()
    assert captured["stages_at_end"], "no stage was recorded; the comparison proves nothing"
    if eager_terms:
        # every stage holding an eager term still records the terms around it
        recorded_groups = {stage.partition(".")[0] for stage in captured["stages_at_end"]}
        assert {"ObservationManager", "RewardManager", "TerminationManager", "EventManager"} <= recorded_groups
    if rebind:
        assert captured["rebind_advanced_generation"], "the rebind must retire graphs that read freed buffers"
        assert captured["stale_stages"] == [], "every stage must record again after the rebind"
        assert captured["stages_at_end"] == captured["stages_before_rebind"]
    for window in (slice(0, _REBIND_STEP), slice(_REBIND_STEP, _STEPS)):
        dones = captured["terminated"][window] | captured["truncated"][window]
        assert dones.any(), "each half must reset environments so the reset stages replay"
    for key in ("obs", "reward", "terminated", "truncated"):
        assert torch.equal(captured[key], eager[key]), f"{key} differs between eager and captured execution"
