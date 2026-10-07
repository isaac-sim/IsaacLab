# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terminal observations of the Warp environments match the stable environments.

With ``compute_final_obs`` set, a step that resets environments exposes the observation computed before the
reset in ``extras["final_obs"]`` and returns the observation after the reset, as the stable environments do
for Same-Step autoreset.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())))

import gymnasium as gym
import isaaclab_tasks_experimental  # noqa: F401
import pytest
import torch
from isaaclab_experimental.envs.frontend import WarpFrontend

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

_NUM_ENVS = 16
_STEPS = 120
_CART_BOUND = 3.0
_RESET_CART_POS = 0.5
_TASK_IDS = ("Isaac-Cartpole", "Isaac-Cartpole-Direct")


def _cartpole_cfg(task_id: str, compute_final_obs: bool):
    """Cartpole with fixed reset states, so the stable and the Warp random draws agree."""
    env_cfg, _ = resolve_task_config(task_id, "", overrides=("physics=newton_mjwarp",))
    env_cfg.seed = 7
    env_cfg.scene.num_envs = _NUM_ENVS
    env_cfg.compute_final_obs = compute_final_obs
    if task_id == "Isaac-Cartpole":
        env_cfg.events.reset_cart_position.params.update(
            position_range=(_RESET_CART_POS, _RESET_CART_POS), velocity_range=(0.0, 0.0)
        )
        env_cfg.events.reset_pole_position.params.update(position_range=(0.1, 0.1), velocity_range=(0.0, 0.0))
    else:
        env_cfg.initial_cart_position_range = (_RESET_CART_POS, _RESET_CART_POS)
        env_cfg.initial_cart_velocity_range = (0.0, 0.0)
        env_cfg.initial_pole_angle_range = (0.1, 0.1)
        env_cfg.initial_pole_velocity_range = (0.0, 0.0)
    return env_cfg


def _rollout(task_id: str, frontend: str, compute_final_obs: bool) -> dict:
    """Push every cart with a constant force and record the observations of every step."""
    env_cfg = _cartpole_cfg(task_id, compute_final_obs)
    sim_utils.create_new_stage()
    if frontend == "warp":
        env = WarpFrontend.build_env(env_cfg, task_id).unwrapped
    else:
        env = gym.make(task_id, cfg=env_cfg).unwrapped
    rollout = {"obs": [], "dones": [], "final_obs": {}, "final_obs_key": []}
    try:
        env.reset()
        # different pushes drive the carts out of bounds at different steps
        push = torch.linspace(-0.3, 0.3, _NUM_ENVS, device=env.device)[:, None]
        for step in range(_STEPS):
            obs, _, terminated, truncated, extras = env.step(push)
            dones = terminated | truncated
            rollout["obs"].append(obs["policy"].clone())
            rollout["dones"].append(dones.clone())
            rollout["final_obs_key"].append("final_obs" in extras)
            if dones.any() and "final_obs" in extras:
                rollout["final_obs"][step] = extras["final_obs"]["policy"].clone()
        rollout["cart_index"] = (
            0 if task_id != "Isaac-Cartpole" else env.scene["robot"].find_joints("slider_to_cart")[0][0]
        )
    finally:
        env.close()
    rollout["obs"] = torch.stack(rollout["obs"])
    rollout["dones"] = torch.stack(rollout["dones"])
    return rollout


@pytest.fixture(scope="module")
def rollouts() -> dict[tuple[str, str, bool], dict]:
    arms = [(task_id, frontend, True) for task_id in _TASK_IDS for frontend in ("stable", "warp")]
    arms += [(task_id, "warp", False) for task_id in _TASK_IDS]
    return {arm: _rollout(*arm) for arm in arms}


@pytest.mark.parametrize("task_id", _TASK_IDS)
def test_final_obs_is_the_observation_before_reset(task_id: str, rollouts: dict):
    rollout = rollouts[(task_id, "warp", True)]
    cart = rollout["cart_index"]
    reset_steps = torch.nonzero(rollout["dones"].any(dim=1)).squeeze(-1).tolist()
    assert reset_steps, "no environment terminated; the terminal observation is not exercised"
    assert sorted(rollout["final_obs"]) == reset_steps
    for step in reset_steps:
        dones = rollout["dones"][step]
        final_obs, obs = rollout["final_obs"][step], rollout["obs"][step]
        # the terminal observation still has the cart out of bounds, the returned one is reset
        assert (final_obs[dones, cart].abs() > _CART_BOUND).all()
        assert torch.equal(obs[dones, cart], torch.full_like(obs[dones, cart], _RESET_CART_POS))
        assert torch.equal(final_obs[~dones], obs[~dones])


@pytest.mark.parametrize("task_id", _TASK_IDS)
def test_final_obs_is_absent_when_disabled(task_id: str, rollouts: dict):
    rollout = rollouts[(task_id, "warp", False)]
    assert rollout["dones"].any()
    assert not any(rollout["final_obs_key"])


@pytest.mark.parametrize(
    ("task_id", "atol"),
    # the direct twin differs from the stable task in the last bit only (at most 4.8e-7 at |obs| < 8)
    [("Isaac-Cartpole", 0.0), ("Isaac-Cartpole-Direct", 1e-6)],
)
def test_final_obs_matches_stable(task_id: str, atol: float, rollouts: dict):
    stable, warp = rollouts[(task_id, "stable", True)], rollouts[(task_id, "warp", True)]
    assert torch.equal(warp["dones"], stable["dones"])
    torch.testing.assert_close(warp["obs"], stable["obs"], rtol=0.0, atol=atol)
    assert sorted(warp["final_obs"]) == sorted(stable["final_obs"])
    for step, final_obs in stable["final_obs"].items():
        torch.testing.assert_close(warp["final_obs"][step], final_obs, rtol=0.0, atol=atol, msg=f"step {step}")
