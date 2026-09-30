# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Same-state parity of the Warp in-hand reorientation twins against the stable Direct task.

Each twin is built from its stable registration through :class:`WarpFrontend` and stepped on its
default captured path. After every step, the stable torch implementation recomputes observations
and rewards from the twin's own state, so the comparison isolates the twin's kernels from
simulation noise.
"""

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_experimental.envs.frontend import WarpFrontend

from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_error_magnitude

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.reorient.reorient_direct_env import ReorientDirectEnv, reorient_reward
from isaaclab_tasks.utils import resolve_task_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="The Warp twins capture CUDA graphs.")

_TASKS = ["Isaac-Reorient-Cube-Allegro-Direct"]


@pytest.fixture(scope="module", params=_TASKS)
def env(request):
    """A 16-environment twin of the parametrized stable task, shared by this module's tests."""
    cfg, _ = resolve_task_config(request.param, "", overrides=("physics=newton_mjwarp",))
    cfg.scene.num_envs = 16
    cfg.seed = 42
    env = WarpFrontend.build_env(cfg, request.param)
    yield env
    env.close()
    SimulationContext.clear_instance()


def _stable_state(env, actions: torch.Tensor) -> SimpleNamespace:
    """The state the stable task derives its observations and rewards from, read off the twin."""
    state = SimpleNamespace(
        cfg=env.cfg,
        num_envs=env.num_envs,
        hand=env.hand,
        object=env.object,
        scene=env.scene,
        finger_bodies=env.hand.find_bodies(env.cfg.fingertip_body_names)[0],
        actions=actions,
        goal_rot=wp.to_torch(env.goal_rot),
    )
    state.num_fingertips = len(state.finger_bodies)
    ReorientDirectEnv._compute_intermediate_values(state)
    joint_limits = env.hand.data.joint_limits.torch
    state.hand_dof_lower_limits, state.hand_dof_upper_limits = joint_limits[..., 0], joint_limits[..., 1]
    state.in_hand_pos = env.object.data.default_root_pose.torch[:, :3] + torch.tensor(
        env.cfg.in_hand_pos_offset, device=env.device
    )
    return state


def test_observations_and_rewards_match_stable_task(env):
    """Observations and rewards equal the stable task's on the same state.

    Rewards are computed before the step's resets, so they are compared on the environments whose
    state no episode reset or goal resample changed afterwards.
    """
    env.reset()
    generator = torch.Generator(device=env.device).manual_seed(0)
    num_compared = 0
    for _ in range(40):
        goal_rot = wp.to_torch(env.goal_rot).clone()
        # unit Gaussian: a policy's samples leave the [-1, 1] action range too
        actions = torch.randn((env.num_envs, env.cfg.action_space), device=env.device, generator=generator)
        obs, reward, terminated, truncated, _ = env.step(actions)
        stable = _stable_state(env, actions)

        torch.testing.assert_close(obs["policy"], ReorientDirectEnv.compute_full_observations(stable))

        untouched = ~(terminated | truncated) & (stable.goal_rot == goal_rot).all(dim=-1)
        error = quat_error_magnitude(stable.object_rot, stable.goal_rot)
        no_reset = torch.zeros_like(untouched)
        expected, *_ = reorient_reward(
            no_reset,
            no_reset,
            torch.zeros_like(error),
            torch.zeros(1, device=env.device),
            stable.object_pos,
            stable.in_hand_pos,
            error <= env.cfg.success_tolerance,
            error,
            actions,
            env.cfg.dist_reward_scale,
            env.cfg.rot_reward_scale,
            env.cfg.rot_eps,
            env.cfg.action_penalty_scale,
            env.cfg.reach_goal_bonus,
            env.cfg.fall_dist,
            env.cfg.fall_penalty,
            env.cfg.av_factor,
        )
        torch.testing.assert_close(reward[untouched], expected[untouched], rtol=1e-4, atol=1e-4)
        num_compared += int(untouched.sum())
    assert num_compared > 0
