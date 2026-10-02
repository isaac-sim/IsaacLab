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

_SHADOW_TASK = "Isaac-Reorient-Cube-Shadow-Direct"
_TASKS = ["Isaac-Reorient-Cube-Allegro-Direct", _SHADOW_TASK]


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


@pytest.mark.parametrize("env", [_SHADOW_TASK], indirect=True)
def test_tendon_actions_curl_the_coupled_finger_joints(env):
    """The tendon action columns drive the middle and distal finger joints, step after step.

    Every finger is first opened, then half of the environments curl theirs; a tendon command
    that never reached the solver, or reached it only once, leaves both halves open.
    """
    env.reset()
    coupled_joints, _ = env.hand.find_joints("rh_(FF|MF|RF|LF)J[12]")
    num_joint_actions = len(env.cfg.actuated_joint_names)
    curling = torch.arange(env.num_envs, device=env.device) < env.num_envs // 2
    actions = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    actions[:, num_joint_actions:] = -1.0
    for _ in range(10):
        env.step(actions)
    actions[curling, num_joint_actions:] = 1.0
    was_reset = torch.zeros_like(curling)
    for _ in range(30):
        _, _, terminated, truncated, _ = env.step(actions)
        was_reset |= terminated | truncated

    curl = env.hand.data.joint_pos.torch[:, coupled_joints].sum(dim=-1)
    assert (curl[curling & ~was_reset].min() - curl[~curling & ~was_reset].max()) > 1.0


@pytest.mark.parametrize("env", [_SHADOW_TASK], indirect=True)
def test_assigned_episode_lengths_time_out_only_their_envs(env):
    """An episode-length buffer assigned from outside, as RSL-RL does, times out only the envs it ends.

    The masked reset restarts those episodes and resamples their goals, leaving the others' alone.
    """
    env.reset()
    episode_lengths = torch.zeros_like(env.episode_length_buf)
    episode_lengths[0] = env.max_episode_length - 2
    env.episode_length_buf = episode_lengths
    goal_rot = wp.to_torch(env.goal_rot).clone()

    _, _, terminated, truncated, _ = env.step(torch.zeros((env.num_envs, env.cfg.action_space), device=env.device))

    assert truncated[0] and not truncated[1:].any()
    continuing = ~(terminated | truncated)
    assert env.episode_length_buf[0] == 0
    assert (env.episode_length_buf[continuing] == 1).all()
    new_goal_rot = wp.to_torch(env.goal_rot)
    assert not torch.equal(new_goal_rot[0], goal_rot[0])
    unreached = continuing & ~wp.to_torch(env.goal_reached)
    torch.testing.assert_close(new_goal_rot[unreached], goal_rot[unreached])
