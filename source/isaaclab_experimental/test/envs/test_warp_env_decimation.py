# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Warp manager-based environment steps physics at the stable environment's cadence.

When every actuator runs inside Newton, the physics backend folds the whole decimation loop into one
step: actions are applied, the simulation stepped and the scene updated once per environment step.
Contact sensors then sample once per environment step, so their history, the air times and every
reward and termination reading them depend on that cadence. The same seeded Go2 rollout on the stable
and the Warp frontend must make the same calls and produce the same contact data.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())))

import math

import isaaclab_tasks_experimental  # noqa: F401
import pytest
import torch
from isaaclab_experimental.envs.frontend import WarpFrontend

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.core.velocity.velocity_env_cfg import RewardsCfg
from isaaclab_tasks.utils import resolve_task_config

_TASK_ID = "Isaac-Velocity-Flat-UnitreeGo2"
_NUM_ENVS = 64
_STEPS = 200


def _go2_cfg():
    """Go2 flat with every random draw removed, so both frontends see identical physics inputs."""
    env_cfg, _ = resolve_task_config(_TASK_ID, "", overrides=("physics=newton_mjwarp",))
    env_cfg.seed = 7
    env_cfg.scene.num_envs = _NUM_ENVS
    # contacts are only reproducible between runs with deterministic contact ordering
    env_cfg.sim.physics.deterministic_mode = "run_to_run"
    env_cfg.sim.physics.solver_cfg.disable_sensors = True
    env_cfg.events.add_base_mass = None
    env_cfg.events.base_com = None
    env_cfg.events.push_robot = None
    env_cfg.events.reset_base.params["pose_range"] = {}
    env_cfg.events.reset_base.params["velocity_range"] = {}
    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    command = env_cfg.commands.base_velocity
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.debug_vis = False
    command.ranges.lin_vel_x = (0.5, 0.5)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    # Go2 drops the thigh-contact penalty; restore it to cover a second contact-history reader
    env_cfg.rewards.undesired_contacts = RewardsCfg().undesired_contacts
    env_cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"
    return env_cfg


def _rollout(frontend: str) -> dict:
    """Step a fixed action sequence and record the physics calls and the contact data of every step."""
    env_cfg = _go2_cfg()
    sim_utils.create_new_stage()
    if frontend == "warp":
        env = WarpFrontend.build_env(env_cfg, _TASK_ID).unwrapped
    else:
        env = ManagerBasedRLEnv(cfg=env_cfg)
    calls = {"sim_step": 0, "apply_action": 0}
    sim_step = env.sim.step

    def count_sim_step(*args, **kwargs):
        calls["sim_step"] += 1
        return sim_step(*args, **kwargs)

    env.sim.step = count_sim_step
    apply_action = env.action_manager.apply_action

    def count_apply_action():
        calls["apply_action"] += 1
        apply_action()

    env.action_manager.apply_action = count_apply_action
    if frontend == "warp":
        step_reward = env.reward_manager._step_reward_tensor_view
        term_dones = env.termination_manager._term_dones_tensor_view
    else:
        step_reward = env.reward_manager._step_reward
        term_dones = env.termination_manager._term_dones
    reward_names = env.reward_manager._term_names
    base_contact = env.termination_manager._term_names.index("base_contact")
    sensor = env.scene["contact_forces"].data
    rollout = {"calls": [], "handles_decimation": env._physics_handles_decimation}
    records = {
        "net_forces_w_history": lambda: sensor.net_forces_w_history.torch,
        "current_air_time": lambda: sensor.current_air_time.torch,
        "last_air_time": lambda: sensor.last_air_time.torch,
        "undesired_contacts": lambda: step_reward[:, reward_names.index("undesired_contacts")],
        "feet_air_time": lambda: step_reward[:, reward_names.index("feet_air_time")],
        "base_contact": lambda: term_dones[:, base_contact],
    }
    try:
        env.reset()
        action_dim = env.action_space.shape[-1]
        envs = torch.arange(_NUM_ENVS, dtype=torch.float32)[:, None]
        joints = torch.arange(action_dim, dtype=torch.float32)[None, :]
        # growing amplitudes make environments fall, and reset, at different steps
        amplitude = 1.0 + 3.0 * envs / (_NUM_ENVS - 1)
        for step in range(_STEPS):
            if step < 20:
                action = torch.zeros((_NUM_ENVS, action_dim))
            else:
                action = amplitude * torch.sin(2 * math.pi * step / 25.0 + 0.7 * joints + 0.05 * envs)
            calls["sim_step"] = calls["apply_action"] = 0
            env.step(action.to(env.device))
            rollout["calls"].append((calls["sim_step"], calls["apply_action"]))
            for key, read in records.items():
                rollout.setdefault(key, []).append(read().clone())
    finally:
        env.close()
    for key in records:
        rollout[key] = torch.stack(rollout[key])
    return rollout


@pytest.fixture(scope="module")
def rollouts() -> dict[str, dict]:
    return {frontend: _rollout(frontend) for frontend in ("stable", "warp")}


def test_physics_loop_cadence_matches_stable(rollouts: dict[str, dict]):
    stable, warp = rollouts["stable"], rollouts["warp"]
    assert stable["handles_decimation"], "Go2 runs its actuators inside Newton, so physics must own decimation"
    assert warp["handles_decimation"]
    assert warp["calls"] == stable["calls"] == [(1, 1)] * _STEPS


def test_contact_data_matches_stable(rollouts: dict[str, dict]):
    stable, warp = rollouts["stable"], rollouts["warp"]
    assert stable["base_contact"].any(), "no environment terminated; the resets are not exercised"
    for key in ("net_forces_w_history", "current_air_time", "last_air_time", "undesired_contacts", "base_contact"):
        assert torch.equal(warp[key], stable[key]), key
    # stable stores ``raw * weight * dt / dt`` and reduces the feet with torch, the Warp manager stores
    # ``raw * weight`` and its term sums the feet in a loop: the two differ in the last bit only
    torch.testing.assert_close(warp["feet_air_time"], stable["feet_air_time"], rtol=0.0, atol=1e-6)
