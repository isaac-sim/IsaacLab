# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for RMPFlow task-space control stability after a reset.

Under a zero *relative* command, an RMPFlow-controlled arm must hold the pose it was reset to:
the commanded end-effector target equals the current end-effector pose, so the arm should settle
and stay put. A quaternion-convention bug in the controller used to scramble the orientation
target, making the arm diverge by many radians from its reset configuration over a handful of
steps (e.g. ~12 rad of total joint drift on the Agibot right arm). These tests pin the corrected
behavior: the arm settles quickly and stays close to its reset pose.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import pytest
import torch

import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Number of environment steps to settle under a zero relative command.
_SETTLE_STEPS = 60
# Window (last steps) over which a settled arm must be effectively motionless.
_SETTLE_WINDOW = 15
# Max total |Δjoint| (rad) allowed across the arm over the settle window: if the arm is still
# moving this much near the end, it has not converged (the bug caused continuous divergence).
_SETTLE_TOL = 0.3

# Per-task budget for total |Δjoint| (rad) drift away from the reset pose after settling.
# Values sit comfortably above the corrected behavior yet far below the multi-radian divergence
# the orientation-convention bug produced (~12 rad on the Agibot right arm). The wide separation
# between corrected drift and the bug's divergence leaves room to tighten these once the corrected
# per-task drift is characterized more precisely. The suction task carries a larger null-space
# residual, hence its higher budget.
_RMPFLOW_TASKS = [
    ("Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0", 1.5),
    ("Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0", 1.5),
    ("Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-RmpFlow-v0", 1.5),
    ("Isaac-Stack-Cube-Galbot-Right-Arm-Suction-RmpFlow-v0", 2.5),
]


@pytest.mark.parametrize("task_name, drift_tol", _RMPFLOW_TASKS)
def test_rmpflow_reset_pose_is_stable(task_name: str, drift_tol: float):
    """The RMPFlow arm settles and holds its reset pose under a zero relative command."""
    # RMPFlow tasks (incl. the suction gripper) run on CPU physics.
    device = "cpu"
    sim_utils.create_new_stage()
    try:
        env_cfg = parse_env_cfg(task_name, device=device, num_envs=1)
        env = gym.make(task_name, cfg=env_cfg)
    except Exception as e:  # noqa: BLE001
        if "env" in locals() and hasattr(env, "_is_closed"):
            env.close()
        pytest.fail(f"Failed to set up the environment for task {task_name}. Error: {e}")

    # disable control on stop
    env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

    try:
        robot = env.unwrapped.scene["robot"]
        # Joints actively controlled by the RMPFlow arm action term. The action term has no public
        # accessor for its resolved joint indices, so read the private attribute directly; switch to
        # a public property here if one is ever added.
        arm_joint_ids = env.unwrapped.action_manager.get_term("arm_action")._joint_ids
        if isinstance(arm_joint_ids, slice):
            arm_joint_ids = list(range(robot.num_joints))

        env.reset()
        q_reset = robot.data.joint_pos.torch[0, arm_joint_ids].clone()

        # Zero action => zero relative delta => "hold current pose" command.
        zero_action = torch.zeros((env.unwrapped.num_envs, env.action_space.shape[-1]), device=device)
        q_window_start = None
        with torch.inference_mode():
            for step in range(_SETTLE_STEPS):
                env.step(zero_action)
                if step == _SETTLE_STEPS - _SETTLE_WINDOW - 1:
                    q_window_start = robot.data.joint_pos.torch[0, arm_joint_ids].clone()
        q_final = robot.data.joint_pos.torch[0, arm_joint_ids].clone()

        # 1) The arm has settled: it is effectively motionless over the final window.
        settle_motion = float((q_final - q_window_start).abs().sum())
        assert settle_motion < _SETTLE_TOL, (
            f"{task_name}: arm still moving {settle_motion:.3f} rad over the last {_SETTLE_WINDOW} steps"
            f" (tol {_SETTLE_TOL}); RMPFlow did not converge to a steady hold."
        )

        # 2) The settled pose stays close to the reset pose (no large drift).
        total_drift = float((q_final - q_reset).abs().sum())
        assert total_drift < drift_tol, (
            f"{task_name}: arm drifted {total_drift:.3f} rad from its reset pose under a zero command"
            f" (tol {drift_tol}); expected it to hold."
        )
    finally:
        env.close()
