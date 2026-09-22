# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A scripted cube-stacking policy, enough to make a rollout show real motion.

This is not a learned policy and is not meant to become one. It exists so that
visual DR can be exercised against a moving scene -- a static arm hides exactly
the problems worth seeing, such as the preserved mask lagging the robot or the
background flickering between frames.

It drives the end effector with a proportional controller through a fixed phase
sequence, per environment, using privileged state straight from the scene.
"""

from __future__ import annotations

import torch

# Phases, in order. Each environment advances independently.
APPROACH, DESCEND, GRASP, LIFT, TRAVEL, PLACE, RELEASE, RETREAT = range(8)

_OPEN = 1.0
_CLOSED = -1.0


class ScriptedStackPolicy:
    """Pick up one cube and stack it on another.

    Args:
        env: The environment to drive.
        upper_cube: Name of the cube to pick up.
        lower_cube: Name of the cube to stack it on.
        step_limit: Largest end-effector move per step [m]. Keeps the differential
            IK solver inside the range where its linearization holds.
        gain: Proportional gain on end-effector position error.
    """

    def __init__(
        self,
        env,
        upper_cube: str = "cube_2",
        lower_cube: str = "cube_1",
        step_limit: float = 0.02,
        gain: float = 3.0,
    ):
        self.env = env
        self.upper_cube = upper_cube
        self.lower_cube = lower_cube
        self.step_limit = step_limit
        self.gain = gain
        # The arm term consumes a relative pose, so a command of ``d`` moves the
        # end effector by ``d * scale``; invert that to speak in metres.
        self.action_scale = float(getattr(env.action_manager.get_term("arm_action").cfg, "scale", 1.0))

        self.num_envs = env.num_envs
        self.device = env.device
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Where the gripper was when it closed. Lifting has to aim at a fixed point:
        # once the cube is held it travels with the gripper, so a target defined
        # relative to the cube recedes as fast as the arm rises and never arrives.
        self.grasp_anchor = torch.zeros(self.num_envs, 3, device=self.device)
        self.action_dim = env.action_manager.total_action_dim

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Send the named environments back to the first phase."""
        if env_ids is None:
            self.phase.zero_()
            self.dwell.zero_()
            self.grasp_anchor.zero_()
        else:
            self.phase[env_ids] = 0
            self.dwell[env_ids] = 0
            self.grasp_anchor[env_ids] = 0.0

    def _positions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ee = self.env.scene["ee_frame"].data.target_pos_w.torch[:, 0, :]
        upper = self.env.scene[self.upper_cube].data.root_pos_w.torch
        lower = self.env.scene[self.lower_cube].data.root_pos_w.torch
        return ee, upper, lower

    def act(self) -> torch.Tensor:
        """Return one action for every environment, advancing the phase machine."""
        ee, upper, lower = self._positions()
        # Environments reset by the simulator restart the sequence; otherwise the
        # policy would keep reaching for a cube that has already moved away.
        self.reset((self.env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1))

        target = torch.zeros_like(ee)
        gripper = torch.full((self.num_envs,), _OPEN, device=self.device)

        above_upper = upper + torch.tensor([0.0, 0.0, 0.12], device=self.device)
        at_upper = upper + torch.tensor([0.0, 0.0, 0.01], device=self.device)
        lifted = self.grasp_anchor + torch.tensor([0.0, 0.0, 0.20], device=self.device)
        above_lower = lower + torch.tensor([0.0, 0.0, 0.20], device=self.device)
        # Cubes are stacked centre over centre, one cube height apart.
        on_lower = lower + torch.tensor([0.0, 0.0, 0.055], device=self.device)

        for phase, goal, grip in (
            (APPROACH, above_upper, _OPEN),
            (DESCEND, at_upper, _OPEN),
            (GRASP, at_upper, _CLOSED),
            (LIFT, lifted, _CLOSED),
            (TRAVEL, above_lower, _CLOSED),
            (PLACE, on_lower, _CLOSED),
            (RELEASE, on_lower, _OPEN),
            (RETREAT, above_lower, _OPEN),
        ):
            mask = self.phase == phase
            target[mask] = goal[mask]
            gripper[mask] = grip

        error = target - ee
        distance = torch.linalg.vector_norm(error, dim=1)

        # Closing and opening the fingers takes time, so those phases hold still
        # and wait rather than advancing the moment the arm is in position.
        holding = (self.phase == GRASP) | (self.phase == RELEASE)
        self.dwell = torch.where(holding, self.dwell + 1, torch.zeros_like(self.dwell))
        arrived = torch.where(holding, self.dwell >= 8, distance < 0.015)
        entering_lift = (self.phase == GRASP) & arrived
        self.grasp_anchor[entering_lift] = ee[entering_lift]
        self.phase = torch.clamp(self.phase + arrived.long(), max=RETREAT)

        delta = torch.clamp(error * self.gain, -self.step_limit, self.step_limit)
        actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        actions[:, :3] = delta / self.action_scale
        # Leave orientation alone: the reset pose already points the gripper down,
        # and commanding rotation here only fights the IK solver.
        actions[:, -1] = gripper
        return actions
