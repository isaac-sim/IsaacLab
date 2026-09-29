# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Operational-space control of a locally authored Newton chain through the articulation's dynamics accessors.

The Isaac Lab :class:`~isaaclab.controllers.OperationalSpaceController` consumes
:attr:`~isaaclab.assets.BaseArticulationData.body_link_jacobian_w`,
:attr:`~isaaclab.assets.BaseArticulationData.mass_matrix`, and
:attr:`~isaaclab.assets.BaseArticulationData.gravity_compensation_forces` every step. These sentinels close the
loop on a six-DOF fixed chain whose links carry center-of-mass offsets, so a wrong Jacobian, mass matrix, gravity
force, or DoF ordering pushes the steady-state error well past the bounds. The chain's joints are unpowered, so the
controller's joint efforts are the only actuation.
"""

from isaaclab_newton.physics import NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg()))

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from isaaclab_newton.assets import Articulation

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.sim import SimulationContext, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.math import compute_pose_error, matrix_from_quat, quat_inv, subtract_frame_transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from articulation_test_utils import NUM_ENVS, local_usd, newton_sim_cfg, spawn_assets, world_gravity  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


@dataclass
class _Chain:
    """A passive six-DOF chain and the end-effector indices the controller reads."""

    sim: SimulationContext
    device: str
    robot: Articulation
    ee_frame_idx: int
    ee_jacobi_idx: int
    arm_joint_ids: list[int]

    def rest(self) -> None:
        """Hold the default configuration at rest with no effort command."""
        default_joint_pos = self.robot.data.default_joint_pos.torch.clone()
        self.robot.write_joint_state_to_sim_index(
            position=default_joint_pos, velocity=torch.zeros_like(default_joint_pos)
        )
        self.robot.actuators.target_command.set_effort_index(value=torch.zeros_like(default_joint_pos))
        self.robot.reset()
        self.robot.write_data_to_sim()
        self.sim.step()
        self.robot.update(self.sim.cfg.dt)


@pytest.fixture(scope="module", params=test_devices(DeviceScope.CPU))
def chain(request: pytest.FixtureRequest) -> Iterator[_Chain]:
    """Build the unpowered chain once, at a configuration away from the joint limits."""
    robot_cfg = ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        spawn=local_usd("fixed_spatial_chain.usda"),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0), joint_pos={"Joint_3": 0.3, "Joint_4": -0.2, "Joint_5": 0.4}
        ),
        actuators={"arm": ImplicitActuatorCfg(joint_names_expr=["Joint_.*"], stiffness=0.0, damping=0.0)},
    )
    device = request.param
    with build_simulation_context(sim_cfg=newton_sim_cfg(device)) as sim:
        robot = spawn_assets({"robot": robot_cfg})["robot"]
        sim.reset()
        assert robot.is_initialized and robot.is_fixed_base
        ee_frame_idx = robot.find_bodies("Link_5")[0][0]
        yield _Chain(
            sim=sim,
            device=device,
            robot=robot,
            ee_frame_idx=ee_frame_idx,
            # the fixed root has no Jacobian row
            ee_jacobi_idx=ee_frame_idx - 1,
            arm_joint_ids=robot.find_joints(["Joint_.*"])[0],
        )


def _compute_ee_pose_root(robot: Articulation, ee_frame_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the end-effector position [m] and quaternion ``(x, y, z, w)`` in the root frame."""
    ee_pose_w = robot.data.body_pose_w.torch[:, ee_frame_idx]
    root_pose_w = robot.data.root_pose_w.torch
    return subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])


def _compute_jacobian_root_frame(robot: Articulation, ee_jacobi_idx: int, arm_joint_ids: list[int]) -> torch.Tensor:
    """Return the end-effector Jacobian sliced to ``arm_joint_ids`` and rotated to the root frame, shape [N, 6, D]."""
    jacobian = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, arm_joint_ids]
    base_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_pose_w.torch[:, 3:7]))
    jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
    return jacobian


def _build_relative_pose_target(
    robot: Articulation, ee_frame_idx: int, delta_xyz: tuple[float, float, float]
) -> torch.Tensor:
    """Return the current end-effector pose in the root frame offset by ``delta_xyz`` [m], keeping its orientation."""
    initial_ee_pos_b, initial_ee_quat_b = _compute_ee_pose_root(robot, ee_frame_idx)
    target_pos_b = initial_ee_pos_b + initial_ee_pos_b.new_tensor(delta_xyz)
    return torch.cat([target_pos_b, initial_ee_quat_b], dim=-1)


def _make_osc(device: str) -> OperationalSpaceController:
    """Return a fixed-impedance absolute-pose controller with inertial decoupling and no gravity compensation."""
    return OperationalSpaceController(
        OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            impedance_mode="fixed",
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=False,
            motion_stiffness_task=500.0,
            motion_damping_ratio_task=1.0,
        ),
        num_envs=NUM_ENVS,
        device=device,
    )


def _run_osc(
    chain: _Chain, osc: OperationalSpaceController, target_pose_b: torch.Tensor, num_steps: int, *, gravity: bool
) -> tuple[list[float], list[float]]:
    """Close the OSC loop for ``num_steps`` steps; return the per-step max position and rotation errors."""
    robot = chain.robot
    pos_history: list[float] = []
    rot_history: list[float] = []
    for _ in range(num_steps):
        jacobian_b = _compute_jacobian_root_frame(robot, chain.ee_jacobi_idx, chain.arm_joint_ids)
        mass_matrix = robot.data.mass_matrix.torch[:, chain.arm_joint_ids, :][:, :, chain.arm_joint_ids]
        gravity_forces = robot.data.gravity_compensation_forces.torch[:, chain.arm_joint_ids] if gravity else None
        ee_pos_b, ee_quat_b = _compute_ee_pose_root(robot, chain.ee_frame_idx)
        ee_pose_b = torch.cat([ee_pos_b, ee_quat_b], dim=-1)
        # OSC's damping term ``kd * ee_vel_b`` needs the end-effector velocity ``J · q_dot``; a zero velocity
        # leaves the impedance undamped.
        joint_vel = robot.data.joint_vel.torch[:, chain.arm_joint_ids]
        ee_vel_b = torch.bmm(jacobian_b, joint_vel.unsqueeze(-1)).squeeze(-1)

        osc.set_command(target_pose_b, current_ee_pose_b=ee_pose_b)
        joint_efforts = osc.compute(
            jacobian_b=jacobian_b,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=ee_vel_b,
            mass_matrix=mass_matrix,
            gravity=gravity_forces,
        )
        robot.actuators.target_command.set_effort_index(value=joint_efforts, joint_ids=chain.arm_joint_ids)
        robot.write_data_to_sim()
        chain.sim.step()
        robot.update(chain.sim.cfg.dt)

        pos_error, rot_error = compute_pose_error(ee_pos_b, ee_quat_b, target_pose_b[:, 0:3], target_pose_b[:, 3:7])
        pos_history.append(pos_error.norm(dim=-1).max().item())
        rot_history.append(rot_error.norm(dim=-1).max().item())
    return pos_history, rot_history


@pytest.mark.isaacsim_ci
def test_osc_tracking_accuracy(chain: _Chain) -> None:
    """OSC pose tracking sentinel for the Jacobian and mass-matrix bridge.

    OSC runs with ``gravity_compensation=False`` and scene gravity disabled so the sentinel isolates the J/M
    bridge; the gravity-compensation path is covered by :func:`test_osc_gravity_compensation_precision`.
    ``inertial_dynamics_decoupling=True`` exercises ``mass_matrix`` and the COM-referenced J → M_b → J product.
    """
    chain.rest()
    target_pose_b = _build_relative_pose_target(chain.robot, chain.ee_frame_idx, (0.05, 0.0, 0.0))
    pos_history, rot_history = _run_osc(chain, _make_osc(chain.device), target_pose_b, 300, gravity=False)

    pos_mean = sum(pos_history[-200:]) / 200
    rot_mean = sum(rot_history[-200:]) / 200

    # Regression sentinel: assert on tail mean rather than min. With ``current_ee_vel_b = J · q_dot`` providing
    # OSC's damping term and no joint PD, the impedance settles to machine precision. A wrong J, wrong mass
    # matrix, or DoF mis-ordering pushes the steady-state error well past the 5 mm bound because OSC consumes
    # both ``body_link_jacobian_w`` and ``mass_matrix`` per step.
    assert pos_mean < 5e-3, f"OSC pos_mean {pos_mean:.5f} > 5 mm — bridge regression?"
    assert rot_mean < 5e-2, f"OSC rot_mean {rot_mean:.5f} > 0.05 rad — bridge regression?"


@pytest.mark.isaacsim_ci
def test_osc_gravity_compensation_precision(chain: _Chain) -> None:
    """Two-phase EE hold: gravity sag without compensation, tight hold with it.

    Same OSC pose-hold loop as :func:`test_osc_tracking_accuracy`, but with gravity on and the target pinned
    to the initial EE pose, so any steady-state error is pure gravity sag. Phase 1 runs with
    ``gravity_compensation=False`` and must sag past a floor; phase 2 flips ``osc.cfg.gravity_compensation``
    — read per :meth:`compute` call, so the flag is the only variable across phases (the gravity tensor is
    fetched and passed in both) — and must recover the hold to under 0.1 mm.

    The floor assertion keeps the test discriminating: if the task stiffness is ever raised high enough to
    mask gravity, phase 1 stops clearing the floor and the test fails loudly instead of silently passing on a
    non-discriminating setup. The gravity feed-forward consumes ``gravity_compensation_forces`` (Newton RNEA via
    ``eval_inverse_dynamics_passive``) live in the loop, covering the FK-staleness refresh on every step of
    phase 2. Both phases reach a true steady state, enforced by tail-half stationarity guards.
    """
    chain.rest()
    osc = _make_osc(chain.device)
    # Hold the initial EE pose: phase-1 steady-state error is pure gravity sag.
    target_pose_b = _build_relative_pose_target(chain.robot, chain.ee_frame_idx, (0.0, 0.0, 0.0))

    def _stationary_tail_mean(history: list[float], label: str) -> float:
        """Mean of the last 200 samples, asserting the two tail halves agree within 25%.

        The relative check carries a 10 µm absolute floor: at the solver noise floor of the compensated hold,
        tail jitter is far below the 0.1 mm verdict threshold and cannot flip the outcome.
        """
        a = sum(history[-200:-100]) / 100
        b = sum(history[-100:]) / 100
        mean = (a + b) / 2.0
        assert abs(a - b) < 0.25 * max(mean, 1e-5), (
            f"{label} not stationary: tail halves {a:.6f} vs {b:.6f} — extend the phase"
        )
        return mean

    with world_gravity((0.0, 0.0, -9.81)):
        hist_off, _ = _run_osc(chain, osc, target_pose_b, 300, gravity=True)
        osc.cfg.gravity_compensation = True
        hist_on, _ = _run_osc(chain, osc, target_pose_b, 300, gravity=True)

    pos_off = _stationary_tail_mean(hist_off, "phase-1 sag")
    pos_on = _stationary_tail_mean(hist_on, "phase-2 hold")

    assert pos_off > 1.2e-2, f"uncompensated sag {pos_off:.5f} < 1.2 cm — setup no longer discriminates gravity"
    assert pos_on < 1e-4, f"compensated hold {pos_on:.6f} > 0.1 mm — gravity compensation inaccurate"
    assert pos_on < pos_off / 10.0, f"compensation only improved sag {pos_off:.5f} -> {pos_on:.6f} (<10x)"
