# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kamino adapters for simulation parameter-validation fixtures."""

from isaaclab.app import AppLauncher

# Launch Omniverse before importing simulator-dependent modules.
simulation_app = AppLauncher(headless=True).app

import pytest
import torch
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg

from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.physics.parameter_validation.fixtures import (
    ACTIVE_LOWER,
    ACTIVE_UPPER,
    INACTIVE_LOWER,
    INACTIVE_UPPER,
    LIMIT_LOWER,
    PROBE_TARGET,
    build_single_dof,
    make_single_dof_cfg,
)
from isaaclab.test.physics.parameter_validation.oracles import PROFILE_DOF_DT, PROFILE_FREE_DT

DEVICE = "cuda:0"
KAMINO_RTOL = 5.0e-3
KAMINO_ATOL = 2.0e-4
BASE_STIFFNESS = 100.0
BASE_DAMPING = 10.0


class KaminoParameterAdapter:
    """Backend adapter for the pinned Kamino validation profiles."""

    def profile_dof_cfg(self, *, alpha: float = 0.0, beta: float = 0.0) -> SimulationCfg:
        """Create the pinned Kamino single-DOF simulation profile."""
        return self._sim_cfg(
            dt=PROFILE_DOF_DT,
            gravity=(0.0, 0.0, 0.0),
            alpha=alpha,
            beta=beta,
        )

    def profile_free_cfg(
        self,
        *,
        gravity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> SimulationCfg:
        """Create the pinned Kamino collision-free body profile."""
        return self._sim_cfg(dt=PROFILE_FREE_DT, gravity=gravity, alpha=0.0, beta=0.0)

    def run_single_dof_step(
        self,
        joint_type: str,
        authoring: str,
        *,
        stiffness: float,
        damping: float,
        armature: float,
        position_target: float,
        velocity_target: float = 0.0,
        effort: float = 0.0,
        position: float = 0.0,
        velocity: float = 0.0,
        passive_damping: float = 0.0,
    ) -> dict[str, float]:
        """Author a parameter, run one Kamino DOF step, and return observed state."""
        spec = self._dof_authoring_spec(
            authoring,
            stiffness=stiffness,
            damping=damping,
            armature=armature,
        )
        usd_stiffness, usd_damping, usd_armature = spec["usd"]
        cfg_stiffness, cfg_damping, cfg_armature = spec["cfg"]

        with build_simulation_context(device=DEVICE, sim_cfg=self.profile_dof_cfg()) as sim:
            sim._app_control_on_stop_handle = None
            build_single_dof(
                joint_type,
                usd_stiffness=usd_stiffness,
                usd_drive_damping=usd_damping,
                usd_armature=usd_armature,
                usd_passive_damping=passive_damping,
            )
            articulation = Articulation(make_single_dof_cfg(cfg_stiffness, cfg_damping, cfg_armature))
            sim.reset()
            articulation.update(0.0)
            body_inertia = float(articulation.data.mass_matrix.torch[0, 0, 0])

            if spec["runtime"] is not None:
                runtime_stiffness, runtime_damping, runtime_armature = spec["runtime"]
                articulation.write_joint_stiffness_to_sim_index(stiffness=runtime_stiffness)
                articulation.write_joint_damping_to_sim_index(damping=runtime_damping)
                articulation.write_joint_armature_to_sim_index(armature=runtime_armature)

            articulation.write_joint_position_to_sim_index(position=torch.full((1, 1), position, device=DEVICE))
            articulation.write_joint_velocity_to_sim_index(velocity=torch.full((1, 1), velocity, device=DEVICE))
            articulation.set_joint_position_target_index(target=torch.full((1, 1), position_target, device=DEVICE))
            articulation.set_joint_velocity_target_index(target=torch.full((1, 1), velocity_target, device=DEVICE))
            articulation.set_joint_effort_target_index(target=torch.full((1, 1), effort, device=DEVICE))
            articulation.update(0.0)
            position_before = float(articulation.data.joint_pos.torch[0, 0])
            velocity_before = float(articulation.data.joint_vel.torch[0, 0])

            articulation.write_data_to_sim()
            sim.step()
            articulation.update(PROFILE_DOF_DT)
            return {
                "body_inertia": body_inertia,
                "position_before": position_before,
                "velocity_before": velocity_before,
                "position_after": float(articulation.data.joint_pos.torch[0, 0]),
                "velocity_after": float(articulation.data.joint_vel.torch[0, 0]),
            }

    def run_position_limit_probe(
        self, joint_type: str, authoring: str, limit: str, bound: float
    ) -> tuple[float, float]:
        """Drive toward one limit and return the extreme and final position."""
        spec = self._position_limit_authoring_spec(authoring, limit, bound)
        target = PROBE_TARGET if limit == "upper" else -PROBE_TARGET
        with build_simulation_context(
            device=DEVICE,
            sim_cfg=self.profile_dof_cfg(alpha=0.01, beta=0.01),
        ) as sim:
            sim._app_control_on_stop_handle = None
            build_single_dof(
                joint_type,
                usd_stiffness=30.0,
                usd_lower=spec["usd_lower"],
                usd_upper=spec["usd_upper"],
            )
            articulation = Articulation(make_single_dof_cfg(30.0, 60.0, None))
            sim.reset()
            if spec["runtime_limit"] is not None:
                lower, runtime_upper = spec["runtime_limit"]
                articulation.write_joint_position_limit_to_sim_index(
                    limits=torch.tensor([[[lower, runtime_upper]]], device=DEVICE)
                )
            articulation.set_joint_position_target_index(target=torch.full((1, 1), target, device=DEVICE))
            position_extreme = -1.0e9 if limit == "upper" else 1.0e9
            for _ in range(600):
                articulation.write_data_to_sim()
                sim.step()
                articulation.update(PROFILE_DOF_DT)
                position = float(articulation.data.joint_pos.torch[0, 0])
                if limit == "upper":
                    position_extreme = max(position_extreme, position)
                else:
                    position_extreme = min(position_extreme, position)
            return position_extreme, float(articulation.data.joint_pos.torch[0, 0])

    def run_joint_state(
        self,
        joint_type: str,
        authoring: str,
        *,
        position: float,
        velocity: float,
    ) -> dict[str, float]:
        """Restore or write joint state, then observe one unforced Kamino step."""
        cfg_position = position if authoring == "cfg" else 0.0
        cfg_velocity = velocity if authoring == "cfg" else 0.0
        with build_simulation_context(device=DEVICE, sim_cfg=self.profile_dof_cfg()) as sim:
            sim._app_control_on_stop_handle = None
            build_single_dof(joint_type, usd_stiffness=0.0)
            articulation = Articulation(
                make_single_dof_cfg(
                    0.0,
                    0.0,
                    0.0,
                    joint_position=cfg_position,
                    joint_velocity=cfg_velocity,
                )
            )
            sim.reset()
            articulation.update(0.0)

            disturbance = torch.full((1, 1), -0.1, device=DEVICE)
            articulation.write_joint_position_to_sim_index(position=disturbance)
            articulation.write_joint_velocity_to_sim_index(velocity=disturbance)
            if authoring == "cfg":
                articulation.write_joint_position_to_sim_index(position=articulation.data.default_joint_pos.torch)
                articulation.write_joint_velocity_to_sim_index(velocity=articulation.data.default_joint_vel.torch)
            elif authoring == "runtime":
                articulation.write_joint_position_to_sim_index(position=torch.full((1, 1), position, device=DEVICE))
                articulation.write_joint_velocity_to_sim_index(velocity=torch.full((1, 1), velocity, device=DEVICE))
            else:
                raise ValueError(f"Unknown joint-state authoring path: {authoring}")

            articulation.update(0.0)
            result = {
                "position_before": float(articulation.data.joint_pos.torch[0, 0]),
                "velocity_before": float(articulation.data.joint_vel.torch[0, 0]),
            }
            sim.step()
            articulation.update(PROFILE_DOF_DT)
            result["position_after"] = float(articulation.data.joint_pos.torch[0, 0])
            result["velocity_after"] = float(articulation.data.joint_vel.torch[0, 0])
            return result

    def run_com_gravity_probe(self, center_of_mass: tuple[float, float, float]) -> tuple[float, float]:
        """Step a fixed-pivot body under gravity and return velocity and effective inertia."""
        with build_simulation_context(
            device=DEVICE,
            sim_cfg=self._sim_cfg(
                dt=PROFILE_DOF_DT,
                gravity=(0.0, -9.81, 0.0),
                alpha=0.0,
                beta=0.0,
            ),
        ) as sim:
            sim._app_control_on_stop_handle = None
            build_single_dof(
                "revolute",
                usd_stiffness=0.0,
                center_of_mass=center_of_mass,
            )
            articulation = Articulation(make_single_dof_cfg(0.0, 0.0, 0.0))
            sim.reset()
            articulation.update(0.0)
            body_inertia = float(articulation.data.mass_matrix.torch[0, 0, 0])
            sim.step()
            articulation.update(PROFILE_DOF_DT)
            return float(articulation.data.joint_vel.torch[0, 0]), body_inertia

    @staticmethod
    def _sim_cfg(
        *,
        dt: float,
        gravity: tuple[float, float, float],
        alpha: float,
        beta: float,
    ) -> SimulationCfg:
        return SimulationCfg(
            dt=dt,
            gravity=gravity,
            device=DEVICE,
            physics=NewtonCfg(
                solver_cfg=KaminoSolverCfg(
                    integrator="euler",
                    constraints_alpha=alpha,
                    constraints_beta=beta,
                ),
                num_substeps=1,
                use_cuda_graph=False,
            ),
        )

    @staticmethod
    def _dof_authoring_spec(
        authoring: str,
        *,
        stiffness: float,
        damping: float,
        armature: float,
    ) -> dict:
        # Non-zero USD gains establish Kamino target mode until IsaacLab#6649 is resolved.
        usd_stiffness = BASE_STIFFNESS if stiffness > 0.0 else 0.0
        usd_damping = BASE_DAMPING if damping > 0.0 else 0.0
        if authoring == "usd":
            return {
                "usd": (stiffness, damping, armature),
                "cfg": (None, None, None),
                "runtime": None,
            }
        if authoring == "cfg":
            return {
                "usd": (usd_stiffness, usd_damping, 0.0),
                "cfg": (stiffness, damping, armature),
                "runtime": None,
            }
        if authoring == "runtime":
            return {
                "usd": (usd_stiffness, usd_damping, armature / 10.0),
                "cfg": (None, None, None),
                "runtime": (stiffness, damping, armature),
            }
        if authoring == "runtime-error":
            if stiffness == 0.0 and damping == 0.0 and armature == 0.0:
                return {
                    "usd": (BASE_STIFFNESS, 0.0, 0.0),
                    "cfg": (None, None, None),
                    "runtime": (0.0, 0.0, 0.0),
                }
            return {
                "usd": (0.0, 0.0, 0.0),
                "cfg": (None, None, None),
                "runtime": (stiffness, damping, armature),
            }
        raise ValueError(f"Unknown authoring path: {authoring}")

    @staticmethod
    def _position_limit_authoring_spec(authoring: str, limit: str, bound: float) -> dict:
        if limit not in {"lower", "upper"}:
            raise ValueError(f"Unsupported position-limit direction: {limit}")
        inactive_bound = INACTIVE_UPPER if limit == "upper" else INACTIVE_LOWER
        opposite_bound = INACTIVE_LOWER if limit == "upper" else INACTIVE_UPPER
        usd_lower = opposite_bound if limit == "upper" else bound
        usd_upper = bound if limit == "upper" else opposite_bound
        if authoring == "usd":
            return {
                "usd_lower": usd_lower,
                "usd_upper": usd_upper,
                "runtime_limit": None,
            }
        if authoring == "runtime":
            initial_bound = (
                inactive_bound
                if abs(bound) < abs(PROBE_TARGET)
                else (ACTIVE_UPPER if limit == "upper" else ACTIVE_LOWER)
            )
            initial_lower = opposite_bound if limit == "upper" else initial_bound
            initial_upper = initial_bound if limit == "upper" else opposite_bound
            runtime_lower = opposite_bound if limit == "upper" else bound
            runtime_upper = bound if limit == "upper" else opposite_bound
            return {
                "usd_lower": initial_lower,
                "usd_upper": initial_upper,
                "runtime_limit": (runtime_lower, runtime_upper),
            }
        if authoring == "runtime-error":
            return {
                "usd_lower": None,
                "usd_upper": None,
                "runtime_limit": (
                    bound if limit == "lower" else LIMIT_LOWER,
                    bound if limit == "upper" else INACTIVE_UPPER,
                ),
            }
        raise ValueError(f"Unknown authoring path: {authoring}")


@pytest.fixture
def kamino() -> KaminoParameterAdapter:
    """Provide the pinned Kamino parameter-validation adapter."""
    if not torch.cuda.is_available():
        pytest.skip("Kamino solver tests require a CUDA device")
    return KaminoParameterAdapter()
