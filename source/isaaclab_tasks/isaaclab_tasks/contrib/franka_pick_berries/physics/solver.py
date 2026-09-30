# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local robot gravity compensation for the native MuJoCo IK controller."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonManager
from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager

from isaaclab.utils.configclass import configclass


@configclass
class BerrySolverCfg(MJWarpSolverCfg):
    impratio: float = 10.0


class NewtonBerryManager(NewtonMJWarpManager):
    @classmethod
    def _initialize_contacts(cls):
        if cls._solver.use_mujoco_cpu:
            if cls._report_contacts:
                raise ValueError("The CPU berry task does not export Newton contact sensors")
            NewtonManager._contacts = None
        else:
            super()._initialize_contacts()

    @classmethod
    def _create_solver(cls, model, solver_cfg):
        # PhysX disable_gravity does not author MuJoCo's gravcomp field in this
        # checkout. Compensate the arm explicitly; the berry retains gravity.
        compensation = model.mujoco.gravcomp.numpy()
        for i, label in enumerate(model.body_label):
            if "/Robot/" in label:
                compensation[i] = 1.0
        model.mujoco.gravcomp.assign(compensation)
        return super()._create_solver(model, solver_cfg)
