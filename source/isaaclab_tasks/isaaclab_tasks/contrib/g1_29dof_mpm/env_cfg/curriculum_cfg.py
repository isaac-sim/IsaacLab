# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils.configclass import configclass

from .. import mdp as g1_mdp


@configclass
class G1CurriculumCfg:
    """Curriculum terms for the MDP."""

    """
    walking
    """
    # command_vel = CurrTerm(
    #     func=mdp.commands_vel,
    #     params={
    #         "command_name": "base_velocity",
    #         "velocity_stages": [
    #             {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
    #             {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "ang_vel_z": (-0.7, 0.7)},
    #             {"step": 10000 * 24, "lin_vel_x": (-1.0, 2.5), "ang_vel_z": (-1.0, 1.0)},
    #         ],
    #     },
    # )

    # track_lin_vel = CurrTerm(
    #     func=mdp.modify_reward_std,
    #     params={"term_name": "track_lin_vel_xy", "std": 0.25, "num_steps": 10000 * 24},
    # )

    # track_heading = CurrTerm(
    #     func=mdp.modify_reward_std,
    #     params={"term_name": "track_heading", "std": 0.25, "num_steps": 10000 * 24},
    # )

    # # track_ang_vel = CurrTerm(
    # #     func=mdp.modify_reward_std,
    # #     params={"term_name": "track_ang_vel_z", "std": 0.25, "num_steps": 15000 * 24}
    # #     # params={"term_name": "track_ang_vel_z", "std": 0.25, "num_steps": 7000 * 24}
    # # )

    """
    running
    """
    command_vel = CurrTerm(
        func=g1_mdp.commands_vel,  # type: ignore
        params={
            "command_name": "base_velocity",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                {"step": 10000 * 24, "lin_vel_x": (-1.0, 1.7), "ang_vel_z": (-0.7, 0.7)},
                {"step": 15000 * 24, "lin_vel_x": (-1.0, 2.5), "ang_vel_z": (-1.0, 1.0)},
            ],
        },
    )

    # gaussian std curriculum
    # track_lin_vel = CurrTerm(
    #     func=g1_mdp.modify_reward_std,
    #     # params={"term_name": "track_lin_vel_xy", "std": 0.25, "num_steps": 15000 * 24},
    #     # params={"term_name": "track_lin_vel_xy", "std": 0.25, "num_steps": 10000 * 24},
    #     params={"term_name": "track_lin_vel_xy", "std": 0.25, "num_steps": 20000 * 24},
    #     # params={"term_name": "track_lin_vel_xy", "std": 0.25, "num_steps": 50_000}, # fastSAC
    # )

    # track_ang_vel = CurrTerm(
    #     func=g1_mdp.modify_reward_std,
    #     # params={"term_name": "track_ang_vel_z", "std": 0.25, "num_steps": 15000 * 24},
    #     # params={"term_name": "track_ang_vel_z", "std": 0.25, "num_steps": 10000 * 24},
    #     params={"term_name": "track_ang_vel_z", "std": 0.25, "num_steps": 20000 * 24},
    #     # params={"term_name": "track_ang_vel_z", "std": 0.25, "num_steps": 50_000}, # fastSAC
    # )

    # track_heading = CurrTerm(
    #     func=g1_mdp.modify_reward_std,
    #     # params={"term_name": "track_heading", "std": 0.25, "num_steps": 15000 * 24},
    #     # params={"term_name": "track_heading", "std": 0.25, "num_steps": 10000 * 24},
    #     params={"term_name": "track_heading", "std": 0.25, "num_steps": 20000 * 24},
    #     # params={"term_name": "track_heading", "std": 0.25, "num_steps": 50_000}, # fastSAC
    # )

    track_lin_vel = CurrTerm(
        func=g1_mdp.ramp_reward_param,
        params={
            "term_name": "track_lin_vel_xy",
            "param_name": "std",
            "val_0": math.sqrt(1.0),
            "val_1": math.sqrt(0.1),
            "step_0": 0,
            "step_1": 15000 * 24,
        },
    )

    track_ang_vel = CurrTerm(
        func=g1_mdp.ramp_reward_param,
        params={
            "term_name": "track_ang_vel_z",
            "param_name": "std",
            "val_0": math.sqrt(1.0),
            "val_1": math.sqrt(0.1),
            "step_0": 0,
            "step_1": 15000 * 24,
        },
    )

    track_heading = CurrTerm(
        func=g1_mdp.ramp_reward_param,
        params={
            "term_name": "track_heading",
            "param_name": "std",
            "val_0": math.sqrt(1.0),
            "val_1": math.sqrt(0.1),
            "step_0": 0,
            "step_1": 15000 * 24,
        },
    )

    # weight curriculum
    track_lin_vel_weight = CurrTerm(
        func=g1_mdp.ramp_reward_weight,
        params={
            "term_name": "track_lin_vel_xy",
            "weight_0": 4.0,
            "weight_1": 8.0,
            "step_0": 0,
            "step_1": 15000 * 24,
        },
    )

    # track_ang_vel_weight = CurrTerm(
    #     func=g1_mdp.ramp_reward_weight,
    #     params={
    #         "term_name": "track_ang_vel_z",
    #         "weight_0": 1.0,
    #         "weight_1": 4.0,
    #         "step_0": 8000 * 24,
    #         "step_1": 15000 * 24,
    #     }
    # )

    # track_heading_weight = CurrTerm(
    #     func=g1_mdp.ramp_reward_weight,
    #     params={
    #         "term_name": "track_heading",
    #         "weight_0": 4.0,
    #         "weight_1": 8.0,
    #         "step_0": 0,
    #         "step_1": 15000 * 24,
    #     }
    # )

    # feet_air_time_weight = CurrTerm(
    #     func=g1_mdp.ramp_reward_weight,
    #     params={
    #         "term_name": "feet_air_time",
    #         "weight_0": 0.5,
    #         "weight_1": 4.0,
    #         "step_0": 0,
    #         "step_1": 15000 * 24,
    #     }
    # )

    # foot_clearance_weight = CurrTerm(
    #     func=g1_mdp.ramp_reward_weight,
    #     params={
    #         "term_name": "foot_clearance",
    #         "weight_0": 5.0,
    #         "weight_1": 10.0,
    #         "step_0": 0,
    #         "step_1": 15000 * 24,
    #     }
    # )
