from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from . import mdp


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    adr = CurrTerm(
        func=mdp.DifficultyScheduler,
        params={"init_difficulty": 0, "min_difficulty": 0, "max_difficulty": 10}
    )
    
    joint_pos_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.joint_pos.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 0., "fv": -.1, "difficulty_term_str": "adr"}
        }
    )

    joint_pos_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.joint_pos.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 0., "fv": .1, "difficulty_term_str": "adr"}
        }
    )

    joint_vel_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.joint_vel.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 0., "fv": -.2, "difficulty_term_str": "adr"}
        }
    )

    joint_vel_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.joint_vel.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 0., "fv": .2, "difficulty_term_str": "adr"}
        }
    )

    hand_tips_pos_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.hand_tips_state_b.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 0., "fv": -.01, "difficulty_term_str": "adr"}
        }
    )

    hand_tips_pos_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.hand_tips_state_b.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 0., "fv": .01, "difficulty_term_str": "adr"}
        }
    )

    object_pose_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.object_pose_b.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 0., "fv": -.03, "difficulty_term_str": "adr"}
        }
    )

    object_pose_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.object_pose_b.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 0., "fv": .03, "difficulty_term_str": "adr"}
        }
    )
    
    object_obs_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.object_observation_b.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 0., "fv": -.01, "difficulty_term_str": "adr"}
        }
    )
    
    object_obs_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.object_observation_b.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 0., "fv": -.01, "difficulty_term_str": "adr"}
        }
    )

    joint_stiffness_scale_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.joint_stiffness_and_damping.params.stiffness_distribution_params",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": (1., 1.), "fv": (.5, 2.), "difficulty_term_str": "adr"}
        }
    )

    joint_damping_scale_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.joint_stiffness_and_damping.params.damping_distribution_params",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": (1., 1.), "fv": (.5, 2.), "difficulty_term_str": "adr"}
        }
    )

    object_mass_scale_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.object_scale_mass.params.mass_distribution_params",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": (1., 1.), "fv": (0.2, 2.0), "difficulty_term_str": "adr"}
        }
    )

    joint_friction_scale_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.joint_friction.params.friction_distribution_params",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": (0., 0.), "fv": (0., 5.), "difficulty_term_str": "adr"}
        }
    )
    
    gravity_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.variable_gravity.params.gravity_distribution_params",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {
                "iv": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                "fv": ((0.0, 0.0, -9.81), (0.0, 0.0, -9.81)),
                "difficulty_term_str": "adr"
            }
        }
    )
