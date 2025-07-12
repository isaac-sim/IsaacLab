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
            "address": "observations.policy.hand_tips_pos.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 0., "fv": -.01, "difficulty_term_str": "adr"}
        }
    )

    hand_tips_pos_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.hand_tips_pos.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 0., "fv": .01, "difficulty_term_str": "adr"}
        }
    )

    object_pose_unoise_min_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.object_pose.noise.n_min",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 0., "fv": -.03, "difficulty_term_str": "adr"}
        }
    )

    object_pose_unoise_max_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "observations.policy.object_pose.noise.n_max",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 0., "fv": .03, "difficulty_term_str": "adr"}
        }
    )

    # joint_stiffness_scale_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.joint_stiffness_and_damping.params.stiffness_distribution_params",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {"iv": (1., 1.), "fv": (.5, 2.), "difficulty_term_str": "adr"}
    #     }
    # )

    # joint_damping_scale_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.joint_stiffness_and_damping.params.damping_distribution_params",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {"iv": (1., 1.), "fv": (.5, 2.), "difficulty_term_str": "adr"}
    #     }
    # )

    # object_mass_scale_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.object_scale_mass.params.mass_distribution_params",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {"iv": (1., 1.), "fv": (.5, 3.), "difficulty_term_str": "adr"}
    #     }
    # )

    # robot_physics_material_static_friction_range_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.robot_physics_material.params.static_friction_range",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {"iv": (1., 1.), "fv": (.5, 1.), "difficulty_term_str": "adr"}
    #     }
    # )

    # robot_physics_material_dynamic_friction_range_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.robot_physics_material.params.dynamic_friction_range",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {"iv": (1., 1.), "fv": (.3, 1.), "difficulty_term_str": "adr"}
    #     }
    # )

    # robot_physics_material_restitution_range_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.robot_physics_material.params.restitution_range",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {"iv": (0., 0.), "fv": (0., 1.), "difficulty_term_str": "adr"}
    #     }
    # )

    # joint_friction_scale_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.joint_friction.params.friction_distribution_params",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {"iv": (0., 0.), "fv": (0., 5.), "difficulty_term_str": "adr"}
    #     }
    # )

    # object_physics_material_static_friction_range_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.object_physics_material.params.static_friction_range",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {"iv": (1., 1.), "fv": (.5, 1.), "difficulty_term_str": "adr"}
    #     }
    # )

    # object_physics_material_dynamic_friction_range_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.object_physics_material.params.dynamic_friction_range",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {"iv": (1., 1.), "fv": (.3, 1.), "difficulty_term_str": "adr"}
    #     }
    # )

    # object_physics_material_restitution_range_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.object_physics_material.params.restitution_range",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {"iv": (0., 0.), "fv": (0., 1.), "difficulty_term_str": "adr"}
    #     }
    # )      

    # lift_weight_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.lift.weight",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {"iv": 2.0, "fv": 0., "difficulty_term_str": "adr"}
    #     }
    # )
    
    # action_rate_l2_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.action_rate_l2.weight",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {"iv": -.005, "fv": -.025, "difficulty_term_str": "adr"}
    #     }
    # )
    
    # action_l2_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.action_l2.weight",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {"iv": -.005, "fv": -.025, "difficulty_term_str": "adr"}
    #     }
    # )

    # joint_pos_reg_weight_adr = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.joint_pos_reg.weight",
    #         "modify_fn": mdp.initial_final_interpolate_fn,
    #         "modify_params": {"iv": -.025, "fv": -.075, "difficulty_term_str": "adr"}
    #     }
    # )

    fingers_to_object_weight_adr = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "rewards.fingers_to_object.weight",
            "modify_fn": mdp.initial_final_interpolate_fn,
            "modify_params": {"iv": 1., "fv": 0., "difficulty_term_str": "adr"}
        }
    )
