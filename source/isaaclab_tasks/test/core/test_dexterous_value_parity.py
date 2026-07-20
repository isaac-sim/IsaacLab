# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct-vs-manager scalar value parity for the dexterous task families.

The Direct and manager configurations define their task scalars separately
(per repo convention); these checks catch one side being re-tuned without the
other. Each case maps a Direct cfg field to the manager cfg location that must
carry the same value.
"""

import pytest


def _resolve(obj, path: str):
    for part in path.split("."):
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
    return obj


def _reorient_cases(direct_cfg, manager_cfg):
    """Field-to-term mapping shared by the Shadow, OpenAI, and Allegro pairs."""
    return [
        (direct_cfg.sim.dt, manager_cfg.sim.dt),
        (direct_cfg.sim.render_interval, manager_cfg.sim.render_interval),
        (direct_cfg.success_tolerance, manager_cfg.commands.object_pose.orientation_success_threshold),
        (direct_cfg.success_tolerance, _resolve(manager_cfg, "rewards.reorient.params.success_tolerance")),
        (direct_cfg.dist_reward_scale, _resolve(manager_cfg, "rewards.reorient.params.distance_scale")),
        (direct_cfg.rot_reward_scale, _resolve(manager_cfg, "rewards.reorient.params.rotation_scale")),
        (direct_cfg.rot_eps, _resolve(manager_cfg, "rewards.reorient.params.rotation_epsilon")),
        (direct_cfg.action_penalty_scale, _resolve(manager_cfg, "rewards.reorient.params.action_penalty_scale")),
        (direct_cfg.reach_goal_bonus, _resolve(manager_cfg, "rewards.reorient.params.success_bonus")),
        (direct_cfg.fall_dist, _resolve(manager_cfg, "rewards.reorient.params.fall_distance")),
        (direct_cfg.fall_penalty, _resolve(manager_cfg, "rewards.reorient.params.fall_penalty")),
        (direct_cfg.av_factor, _resolve(manager_cfg, "rewards.reorient.params.averaging_factor")),
        (direct_cfg.act_moving_average, manager_cfg.actions.joint_pos.alpha),
        (direct_cfg.decimation, manager_cfg.decimation),
        (direct_cfg.episode_length_s, manager_cfg.episode_length_s),
        (direct_cfg.reset_position_noise, _reset_event_params(manager_cfg)["position_noise"]),
        (direct_cfg.reset_dof_pos_noise, _reset_event_params(manager_cfg)["joint_position_noise"]),
        (direct_cfg.reset_dof_vel_noise, _reset_event_params(manager_cfg)["joint_velocity_noise"]),
        (direct_cfg.max_consecutive_success, _timeout_max_successes(manager_cfg)),
    ]


def _reset_event_params(manager_cfg):
    """Params of the reset event term (the OpenAI events are preset-wrapped)."""
    events = manager_cfg.events
    term = getattr(events, "reset_state", None) or events.default.reset_state
    return term.params


def _timeout_max_successes(manager_cfg):
    """Successes-based timeout threshold; 0 when the manager uses the plain timeout.

    Mirrors the Direct convention where ``max_consecutive_success = 0`` disables
    the mechanism (state and Allegro), while the OpenAI variants enable it.
    """
    params = getattr(manager_cfg.terminations.time_out, "params", None) or {}
    return params.get("max_successes", 0)


def _pairs():
    from isaaclab_tasks.core.handover.handover_env_cfg import HandoverEnvCfg
    from isaaclab_tasks.core.handover.handover_manager_env_cfg import HandoverManagerEnvCfg
    from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_direct_env_cfg import AllegroHandEnvCfg
    from isaaclab_tasks.core.reorient.config.allegro_hand.allegro_hand_manager_env_cfg import AllegroCubeEnvCfg
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_direct_env_cfg import (
        ShadowHandEnvCfg,
        ShadowHandOpenAIEnvCfg,
    )
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_manager_env_cfg import ShadowHandManagerEnvCfg
    from isaaclab_tasks.core.reorient.config.shadow_hand.shadow_hand_openai_manager_env_cfg import (
        ShadowHandOpenAIManagerEnvCfg,
    )

    return {
        "shadow": (ShadowHandEnvCfg(), ShadowHandManagerEnvCfg()),
        "openai": (ShadowHandOpenAIEnvCfg(), ShadowHandOpenAIManagerEnvCfg()),
        "allegro": (AllegroHandEnvCfg(), AllegroCubeEnvCfg()),
        "handover": (HandoverEnvCfg(), HandoverManagerEnvCfg()),
    }


@pytest.mark.parametrize("family", ["shadow", "openai", "allegro"])
def test_reorient_direct_manager_scalars_match(family):
    """Direct cfg scalars equal the manager term params they mirror."""
    direct_cfg, manager_cfg = _pairs()[family]
    for i, (direct_value, manager_value) in enumerate(_reorient_cases(direct_cfg, manager_cfg)):
        assert direct_value == manager_value, f"{family} case {i}: direct={direct_value} manager={manager_value}"


def test_handover_direct_manager_scalars_match():
    """Handover Direct cfg scalars equal the manager term params they mirror."""
    direct_cfg, manager_cfg = _pairs()["handover"]
    obs = manager_cfg.observations
    cases = [
        (direct_cfg.sim.dt, manager_cfg.sim.dt),
        (direct_cfg.sim.render_interval, manager_cfg.sim.render_interval),
        (direct_cfg.dist_reward_scale, _resolve(manager_cfg, "rewards.handover.params.distance_scale")),
        (
            direct_cfg.success_distance_threshold,
            _resolve(manager_cfg, "rewards.handover.params.success_distance_threshold"),
        ),
        (direct_cfg.vel_obs_scale, _resolve(obs, "policy.right_object_goal.params.vel_obs_scale")),
        (direct_cfg.vel_obs_scale, _resolve(obs, "policy.left_object_goal.params.vel_obs_scale")),
        (direct_cfg.act_moving_average, manager_cfg.actions.right_hand.alpha),
        (direct_cfg.act_moving_average, manager_cfg.actions.left_hand.alpha),
        (direct_cfg.decimation, manager_cfg.decimation),
        (direct_cfg.episode_length_s, manager_cfg.episode_length_s),
    ]
    for i, (direct_value, manager_value) in enumerate(cases):
        assert direct_value == manager_value, f"handover case {i}: direct={direct_value} manager={manager_value}"
