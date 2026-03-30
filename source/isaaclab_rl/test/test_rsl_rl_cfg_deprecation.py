# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for handle_deprecated_rsl_rl_cfg across rsl-rl version boundaries."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

from dataclasses import MISSING

import pytest

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)
from isaaclab_rl.rsl_rl.utils import _is_missing, handle_deprecated_rsl_rl_cfg


def _ppo_algo():
    return RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.01,
        desired_kl=0.01,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )


def _distillation_algo():
    return RslRlDistillationAlgorithmCfg(num_learning_epochs=5, learning_rate=1e-3, gradient_length=1)


def _ppo_mlp_policy():
    return RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )


def _ppo_rnn_policy():
    return RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[128, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )


def _distillation_mlp_policy():
    return RslRlDistillationStudentTeacherCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=True,
        student_hidden_dims=[256, 256],
        teacher_hidden_dims=[128, 128],
        activation="elu",
    )


def _distillation_rnn_policy():
    return RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=True,
        student_hidden_dims=[256, 256],
        teacher_hidden_dims=[128, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=2,
        teacher_recurrent=True,
    )


def _mlp_model():
    return RslRlMLPModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        stochastic=True,
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
    )


def _rnn_model():
    return RslRlRNNModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        stochastic=True,
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )


def _on_policy_runner(**kw):
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=100,
        save_interval=50,
        experiment_name="test",
        **kw,
    )


def _distillation_runner(**kw):
    return RslRlDistillationRunnerCfg(
        num_steps_per_env=24,
        max_iterations=100,
        save_interval=50,
        experiment_name="test",
        **kw,
    )


# ===================================================================
# rsl-rl < 4.0.0
# ===================================================================
class TestBelow4:
    def test_no_policy_raises(self):
        with pytest.raises(ValueError, match="policy"):
            handle_deprecated_rsl_rl_cfg(_on_policy_runner(algorithm=_ppo_algo()), "3.0.0")

    def test_distillation_no_policy_raises(self):
        with pytest.raises(ValueError, match="policy"):
            handle_deprecated_rsl_rl_cfg(_distillation_runner(algorithm=_distillation_algo()), "3.0.0")

    def test_preserves_policy(self):
        cfg = _on_policy_runner(policy=_ppo_mlp_policy(), algorithm=_ppo_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "3.0.0")
        assert not _is_missing(cfg.policy)
        assert cfg.policy.actor_hidden_dims == [256, 256]

    def test_removes_optimizer_silently_for_adam(self, capsys):
        cfg = _on_policy_runner(policy=_ppo_mlp_policy(), algorithm=_ppo_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "3.0.0")
        assert "optimizer" not in capsys.readouterr().out

    def test_removes_optimizer_with_warning_for_non_adam(self, capsys):
        algo = _ppo_algo()
        algo.optimizer = "adamw"
        cfg = _on_policy_runner(policy=_ppo_mlp_policy(), algorithm=algo)
        handle_deprecated_rsl_rl_cfg(cfg, "3.0.0")
        assert "optimizer" in capsys.readouterr().out.lower()

    def test_distillation_optimizer_untouched(self):
        cfg = _distillation_runner(policy=_distillation_mlp_policy(), algorithm=_distillation_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "3.0.0")
        assert cfg.algorithm.optimizer == "adam"

    def test_empirical_normalization_migrates(self):
        p = _ppo_mlp_policy()
        p.actor_obs_normalization = MISSING
        p.critic_obs_normalization = MISSING
        cfg = _on_policy_runner(policy=p, algorithm=_ppo_algo(), empirical_normalization=True)
        handle_deprecated_rsl_rl_cfg(cfg, "3.0.0")
        assert cfg.policy.actor_obs_normalization is True
        assert cfg.policy.critic_obs_normalization is True
        assert _is_missing(cfg.empirical_normalization)

    def test_empirical_normalization_does_not_overwrite(self):
        cfg = _on_policy_runner(policy=_ppo_mlp_policy(), algorithm=_ppo_algo(), empirical_normalization=True)
        handle_deprecated_rsl_rl_cfg(cfg, "3.0.0")
        assert cfg.policy.actor_obs_normalization is False
        assert cfg.policy.critic_obs_normalization is False

    def test_clears_model_cfgs(self):
        critic = _mlp_model()
        critic.stochastic = False
        cfg = _on_policy_runner(policy=_ppo_mlp_policy(), algorithm=_ppo_algo(), actor=_mlp_model(), critic=critic)
        handle_deprecated_rsl_rl_cfg(cfg, "3.0.0")
        assert _is_missing(cfg.actor)
        assert _is_missing(cfg.critic)

    def test_distillation_clears_model_cfgs(self):
        cfg = _distillation_runner(
            policy=_distillation_mlp_policy(),
            algorithm=_distillation_algo(),
            student=_mlp_model(),
            teacher=_mlp_model(),
        )
        handle_deprecated_rsl_rl_cfg(cfg, "3.0.0")
        assert _is_missing(cfg.student)
        assert _is_missing(cfg.teacher)


# ===================================================================
# 4.0.0 <= rsl-rl < 5.0.0
# ===================================================================
class TestV4:
    # PPO tests
    def test_infers_mlp_actor_critic(self):
        p = RslRlPpoActorCriticCfg(
            actor_hidden_dims=[512],
            critic_hidden_dims=[64],
            activation="elu",
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            init_noise_std=0.5,
            noise_std_type="log",
            state_dependent_std=True,
        )
        cfg = _on_policy_runner(policy=p, algorithm=_ppo_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")

        assert isinstance(cfg.actor, RslRlMLPModelCfg) and not isinstance(cfg.actor, RslRlRNNModelCfg)
        assert cfg.actor.hidden_dims == [512]
        assert cfg.actor.stochastic is True
        assert cfg.actor.init_noise_std == 0.5
        assert cfg.actor.noise_std_type == "log"
        assert cfg.actor.state_dependent_std is True

        assert isinstance(cfg.critic, RslRlMLPModelCfg) and not isinstance(cfg.critic, RslRlRNNModelCfg)
        assert cfg.critic.hidden_dims == [64]
        assert cfg.critic.stochastic is False

    def test_infers_rnn_actor_critic(self):
        p = RslRlPpoActorCriticRecurrentCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[128, 128],
            activation="elu",
            rnn_type="gru",
            rnn_hidden_dim=128,
            rnn_num_layers=3,
        )
        cfg = _on_policy_runner(policy=p, algorithm=_ppo_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")

        assert isinstance(cfg.actor, RslRlRNNModelCfg)
        assert cfg.actor.rnn_type == "gru"
        assert cfg.actor.rnn_hidden_dim == 128
        assert cfg.actor.rnn_num_layers == 3
        assert isinstance(cfg.critic, RslRlRNNModelCfg)

    def test_clears_policy_after_inference(self):
        cfg = _on_policy_runner(policy=_ppo_mlp_policy(), algorithm=_ppo_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")
        assert _is_missing(cfg.policy)

    def test_skips_existing_actor(self):
        actor = _mlp_model()
        actor.hidden_dims = [999]
        cfg = _on_policy_runner(policy=_ppo_mlp_policy(), algorithm=_ppo_algo(), actor=actor)
        handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")
        assert cfg.actor.hidden_dims == [999]

    def test_skips_existing_critic(self):
        critic = _mlp_model()
        critic.stochastic = False
        critic.hidden_dims = [777]
        cfg = _on_policy_runner(policy=_ppo_mlp_policy(), algorithm=_ppo_algo(), critic=critic)
        handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")
        assert cfg.critic.hidden_dims == [777]

    def test_empirical_norm_then_inference(self):
        p = _ppo_mlp_policy()
        p.actor_obs_normalization = MISSING
        p.critic_obs_normalization = MISSING
        cfg = _on_policy_runner(policy=p, algorithm=_ppo_algo(), empirical_normalization=True)
        handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")
        assert cfg.actor.obs_normalization is True
        assert cfg.critic.obs_normalization is True
        assert _is_missing(cfg.empirical_normalization)

    # Distillation tests
    def test_distillation_infers_mlp_student_teacher(self):
        p = RslRlDistillationStudentTeacherCfg(
            student_hidden_dims=[512],
            teacher_hidden_dims=[64],
            activation="elu",
            student_obs_normalization=False,
            teacher_obs_normalization=True,
            init_noise_std=0.8,
            noise_std_type="log",
        )
        cfg = _distillation_runner(policy=p, algorithm=_distillation_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")

        assert isinstance(cfg.student, RslRlMLPModelCfg)
        assert cfg.student.hidden_dims == [512]
        assert cfg.student.init_noise_std == 0.8
        assert cfg.student.noise_std_type == "log"
        assert isinstance(cfg.teacher, RslRlMLPModelCfg)
        assert cfg.teacher.hidden_dims == [64]
        assert cfg.teacher.init_noise_std == 0.0  # hardcoded
        assert _is_missing(cfg.policy)

    def test_distillation_infers_rnn_student_teacher(self):
        cfg = _distillation_runner(policy=_distillation_rnn_policy(), algorithm=_distillation_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")

        assert isinstance(cfg.student, RslRlRNNModelCfg)
        assert cfg.student.rnn_type == "gru"
        assert isinstance(cfg.teacher, RslRlRNNModelCfg)
        assert cfg.teacher.init_noise_std == 0.0

    def test_distillation_skips_existing_student_teacher(self):
        student = _mlp_model()
        student.hidden_dims = [999]
        teacher = _mlp_model()
        teacher.hidden_dims = [777]
        cfg = _distillation_runner(
            policy=_distillation_mlp_policy(),
            algorithm=_distillation_algo(),
            student=student,
            teacher=teacher,
        )
        handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")
        assert cfg.student.hidden_dims == [999]
        assert cfg.teacher.hidden_dims == [777]

    # Stochastic tests
    def test_validates_stochastic_on_existing_models(self):
        critic = _mlp_model()
        critic.stochastic = False
        cfg = _on_policy_runner(algorithm=_ppo_algo(), actor=_mlp_model(), critic=critic)
        handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")  # should not raise

    def test_missing_stochastic_raises(self):
        a = _mlp_model()
        a.stochastic = MISSING
        cfg = _on_policy_runner(algorithm=_ppo_algo(), actor=a)
        with pytest.raises(ValueError, match="stochastic"):
            handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")

    def test_removes_distribution_cfg(self):
        a = _mlp_model()
        a.distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0)
        cfg = _on_policy_runner(algorithm=_ppo_algo(), actor=a)
        handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")
        assert not hasattr(cfg.actor, "distribution_cfg")

    def test_no_policy_no_models_is_noop(self):
        cfg = _on_policy_runner(algorithm=_ppo_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "4.0.0")
        assert _is_missing(cfg.actor) and _is_missing(cfg.critic)


# ===================================================================
# rsl-rl >= 5.0.0
# ===================================================================
class TestV5:
    # Distribution tests
    def test_gaussian_from_stochastic(self):
        a = _mlp_model()
        a.init_noise_std = 0.5
        a.noise_std_type = "log"
        cfg = _on_policy_runner(algorithm=_ppo_algo(), actor=a)
        handle_deprecated_rsl_rl_cfg(cfg, "5.0.0")

        d = cfg.actor.distribution_cfg
        assert isinstance(d, RslRlMLPModelCfg.GaussianDistributionCfg)
        assert not isinstance(d, RslRlMLPModelCfg.HeteroscedasticGaussianDistributionCfg)
        assert d.init_std == 0.5
        assert d.std_type == "log"

    def test_heteroscedastic_from_stochastic(self):
        a = _mlp_model()
        a.state_dependent_std = True
        a.init_noise_std = 0.3
        cfg = _on_policy_runner(algorithm=_ppo_algo(), actor=a)
        handle_deprecated_rsl_rl_cfg(cfg, "5.0.0")

        assert isinstance(cfg.actor.distribution_cfg, RslRlMLPModelCfg.HeteroscedasticGaussianDistributionCfg)
        assert cfg.actor.distribution_cfg.init_std == 0.3

    def test_keeps_existing_distribution_cfg(self):
        a = _mlp_model()
        a.state_dependent_std = True
        a.distribution_cfg = RslRlMLPModelCfg.HeteroscedasticGaussianDistributionCfg(init_std=2.0, std_type="log")
        cfg = _on_policy_runner(algorithm=_ppo_algo(), actor=a)
        handle_deprecated_rsl_rl_cfg(cfg, "5.0.0")

        assert isinstance(cfg.actor.distribution_cfg, RslRlMLPModelCfg.HeteroscedasticGaussianDistributionCfg)
        assert cfg.actor.distribution_cfg.init_std == 2.0
        assert cfg.actor.distribution_cfg.std_type == "log"

    def test_non_stochastic_no_distribution(self):
        a = _mlp_model()
        a.stochastic = False
        cfg = _on_policy_runner(algorithm=_ppo_algo(), actor=a)
        handle_deprecated_rsl_rl_cfg(cfg, "5.0.0")
        assert cfg.actor.distribution_cfg is None

    def test_removes_deprecated_params(self):
        cfg = _on_policy_runner(algorithm=_ppo_algo(), actor=_mlp_model())
        handle_deprecated_rsl_rl_cfg(cfg, "5.0.0")
        assert not hasattr(cfg.actor, "stochastic")
        assert not hasattr(cfg.actor, "init_noise_std")
        assert not hasattr(cfg.actor, "noise_std_type")
        assert not hasattr(cfg.actor, "state_dependent_std")

    def test_migrates_rnn_models(self):
        a = _rnn_model()
        a.state_dependent_std = True
        c = _rnn_model()
        c.stochastic = False
        cfg = _on_policy_runner(algorithm=_ppo_algo(), actor=a, critic=c)
        handle_deprecated_rsl_rl_cfg(cfg, "5.0.0")

        assert isinstance(cfg.actor.distribution_cfg, RslRlMLPModelCfg.HeteroscedasticGaussianDistributionCfg)
        assert cfg.critic.distribution_cfg is None

    # Full conversion pipeline tests
    def test_mlp_policy_full_pipeline(self):
        p = _ppo_mlp_policy()
        p.init_noise_std = 0.7
        p.noise_std_type = "log"
        cfg = _on_policy_runner(policy=p, algorithm=_ppo_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "5.0.0")

        assert _is_missing(cfg.policy)
        assert isinstance(cfg.actor.distribution_cfg, RslRlMLPModelCfg.GaussianDistributionCfg)
        assert cfg.actor.distribution_cfg.init_std == 0.7
        assert cfg.actor.distribution_cfg.std_type == "log"
        assert cfg.critic.distribution_cfg is None

    def test_mlp_policy_state_dependent_pipeline(self):
        p = _ppo_mlp_policy()
        p.init_noise_std = 0.5
        p.state_dependent_std = True
        cfg = _on_policy_runner(policy=p, algorithm=_ppo_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "5.0.0")

        assert isinstance(cfg.actor.distribution_cfg, RslRlMLPModelCfg.HeteroscedasticGaussianDistributionCfg)

    def test_rnn_policy_full_pipeline(self):
        p = _ppo_rnn_policy()
        p.init_noise_std = 1.5
        p.rnn_type = "gru"
        p.rnn_hidden_dim = 64
        p.rnn_num_layers = 3
        cfg = _on_policy_runner(policy=p, algorithm=_ppo_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "5.0.0")

        assert _is_missing(cfg.policy)
        assert isinstance(cfg.actor, RslRlRNNModelCfg)
        assert cfg.actor.rnn_type == "gru"
        assert isinstance(cfg.actor.distribution_cfg, RslRlMLPModelCfg.GaussianDistributionCfg)
        assert cfg.actor.distribution_cfg.init_std == 1.5
        assert isinstance(cfg.critic, RslRlRNNModelCfg)
        assert cfg.critic.distribution_cfg is None

    def test_distillation_mlp_policy_full_pipeline(self):
        p = _distillation_mlp_policy()
        p.init_noise_std = 0.9
        p.noise_std_type = "log"
        cfg = _distillation_runner(policy=p, algorithm=_distillation_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "5.0.0")

        assert _is_missing(cfg.policy)
        assert isinstance(cfg.student.distribution_cfg, RslRlMLPModelCfg.GaussianDistributionCfg)
        assert cfg.student.distribution_cfg.init_std == 0.9
        assert cfg.student.distribution_cfg.std_type == "log"
        assert isinstance(cfg.teacher.distribution_cfg, RslRlMLPModelCfg.GaussianDistributionCfg)
        assert cfg.teacher.distribution_cfg.init_std == 0.0

    def test_distillation_rnn_policy_full_pipeline(self):
        p = _distillation_rnn_policy()
        p.init_noise_std = 1.2
        cfg = _distillation_runner(policy=p, algorithm=_distillation_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "5.0.0")

        assert _is_missing(cfg.policy)
        assert isinstance(cfg.student, RslRlRNNModelCfg)
        assert isinstance(cfg.student.distribution_cfg, RslRlMLPModelCfg.GaussianDistributionCfg)
        assert isinstance(cfg.teacher, RslRlRNNModelCfg)
        assert cfg.teacher.distribution_cfg.init_std == 0.0

    def test_no_policy_no_models_is_noop(self):
        cfg = _on_policy_runner(algorithm=_ppo_algo())
        handle_deprecated_rsl_rl_cfg(cfg, "5.0.0")
        assert _is_missing(cfg.actor) and _is_missing(cfg.critic)

    def test_empirical_norm_rnn_policy_full_pipeline(self):
        p = _ppo_rnn_policy()
        p.init_noise_std = 2.0
        p.state_dependent_std = True
        p.actor_obs_normalization = MISSING
        p.critic_obs_normalization = MISSING
        cfg = _on_policy_runner(policy=p, algorithm=_ppo_algo(), empirical_normalization=True)
        handle_deprecated_rsl_rl_cfg(cfg, "5.0.0")

        assert _is_missing(cfg.policy) and _is_missing(cfg.empirical_normalization)
        assert cfg.actor.obs_normalization is True
        assert isinstance(cfg.actor.distribution_cfg, RslRlMLPModelCfg.HeteroscedasticGaussianDistributionCfg)

    # Edge tests
    def test_returns_same_object(self):
        cfg = _on_policy_runner(policy=_ppo_mlp_policy(), algorithm=_ppo_algo())
        assert handle_deprecated_rsl_rl_cfg(cfg, "5.0.0") is cfg
