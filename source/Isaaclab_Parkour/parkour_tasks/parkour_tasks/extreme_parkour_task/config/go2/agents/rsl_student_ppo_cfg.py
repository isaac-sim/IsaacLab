from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
ParkourRslRlOnPolicyRunnerCfg,
ParkourRslRlPpoActorCriticCfg,
ParkourRslRlActorCfg,
ParkourRslRlStateHistEncoderCfg,
ParkourRslRlEstimatorCfg,
ParkourRslRlDistillationAlgorithmCfg,
ParkourRslRlDepthEncoderCfg
)
from isaaclab.utils import configclass

@configclass
class UnitreeGo2ParkourStudentPPORunnerCfg(ParkourRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24 
    max_iterations = 50000 
    save_interval = 100
    experiment_name = "unitree_go2_parkour_student_ppo"
    empirical_normalization = False
    policy = ParkourRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        scan_encoder_dims = [128, 64, 32],
        priv_encoder_dims = [64, 20],
        activation="elu",
        actor = ParkourRslRlActorCfg(
            class_name = "Actor",
            state_history_encoder = ParkourRslRlStateHistEncoderCfg(
                class_name = "StateHistoryEncoder" 
            )
        )
    )
    estimator = ParkourRslRlEstimatorCfg(
            hidden_dims = [128, 64]
    )
    depth_encoder = ParkourRslRlDepthEncoderCfg(
        hidden_dims = 512,
        learning_rate= 1e-3,
        num_steps_per_env = 24*5
    )

    algorithm = ParkourRslRlDistillationAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate = 2.e-4, 
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

