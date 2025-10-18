
import torch 
import os
from core.nerual_networks.feature_extractors.estimator import DefaultEstimator
from core.nerual_networks.actor_critic_with_encoder import ActorCriticRMA
from core.nerual_networks.distillation_with_extractor import DistillationWithExtractor 
from core.nerual_networks.ppo_with_extractor import PPOWithExtractor 
from rsl_rl.modules import (
    EmpiricalNormalization,
)
import warnings 

class Runner():
    def __init__(
        self,
        agent_cfg,
        env,
        device
        ):
        self.device = device
        self.cfg = agent_cfg
        self.alg_cfg = agent_cfg["algorithm"]
        self.estimator_cfg = agent_cfg["estimator"]
        self.depth_encoder_cfg = agent_cfg["depth_encoder"]
        self.policy_cfg = agent_cfg["policy"]
        self.env = env
        self.mean_hist_latent_loss = 0.
        self._configure_multi_gpu()
        if self.alg_cfg["class_name"] == "PPOWithExtractor":
            self.training_type = "rl"
        elif self.alg_cfg["class_name"] == "DistillationWithExtractor":
            self.training_type = "distillation"
        else:
            raise ValueError(f"Training type not found for algorithm {self.alg_cfg['class_name']}.")

        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]

        if self.training_type == "rl":
            if "critic" in extras["observations"]:
                self.privileged_obs_type = "critic"  # actor-critic reinforcement learnig, e.g., PPO
            else:
                self.privileged_obs_type = None
        if self.training_type == "distillation":
            if "teacher" in extras["observations"]:
                self.privileged_obs_type = "teacher"  # policy distillation
            else:
                self.privileged_obs_type = None

        if self.privileged_obs_type is not None:
            num_privileged_obs = extras["observations"][self.privileged_obs_type].shape[1]
        else:
            num_privileged_obs = num_obs

        estimator_class = eval(self.estimator_cfg.pop("class_name"))
        estimator: DefaultEstimator = estimator_class(**self.estimator_cfg).to(self.device)
        
        policy_class = eval(self.policy_cfg.pop("class_name"))
        policy: ActorCriticRMA = policy_class(
            num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        if "rnd_cfg" in self.alg_cfg and self.alg_cfg["rnd_cfg"] is not None:
            # check if rnd gated state is present
            rnd_state = extras["observations"].get("rnd_state")
            if rnd_state is None:
                raise ValueError("Observations for the key 'rnd_state' not found in infos['observations'].")
            # get dimension of rnd gated state
            num_rnd_state = rnd_state.shape[1]
            # add rnd gated state to config
            self.alg_cfg["rnd_cfg"]["num_states"] = num_rnd_state
            # scale down the rnd weight with timestep (similar to how rewards are scaled down in legged_gym envs)
            self.alg_cfg["rnd_cfg"]["weight"] *= env.unwrapped.step_dt

        # if using symmetry then pass the environment config object
        if "symmetry_cfg" in self.alg_cfg and self.alg_cfg["symmetry_cfg"] is not None:
            # this is used by the symmetry function for handling different observation terms
            self.alg_cfg["symmetry_cfg"]["_env"] = env

        # initialize algorithm

        if self.depth_encoder_cfg is not None:
            alg_class = eval(self.alg_cfg.pop("class_name"))
            self.alg: DistillationWithExtractor = alg_class(
                                                    policy = policy, 
                                                    estimator= estimator, 
                                                    estimator_paras= self.estimator_cfg,
                                                    depth_encoder_cfg = self.depth_encoder_cfg,
                                                    learning_rate = self.alg_cfg['learning_rate'],
                                                    policy_cfg = self.policy_cfg, 
                                                    max_grad_norm = self.alg_cfg['max_grad_norm'],
                                                    device=self.device, 
                                                    multi_gpu_cfg=self.multi_gpu_cfg
                                                    )
        else:
            self.dagger_update_freq = self.alg_cfg.pop("dagger_update_freq")
            alg_class = eval(self.alg_cfg.pop("class_name"))
            self.alg: PPOWithExtractor = alg_class(
                                                    policy, 
                                                    estimator, 
                                                    self.estimator_cfg,
                                                    **self.alg_cfg, 
                                                    device=self.device, 
                                                    multi_gpu_cfg=self.multi_gpu_cfg
                                                    )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.empirical_normalization = self.cfg["empirical_normalization"]

        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(shape=[num_privileged_obs], until=1.0e8).to(
                self.device
            )
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        if self.depth_encoder_cfg is None:

            self.alg.init_storage(
                self.training_type,
                self.env.num_envs,
                self.num_steps_per_env,
                [num_obs],
                [num_privileged_obs],
                [self.env.num_actions],
            )

        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0
        # Logging
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    def _configure_multi_gpu(self):
        """Configure multi-gpu training."""
        # check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # if not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        # get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # make a configuration dictionary
        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,  # rank of the main process
            "local_rank": self.gpu_local_rank,  # rank of the current process
            "world_size": self.gpu_world_size,  # total number of processes
        }

        # check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'.")
        # validate multi-gpu configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'.")

        # initialize torch distributed
        torch.distributed.init_process_group(
            backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
        )
        # set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)

    def load(self, path: str, load_optimizer: bool = True):
        loaded_dict = torch.load(path, weights_only=False)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        if self.alg.rnd:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if self.empirical_normalization:
            if resumed_training:
                self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["privileged_obs_norm_state_dict"])
            else:
                self.privileged_obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])

        if self.depth_encoder_cfg is not None:
            if 'depth_encoder_state_dict' not in loaded_dict:
                warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
            else:
                print("Saved depth encoder detected, loading...")
                self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
            if 'depth_actor_state_dict' in loaded_dict:
                print("Saved depth actor detected, loading...")
                self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
            else:
                print("No saved depth actor, Copying actor critic actor to depth actor...")
                self.alg.depth_actor.load_state_dict(self.alg.policy.actor.state_dict())

        if load_optimizer and resumed_training:
            # -- algorithm optimizer
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            # -- RND optimizer if used
            if self.alg.rnd:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        # -- load current learning iteration
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_estimator_inference_policy(self, device=None):
        self.alg: PPOWithExtractor
        self.alg.estimator.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.estimator.to(device)
        return self.alg.estimator.inference
    
    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.policy.to(device)
        policy = self.alg.policy.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.policy.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy
    
    def get_depth_encoder_inference_policy(self, device=None):
        self.alg.depth_encoder.eval()
        if device is not None:
            self.alg.depth_encoder.to(device)
        return self.alg.depth_encoder

    def eval_mode(self):
        # -- PPO
        self.alg.policy.eval()
        # -- RND
        if self.alg.rnd:
            self.alg.rnd.eval()
        # -- Normalization
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.privileged_obs_normalizer.eval()
