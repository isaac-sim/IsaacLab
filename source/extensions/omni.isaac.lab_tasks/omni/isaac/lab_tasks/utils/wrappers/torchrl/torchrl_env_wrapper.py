from __future__ import annotations

import torch
import numpy as np 
import contextlib
import os 
from dataclasses import asdict

import time 

from typing import Any, Dict, Optional, Iterator
from tensordict.nn import TensorDictModule
from tensordict import TensorDict, TensorDictBase
from omni.isaac.lab.envs import ManagerBasedRLEnv
from torchrl.envs.libs.gym import  default_info_dict_reader
from torchrl.trainers import Trainer
from torchrl.envs import GymWrapper
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss
from torchrl.record.loggers.wandb import WandbLogger
from torchrl.trainers.trainers import LOGGER_METHODS
from torchrl.trainers import Trainer
import warnings

from torchrl.data.tensor_specs import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)

from torchrl.collectors.utils import split_trajectories

from torchrl.envs.utils import (
    _terminated_or_truncated,
)
from torchrl.objectives.utils import (
    _reduce,
)

from tensordict.nn import (
    dispatch,
    ProbabilisticTensorDictSequential,
    TensorDictModule
)

class TorchRLEnvWrapper(GymWrapper):
    def __init__(self, env: ManagerBasedRLEnv, categorical_action_encoding=False, **kwargs):
        super().__init__(env, categorical_action_encoding=categorical_action_encoding, **kwargs)
        self._ep_len_buf = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
        self._ep_reward_buf = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
        self._curr_ep_len = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
        self._curr_reward_sum = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
        self._env_reset_mask = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.bool)
    
    def maybe_reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Checks the done keys of the input tensordict. Unlike the base GymWrapper implementation, we do not 
        call env.reset() 

        Args:
            tensordict (TensorDictBase): a tensordict coming from the output of :func:`~torchrl.envs.utils.step_mdp`.

        Returns:
            A tensordict that is identical to the input where the environment was
            not reset and contains the new reset data where the environment was reset.

        """
        if self._simple_done:
            done = tensordict._get_str("done", default=None)
            any_done = done.any()
            if any_done:
                tensordict._set_str(
                    "_reset",
                    done.clone(),
                    validated=True,
                    inplace=False,
                    non_blocking=False,
                )
        else:
            any_done = _terminated_or_truncated(
                tensordict,
                full_done_spec=self.output_spec["full_done_spec"],
                key="_reset",
            )
            
        return tensordict

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Makes a step in the environment.

        Step accepts a single argument, tensordict, which usually carries an 'action' key which indicates the action
        to be taken.
        Step will call an out-place private method, _step, which is the method to be re-written by EnvBase subclasses.

        Args:
            tensordict (TensorDictBase): Tensordict containing the action to be taken.
                If the input tensordict contains a ``"next"`` entry, the values contained in it
                will prevail over the newly computed values. This gives a mechanism
                to override the underlying computations.

        Returns:
            the input tensordict, modified in place with the resulting observations, done state and reward
            (+ others if needed).

        """
        # sanity check
        self._assert_tensordict_shape(tensordict)
        next_preset = tensordict.get("next", None)

        next_tensordict = self._step(tensordict)
        next_tensordict = self._step_proc_data(next_tensordict)
        if next_preset is not None:
            # tensordict could already have a "next" key
            # this could be done more efficiently by not excluding but just passing
            # the necessary keys
            next_tensordict.update(
                next_preset.exclude(*next_tensordict.keys(True, True))
            )
        tensordict.set("next", next_tensordict)
        # Information from the "extras" dict of dicts passed from IsaacLab contains reward and metrics information.
        # The _data postfix is added to distinguish the field containing the actual values from the dummy {key} field
        # in the tensordict required for faster rollouts. To get logging to work, we need to move the data from the "next" tensordict
        # to the top level tensordict. 
        # TODO: Find a more elegant solution to enable rewards and metrics logging. " 
        for key in self.env.unwrapped.extras:
            tensordict[f"{key}_data"] = tensordict["next"][f"{key}_data"]
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key)
        reward = 0
        for _ in range(self.wrapper_frame_skip):
            (
                obs,
                _reward,
                terminated,
                truncated,
                done,
                info_dict,
            ) = self._output_transform(self._env.step(action))

            if _reward is not None:
                reward = reward + _reward

            terminated, truncated, done, do_break = self.read_done(
                terminated=terminated, truncated=truncated, done=done
            )
            if do_break:
                break
        reward = self.read_reward(reward)
        obs_dict = self.read_obs(obs)
        obs_dict[self.reward_key] = reward
        
        # if truncated/terminated is not in the keys, we just don't pass it even if it
        # is defined.
        if terminated is None:
            terminated = done
        if truncated is not None:
            obs_dict["truncated"] = truncated

        obs_dict["done"] = done
        obs_dict["terminated"] = terminated
        validated = self.validated
        if not validated:
            tensordict_out = TensorDict(obs_dict, batch_size=tensordict.batch_size)
            if validated is None:
                # check if any value has to be recast to something else. If not, we can safely
                # build the tensordict without running checks
                self.validated = all(
                    val is tensordict_out.get(key)
                    for key, val in TensorDict(obs_dict, []).items(True, True)
                )
        else:
            tensordict_out = TensorDict._new_unsafe(
                obs_dict,
                batch_size=tensordict.batch_size,
            )
        if self.device is not None:
            tensordict_out = tensordict_out.to(self.device)

        self._curr_ep_len += 1 
        self._curr_reward_sum += reward
        done_envs = tensordict_out["done"] | tensordict_out["terminated"] | tensordict_out["truncated"]
        new_ids = (done_envs > 0).nonzero(as_tuple=False)
        
        self._ep_len_buf = torch.where(self._env_reset_mask == False, self._curr_ep_len.clone(), self._ep_len_buf)
        self._ep_reward_buf = torch.where(self._env_reset_mask == False, self._curr_reward_sum.clone(), self._ep_reward_buf)
        if len(new_ids) > 0:
            self._env_reset_mask[new_ids] = True
            self._ep_len_buf[new_ids] = self._curr_ep_len[new_ids].clone()
            self._ep_reward_buf[new_ids] = self._curr_reward_sum[new_ids].clone()

        self._curr_ep_len[new_ids] = 0
        self._curr_reward_sum[new_ids] = 0

        tensordict.set("episode_length", self._ep_len_buf)
        tensordict.set("episode_reward", self._ep_reward_buf)
        tensordict_out.set("episode_length", self._ep_len_buf)
        tensordict_out.set("episode_reward", self._ep_reward_buf)
        if self.info_dict_reader and (info_dict is not None):
            if not isinstance(info_dict, dict):
                warnings.warn(
                    f"Expected info to be a dictionary but got a {type(info_dict)} with values {str(info_dict)[:100]}."
                )
            else:
                for info_dict_reader in self.info_dict_reader:
                    out = info_dict_reader(info_dict, tensordict_out)
                    if out is not None:
                        tensordict_out = out
        return tensordict_out
    
class InfoDictReaderWrapper(default_info_dict_reader):
    def __call__(
        self, info_dict: Dict[str, Any], tensordict: TensorDictBase
    ) -> TensorDictBase:
        if not isinstance(info_dict, (dict, TensorDictBase)) and len(self.keys):
            warnings.warn(
                f"Found an info_dict of type {type(info_dict)} "
                f"but expected type or subtype `dict`."
            )
        keys = self.keys
        if keys is None:
            keys = info_dict.keys()
            if self.ignore_private:
                keys = [key for key in keys if not key.startswith("_")]
            self.keys = keys
        # create an info_spec only if there is none
        info_spec = None if self.info_spec is not None else CompositeSpec()
        
        def add_value_to_tensordict(tensordict: TensorDict, key: str, value: torch.Tensor):
            if value.dtype == np.dtype("O"):
                value = np.stack(value)
            # if value.shape != tensordict.shape:
            value = value.expand(tensordict.shape)
            tensordict.set(key, value)
            if info_spec is not None:
                value = tensordict.get(key)
                info_spec[key] = UnboundedContinuousTensorSpec(
                    value.shape, device=value.device, dtype=value.dtype
                )
                
        def _traverse_dict(tensordict: TensorDict, key, val):
            """
            Recursively traverse key-value pairs of an arbitrarily nested dictionary 
            and add non-dict leaf values to the info tensordict while preserving structure. 
            """
            if isinstance(val, dict):
                nested_tensordict = TensorDict(batch_size=tensordict.shape, device=tensordict.device)
                for k, v in val.items():
                    _traverse_dict(nested_tensordict, k, v)
                tensordict.set(key, nested_tensordict)
            else:
                val = torch.as_tensor(val, device=tensordict.device)
                add_value_to_tensordict(tensordict, key, val)
        
        for key in keys:
            if key in info_dict:
                val = info_dict[key]
                if isinstance(val, dict):
                    # Add all key strings that map to dictionaries to the tensordict. 
                    # This is done because sampling throughput is reduced 
                    # if the tensordict key set does not match the expected.
                    tensordict.set(key, torch.tensor(0).expand(tensordict.shape))
                    nested_tensordict = TensorDict(batch_size=tensordict.shape, device=tensordict.device)
                    key = key + "_data"
                    for k, v in val.items():
                        v = torch.as_tensor(v, device=tensordict.device)
                        add_value_to_tensordict(nested_tensordict, k, v)
                    tensordict.set(key, nested_tensordict)
            elif info_spec is not None:
                if key in info_spec:
                    # Fill missing with 0s
                    tensordict.set(key, info_spec[key].zero())
            else:
                raise KeyError(f"The key {key} could not be found or inferred.")
        if info_spec is not None:
            if tensordict.device is not None:
                info_spec = info_spec.to(tensordict.device)
            self._info_spec = info_spec
        # add episode length field to tensordict for logging 
        tensordict.set("episode_length", torch.tensor(0.0).expand(tensordict.shape))
        tensordict.set("episode_reward", torch.tensor(0.0).expand(tensordict.shape))
        return tensordict

class SyncDataCollectorWrapper(SyncDataCollector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def iterator(self) -> Iterator[TensorDictBase]:
        """Iterates through the DataCollector.

        Yields: TensorDictBase objects containing (chunks of) trajectories

        """
        # The portion of the code handling cuda streams has been removed in this inherited method, which
        # caused CUDA memory allocation issues with IsaacSim during env stepping.  
        total_frames = self.total_frames

        while self._frames < self.total_frames:
            self._iter += 1
            # Log the time taken by self.rollout()
            start_time = time.perf_counter()
            tensordict_out = self.rollout()
            end_time = time.perf_counter()
            rollout_time = end_time - start_time
            # Add rollout time to tensordict
            tensordict_out.set("rollout_time", torch.tensor(rollout_time).expand(tensordict_out.shape))
            self._frames += tensordict_out.numel()
            if self._frames >= total_frames:
                self.env.close()

            if self.split_trajs:
                tensordict_out = split_trajectories(
                    tensordict_out, prefix="collector"
                )
            if self.postproc is not None:
                tensordict_out = self.postproc(tensordict_out)
            if self._exclude_private_keys:

                def is_private(key):
                    if isinstance(key, str) and key.startswith("_"):
                        return True
                    if isinstance(key, tuple) and any(
                        _key.startswith("_") for _key in key
                    ):
                        return True
                    return False

                excluded_keys = [
                    key for key in tensordict_out.keys(True) if is_private(key)
                ]
                tensordict_out = tensordict_out.exclude(
                    *excluded_keys, inplace=True
                )
            if self.return_same_td:
                # This is used with multiprocessed collectors to use the buffers
                # stored in the tensordict.
                # if events:
                #     for event in events:
                #         event.record()
                #         event.synchronize()
                yield tensordict_out
            else:
                # we must clone the values, as the tensordict is updated in-place.
                # otherwise the following code may break:
                # >>> for i, data in enumerate(collector):
                # >>>      if i == 0:
                # >>>          data0 = data
                # >>>      elif i == 1:
                # >>>          data1 = data
                # >>>      else:
                # >>>          break
                # >>> assert data0["done"] is not data1["done"]
                yield tensordict_out.clone()

class ClipPPOLossWrapper(ClipPPOLoss):
    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        critic_network: TensorDictModule | None = None,
        *,
        clip_epsilon: float = 0.2,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        desired_kl: float = 0.01,
        loss_critic_type: str = "l2",
        normalize_advantage: bool = False,
        gamma: float = None,
        separate_losses: bool = False,
        reduction: str = None,
        clip_value: bool | float | None = None,
        **kwargs,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=critic_network,
            clip_epsilon=clip_epsilon,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            loss_critic_type=loss_critic_type,
            normalize_advantage=normalize_advantage,
            gamma=gamma,
            separate_losses=separate_losses,
            reduction=reduction,
            clip_value=clip_value,
            **kwargs
        )
    
        self.desired_kl = desired_kl 

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = tensordict.clone(False)
        try:
            previous_dist = self.actor_network.build_dist_from_params(tensordict)
        except KeyError as err:
            raise KeyError(
                "The parameters of the distribution were not found. "
                f"Make sure they are provided to {type(self).__name__}."
            ) from err
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean()
            scale = advantage.std().clamp_min(1e-6)
            advantage = (advantage - loc) / scale

        log_weight, dist, kl_approx = self._log_weight(tensordict)
        # ESS for logging
        with torch.no_grad():
            # In theory, ESS should be computed on particles sampled from the same source. Here we sample according
            # to different, unrelated trajectories, which is not standard. Still it can give a idea of the dispersion
            # of the weights.
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]

        gain1 = log_weight.exp() * advantage

        log_weight_clip = log_weight.clamp(*self._clip_bounds)
        clip_fraction = (log_weight_clip != log_weight).to(log_weight.dtype).mean()
        ratio = log_weight_clip.exp()
        gain2 = ratio * advantage

        gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]
        td_out = TensorDict({"loss_objective": -gain.mean()}, batch_size=[])
        td_out.set("clip_fraction", clip_fraction)

        with self.actor_network_params.to_module(
            self.actor_network
        ) if self.functional else contextlib.nullcontext():
            current_dist = self.actor_network.get_dist(tensordict)
            td_out.set("action_noise", current_dist.scale.mean())
        try:
            kl = torch.distributions.kl.kl_divergence(previous_dist, current_dist).mean()
        except NotImplementedError:
            x = previous_dist.sample((1))
            kl = (previous_dist.log_prob(x) - current_dist.log_prob(x)).mean(0)
        kl = kl.unsqueeze(-1)
        td_out.set("kl", kl)
        if self.entropy_bonus:
            entropy = self.get_entropy_bonus(dist)
            td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("kl_approx", kl_approx.detach().mean())  # for logging
            td_out.set("loss_entropy", -self.entropy_coef * entropy.sum(dim=-1).mean())
        if self.critic_coef:
            loss_critic, value_clip_fraction = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)

        td_out.set("ESS", _reduce(ess, self.reduction) / batch)
        td_out = td_out.named_apply(
            lambda name, value: _reduce(value, reduction=self.reduction).squeeze(-1)
            if name.startswith("loss_")
            else value,
            batch_size=[],
        )
        return td_out

class TrainerWrapper(Trainer):
    def __init__(self, num_mini_batches, lr_schedule: str, **kwargs):
        super().__init__(**kwargs)
        self.num_mini_batches = num_mini_batches
        self.lr_schedule = lr_schedule
        self.log_path, self.file_name_ext = os.path.split(self.save_trainer_file)
        self.file_name, self.file_ext = os.path.splitext(self.file_name_ext)
        self.learning_rate = 1e-3

    def save_trainer(self, force_save: bool = False) -> None:
        # overwrite the model name with iteration count 
        file_name = f"{self.file_name}_{self._optim_count}"
        self.save_trainer_file = os.path.join(self.log_path, file_name+self.file_ext)

        _save = force_save
        if self.save_trainer_file is not None:
            if (self.collected_frames - self._last_save) > self.save_trainer_interval:
                self._last_save = self.collected_frames
                _save = True
        
        if _save and self.save_trainer_file:
            file_path, file_ext = os.path.splitext(self.save_trainer_file)
            # Append iteration count to file name and save 
            file_path += f"_{iter}{file_ext}"
            
            self._save_trainer()
            if hasattr(self.logger, "log_model"):
                self.logger.log_model(self.save_trainer_file, iter=self._optim_count)

    def _log(self, log_pbar=False, **kwargs) -> None:
        collected_frames = self.collected_frames
        iter = collected_frames // self.collector.requested_frames_per_batch
        for key, item in kwargs.items():
            self._log_dict[key].append(item)
            if (collected_frames - self._last_log.get(key, 0)) >= self._log_interval:
                self._last_log[key] = collected_frames
                _log = True
            else:
                _log = False
            method = LOGGER_METHODS.get(key, "log_scalar")
            if _log and self.logger is not None:
                getattr(self.logger, method)(key, item, step=iter)
            if method == "log_scalar" and self.progress_bar and log_pbar:
                if isinstance(item, torch.Tensor):
                    item = item.item()
                self._pbar_str[key] = item

    def optim_steps(self, batch: TensorDictBase) -> None:
        average_losses = None

        self._pre_optim_hook()
        # Compute minibatch
        n_envs = batch.shape[0]
        n_transitions_per_env = batch.shape[1]
        batch_size = n_envs * n_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        indices = torch.randperm(self.num_mini_batches * mini_batch_size, requires_grad=False, device=batch.device)
        flat_batch = batch.flatten(0, 1)

        for _ in range(self.optim_steps_per_batch):
            self._optim_count += 1
            # Initialize average_losses for this optimization step
            average_losses = None
            # Compute mini-batch
            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                end = start + mini_batch_size
                batch_indices = indices[start:end]
                sub_batch = flat_batch[batch_indices] 
                losses_td = self.loss_module(sub_batch)
                if self.lr_schedule == "adaptive":
                    desired_kl = self.loss_module.desired_kl
                    kl = losses_td["kl"]
                    if kl > desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl < desired_kl / 2.0 and kl > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate
                batch.set("learning_rate", torch.tensor(self.learning_rate).expand(batch.shape))
                self._post_loss_hook(sub_batch)
                losses_detached = self._optimizer_hook(losses_td)
                # Update average_losses after each mini-batch
                if average_losses is None:
                    average_losses = losses_detached.copy()
                else:
                    for key, item in losses_detached.items():
                        # compute updated average loss
                        val = average_losses.get(key)
                        average_losses.set(key, val * i / (i + 1) + item / (i + 1))

            self._post_optim_hook()
            self._post_optim_log(batch)

            del sub_batch, losses_td, losses_detached
        if self.optim_steps_per_batch > 0:
            self._log(optim_steps=self._optim_count, **average_losses,)

class WandbLoggerWrapper(WandbLogger):
    def __init__(self, 
                 exp_name: str,
                 offline: bool = False,
                 save_dir: str = None,
                 id: str = None,
                 project: str = None, 
                 **kwargs
        ):
        super().__init__(exp_name=exp_name, offline=offline, save_dir=save_dir, id=id, project=project, **kwargs)
    
    def log_model(self, model_path, iter):
        import wandb
        # Split the file name and extension
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def log_config(self, env_cfg) -> None:
        import wandb 
        wandb.config.update({"env_cfg" : asdict(env_cfg)})


    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Logs a scalar value to wandb.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged.
                Defaults to None.
        """
        if step is not None:
            self.experiment.log({name: value, "trainer/step": step}, step=step)
        else:
            self.experiment.log({name: value})