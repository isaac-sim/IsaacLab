* **Breaking:** :meth:`~isaaclab.envs.ManagerBasedRLEnv.step` resets terminated environments through a boolean
  mask on every step instead of compacting ``reset_buf`` into indices, so the step no longer waits on the device.
  :meth:`~isaaclab.envs.ManagerBasedEnv._reset_mask` implements the reset and ``_reset_idx`` wraps it; subclasses
  that override ``_reset_idx`` keep receiving indices.
* **Breaking:** Built-in event, command, curriculum, action, observation, reward and modifier terms select
  environments with ``env_mask`` (a boolean tensor of shape ``(num_envs,)``) instead of ``env_ids``. They compute for
  all environments and leave unselected ones unchanged. Terms that still take ``env_ids`` keep working: managers pass
  them indices, which synchronizes the device, and skip the call when no environment is selected.
* **Breaking:** Command metrics, the rewards' episodic sums and curriculum states are logged as tensors averaged over
  the reset environments. A logged value keeps its previous value on steps where no environment reset.
* Mask-native curriculum terms run on every step. ``modify_reward_weight`` therefore switches at the first step
  after ``num_steps`` instead of at the next reset after it.
* Action terms set joint targets through ``articulation.actuators.target_command`` instead of the deprecated
  ``set_joint_*_target_index`` methods, which warned on every step.
