* **Breaking:** :class:`~isaaclab.envs.ManagerBasedRLEnv` and :class:`~isaaclab.envs.DirectRLEnv` capture their step
  by default on Newton. Returned observations, rewards and dones are reused buffers; copy them to keep them, or set
  ``capture_step = False``. Terms that branch on host state changing between steps must set
  ``supports_graph_capture = False``.
* **Breaking:** The step resets terminated environments through a boolean mask instead of indices, so it no longer
  synchronizes the device. Built-in terms take ``env_mask`` (a :class:`~isaaclab.utils.warp.ProxyArray`) instead of
  ``env_ids``; terms that still take ``env_ids`` keep working at the cost of a device sync. Overrides of
  ``_reset_idx`` still receive indices.
* **Breaking:** Logged command metrics, episodic reward sums and curriculum states are tensors averaged over reset
  environments, and ``modify_reward_weight`` switches at the first step after ``num_steps``.
* **Breaking:** Physics manager runtime methods are instance methods; use ``sim.physics_manager``.
