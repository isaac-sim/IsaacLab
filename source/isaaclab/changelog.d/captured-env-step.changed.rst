* **Breaking:** :class:`~isaaclab.envs.ManagerBasedRLEnv` captures its step by default on Newton physics. The returned
  observations, rewards and done flags are then buffers that the next step overwrites; copy them to keep them. Set
  ``capture_step = False`` to step eagerly. Terms that branch on host state that changes between steps must set
  ``supports_graph_capture = False``; ``modify_reward_weight``, ``modify_env_param``, ``modify_term_cfg`` and
  ``DifficultyScheduler`` do.
* ``set_term_cfg`` of the reward, termination and event managers, and curriculum changes made by ``modify_env_param``,
  record the captured step again. ``modify_reward_weight`` writes the weight only when it changes.
* A captured step stays equivalent to eager stepping when terms replace their buffers each step
  (``self.x = f(self.x)``): each replay copies the new value into the buffer that earlier recorded operations read.
  An eagerly run operation that replaces environment, manager or term buffers every step makes the environment step
  eagerly, naming the buffers. Operations that synchronize the device run eagerly in place, and a subclass override of
  ``_reset_mask`` runs eagerly as a whole.
* Episode logs in ``extras["log"]`` keep one tensor per key, updated in place.
