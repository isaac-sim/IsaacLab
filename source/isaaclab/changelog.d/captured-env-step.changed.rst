* **Breaking:** :class:`~isaaclab.envs.ManagerBasedRLEnv` captures its step by default on Newton physics. The returned
  observations, rewards and done flags are then buffers that the next step overwrites; copy them to keep them. Set
  ``capture_step = False`` to step eagerly. Terms that branch on host state that changes between steps must set
  ``supports_graph_capture = False``; ``modify_reward_weight``, ``modify_env_param``, ``modify_term_cfg`` and
  ``DifficultyScheduler`` do.
* ``set_term_cfg`` of the reward, termination and event managers, and curriculum changes made by ``modify_env_param``,
  record the captured step again. ``modify_reward_weight`` writes the weight only when it changes.
