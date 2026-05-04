Added
^^^^^

* Added Newton rough terrain support for the G1 biped locomotion velocity
  env. The only engine-specific change is a ~1.7x ``max_iterations`` preset on
  :class:`~isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg.G1RoughPPORunnerCfg`
  (Newton = 5000, PhysX = 3000). PhysX saturates near iter 3000 on both
  reward (≈ +18) and episode length (≈ 980) and does not meaningfully
  improve further; Newton reaches the same (reward, ep_len) quality at
  iter 5000. The iteration budget is bumped rather than tuning physics
  or reward terms.
