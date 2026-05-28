Changed
^^^^^^^

* Reduced default ``max_iterations`` for the Shadow-Hand Vision (in-hand
  cube reposing with camera observations) PPO agents from ``50000`` to
  ``5000`` in
  :class:`~isaaclab_tasks.direct.shadow_hand.agents.rsl_rl_ppo_cfg.ShadowHandVisionFFPPORunnerCfg`
  and the matching ``rl_games`` ``max_epochs`` in
  ``shadow_hand/agents/rl_games_ppo_vision_cfg.yaml``. The 50k default
  was a 10-30 hour wall-clock job on a current GPU; the prior empirical
  signal showed convergence well before 5k iterations. Operators
  who need the longer schedule can still pass ``--max_iterations 50000``
  on the CLI.
