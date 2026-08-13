Fixed
^^^^^

* Fixed the skrl entrypoints reporting an observation/action space name as the
  training algorithm. ``--agent`` keys do not all encode an algorithm: the
  Cartpole showcase tasks key theirs by space combination, so
  ``skrl_box_discrete_cfg_entry_point`` was decoded as the algorithm
  ``box_discrete``. That value reached run manifests, benchmark KPI metadata, and
  log directory names, and switched off the MARL-to-single-agent conversion that
  PPO requires. Only ``amp``, ``ppo``, ``ippo``, and ``mappo`` are now read as
  algorithms; any other key keeps the algorithm from ``--algorithm``.

Changed
^^^^^^^

* Added :func:`~isaaclab_rl.entrypoints.common.resolve_skrl_agent_entry_point`,
  replacing the four copies of the skrl agent/algorithm resolution in the train,
  play, and benchmark entrypoints.
