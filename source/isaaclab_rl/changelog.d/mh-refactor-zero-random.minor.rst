Added
^^^^^

* Added the checkpoint-free playback workflows :func:`~isaaclab_rl.zero_agent` and
  :func:`~isaaclab_rl.random_agent`, their :class:`~isaaclab_rl.SimpleAgentRequest`
  parameters, and the :func:`~isaaclab_rl.run_zero_agent_cli` and
  :func:`~isaaclab_rl.run_random_agent_cli` command-line dispatchers.

Changed
^^^^^^^

* Changed ``scripts/environments/zero_agent.py`` and ``scripts/environments/random_agent.py``
  to thin delegates of the reusable entrypoints in :mod:`isaaclab_rl.entrypoints`. Their
  command-line interface is unchanged.
* Changed the zero and random agents to simulate 4 environments by default instead of the
  task's configured environment count. Pass ``--num_envs`` to select a different count.
