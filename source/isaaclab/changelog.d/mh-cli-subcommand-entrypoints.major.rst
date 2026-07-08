Changed
^^^^^^^

* **Breaking:** Removed the standalone ``train``, ``play``, and ``train_multigpu``
  console scripts. These are now subcommands of the ``isaaclab`` entry point
  (``isaaclab train`` / ``isaaclab play`` / ``isaaclab train_multigpu``, e.g.
  ``uv run isaaclab train ...``) so they no longer clash with ``train``/``play``
  commands provided by other installed packages.
