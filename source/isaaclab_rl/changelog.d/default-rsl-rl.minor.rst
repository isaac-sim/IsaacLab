Changed
^^^^^^^

* **Breaking:** Changed the ``train`` and ``play`` CLI commands to use a task's
  registered default RL library when ``--rl_library`` is omitted. Pass
  ``--rl_library`` explicitly to select a different library.
