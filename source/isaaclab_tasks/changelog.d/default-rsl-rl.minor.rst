Changed
^^^^^^^

* **Breaking:** Changed core tasks to use RSL-RL when ``--rl_library`` is omitted, except the multi-agent Pendulum task, which uses SKRL.
  Pass ``--rl_library`` explicitly to select a different RL library.
