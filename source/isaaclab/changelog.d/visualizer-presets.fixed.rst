* Preserved shared renderer references inside tuples during launch.
* Applied distributed and runtime-selected devices to the ``SimulationCfg`` found in the config tree,
  including standalone configs and configs nested in dictionaries or lists, in ``launch_simulation``.
