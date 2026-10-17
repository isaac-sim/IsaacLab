* Preserved shared renderer references and unchanged named tuples during launch.
* Applied distributed and runtime-selected devices to the ``SimulationCfg`` found in the config tree,
  including standalone configs and configs nested in dictionaries or lists, in ``launch_simulation``.
* Reused each recorded perspective producer across multiple recorders and enabled streaming views requested by video sources.
