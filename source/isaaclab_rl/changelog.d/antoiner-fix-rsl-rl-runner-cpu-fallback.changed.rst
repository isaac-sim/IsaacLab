* **Breaking:** Changed the RSL-RL, RL-Games, and Stable-Baselines3 train, play, and export workflows to run the
  agent on the device :func:`~isaaclab.app.launch_simulation` resolved for the simulation, as skrl and TorchRL
  already did. Outside distributed runs they used the agent configuration's device, ``cuda:0`` by default, so
  ``--device cpu`` trained on ``cuda:0`` and ``--device cuda:1`` trained on GPU 0. Pass ``--device`` to choose
  where both the simulation and the agent run; the agent configuration's ``device`` is now overwritten.
* Changed each library's agent configuration overrides to live in one shared function that every train, play,
  export, and benchmark workflow calls: ``update_rsl_rl_cfg``, and the new ``update_rl_games_cfg`` and
  ``update_sb3_cfg``.
