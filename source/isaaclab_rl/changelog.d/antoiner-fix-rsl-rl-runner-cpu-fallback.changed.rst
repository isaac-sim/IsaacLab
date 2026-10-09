* **Breaking:** Changed the RSL-RL, RL-Games, and Stable-Baselines3 workflows to run the agent on the simulation's
  device, as skrl and TorchRL do. They ignored ``--device`` for the agent outside distributed runs and used the
  agent configuration's ``cuda:0``; pass ``--device`` to choose where both run.
