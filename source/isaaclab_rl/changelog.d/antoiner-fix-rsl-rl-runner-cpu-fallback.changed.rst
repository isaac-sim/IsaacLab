* **Breaking:** Changed the RSL-RL, RL-Games, and Stable-Baselines3 train, play, and export workflows to run the
  agent on the device :func:`~isaaclab.app.launch_simulation` resolved for the simulation, as skrl and TorchRL
  already did. Outside distributed runs they used the agent configuration's device, ``cuda:0`` by default, so
  ``--device cpu`` trained on ``cuda:0``, ``--device cuda:1`` trained on GPU 0, and hosts without CUDA, such as
  macOS, failed. Pass ``--device`` to choose where both the simulation and the agent run; the agent
  configuration's ``device`` is now overwritten by these workflows.
