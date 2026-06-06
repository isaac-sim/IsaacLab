Changed
^^^^^^^

* **Breaking:** Renamed the reach Gym environment IDs to drop the ``-v0`` version suffix. The
  robot name is kept in the ID and the manager-based workflow carries no workflow suffix. Update
  ``gym.make`` / ``--task`` calls:

  * ``Isaac-Reach-Franka-v0`` → ``Isaac-Reach-Franka``.
  * ``Isaac-Reach-Franka-Play-v0`` → ``Isaac-Reach-Franka-Play``.
  * ``Isaac-Reach-Franka-OSC-v0`` → ``Isaac-Reach-Franka-OSC``.
  * ``Isaac-Reach-Franka-OSC-Play-v0`` → ``Isaac-Reach-Franka-OSC-Play``.
  * ``Isaac-Reach-UR10-v0`` → ``Isaac-Reach-UR10``.
  * ``Isaac-Reach-UR10-Play-v0`` → ``Isaac-Reach-UR10-Play``.
* Renamed the RSL-RL experiment name for the Franka reach task from ``franka_reach`` to
  ``reach_franka`` so it matches the other Franka reach agent configs and the UR10 reach task.
* **Breaking:** Moved the reach pose-tracking reward terms ``position_command_error``,
  ``position_command_error_tanh`` and ``orientation_command_error`` to the shared
  :mod:`isaaclab.envs.mdp` terms and removed the ``isaaclab_tasks.core.reach.mdp`` package. Import
  these terms from :mod:`isaaclab.envs.mdp` instead, e.g. replace
  ``import isaaclab_tasks.core.reach.mdp as mdp`` with ``import isaaclab.envs.mdp as mdp``.
