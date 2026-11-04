* Added ``--policy_ghosts``, ``--policy_ghost_slots``, and ``--policy_ghost_root`` to RSL-RL playback to overlay
  several checkpoints of a task in the Newton GL viewer. ``--policy_ghost_slots`` defaults to eight and has no effect
  unless ``--policy_ghosts`` is set; enabling the feature lowers the step rate by roughly 10% to 20% with eight slots shown in a
  four-environment test.
* Added a restart of RSL-RL playback, requested from the Newton GL viewer HUD, that switches Policy Ghosts on or off.
