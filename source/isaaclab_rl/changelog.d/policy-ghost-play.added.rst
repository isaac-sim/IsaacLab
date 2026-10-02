* Added ``--policy_ghosts``, ``--policy_ghost_slots``, and ``--policy_ghost_root`` to RSL-RL playback to overlay
  several checkpoints of a task in the Newton GL viewer. ``--policy_ghost_slots`` defaults to eight and has no effect
  unless ``--policy_ghosts`` is set; enabling the feature costs about 8% of the step rate in a four-environment test.
