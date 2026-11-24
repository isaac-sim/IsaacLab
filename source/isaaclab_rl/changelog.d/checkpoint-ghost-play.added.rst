* Added ``--checkpoint_ghosts`` and ``--checkpoint_ghost_slots`` to RSL-RL, rl_games, skrl, and Stable-Baselines3 playback, the Newton GL viewer's checkpoint
  visualization. ``--num_envs`` is the total simulated and is split evenly among the checkpoints shown: the reference
  starts with all of them, and adding a ghost resizes every slot (32 environments give 16 each with two checkpoints).
  ``--checkpoint_ghost_slots`` caps the checkpoints shown at once (default eight, at most ``--num_envs``).
  ``--checkpoint_ghost_root`` searches another directory for checkpoints. Plain playback allocates nothing for the feature.
* Rejected ``--checkpoint_ghosts`` for tasks whose curriculum term sets the reset state, since ghosts disable curriculum
  terms.
