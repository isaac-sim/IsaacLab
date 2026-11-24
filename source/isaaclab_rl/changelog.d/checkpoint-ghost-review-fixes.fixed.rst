* Fixed ``--checkpoint_ghosts`` rejecting a run with several viewers such as ``--viz newton,viser``.
* Fixed ``--checkpoint latest`` and ``--checkpoint best`` failing in RSL-RL playback after the Checkpoint Ghosts change
  dropped the ``library`` and ``agent`` arguments of the checkpoint selector.
* Fixed Checkpoint Ghosts leaving the environment's reset and command hooks installed when no Newton viewer could
  attach.
