* Fixed ``--checkpoint_ghosts`` rejecting a run with
  several viewers such as ``--viz newton,viser``, and recurrent ghost policies keeping their memory across an episode
  reset.
* Fixed ``--checkpoint latest`` and ``--checkpoint best`` failing in RSL-RL playback after the Checkpoint Ghosts change
  dropped the ``library`` and ``agent`` arguments of the checkpoint selector.
* Fixed Checkpoint Ghosts leaving the environment's reset and command hooks installed when no Newton viewer could
  attach, and allocating a new action tensor on every playback step.
* Fixed Checkpoint Ghosts failing with recurrent policies when a ghost is added or removed, because the new slice size
  no longer matched the hidden state. Adding, changing, or removing a ghost on an empty or unknown slot, and a failed
  re-split, are now handled without ending the session.
