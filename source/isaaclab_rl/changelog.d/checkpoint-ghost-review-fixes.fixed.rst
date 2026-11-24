* Fixed ``--checkpoint_ghosts`` rejecting a run with
  several viewers such as ``--viz newton,viser``, and recurrent ghost policies keeping their memory across an episode
  reset. Tasks whose cameras render with OVRTX are now rejected, since Checkpoint Ghosts do not support them yet.
