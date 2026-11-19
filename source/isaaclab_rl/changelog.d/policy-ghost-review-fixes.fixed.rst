* Fixed Rollout Ghosts restarts dropping the original playback arguments, ``--policy_ghosts`` rejecting a run with
  several viewers such as ``--viz newton,viser``, and recurrent ghost policies keeping their memory across an episode
  reset. Tasks whose cameras render with OVRTX are now rejected, because the ghost overlay would shift their scene.
