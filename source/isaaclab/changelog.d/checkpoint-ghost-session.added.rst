* Added :class:`~isaaclab.visualizers.checkpoint_ghost_session.CheckpointGhostSession`, which steps one environment whose
  slices are driven by different checkpoints of one task. It splits the environments among the loaded checkpoints,
  resets them together from the reference's state, and serves the checkpoint browser. It needs only an environment
  wrapper and a policy factory, so a library's play entry point supplies the wiring. Only RSL-RL provides it for now.
* Fixed the session failing with recurrent policies when a ghost is added or removed, and handled a ghost added,
  changed, or removed on an empty or unknown slot, and a failed re-split, without ending the session. It reuses one
  action tensor for every step.
